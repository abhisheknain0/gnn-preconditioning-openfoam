from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from gnn_preconditioner.inference.openfoam_fields import read_openfoam_nonuniform_vector_internal_field
from gnn_preconditioner.inference.openfoam_fields import read_openfoam_scalar_internal_field

from .common import MeshProfile
from .gmsh_generate import generate_gmsh_geo, generate_mesh_with_gmsh


_FORMAT_RE = re.compile(r"^\s*format\s+(\w+)\s*;", re.MULTILINE)


@dataclass(frozen=True)
class BoundaryEntry:
    name: str
    patch_type: str
    n_faces: int
    start_face: int


@dataclass(frozen=True)
class MeshCalibrationResult:
    case_id: int
    case_name: str
    n_cells: int
    case_dir: str



def run_logged(cmd: list[str], cwd: Path, log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            return subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=check, text=True)
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def _extract_list_body(text: str) -> tuple[int, str]:
    match = re.search(r"\n\s*(\d+)\s*\n\s*\(", text)
    if match is None:
        raise ValueError("Could not find OpenFOAM list header")
    count = int(match.group(1))
    start = match.end()
    end = text.rfind(")")
    if end < start:
        raise ValueError("Could not find closing ')' for OpenFOAM list")
    return count, text[start:end]


def _header_format(path: Path) -> str | None:
    text = path.read_bytes()[:4096].decode("latin-1", errors="ignore")
    match = _FORMAT_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip().lower()


def parse_points(path: Path) -> torch.Tensor:
    text = path.read_text(encoding="utf-8", errors="ignore")
    expected_n, body = _extract_list_body(text)
    vecs = re.findall(r"\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)", body)
    if len(vecs) != expected_n:
        raise ValueError(f"{path}: parsed {len(vecs)} points, expected {expected_n}")
    out = torch.empty((expected_n, 3), dtype=torch.float32)
    for i, (x, y, z) in enumerate(vecs):
        out[i, 0] = float(x)
        out[i, 1] = float(y)
        out[i, 2] = float(z)
    return out


def parse_label_list(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    expected_n, body = _extract_list_body(text)
    vals = [int(v) for v in re.findall(r"-?\d+", body)]
    if len(vals) != expected_n:
        raise ValueError(f"{path}: parsed {len(vals)} labels, expected {expected_n}")
    return vals


def parse_faces(path: Path) -> list[list[int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    expected_n, body = _extract_list_body(text)
    faces: list[list[int]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.fullmatch(r"(\d+)\(([^()]*)\)", line)
        if match is None:
            continue
        n = int(match.group(1))
        pts = [int(v) for v in match.group(2).split()]
        if len(pts) != n:
            raise ValueError(f"{path}: malformed face line {line!r}")
        faces.append(pts)
    if len(faces) != expected_n:
        raise ValueError(f"{path}: parsed {len(faces)} faces, expected {expected_n}")
    return faces


def parse_boundary(path: Path) -> list[BoundaryEntry]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    expected_n, body = _extract_list_body(text)
    entries: list[BoundaryEntry] = []
    block_re = re.compile(r"(\w+)\s*\{(.*?)\}", re.DOTALL)
    for name, raw_block in block_re.findall(body):
        type_match = re.search(r"type\s+(\w+)\s*;", raw_block)
        n_faces_match = re.search(r"nFaces\s+(\d+)\s*;", raw_block)
        start_face_match = re.search(r"startFace\s+(\d+)\s*;", raw_block)
        if type_match is None or n_faces_match is None or start_face_match is None:
            continue
        entries.append(
            BoundaryEntry(
                name=name,
                patch_type=type_match.group(1),
                n_faces=int(n_faces_match.group(1)),
                start_face=int(start_face_match.group(1)),
            )
        )
    if len(entries) != expected_n:
        raise ValueError(f"{path}: parsed {len(entries)} boundary entries, expected {expected_n}")
    return entries


def face_area_vector(points: torch.Tensor, face: list[int]) -> torch.Tensor:
    verts = points[torch.tensor(face, dtype=torch.long)]
    if verts.shape[0] < 3:
        raise ValueError("Face must have at least 3 vertices")
    ref = verts[0]
    area_vec = torch.zeros(3, dtype=torch.float32)
    for i in range(1, verts.shape[0] - 1):
        area_vec = area_vec + 0.5 * torch.cross(verts[i] - ref, verts[i + 1] - ref, dim=0)
    return area_vec


def face_center(points: torch.Tensor, face: list[int]) -> torch.Tensor:
    verts = points[torch.tensor(face, dtype=torch.long)]
    return verts.mean(dim=0)


def build_internal_face_geometry(
    points: torch.Tensor,
    faces: list[list[int]],
    neighbour: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    n_internal = len(neighbour)
    sf = torch.empty((n_internal, 3), dtype=torch.float32)
    cf = torch.empty((n_internal, 3), dtype=torch.float32)
    for fi in range(n_internal):
        face = faces[fi]
        sf[fi] = face_area_vector(points, face)
        cf[fi] = face_center(points, face)
    return sf, cf


def classify_boundary_mask(entries: list[BoundaryEntry], owner: list[int], n_cells: int) -> np.ndarray:
    mask = np.zeros((n_cells, 4), dtype=np.uint8)
    for entry in entries:
        if entry.name.lower() == "inlet":
            col = 0
        elif entry.name.lower() == "outlet":
            col = 1
        elif entry.name in {"front", "back"} or entry.patch_type == "empty":
            col = 3
        else:
            col = 2
        start = entry.start_face
        end = start + entry.n_faces
        for face_i in range(start, end):
            mask[owner[face_i], col] = 1
    return mask


def compute_internal_face_flux_xy(
    velocity: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    sf_internal: torch.Tensor,
) -> torch.Tensor:
    u_face = 0.5 * (velocity.index_select(0, owner) + velocity.index_select(0, neighbour))
    return (u_face[:, :2] * sf_internal[:, :2]).sum(dim=-1)


def compute_internal_face_pressure_jump(
    pressure_gauge: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> torch.Tensor:
    return pressure_gauge.index_select(0, owner) - pressure_gauge.index_select(0, neighbour)


def patch_boundary_file(boundary_file: Path, front_back_mode: str = "empty") -> None:
    content = boundary_file.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if "physicalType" not in line]
    content = "\n".join(lines) + "\n"
    patch_fixes = {
        "front": (front_back_mode, front_back_mode),
        "back": (front_back_mode, front_back_mode),
        "fixedWalls": ("wall", "wall"),
    }
    for patch_name, (patch_type, in_group) in patch_fixes.items():
        pattern = rf"({patch_name}\s*\{{[^}}]*?type\s+)\w+(\s*;)"
        content = re.sub(pattern, rf"\1{patch_type}\2", content, flags=re.DOTALL)
        block_pattern = re.compile(rf"({patch_name}\s*\{{)(.*?)(\n\s*\}})", re.DOTALL)
        match = block_pattern.search(content)
        if match is None:
            continue
        block = match.group(2)
        if "inGroups" not in block:
            inject = f"\n        inGroups        List<word> 1({in_group});"
            content = content[: match.end(1)] + inject + content[match.end(1) :]
    boundary_file.write_text(content, encoding="utf-8")


def replace_patch_value(path: Path, patch_name: str, new_value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    patch_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == patch_name or (stripped.startswith(patch_name) and "{" in stripped):
            patch_start = i
            break
    if patch_start is None:
        raise ValueError(f"Patch {patch_name!r} not found in {path}")

    brace_count = 0
    patch_end = -1
    for i in range(patch_start, len(lines)):
        brace_count += lines[i].count("{") - lines[i].count("}")
        if brace_count == 0 and i > patch_start:
            patch_end = i
            break

    for i in range(patch_start + 1, patch_end):
        stripped = lines[i].strip()
        if stripped.startswith("value") and "uniform" in stripped:
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            lines[i] = f"{indent}value           uniform {new_value};"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replace_transport_nu(path: Path, nu: float) -> None:
    content = path.read_text(encoding="utf-8")
    pattern = r"(nu\s+\[[^\]]+\]\s+)([\d.eE+-]+)(\s*;)"
    updated = re.sub(pattern, rf"\g<1>{nu}\g<3>", content)
    path.write_text(updated, encoding="utf-8")


def update_control_dict(control_dict: Path, end_time: float, write_interval: float) -> None:
    content = control_dict.read_text(encoding="utf-8")
    replacements = {
        r"(endTime\s+)([^;]+)(;)": f"\\g<1>{end_time}\\g<3>",
        r"(writeInterval\s+)([^;]+)(;)": f"\\g<1>{write_interval}\\g<3>",
        r"(startFrom\s+)([^;]+)(;)": "\\g<1>startTime\\g<3>",
        r"(startTime\s+)([^;]+)(;)": "\\g<1>0\\g<3>",
    }
    for pattern, repl in replacements.items():
        content = re.sub(pattern, repl, content)
    control_dict.write_text(content, encoding="utf-8")


def prepare_case_directory(base_dir: Path, target_dir: Path, overwrite: bool = False) -> Path:
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Target case directory already exists: {target_dir}")
        shutil.rmtree(target_dir)
    shutil.copytree(base_dir, target_dir)
    return target_dir


def generate_mesh_for_case(
    case_dir: Path,
    ymid: float,
    xmid: float,
    ymax: float,
    profile: MeshProfile,
    front_back_mode: str = "empty",
    check_mesh: bool = True,
) -> None:
    geo_file = case_dir / "system" / "geometry.geo"
    msh_file = case_dir / "system" / "geometry.msh"
    generate_gmsh_geo(geo_file, ymid=ymid, xmid=xmid, ymax=ymax, profile=profile)
    generate_mesh_with_gmsh(geo_file, msh_file)
    run_logged(["gmshToFoam", str(msh_file)], cwd=case_dir, log_path=case_dir / "log.gmshToFoam")
    patch_boundary_file(case_dir / "constant" / "polyMesh" / "boundary", front_back_mode=front_back_mode)
    if check_mesh:
        run_logged(["checkMesh"], cwd=case_dir, log_path=case_dir / "log.checkMesh")
    run_logged(["renumberMesh", "-overwrite"], cwd=case_dir, log_path=case_dir / "log.renumberMesh", check=False)


def run_solver(case_dir: Path, solver: str = "pimpleFoam") -> None:
    run_logged([solver], cwd=case_dir, log_path=case_dir / f"log.{solver}")


def n_cells_from_owner(owner_path: Path, neighbour_path: Path) -> int:
    owner = parse_label_list(owner_path)
    neighbour = parse_label_list(neighbour_path)
    return max(owner + neighbour) + 1 if neighbour else max(owner) + 1


def collect_numeric_time_dirs(case_dir: Path, include_zero: bool = True) -> list[Path]:
    times: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if not include_zero and math.isclose(value, 0.0):
            continue
        times.append((value, child))
    times.sort(key=lambda item: item[0])
    return [p for _, p in times]


def ensure_cell_geometry_ascii(case_dir: Path) -> tuple[Path, Path]:
    c_path = case_dir / "0" / "C"
    v_candidates = [
        case_dir / "0" / "Vc",
        case_dir / "0" / "V",
    ]
    existing_v_path = next((path for path in v_candidates if path.exists()), None)
    needs_rebuild = True
    if (
        c_path.exists()
        and existing_v_path is not None
        and _header_format(c_path) != "binary"
        and _header_format(existing_v_path) != "binary"
    ):
        needs_rebuild = False
    if needs_rebuild:
        if c_path.exists():
            c_path.unlink()
        for path in v_candidates:
            if path.exists():
                path.unlink()
        run_logged(["postProcess", "-func", "writeCellCentres", "-time", "0"], cwd=case_dir, log_path=case_dir / "log.writeCellCentres")
        run_logged(["postProcess", "-func", "writeCellVolumes", "-time", "0"], cwd=case_dir, log_path=case_dir / "log.writeCellVolumes")
        existing_v_path = next((path for path in v_candidates if path.exists()), None)
    if existing_v_path is None:
        raise FileNotFoundError(f"No cell-volume field written in {case_dir / '0'}; expected one of {[p.name for p in v_candidates]}")
    return c_path, existing_v_path


def load_case_static_geometry(case_dir: Path) -> dict[str, object]:
    c_path, v_path = ensure_cell_geometry_ascii(case_dir)
    points = parse_points(case_dir / "constant" / "polyMesh" / "points")
    faces = parse_faces(case_dir / "constant" / "polyMesh" / "faces")
    owner_all = parse_label_list(case_dir / "constant" / "polyMesh" / "owner")
    neighbour = parse_label_list(case_dir / "constant" / "polyMesh" / "neighbour")
    owner = owner_all[: len(neighbour)]
    boundary = parse_boundary(case_dir / "constant" / "polyMesh" / "boundary")
    cell_centers = read_openfoam_nonuniform_vector_internal_field(c_path).to(torch.float32)
    cell_volumes = read_openfoam_scalar_internal_field(v_path, expected_size=int(cell_centers.shape[0])).to(torch.float32)
    sf_internal, cf_internal = build_internal_face_geometry(points, faces, neighbour)
    boundary_mask = classify_boundary_mask(boundary, owner_all, int(cell_centers.shape[0]))
    return {
        "points": points,
        "faces": faces,
        "owner": owner,
        "owner_all": owner_all,
        "neighbour": neighbour,
        "boundary": boundary,
        "cell_centers": cell_centers,
        "cell_volumes": cell_volumes,
        "Sf_internal": sf_internal,
        "Cf_internal": cf_internal,
        "boundary_mask": boundary_mask,
    }


def calibrate_profile_counts(
    base_case_dir: Path,
    profile: MeshProfile,
    cases: Iterable[dict[str, object]],
    front_back_mode: str = "empty",
) -> list[MeshCalibrationResult]:
    results: list[MeshCalibrationResult] = []
    with tempfile.TemporaryDirectory(prefix="iterreduce_meshcal_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        for case in cases:
            case_id = int(case["case_id"])
            case_name = str(case.get("canonical_case_name") or f"case_{case_id:03d}")
            case_dir = tmp_root / case_name
            prepare_case_directory(base_case_dir, case_dir, overwrite=True)
            update_control_dict(case_dir / "system" / "controlDict", end_time=0.0, write_interval=0.01)
            replace_patch_value(case_dir / "0" / "U", "inlet", f"({float(case['Uinlet'])} 0 0)")
            replace_transport_nu(case_dir / "constant" / "transportProperties", float(case["nu"]))
            generate_mesh_for_case(
                case_dir,
                ymid=float(case["ymid"]),
                xmid=float(case["xmid"]),
                ymax=float(case["ymax"]),
                profile=profile,
                front_back_mode=front_back_mode,
                check_mesh=False,
            )
            n_cells = n_cells_from_owner(
                case_dir / "constant" / "polyMesh" / "owner",
                case_dir / "constant" / "polyMesh" / "neighbour",
            )
            results.append(
                MeshCalibrationResult(
                    case_id=case_id,
                    case_name=case_name,
                    n_cells=n_cells,
                    case_dir=str(case_dir),
                )
            )
    return results
