from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ITERREDUCE_ROOT = PROJECT_ROOT / "gnn_preconditioner"
DEFAULT_CASE_OUTPUT_ROOT = ITERREDUCE_ROOT / "data_generation" / "cases" / "pilot_v1"


_TRAJECTORY_RE = re.compile(r"^(.*?)(\d+)$")


def trajectory_numeric_id(key: str) -> int | None:
    match = _TRAJECTORY_RE.match(str(key))
    if match is None:
        return None
    return int(match.group(2))


def trajectory_sort_key(key: str) -> tuple[str, int | float, str]:
    key = str(key)
    numeric = trajectory_numeric_id(key)
    if numeric is None:
        return (key, float("inf"), key)
    prefix = key[: key.rfind(str(numeric))]
    return (prefix, numeric, key)


def sorted_trajectory_keys(keys: Iterable[str]) -> list[str]:
    return sorted((str(k) for k in keys), key=trajectory_sort_key)


@dataclass(frozen=True)
class SourceCaseMeta:
    case_id: int
    trajectory_key: str
    Uinlet: float
    xmid: float
    ymid: float
    ymax: float
    nu: float
    iterreduce_case_dir: str | None

    def canonical_case_name(self) -> str:
        return (
            f"case_{self.case_id:03d}_U{self.Uinlet:.3f}_"
            f"H{self.ymid:.4f}_L{self.xmid:.4f}"
        )

    def as_json(self) -> dict[str, object]:
        data = {
            "case_id": self.case_id,
            "trajectory_key": self.trajectory_key,
            "Uinlet": self.Uinlet,
            "xmid": self.xmid,
            "ymid": self.ymid,
            "ymax": self.ymax,
            "nu": self.nu,
            "canonical_case_name": self.canonical_case_name(),
        }
        if self.iterreduce_case_dir is not None:
            data["iterreduce_case_dir"] = self.iterreduce_case_dir
        else:
            try:
                case_root = DEFAULT_CASE_OUTPUT_ROOT.relative_to(PROJECT_ROOT)
            except ValueError:
                case_root = DEFAULT_CASE_OUTPUT_ROOT
            data["iterreduce_case_dir"] = str(case_root / self.canonical_case_name())
        return data


@dataclass(frozen=True)
class MeshProfile:
    profile_name: str
    mesh_size_coarse: float
    mesh_size_fine: float
    dist_min: float
    dist_max: float
    n_nodes_by_edge: int

    @classmethod
    def from_json(cls, path: str | Path) -> "MeshProfile":
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            profile_name=str(data["profile_name"]),
            mesh_size_coarse=float(data["mesh_size_coarse"]),
            mesh_size_fine=float(data["mesh_size_fine"]),
            dist_min=float(data["dist_min"]),
            dist_max=float(data["dist_max"]),
            n_nodes_by_edge=int(data["n_nodes_by_edge"]),
        )

    def scaled(self, factor: float, name: str | None = None) -> "MeshProfile":
        return MeshProfile(
            profile_name=name or f"{self.profile_name}_x{factor:.3f}",
            mesh_size_coarse=self.mesh_size_coarse * factor,
            mesh_size_fine=self.mesh_size_fine * factor,
            dist_min=self.dist_min * factor,
            dist_max=self.dist_max * factor,
            n_nodes_by_edge=self.n_nodes_by_edge,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "profile_name": self.profile_name,
            "mesh_size_coarse": self.mesh_size_coarse,
            "mesh_size_fine": self.mesh_size_fine,
            "dist_min": self.dist_min,
            "dist_max": self.dist_max,
            "n_nodes_by_edge": self.n_nodes_by_edge,
        }


def resolve_case_dir(case_root: str | Path, case_id: int) -> Path | None:
    case_root = Path(case_root).expanduser().resolve()
    matches = sorted(case_root.glob(f"case_{case_id:03d}_*"))
    if not matches:
        return None
    return matches[0]


def load_source_case_meta(source_h5: str | Path, case_root: str | Path) -> list[SourceCaseMeta]:
    source_h5 = Path(source_h5).expanduser().resolve()
    case_root = Path(case_root).expanduser().resolve()
    rows: list[SourceCaseMeta] = []
    with h5py.File(source_h5, "r") as f:
        for key in sorted_trajectory_keys(f.keys()):
            case_id = trajectory_numeric_id(key)
            if case_id is None:
                continue
            grp = f[key]
            case_dir = resolve_case_dir(case_root, case_id)
            if case_dir is not None:
                try:
                    iterreduce_case_dir = str(case_dir.relative_to(PROJECT_ROOT))
                except ValueError:
                    iterreduce_case_dir = str(case_dir)
            else:
                iterreduce_case_dir = None
            rows.append(
                SourceCaseMeta(
                    case_id=case_id,
                    trajectory_key=key,
                    Uinlet=float(np.asarray(grp["Uinlet"]).reshape(-1)[0]),
                    xmid=float(np.asarray(grp["xmid"]).reshape(-1)[0]),
                    ymid=float(np.asarray(grp["ymid"]).reshape(-1)[0]),
                    ymax=float(np.asarray(grp["ymax"]).reshape(-1)[0]),
                    nu=float(np.asarray(grp["nu"]).reshape(-1)[0]),
                    iterreduce_case_dir=iterreduce_case_dir,
                )
            )
    return rows


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {path} must be a list or {{'cases': [...]}}")
    return data


def resolve_iterreduce_case_dir(case_root: str | Path, row: dict[str, object]) -> Path:
    case_root = Path(case_root).expanduser()
    canonical_name = str(row["canonical_case_name"])
    canonical = (case_root / canonical_name).resolve()
    if canonical.is_dir():
        return canonical

    explicit = row.get("iterreduce_case_dir")
    if explicit is None:
        return canonical

    path = Path(str(explicit)).expanduser()
    if path.is_absolute():
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            return canonical
        return path.resolve()
    if path.parts and path.parts[0] == "gnn_preconditioner":
        return (PROJECT_ROOT / path).resolve()
    return (case_root / path).resolve()
