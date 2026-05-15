#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.inference.openfoam_fields import read_openfoam_scalar_internal_field, read_openfoam_vector_internal_field

from gnn_preconditioner.data_generation.common import (
    DEFAULT_CASE_OUTPUT_ROOT,
    load_manifest,
    resolve_iterreduce_case_dir,
)
from gnn_preconditioner.data_generation.mesh_utils import (
    collect_numeric_time_dirs,
    compute_internal_face_flux_xy,
    compute_internal_face_pressure_jump,
    load_case_static_geometry,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_OUTPUT_ROOT)
    parser.add_argument("--mesh-profile-name", type=str, default="coarse_v1")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _resolve_case_dir(case_root: Path, row: dict[str, object]) -> Path:
    return resolve_iterreduce_case_dir(case_root, row)


def _build_case_payload(task: dict[str, object]) -> dict[str, object]:
    row = task["row"]
    case_root = Path(str(task["case_root"]))
    mesh_profile_name = str(task["mesh_profile_name"])
    case_dir = _resolve_case_dir(case_root, row)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Generated case directory not found: {case_dir}")

    static = load_case_static_geometry(case_dir)
    cell_centers = static["cell_centers"]
    cell_volumes = static["cell_volumes"]
    owner = torch.tensor(static["owner"], dtype=torch.long)
    neighbour = torch.tensor(static["neighbour"], dtype=torch.long)
    sf_internal = static["Sf_internal"]
    cf_internal = static["Cf_internal"]
    boundary_mask = np.asarray(static["boundary_mask"], dtype=np.uint8)

    time_dirs = collect_numeric_time_dirs(case_dir, include_zero=True)
    if not time_dirs:
        raise ValueError(f"No numeric time directories found in {case_dir}")

    velocities = []
    pressures = []
    times = []
    for time_dir in time_dirs:
        times.append(float(time_dir.name))
        velocities.append(
            read_openfoam_vector_internal_field(
                time_dir / "U",
                expected_size=int(cell_centers.shape[0]),
            ).to(torch.float32)
        )
        pressures.append(
            read_openfoam_scalar_internal_field(
                time_dir / "p",
                expected_size=int(cell_centers.shape[0]),
            ).to(torch.float32)
        )

    velocity = torch.stack(velocities, dim=0)
    velocity_xy = velocity[:, :, :2].contiguous()
    pressure_raw = torch.stack(pressures, dim=0)
    pressure_gauge = pressure_raw - pressure_raw.mean(dim=1, keepdim=True)
    phi_xy = torch.stack(
        [compute_internal_face_flux_xy(v, owner, neighbour, sf_internal) for v in velocity_xy],
        dim=0,
    )
    gradp_jump = torch.stack(
        [compute_internal_face_pressure_jump(p, owner, neighbour) for p in pressure_gauge],
        dim=0,
    )

    return {
        "trajectory_key": str(row["trajectory_key"]),
        "times": np.asarray(times, dtype=np.float64),
        "Uinlet": np.asarray([float(row["Uinlet"])], dtype=np.float32),
        "nu": np.asarray([float(row["nu"])], dtype=np.float32),
        "cell_centers": cell_centers.numpy().astype(np.float32),
        "cell_volumes": cell_volumes.numpy().astype(np.float32),
        "boundary_mask": boundary_mask,
        "owner": np.asarray(static["owner"], dtype=np.int64),
        "neighbour": np.asarray(static["neighbour"], dtype=np.int64),
        "Sf_internal": sf_internal.numpy().astype(np.float32),
        "Cf_internal": cf_internal.numpy().astype(np.float32),
        "velocity_xy": velocity_xy.numpy().astype(np.float32),
        "pressure_raw": pressure_raw.numpy().astype(np.float32),
        "pressure_gauge": pressure_gauge.numpy().astype(np.float32),
        "phi_xy_next_internal": phi_xy.numpy().astype(np.float32),
        "gradp_jump_next_internal": gradp_jump.numpy().astype(np.float32),
        "cell_ids": np.arange(int(cell_centers.shape[0]), dtype=np.int64),
        "attrs": {
            "case_id": int(row["case_id"]),
            "source_case_path": str(case_dir),
            "pressure_gauge_mode": "mean_subtracted",
            "front_back_mode": "empty",
            "mesh_profile": mesh_profile_name,
            "n_times": int(len(times)),
            "n_cells": int(cell_centers.shape[0]),
        },
    }


def main() -> None:
    args = _parse_args()
    rows = load_manifest(args.manifest)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)

    case_root = args.case_root.expanduser().resolve()
    tasks = [
        {
            "row": row,
            "case_root": str(case_root),
            "mesh_profile_name": args.mesh_profile_name,
        }
        for row in rows
    ]
    if args.workers <= 1:
        payloads = [_build_case_payload(task) for task in tasks]
    else:
        max_workers = min(args.workers, len(tasks), os.cpu_count() or args.workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            payloads = list(pool.map(_build_case_payload, tasks))

    with h5py.File(args.output_h5, "w") as h5:
        for payload in payloads:
            grp = h5.create_group(payload["trajectory_key"])
            for key in (
                "times",
                "Uinlet",
                "nu",
                "cell_centers",
                "cell_volumes",
                "boundary_mask",
                "owner",
                "neighbour",
                "Sf_internal",
                "Cf_internal",
                "velocity_xy",
                "pressure_raw",
                "pressure_gauge",
                "phi_xy_next_internal",
                "gradp_jump_next_internal",
                "cell_ids",
            ):
                grp.create_dataset(key, data=payload[key])
            for key, value in payload["attrs"].items():
                grp.attrs[key] = value

    print(json.dumps({
        "output_h5": str(args.output_h5.expanduser().resolve()),
        "trajectory_count": len(rows),
        "workers": min(args.workers, len(tasks)) if tasks else 0,
    }, indent=2))


if __name__ == "__main__":
    main()
