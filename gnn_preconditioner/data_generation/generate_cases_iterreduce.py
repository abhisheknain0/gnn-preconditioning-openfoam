#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.data_generation.common import DEFAULT_CASE_OUTPUT_ROOT, MeshProfile, load_manifest
from gnn_preconditioner.data_generation.mesh_utils import (
    generate_mesh_for_case,
    prepare_case_directory,
    replace_patch_value,
    replace_transport_nu,
    run_solver,
    update_control_dict,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-case-dir", type=Path, default=Path("gnn_preconditioner/data_generation/base"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CASE_OUTPUT_ROOT)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--write-interval", type=float, default=0.01)
    parser.add_argument("--front-back-mode", type=str, default="empty")
    parser.add_argument("--mesh-profile", type=Path, required=True)
    parser.add_argument("--solver", type=str, default="pimpleFoam")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-ids", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _run_single_case(task: dict[str, object]) -> dict[str, object]:
    row = task["row"]
    profile = MeshProfile(**task["profile"])
    base_case_dir = Path(str(task["base_case_dir"]))
    output_root = Path(str(task["output_root"]))
    case_name = str(row["canonical_case_name"])
    case_dir = output_root / case_name
    prepare_case_directory(base_case_dir, case_dir, overwrite=bool(task["overwrite"]))
    update_control_dict(case_dir / "system" / "controlDict", end_time=float(task["end_time"]), write_interval=float(task["write_interval"]))
    replace_patch_value(case_dir / "0" / "U", "inlet", f"({float(row['Uinlet'])} 0 0)")
    replace_transport_nu(case_dir / "constant" / "transportProperties", float(row["nu"]))
    generate_mesh_for_case(
        case_dir,
        ymid=float(row["ymid"]),
        xmid=float(row["xmid"]),
        ymax=float(row["ymax"]),
        profile=profile,
        front_back_mode=str(task["front_back_mode"]),
        check_mesh=True,
    )
    if not bool(task["skip_solve"]) and not bool(task["mesh_only"]):
        run_solver(case_dir, solver=str(task["solver"]))
    (case_dir / "case.foam").touch()
    return {
        "case_id": int(row["case_id"]),
        "trajectory_key": row["trajectory_key"],
        "case_dir": str(case_dir),
        "solver_ran": not bool(task["skip_solve"]) and not bool(task["mesh_only"]),
    }


def main() -> None:
    args = _parse_args()
    profile = MeshProfile.from_json(args.mesh_profile)
    rows = load_manifest(args.manifest)
    if args.case_ids:
        requested = set(args.case_ids)
        rows = [row for row in rows if int(row["case_id"]) in requested]
    if args.max_cases is not None:
        rows = rows[: args.max_cases]

    base_case_dir = args.base_case_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    created = []
    tasks = [
        {
            "row": row,
            "profile": profile.to_json_dict(),
            "base_case_dir": str(base_case_dir),
            "output_root": str(output_root),
            "end_time": args.end_time,
            "write_interval": args.write_interval,
            "front_back_mode": args.front_back_mode,
            "solver": args.solver,
            "overwrite": args.overwrite,
            "skip_solve": args.skip_solve,
            "mesh_only": args.mesh_only,
        }
        for row in rows
    ]
    if args.workers <= 1:
        for task in tasks:
            created.append(_run_single_case(task))
    else:
        max_workers = min(args.workers, len(tasks), os.cpu_count() or args.workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for result in pool.map(_run_single_case, tasks):
                created.append(result)
    created.sort(key=lambda item: int(item["case_id"]))

    print(json.dumps({
        "created_count": len(created),
        "output_root": str(output_root),
        "mesh_profile": profile.to_json_dict(),
        "workers": min(args.workers, len(tasks)) if tasks else 0,
        "cases": created,
    }, indent=2))


if __name__ == "__main__":
    main()
