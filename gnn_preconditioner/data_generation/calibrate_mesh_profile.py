#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.data_generation.common import MeshProfile, load_source_case_meta
from gnn_preconditioner.data_generation.mesh_utils import calibrate_profile_counts


DEFAULT_FACTORS = [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--case-ids", type=int, nargs="+", required=True)
    parser.add_argument("--target-min-cells", type=int, required=True)
    parser.add_argument("--target-max-cells", type=int, required=True)
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=Path("gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"),
    )
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path("gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"),
    )
    parser.add_argument("--base-case-dir", type=Path, default=Path("gnn_preconditioner/data_generation/base"))
    parser.add_argument("--front-back-mode", type=str, default="empty")
    parser.add_argument("--write-profile", action="store_true", help="Overwrite --profile with the best in-band candidate")
    return parser.parse_args()


def _score(counts: list[int], target_min: int, target_max: int) -> tuple[bool, float]:
    in_band = min(counts) >= target_min and max(counts) <= target_max
    clipped = np.clip(np.asarray(counts, dtype=np.float64), target_min, target_max)
    penalty = float(np.abs(np.asarray(counts, dtype=np.float64) - clipped).sum())
    band_mid = 0.5 * (target_min + target_max)
    spread_penalty = float(np.abs(np.asarray(counts, dtype=np.float64) - band_mid).mean())
    return in_band, penalty * 1000.0 + spread_penalty


def main() -> None:
    args = _parse_args()
    base_profile = MeshProfile.from_json(args.profile)
    source_rows = [meta.as_json() for meta in load_source_case_meta(args.source_h5, args.case_root)]
    row_by_id = {int(row["case_id"]): row for row in source_rows}
    selected_rows = []
    for case_id in args.case_ids:
        if case_id not in row_by_id:
            raise KeyError(f"Case id {case_id} not found in source metadata")
        selected_rows.append(row_by_id[case_id])

    candidates = [base_profile.scaled(factor, name=f"{base_profile.profile_name}_x{factor:.3f}") for factor in DEFAULT_FACTORS]
    if 1.0 not in DEFAULT_FACTORS:
        candidates.append(base_profile)

    best_profile = None
    best_results = None
    best_score = None
    evaluations = []
    for profile in candidates:
        results = calibrate_profile_counts(
            base_case_dir=args.base_case_dir.expanduser().resolve(),
            profile=profile,
            cases=selected_rows,
            front_back_mode=args.front_back_mode,
        )
        counts = [result.n_cells for result in results]
        in_band, score = _score(counts, args.target_min_cells, args.target_max_cells)
        record = {
            "profile": profile.to_json_dict(),
            "counts": counts,
            "min_cells": min(counts),
            "max_cells": max(counts),
            "mean_cells": float(np.mean(counts)),
            "in_band": in_band,
            "score": score,
        }
        evaluations.append(record)
        if best_score is None or score < best_score:
            best_profile = profile
            best_results = results
            best_score = score

    assert best_profile is not None and best_results is not None

    if args.write_profile:
        args.profile.write_text(json.dumps(best_profile.to_json_dict(), indent=2), encoding="utf-8")

    summary = {
        "target_min_cells": args.target_min_cells,
        "target_max_cells": args.target_max_cells,
        "selected_case_ids": args.case_ids,
        "best_profile": best_profile.to_json_dict(),
        "best_counts": [result.n_cells for result in best_results],
        "evaluations": evaluations,
        "profile_written": bool(args.write_profile),
        "profile_path": str(args.profile.expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
