#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.data_generation.common import load_source_case_meta


FEATURE_NAMES = ("Uinlet", "xmid", "ymid", "ymax")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cases", type=int, default=None, help="Default: use all available cases")
    parser.add_argument("--force-case-id", type=int, default=None)
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--valid-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--num-bins", type=int, default=4)
    return parser.parse_args()


def _compute_bin_ids(rows: list[dict[str, object]], n_bins: int) -> dict[int, tuple[int, ...]]:
    bins: dict[int, tuple[int, ...]] = {}
    for feature in FEATURE_NAMES:
        values = np.asarray([float(r[feature]) for r in rows], dtype=np.float64)
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        ids = np.digitize(values, quantiles[1:-1], right=False)
        for row, bin_id in zip(rows, ids, strict=True):
            case_id = int(row["case_id"])
            bins.setdefault(case_id, tuple())
            bins[case_id] = bins[case_id] + (int(bin_id),)
    return bins


def _round_robin_select(
    by_bin: dict[tuple[int, ...], list[dict[str, object]]],
    target_count: int,
    rng: np.random.RandomState,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    ordered_bins = sorted(by_bin.keys())
    while len(selected) < target_count and ordered_bins:
        progress = False
        rng.shuffle(ordered_bins)
        for key in list(ordered_bins):
            group = by_bin[key]
            if not group:
                ordered_bins.remove(key)
                continue
            selected.append(group.pop())
            progress = True
            if len(selected) >= target_count:
                break
        if not progress:
            break
    return selected


def _select_rows(
    rows: list[dict[str, object]],
    num_cases: int,
    force_case_id: int | None,
    seed: int,
    num_bins: int,
) -> list[dict[str, object]]:
    rng = np.random.RandomState(seed)
    bins = _compute_bin_ids(rows, n_bins=num_bins)
    by_bin: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    forced_row = None
    for row in rows:
        if force_case_id is not None and int(row["case_id"]) == force_case_id:
            forced_row = row
            continue
        by_bin[bins[int(row["case_id"])]].append(row)
    for group in by_bin.values():
        rng.shuffle(group)

    selected: list[dict[str, object]] = []
    if forced_row is not None:
        selected.append(forced_row)
    selected.extend(_round_robin_select(by_bin, target_count=num_cases - len(selected), rng=rng))
    if len(selected) < num_cases:
        remaining = [row for group in by_bin.values() for row in group]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_cases - len(selected)])
    if len(selected) != num_cases:
        raise ValueError(f"Could not select requested {num_cases} cases; got {len(selected)}")
    return sorted(selected, key=lambda row: int(row["case_id"]))


def _stratified_split(
    rows: list[dict[str, object]],
    train_count: int,
    valid_count: int,
    test_count: int,
    force_case_id: int | None,
    seed: int,
    num_bins: int,
) -> dict[str, list[str]]:
    if train_count + valid_count + test_count != len(rows):
        raise ValueError("Split counts must sum to selected rows")
    rng = np.random.RandomState(seed)
    bins = _compute_bin_ids(rows, n_bins=num_bins)
    buckets = {"train": [], "valid": [], "test": []}

    remaining = list(rows)
    if force_case_id is not None:
        forced = [row for row in rows if int(row["case_id"]) == force_case_id]
        if len(forced) != 1:
            raise ValueError("Forced case must appear exactly once in selected rows")
        forced_row = forced[0]
        buckets["test"].append(forced_row["trajectory_key"])
        remaining = [row for row in rows if int(row["case_id"]) != force_case_id]
        test_count -= 1

    by_bin: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    for row in remaining:
        by_bin[bins[int(row["case_id"])]].append(row)
    for group in by_bin.values():
        rng.shuffle(group)

    quotas = {"test": test_count, "valid": valid_count, "train": train_count}
    for split_name in ("test", "valid", "train"):
        need = quotas[split_name]
        picked = _round_robin_select(by_bin, target_count=need, rng=rng)
        if len(picked) < need:
            remaining_rows = [row for group in by_bin.values() for row in group]
            rng.shuffle(remaining_rows)
            picked.extend(remaining_rows[: need - len(picked)])
        chosen = {row["trajectory_key"] for row in picked}
        buckets[split_name].extend(row["trajectory_key"] for row in picked)
        for key in list(by_bin.keys()):
            by_bin[key] = [row for row in by_bin[key] if row["trajectory_key"] not in chosen]
    return buckets


def main() -> None:
    args = _parse_args()
    rows = [meta.as_json() for meta in load_source_case_meta(args.source_h5, args.case_root)]
    num_cases = args.num_cases or len(rows)
    if num_cases > len(rows):
        raise ValueError(f"Requested {num_cases} cases but only {len(rows)} are available")
    if args.train_count + args.valid_count + args.test_count != num_cases:
        raise ValueError("train/valid/test counts must sum to num-cases")
    selected = _select_rows(rows, num_cases, args.force_case_id, args.seed, args.num_bins)
    split = _stratified_split(
        selected,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        force_case_id=args.force_case_id,
        seed=args.seed,
        num_bins=args.num_bins,
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    args.output_split.write_text(json.dumps(split, indent=2), encoding="utf-8")
    holdout_path = args.output_split.with_name(f"{args.output_split.stem}_holdout_manifest.json")
    holdout_rows = [row for row in selected if row["trajectory_key"] in set(split["test"])]
    holdout_path.write_text(json.dumps(holdout_rows, indent=2), encoding="utf-8")

    summary = {
        "selected_count": len(selected),
        "train_count": len(split["train"]),
        "valid_count": len(split["valid"]),
        "test_count": len(split["test"]),
        "forced_case_id": args.force_case_id,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_split": str(args.output_split.resolve()),
        "holdout_manifest": str(holdout_path.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
