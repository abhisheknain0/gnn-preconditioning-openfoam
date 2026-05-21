#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
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
    parser.add_argument(
        "--output-holdout-manifest",
        type=Path,
        default=None,
        help="Optional path for a redundant case-metadata JSON containing only test cases.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--force-case-id", type=int, default=51)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--valid-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    return parser.parse_args()


def _compute_bin_ids(rows: list[dict[str, object]], n_bins: int = 4) -> dict[int, tuple[int, ...]]:
    bins: dict[int, tuple[int, ...]] = {}
    for feature in FEATURE_NAMES:
        values = np.asarray([float(r[feature]) for r in rows], dtype=np.float64)
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        ids = np.digitize(values, quantiles[1:-1], right=False)
        for row, bin_id in zip(rows, ids):
            bins.setdefault(int(row["case_id"]), tuple())
            bins[int(row["case_id"])] = bins[int(row["case_id"])] + (int(bin_id),)
    return bins


def _stratified_sample(rows: list[dict[str, object]], num_cases: int, force_case_id: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.RandomState(seed)
    bins = _compute_bin_ids(rows)
    by_bin: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    forced = None
    for row in rows:
        if int(row["case_id"]) == force_case_id:
            forced = row
            continue
        by_bin[bins[int(row["case_id"])]].append(row)
    if forced is None:
        raise ValueError(f"Forced case id {force_case_id} not found in source set")
    for group in by_bin.values():
        rng.shuffle(group)

    selected = [forced]
    ordered_bins = sorted(by_bin.keys())
    while len(selected) < num_cases and ordered_bins:
        progress = False
        rng.shuffle(ordered_bins)
        for key in list(ordered_bins):
            group = by_bin[key]
            if not group:
                ordered_bins.remove(key)
                continue
            selected.append(group.pop())
            progress = True
            if len(selected) >= num_cases:
                break
        if not progress:
            break
    if len(selected) < num_cases:
        remaining = [row for group in by_bin.values() for row in group]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_cases - len(selected)])
    if len(selected) != num_cases:
        raise ValueError(f"Could not select requested {num_cases} cases; got {len(selected)}")
    return sorted(selected, key=lambda row: int(row["case_id"]))


def _split_selected(
    rows: list[dict[str, object]],
    train_count: int,
    valid_count: int,
    test_count: int,
    force_case_id: int,
    seed: int,
) -> dict[str, list[str]]:
    if train_count + valid_count + test_count != len(rows):
        raise ValueError("Split counts must sum to selected rows")
    rng = np.random.RandomState(seed)
    bins = _compute_bin_ids(rows)
    forced = [row for row in rows if int(row["case_id"]) == force_case_id]
    if len(forced) != 1:
        raise ValueError("Forced case must appear exactly once in selected rows")
    forced_row = forced[0]
    remaining = [row for row in rows if int(row["case_id"]) != force_case_id]

    by_bin: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    for row in remaining:
        by_bin[bins[int(row["case_id"])]].append(row)
    for group in by_bin.values():
        rng.shuffle(group)

    buckets = {"train": [], "valid": [], "test": [forced_row["trajectory_key"]]}
    quotas = {"train": train_count, "valid": valid_count, "test": test_count - 1}
    order = ["test", "valid", "train"]

    for split in order:
        needed = quotas[split]
        ordered_bins = sorted(by_bin.keys())
        while needed > 0 and ordered_bins:
            progress = False
            rng.shuffle(ordered_bins)
            for key in list(ordered_bins):
                group = by_bin[key]
                if not group:
                    ordered_bins.remove(key)
                    continue
                buckets[split].append(group.pop()["trajectory_key"])
                needed -= 1
                progress = True
                if needed == 0:
                    break
            if not progress:
                break
        if needed > 0:
            remaining_rows = [row for group in by_bin.values() for row in group]
            rng.shuffle(remaining_rows)
            take = remaining_rows[:needed]
            buckets[split].extend(row["trajectory_key"] for row in take)
            chosen = {row["trajectory_key"] for row in take}
            for key in list(by_bin.keys()):
                by_bin[key] = [row for row in by_bin[key] if row["trajectory_key"] not in chosen]
    return buckets


def main() -> None:
    args = _parse_args()
    rows = [meta.as_json() for meta in load_source_case_meta(args.source_h5, args.case_root)]
    selected = _stratified_sample(rows, args.num_cases, args.force_case_id, args.seed)
    split = _split_selected(
        selected,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        force_case_id=args.force_case_id,
        seed=args.seed,
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    args.output_split.write_text(json.dumps(split, indent=2), encoding="utf-8")
    holdout_rows = [row for row in selected if row["trajectory_key"] in set(split["test"])]
    if args.output_holdout_manifest is not None:
        args.output_holdout_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_holdout_manifest.write_text(json.dumps(holdout_rows, indent=2), encoding="utf-8")

    summary = {
        "selected_count": len(selected),
        "train_count": len(split["train"]),
        "valid_count": len(split["valid"]),
        "test_count": len(split["test"]),
        "forced_case_id": args.force_case_id,
        "forced_case_in_test": any(int(row["case_id"]) == args.force_case_id for row in holdout_rows),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_split": str(args.output_split.resolve()),
        "output_holdout_manifest": str(args.output_holdout_manifest.resolve())
        if args.output_holdout_manifest is not None
        else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
