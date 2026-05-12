#!/usr/bin/env python3
"""Summarize solver-coupled TorchScript timing from benchmark JSON artifacts.

The script expects a benchmark root containing variant directories with
case_*/abg_metrics_iterreduce_v1.json files. It prints a compact Markdown table
that separates wall-clock, TorchScript load time, TorchScript forward time,
injection count, and residual non-inference wall time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _variant_metadata(run_root: Path) -> dict[str, dict[str, float | int | str]]:
    summary_path = run_root / "summary.csv"
    meta: dict[str, dict[str, float | int | str]] = {}
    if not summary_path.exists():
        return meta

    with summary_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            variant = row.get("variant")
            if not variant or variant in meta:
                continue
            item: dict[str, float | int | str] = {}
            if row.get("rollout_steps"):
                item["rollout_steps"] = int(float(row["rollout_steps"]))
            if row.get("blend_mode"):
                item["blend_mode"] = row["blend_mode"]
            if row.get("gnn_training_dt"):
                item["gnn_training_dt"] = float(row["gnn_training_dt"])
            if row.get("macro_dt"):
                item["macro_dt"] = float(row["macro_dt"])
            meta[variant] = item
    return meta


def _read_case_metrics(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["coldstart"], data["iterreduce"]


def summarize(run_root: Path) -> list[dict[str, float | int | str]]:
    meta = _variant_metadata(run_root)
    rows: list[dict[str, float | int | str]] = []

    variant_dirs = {p.name: p for p in run_root.iterdir() if p.is_dir()}
    ordered_names = list(meta)
    ordered_names.extend(name for name in sorted(variant_dirs) if name not in meta)

    for variant in ordered_names:
        variant_dir = variant_dirs.get(variant)
        if variant_dir is None:
            continue
        json_files = sorted(variant_dir.glob("case_*/abg_metrics_iterreduce_v1.json"))
        if not json_files:
            continue

        rollout_steps = int(meta.get(variant, {}).get("rollout_steps", 0))
        cold_wall = model_wall = load_wall = forward_wall = 0.0
        injections = 0
        cold_total = model_total = cold_p = model_p = 0

        for json_file in json_files:
            cold, model = _read_case_metrics(json_file)
            cold_wall += float(cold.get("clock_time_s") or 0.0)
            model_wall += float(model.get("clock_time_s") or 0.0)
            load_wall += float(model.get("runtime_load_wall_s") or 0.0)
            forward_wall += float(model.get("total_forward_wall_s") or 0.0)
            injections += int(model.get("injections") or 0)
            cold_total += int(cold.get("total_linear_iterations") or 0)
            model_total += int(model.get("total_linear_iterations") or 0)
            cold_p += int(cold.get("counts", {}).get("p") or 0)
            model_p += int(model.get("counts", {}).get("p") or 0)

        wall_delta = model_wall - cold_wall
        inference_wall = load_wall + forward_wall
        extra_non_inference = wall_delta - inference_wall
        forward_per_injection = forward_wall / injections if injections else 0.0
        forward_per_rollout_step = (
            forward_wall / (injections * rollout_steps)
            if injections and rollout_steps
            else 0.0
        )
        total_delta_pct = 100.0 * (model_total - cold_total) / cold_total
        p_delta_pct = 100.0 * (model_p - cold_p) / cold_p

        rows.append(
            {
                "variant": variant,
                "rollout_steps": rollout_steps,
                "cases": len(json_files),
                "cold_wall_s": cold_wall,
                "model_wall_s": model_wall,
                "wall_delta_pct": 100.0 * wall_delta / cold_wall,
                "load_wall_s": load_wall,
                "forward_wall_s": forward_wall,
                "injections": injections,
                "forward_per_injection_s": forward_per_injection,
                "forward_per_rollout_step_ms": 1000.0 * forward_per_rollout_step,
                "extra_non_inference_s": extra_non_inference,
                "total_delta_pct": total_delta_pct,
                "pressure_delta_pct": p_delta_pct,
            }
        )

    return rows


def print_markdown(rows: list[dict[str, float | int | str]]) -> None:
    print(
        "| Setting | r | Wall cold -> model | Load s | Forward s | Inj. | "
        "Fwd/inj s | Fwd/step ms | Extra non-inference s | Total iter % | p iter % |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['variant']} | {row['rollout_steps']} | "
            f"{row['cold_wall_s']:.1f} -> {row['model_wall_s']:.1f} "
            f"(+{row['wall_delta_pct']:.1f}%) | "
            f"{row['load_wall_s']:.1f} | {row['forward_wall_s']:.1f} | "
            f"{row['injections']} | {row['forward_per_injection_s']:.3f} | "
            f"{row['forward_per_rollout_step_ms']:.1f} | "
            f"{row['extra_non_inference_s']:.1f} | "
            f"{row['total_delta_pct']:+.2f} | {row['pressure_delta_pct']:+.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print_markdown(summarize(args.run_root))


if __name__ == "__main__":
    main()
