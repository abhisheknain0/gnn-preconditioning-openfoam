"""Compare cold vs warm internalField p / Ux / Uy at every written timestep.

Skips injection timesteps (t at multiples of macroDt within [startInjectTime,
endTime]), so the comparison only covers fields the solver produced from a
non-overwritten initial guess.

Outputs:
  accuracy_probe_l2_per_time.csv      one row per (case, time, field)
  accuracy_probe_summary.csv          one row per (case, field) with
                                      mean/max relL2 across non-injection times
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from gnn_preconditioner.inference.openfoam_fields import (
    read_openfoam_scalar_internal_field,
    read_openfoam_vector_internal_field,
)

TIME_DIR_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
INJECT_TOL = 5e-3  # treat |t - k*macroDt| < tol as injection step

CASES = ["011", "018", "094"]
MACRO_DT = 1.0
START_INJECT = 0.0
END_TIME = 3.0


def list_time_dirs(case_path: Path) -> list[tuple[float, Path]]:
    times = []
    for p in case_path.iterdir():
        if not p.is_dir():
            continue
        if TIME_DIR_RE.match(p.name):
            try:
                t = float(p.name)
            except ValueError:
                continue
            if t > 0:  # skip t=0 initial condition
                times.append((t, p))
    times.sort()
    return times


def is_injection_time(t: float) -> bool:
    if t < START_INJECT - INJECT_TOL:
        return False
    k = round((t - START_INJECT) / MACRO_DT)
    return abs(t - (START_INJECT + k * MACRO_DT)) < INJECT_TOL


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).float()
    denom = a.float().norm().item()
    if denom == 0:
        return float("nan")
    return diff.norm().item() / denom


def abs_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).float().norm().item()


def main(run_root: Path):
    per_time_rows = []
    summary_rows = []

    for case in CASES:
        cold_dir = run_root / f"case{case}_md1p0_r100" / "coldstart"
        warm_dir = run_root / f"case{case}_md1p0_r100" / "iterreduce"
        if not cold_dir.exists() or not warm_dir.exists():
            print(f"MISSING: {cold_dir} or {warm_dir}")
            continue
        cold_times = dict(list_time_dirs(cold_dir))
        warm_times = dict(list_time_dirs(warm_dir))
        shared = sorted(set(cold_times) & set(warm_times))
        print(f"\ncase {case}: {len(shared)} shared write times "
              f"(cold-only={len(cold_times)-len(shared)}, "
              f"warm-only={len(warm_times)-len(shared)})")

        per_field_acc = {"p": [], "Ux": [], "Uy": []}
        for t in shared:
            inj = is_injection_time(t)
            cpath = cold_times[t]
            wpath = warm_times[t]
            try:
                cP = read_openfoam_scalar_internal_field(cpath / "p")
                wP = read_openfoam_scalar_internal_field(wpath / "p")
                cU = read_openfoam_vector_internal_field(cpath / "U", expected_size=cP.numel())
                wU = read_openfoam_vector_internal_field(wpath / "U", expected_size=wP.numel())
            except Exception as exc:
                print(f"  t={t}: read failed: {exc}")
                continue

            rp = rel_l2(cP, wP)
            rux = rel_l2(cU[:, 0], wU[:, 0])
            ruy = rel_l2(cU[:, 1], wU[:, 1])
            per_time_rows.append(dict(
                case=case, t=t, injection=inj,
                rel_l2_p=round(rp, 6),
                rel_l2_Ux=round(rux, 6),
                rel_l2_Uy=round(ruy, 6),
                abs_l2_p=round(abs_l2(cP, wP), 6),
                cold_norm_p=round(cP.float().norm().item(), 6),
                cold_norm_Ux=round(cU[:, 0].float().norm().item(), 6),
                cold_norm_Uy=round(cU[:, 1].float().norm().item(), 6),
            ))
            if not inj:
                per_field_acc["p"].append(rp)
                per_field_acc["Ux"].append(rux)
                per_field_acc["Uy"].append(ruy)

        for field, vals in per_field_acc.items():
            if not vals:
                continue
            summary_rows.append(dict(
                case=case, field=field,
                n_times=len(vals),
                mean_rel_l2=round(sum(vals) / len(vals), 6),
                max_rel_l2=round(max(vals), 6),
                min_rel_l2=round(min(vals), 6),
            ))

    out_dir = Path(__file__).parent
    pt_csv = out_dir / "accuracy_probe_l2_per_time.csv"
    sm_csv = out_dir / "accuracy_probe_summary.csv"
    with open(pt_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_time_rows[0].keys()))
        w.writeheader(); w.writerows(per_time_rows)
    with open(sm_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nWrote {len(per_time_rows)} rows -> {pt_csv.name}")
    print(f"Wrote {len(summary_rows)} rows -> {sm_csv.name}")

    print("\n=== Summary (non-injection times only, md=1.0/r=100) ===")
    print(f"{'case':<5} {'field':<4} {'n':>3} {'mean relL2':>12} {'max relL2':>12}")
    for r in summary_rows:
        print(f"{r['case']:<5} {r['field']:<4} {r['n_times']:>3} "
              f"{r['mean_rel_l2']:>12.3e} {r['max_rel_l2']:>12.3e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    args = ap.parse_args()
    main(args.run_root)
