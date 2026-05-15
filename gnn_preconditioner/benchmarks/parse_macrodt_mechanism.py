"""Re-parse macroDt 10-case sweep.

Uses logs when present (for timesteps + region breakdown + avg-iters/solve);
falls back to abg_metrics_iterreduce_v1.json otherwise. Flags wallclock
outliers (runs where iterreduce forward_wall far exceeds the cohort median).

Outputs:
  macrodt_10case_mechanism_and_wallclock.csv
  macrodt_10case_region_breakdown.csv   (log-only runs)
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH = PROJECT_ROOT / "gnn_preconditioner/benchmarks/runs"

SWEEP_ROOTS = [
    BENCH / "macrodt_rerun_clean_20260417T222402Z",
    BENCH / "macrodt_10case_sweep_20260414T003031Z",
    BENCH / "macrodt_resume_20260414T015632Z",
    BENCH / "macrodt_sweep_extended_20260413T224537Z",
    BENCH / "macrodt_sweep_case011_20260413T221013Z",
    BENCH / "macrodt_r300_retry_20260414T000833Z",
    BENCH / "macrodt_r300_retry_20260414T000106Z",
]

CASES = ["006", "010", "011", "013", "018", "030", "037", "078", "094", "097"]
CADENCES = [
    ("0p01",  "1",   0.01),
    ("0p05",  "5",   0.05),
    ("0p1",   "10",  0.1),
    ("0p3",   "30",  0.3),
    ("1p0",   "100", 1.0),
    ("1p5",   "150", 1.5),
    ("3p0",   "300", 3.0),
]

REGIONS = [
    ("startup", 0.0, 0.5),
    ("mid",     0.5, 1.5),
    ("steady",  1.5, 1e9),
]

TIME_PREFIX = "Time = "
P_SOLVE_RE  = re.compile(r"Solving for p, .*No Iterations (\d+)")
EXEC_RE     = re.compile(r"^ExecutionTime = ([0-9.eE+-]+)\s+s")


def _find_run(case, md_tag, r_tag):
    patterns = [f"case{case}_md{md_tag}_r{r_tag}",
                f"case_{case}_md{md_tag}_r{r_tag}"]
    for root in SWEEP_ROOTS:
        if not root.exists():
            continue
        for pat in patterns:
            d = root / pat
            if d.exists():
                return d
    return None


def parse_log(path: Path):
    steps = 0
    p_solves = 0
    p_iters = 0
    exec_last = None
    region_iters = {name: 0 for name, _, _ in REGIONS}
    cur_time = 0.0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith(TIME_PREFIX):
                steps += 1
                val = line[len(TIME_PREFIX):].strip().rstrip("s").strip()
                try:
                    cur_time = float(val)
                except ValueError:
                    pass
                continue
            m = P_SOLVE_RE.search(line)
            if m:
                n = int(m.group(1))
                p_solves += 1
                p_iters += n
                for name, lo, hi in REGIONS:
                    if lo <= cur_time < hi:
                        region_iters[name] += n
                        break
                continue
            m = EXEC_RE.match(line)
            if m:
                exec_last = float(m.group(1))
    return dict(steps=steps, p_solves=p_solves, p_iters=p_iters,
                wallclock=exec_last, region_iters=region_iters)


def parse_json(path: Path):
    """Fallback: coarse totals from abg_metrics json (no regions, no p_solves)."""
    d = json.load(open(path))
    cs = d.get("coldstart", {}) or {}
    ir = d.get("iterreduce", {}) or {}
    cs_counts = cs.get("counts", {}) or {}
    ir_counts = ir.get("counts", {}) or {}
    return dict(
        cold_p_iters=cs_counts.get("p"),
        ir_p_iters=ir_counts.get("p"),
        cold_wallclock=cs.get("execution_time_s"),
        ir_wallclock=ir.get("execution_time_s"),
        cold_ux_iters=cs_counts.get("Ux"),
        cold_uy_iters=cs_counts.get("Uy"),
        ir_ux_iters=ir_counts.get("Ux"),
        ir_uy_iters=ir_counts.get("Uy"),
        ir_forward_wall=ir.get("total_forward_wall_s"),
    )


def main():
    mech_rows = []
    region_rows = []

    for case in CASES:
        for md_tag, r_tag, md_val in CADENCES:
            run = _find_run(case, md_tag, r_tag)
            if run is None:
                print(f"SKIP {case} md={md_tag} r={r_tag}: no dir")
                continue
            cl = run / "coldstart" / "log.coldstart"
            il = run / "iterreduce" / "log.iterreduce"
            jp = run / "abg_metrics_iterreduce_v1.json"

            source = ""
            cold = None
            ir = None
            if cl.exists() and il.exists():
                cold = parse_log(cl)
                ir = parse_log(il)
                source = "log"
            elif jp.exists():
                j = parse_json(jp)
                source = "json"
            else:
                print(f"SKIP {case} md={md_tag} r={r_tag}: no log nor json")
                continue

            def pct(n, d):
                return 100.0 * n / d if d else 0.0

            if source == "log":
                dt_steps = ir["steps"] - cold["steps"]
                dt_piters = ir["p_iters"] - cold["p_iters"]
                avg_c = cold["p_iters"] / max(cold["p_solves"], 1)
                avg_i = ir["p_iters"] / max(ir["p_solves"], 1)
                dt_wall = (ir["wallclock"] or 0) - (cold["wallclock"] or 0)
                mech_rows.append(dict(
                    case=case, macro_dt=md_val, rollout=int(r_tag),
                    source=source,
                    cold_timesteps=cold["steps"],
                    ir_timesteps=ir["steps"],
                    timesteps_delta_pct=round(pct(dt_steps, cold["steps"]), 2),
                    cold_p_solves=cold["p_solves"],
                    ir_p_solves=ir["p_solves"],
                    cold_p_iters=cold["p_iters"],
                    ir_p_iters=ir["p_iters"],
                    p_iters_delta_pct=round(pct(dt_piters, cold["p_iters"]), 2),
                    cold_avg_iters_per_solve=round(avg_c, 3),
                    ir_avg_iters_per_solve=round(avg_i, 3),
                    avg_iters_delta_pct=round(pct(avg_i - avg_c, avg_c), 2),
                    cold_wallclock_s=round(cold["wallclock"] or 0, 2),
                    ir_wallclock_s=round(ir["wallclock"] or 0, 2),
                    wallclock_delta_pct=round(pct(dt_wall, cold["wallclock"] or 1.0), 2),
                    ir_forward_wall_s=None,
                ))
                total_red = cold["p_iters"] - ir["p_iters"]
                for name, lo, hi in REGIONS:
                    c = cold["region_iters"][name]
                    i = ir["region_iters"][name]
                    region_rows.append(dict(
                        case=case, macro_dt=md_val, rollout=int(r_tag),
                        region=name, t_lo=lo,
                        t_hi=(hi if hi < 1e8 else None),
                        cold_p_iters=c, ir_p_iters=i,
                        region_delta_pct=round(pct(i - c, c), 2),
                        share_of_total_reduction_pct=round(pct(c - i, total_red), 2),
                    ))
            else:  # json fallback
                dt_p = (j["ir_p_iters"] or 0) - (j["cold_p_iters"] or 0)
                dt_wall = (j["ir_wallclock"] or 0) - (j["cold_wallclock"] or 0)
                mech_rows.append(dict(
                    case=case, macro_dt=md_val, rollout=int(r_tag),
                    source=source,
                    cold_timesteps=None,
                    ir_timesteps=None,
                    timesteps_delta_pct=None,
                    cold_p_solves=None,
                    ir_p_solves=None,
                    cold_p_iters=j["cold_p_iters"],
                    ir_p_iters=j["ir_p_iters"],
                    p_iters_delta_pct=round(pct(dt_p, j["cold_p_iters"]), 2),
                    cold_avg_iters_per_solve=None,
                    ir_avg_iters_per_solve=None,
                    avg_iters_delta_pct=None,
                    cold_wallclock_s=round(j["cold_wallclock"] or 0, 2),
                    ir_wallclock_s=round(j["ir_wallclock"] or 0, 2),
                    wallclock_delta_pct=round(pct(dt_wall, j["cold_wallclock"] or 1.0), 2),
                    ir_forward_wall_s=round(j["ir_forward_wall"] or 0, 2),
                ))

    # Flag wallclock outliers within each macroDt cohort
    for md_tag, r_tag, md_val in CADENCES:
        cohort = [r for r in mech_rows if abs(r["macro_dt"] - md_val) < 1e-9
                  and r["ir_wallclock_s"]]
        if len(cohort) < 3:
            continue
        vals = [r["ir_wallclock_s"] for r in cohort]
        med = median(vals)
        for r in cohort:
            r["wallclock_outlier"] = r["ir_wallclock_s"] > 2.5 * med
    for r in mech_rows:
        r.setdefault("wallclock_outlier", False)

    out_dir = Path(__file__).parent
    mech_csv = out_dir / "macrodt_10case_mechanism_and_wallclock.csv"
    region_csv = out_dir / "macrodt_10case_region_breakdown.csv"

    fieldnames = list(mech_rows[0].keys())
    # move outlier col to end for readability
    if "wallclock_outlier" in fieldnames:
        fieldnames.remove("wallclock_outlier")
        fieldnames.append("wallclock_outlier")
    with open(mech_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(mech_rows)
    with open(region_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()))
        w.writeheader()
        w.writerows(region_rows)

    n_log = sum(1 for r in mech_rows if r["source"] == "log")
    n_json = sum(1 for r in mech_rows if r["source"] == "json")
    n_outl = sum(1 for r in mech_rows if r["wallclock_outlier"])
    print(f"\nWrote {len(mech_rows)} rows -> {mech_csv.name}  "
          f"(log={n_log} json={n_json} wallclock_outliers={n_outl})")
    print(f"Wrote {len(region_rows)} rows -> {region_csv.name}  "
          f"(log-only coverage)")

    # Summary @ md=1.0
    print("\n=== Mechanism split @ macroDt=1.0 / rollout=100 ===")
    print(f"{'case':<5} {'src':<4} {'Dsteps%':>8} {'Dpiters%':>9} "
          f"{'Davg/solve%':>12} {'coldWall':>9} {'warmWall':>9} "
          f"{'Dwall%':>8} {'outlier'}")
    for r in mech_rows:
        if abs(r["macro_dt"] - 1.0) < 1e-9:
            ts  = r["timesteps_delta_pct"]
            avg = r["avg_iters_delta_pct"]
            print(f"{r['case']:<5} {r['source']:<4} "
                  f"{('' if ts is None else f'{ts:+.2f}'):>8} "
                  f"{r['p_iters_delta_pct']:>+9.2f} "
                  f"{('' if avg is None else f'{avg:+.2f}'):>12} "
                  f"{r['cold_wallclock_s']:>9.1f} "
                  f"{r['ir_wallclock_s']:>9.1f} "
                  f"{r['wallclock_delta_pct']:>+8.2f} "
                  f"{'YES' if r['wallclock_outlier'] else ''}")

    print("\n=== Region share of iter-reduction @ macroDt=1.0 / rollout=100 ===")
    print(f"{'case':<5}  {'startup%':>9} {'mid%':>7} {'steady%':>8}  "
          f"{'startup/own':>12} {'mid/own':>9} {'steady/own':>11}")
    cases_seen = set()
    for case in CASES:
        buckets = {r["region"]: r for r in region_rows
                   if r["case"] == case and abs(r["macro_dt"] - 1.0) < 1e-9}
        if not buckets:
            continue
        cases_seen.add(case)
        s_share = buckets.get("startup", {}).get("share_of_total_reduction_pct", 0)
        m_share = buckets.get("mid",     {}).get("share_of_total_reduction_pct", 0)
        y_share = buckets.get("steady",  {}).get("share_of_total_reduction_pct", 0)
        s_own = buckets.get("startup", {}).get("region_delta_pct", 0)
        m_own = buckets.get("mid",     {}).get("region_delta_pct", 0)
        y_own = buckets.get("steady",  {}).get("region_delta_pct", 0)
        print(f"{case:<5}  {s_share:>+9.1f} {m_share:>+7.1f} {y_share:>+8.1f}  "
              f"{s_own:>+12.2f} {m_own:>+9.2f} {y_own:>+11.2f}")
    missing_region = [c for c in CASES if c not in cases_seen]
    if missing_region:
        print(f"\nRegion breakdown missing (logs cleaned): {missing_region}")


if __name__ == "__main__":
    main()
