#!/usr/bin/env python3
"""Consolidate ONE timing+benchmark run into every number reported in the thesis.

Reads a run produced by slurm_timing_full_30case.sh and emits a single,
self-consistent report covering:
  - panel A  transient matched  (dt=0.01, alpha=1, every step)
  - panel B  transient fine-step (dt=0.002, alpha=0.2, every step)
  - panel C  steady simpleFoam startup-rollout (r=30, r=100) + offline gen cost
  - panel D  sanity controls (identityNoop / identity_model / bad_zero)

Iteration deltas feed tbl in section 4; the load/forward/inject aggregates feed
tbl-discussion-overhead and the section 4.5 / 5.1 wall-clock numbers. All from
the same exclusive-node run, so nothing is cross-sourced.

Usage:
    python3 summarize_timing_run.py <run_root> [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys


def _case_id(path: str) -> str:
    """Case id from the immediate parent dir (avoids matching '30case_2026...')."""
    parent = os.path.basename(os.path.dirname(path))
    m = re.findall(r"case_(\d+)", parent)
    return str(int(m[-1])) if m else parent


def _load_abg(panel_root: str) -> dict:
    """Aggregate a transient panel from its per-case abg_metrics JSONs."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(panel_root, "**", "abg_metrics_iterreduce_v1.json"), recursive=True)):
        obj = json.load(open(f))
        cold = obj.get("coldstart") or {}
        model = obj.get("iterreduce") or {}
        comp = obj.get("comparison") or {}
        if model.get("runtime_load_wall_s") is None:
            continue
        rows[_case_id(f)] = {
            "cold_total": cold.get("total_linear_iterations"),
            "model_total": model.get("total_linear_iterations"),
            "cold_p": cold.get("counts", {}).get("p"),
            "model_p": model.get("counts", {}).get("p"),
            "delta_total": comp.get("total_iteration_delta"),
            "delta_p": comp.get("pressure_iteration_delta"),
            "load": model.get("runtime_load_wall_s"),
            "forward": model.get("total_forward_wall_s"),
            "inject": model.get("total_inject_wall_s"),
            "injections": model.get("injections") or model.get("num_injections"),
            "cold_exec": cold.get("execution_time_s"),
            "model_exec": model.get("execution_time_s"),
        }
    return rows


def _pct(delta, base):
    return 100.0 * delta / base if base else 0.0


def _summarize_transient(name, rows):
    n = len(rows)
    if not n:
        return {"name": name, "n": 0}
    sum_cold = sum(r["cold_total"] for r in rows.values())
    sum_model = sum(r["model_total"] for r in rows.values())
    sum_cold_p = sum(r["cold_p"] for r in rows.values())
    sum_model_p = sum(r["model_p"] for r in rows.values())
    load = sum(r["load"] for r in rows.values())
    fwd = sum(r["forward"] for r in rows.values())
    inj = sum(r["inject"] for r in rows.values())
    calls = sum(r["injections"] for r in rows.values())
    cold_exec = sum((r["cold_exec"] or 0) for r in rows.values())
    model_exec = sum((r["model_exec"] or 0) for r in rows.values())
    improved = sum(1 for r in rows.values() if r["model_total"] < r["cold_total"])
    return {
        "name": name, "n": n, "calls": calls,
        "total_delta": sum_model - sum_cold, "total_delta_pct": _pct(sum_model - sum_cold, sum_cold),
        "p_delta": sum_model_p - sum_cold_p, "p_delta_pct": _pct(sum_model_p - sum_cold_p, sum_cold_p),
        "improved": improved,
        "load_s": load, "forward_s": fwd, "inject_s": inj, "field_handling_s": inj - fwd,
        "cold_exec_s": cold_exec, "model_exec_s": model_exec,
        "exec_delta_s": model_exec - cold_exec, "exec_delta_pct": _pct(model_exec - cold_exec, cold_exec),
        "fwd_ms_per_call": 1000.0 * fwd / calls if calls else 0.0,
    }


def _summarize_steady(run_root):
    summ = os.path.join(run_root, "steady", "summary.csv")
    out = {}
    if not os.path.exists(summ):
        return out
    by_r = {}
    with open(summ) as fh:
        for row in csv.DictReader(fh):
            try:
                R = int(row["rollout"])
            except (KeyError, ValueError):
                continue
            d = by_r.setdefault(R, {"cold": 0, "warm": 0, "cold_p": 0, "warm_p": 0, "n": 0, "improved": 0})
            ct, wt = float(row["cold_total"]), float(row["warm_total"])
            d["cold"] += ct; d["warm"] += wt
            d["cold_p"] += float(row["cold_p"]); d["warm_p"] += float(row["warm_p"])
            d["n"] += 1
            d["improved"] += 1 if wt < ct else 0
    # offline generation cost from instrumented manifests
    gen = {}
    for f in glob.glob(os.path.join(run_root, "steady", "**", "startup_rollout_manifest.json"), recursive=True):
        man = json.load(open(f))
        R = int(man.get("rollout_steps", 0))
        g = gen.setdefault(R, {"load": 0.0, "fwd": 0.0, "n": 0})
        if man.get("gen_rollout_forward_wall_s") is not None:
            g["load"] += man.get("gen_model_load_wall_s", 0.0)
            g["fwd"] += man.get("gen_rollout_forward_wall_s", 0.0)
            g["n"] += 1
    for R, d in sorted(by_r.items()):
        g = gen.get(R, {})
        out[R] = {
            "n": d["n"], "improved": d["improved"],
            "total_delta_pct": _pct(d["warm"] - d["cold"], d["cold"]),
            "p_delta_pct": _pct(d["warm_p"] - d["cold_p"], d["cold_p"]),
            "gen_load_s": g.get("load"), "gen_forward_s": g.get("fwd"), "gen_cases": g.get("n"),
            "gen_total_s": (g.get("load", 0.0) + g.get("fwd", 0.0)) if g else None,
        }
    return out


def _summarize_sanity(run_root):
    out = []
    for f in sorted(glob.glob(os.path.join(run_root, "sanity", "**", "abg_metrics_iterreduce_v1.json"), recursive=True)):
        obj = json.load(open(f))
        cold = obj.get("coldstart") or {}
        model = obj.get("iterreduce") or {}
        comp = obj.get("comparison") or {}
        mode = os.path.basename(os.path.dirname(os.path.dirname(f)))
        ct = cold.get("total_linear_iterations")
        mt = model.get("total_linear_iterations")
        out.append({
            "mode": mode, "case": _case_id(f),
            "cold_total": ct, "model_total": mt,
            "total_delta_pct": _pct((mt - ct) if (mt is not None and ct is not None) else 0, ct),
            "p_delta_pct": _pct(comp.get("pressure_iteration_delta", 0), cold.get("counts", {}).get("p")),
            "forward_s": model.get("total_forward_wall_s"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rr = args.run_root.rstrip("/")

    matched = _summarize_transient("transient_matched", _load_abg(os.path.join(rr, "transient_matched")))
    finestep = _summarize_transient("transient_finestep", _load_abg(os.path.join(rr, "transient_finestep")))
    steady = _summarize_steady(rr)
    sanity = _summarize_sanity(rr)

    def line(s):
        if not s.get("n"):
            print(f"  [missing] {s['name']}")
            return
        print(f"  {s['name']:<20} n={s['n']:>2} calls={s['calls']:>6}  "
              f"total {s['total_delta_pct']:+6.2f}%  p {s['p_delta_pct']:+6.2f}%  improved {s['improved']}/{s['n']}")
        print(f"      timing: load {s['load_s']:6.2f}s  forward {s['forward_s']:6.2f}s "
              f"({s['fwd_ms_per_call']:.2f} ms/call)  inject {s['inject_s']:6.2f}s  "
              f"field-handling {s['field_handling_s']:6.2f}s")
        print(f"      solver exec: cold {s['cold_exec_s']:6.1f}s -> model {s['model_exec_s']:6.1f}s "
              f"({s['exec_delta_pct']:+.2f}%)")

    print(f"\n=== CONSOLIDATED TIMING REPORT  {rr} ===\n")
    print("TRANSIENT panels:")
    line(matched)
    line(finestep)
    print("\nSTEADY simpleFoam startup-rollout:")
    for R, d in sorted(steady.items()):
        gen = (f"  offline gen: load {d['gen_load_s']:.2f}s + forward {d['gen_forward_s']:.2f}s "
               f"= {d['gen_total_s']:.2f}s ({d['gen_cases']} cases)") if d.get("gen_total_s") is not None else "  (no gen timing)"
        print(f"  r={R:<4} n={d['n']:>2}  total {d['total_delta_pct']:+6.2f}%  p {d['p_delta_pct']:+6.2f}%  "
              f"improved {d['improved']}/{d['n']}")
        print(f"     {gen}")
    print("\nSANITY controls:")
    for s in sanity:
        print(f"  {s['mode']:<15} case {s['case']:>3}: total {s['total_delta_pct']:+7.2f}%  "
              f"p {s['p_delta_pct']:+7.2f}%  (cold {s['cold_total']} -> {s['model_total']})")

    blob = {"run_root": rr, "transient_matched": matched, "transient_finestep": finestep,
            "steady": steady, "sanity": sanity}
    out_json = args.json or os.path.join(rr, "consolidated_timing.json")
    with open(out_json, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nwrote {out_json}\n")


if __name__ == "__main__":
    main()
