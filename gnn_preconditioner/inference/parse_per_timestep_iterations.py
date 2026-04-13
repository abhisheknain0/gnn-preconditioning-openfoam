#!/usr/bin/env python3
"""Parse an OpenFOAM solver log into per-time-step iteration counts.

Handles adaptive time stepping — each row is one solver time step with its
actual simulation time and deltaT.

Usage:
    python parse_per_timestep_iterations.py <log_path> [--steady-start <time>] [--json]
"""
import argparse
import json
import re
import sys
from pathlib import Path


def parse_log(path: Path):
    """Return a list of dicts, one per time step."""
    text = path.read_text()
    steps = []
    current = None

    for line in text.splitlines():
        # New time step: "Time = 0.01" or "Time = 0.01s"
        m = re.match(r"^Time = ([0-9.eE+-]+)s?$", line)
        if m:
            if current is not None:
                steps.append(current)
            current = {
                "time": float(m.group(1)),
                "deltaT": None,
                "p": 0,
                "Ux": 0,
                "Uy": 0,
            }
            continue

        if current is None:
            continue

        # deltaT: "deltaT = 0.001"
        m = re.match(r"^deltaT = ([0-9.eE+-]+)$", line)
        if m:
            current["deltaT"] = float(m.group(1))
            continue

        # Solver iterations: "Solving for Ux, ... No Iterations 5"
        m = re.search(r"Solving for (Ux|Uy|p),.*No Iterations (\d+)", line)
        if m:
            current[m.group(1)] += int(m.group(2))

    if current is not None:
        steps.append(current)

    for s in steps:
        s["total"] = s["p"] + s["Ux"] + s["Uy"]

    return steps


def summarize_window(steps, t_start=None, t_end=None):
    """Summarize iteration counts within a time window."""
    filtered = steps
    if t_start is not None:
        filtered = [s for s in filtered if s["time"] >= t_start]
    if t_end is not None:
        filtered = [s for s in filtered if s["time"] <= t_end]

    if not filtered:
        return None

    return {
        "n_steps": len(filtered),
        "t_min": filtered[0]["time"],
        "t_max": filtered[-1]["time"],
        "total": sum(s["total"] for s in filtered),
        "p": sum(s["p"] for s in filtered),
        "Ux": sum(s["Ux"] for s in filtered),
        "Uy": sum(s["Uy"] for s in filtered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--steady-start", type=float, default=None,
                        help="Start time for steady-state window")
    parser.add_argument("--steady-end", type=float, default=None,
                        help="End time for steady-state window")
    parser.add_argument("--json", action="store_true",
                        help="Output full per-step data as JSON")
    args = parser.parse_args()

    steps = parse_log(args.log_path)

    if not steps:
        print("No time steps found in log.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {"steps": steps}
        if args.steady_start is not None:
            output["steady_summary"] = summarize_window(
                steps, t_start=args.steady_start, t_end=args.steady_end
            )
            output["transient_summary"] = summarize_window(
                steps, t_end=args.steady_start
            )
        output["full_summary"] = summarize_window(steps)
        print(json.dumps(output, indent=2))
    else:
        # Human-readable table
        print(f"{'Time':>10s}  {'deltaT':>10s}  {'p':>5s}  {'Ux':>5s}  {'Uy':>5s}  {'total':>6s}")
        print("-" * 55)
        for s in steps:
            dt_str = f"{s['deltaT']:.6g}" if s["deltaT"] is not None else "?"
            print(f"{s['time']:10.6f}  {dt_str:>10s}  {s['p']:5d}  {s['Ux']:5d}  {s['Uy']:5d}  {s['total']:6d}")

        full = summarize_window(steps)
        print(f"\nFull run: {full['n_steps']} steps, "
              f"total={full['total']}, p={full['p']}, Ux={full['Ux']}, Uy={full['Uy']}")

        if args.steady_start is not None:
            trans = summarize_window(steps, t_end=args.steady_start)
            steady = summarize_window(steps, t_start=args.steady_start,
                                      t_end=args.steady_end)
            if trans:
                print(f"Transient (t < {args.steady_start}): {trans['n_steps']} steps, "
                      f"total={trans['total']}, p={trans['p']}")
            if steady:
                print(f"Steady (t >= {args.steady_start}): {steady['n_steps']} steps, "
                      f"total={steady['total']}, p={steady['p']}")


if __name__ == "__main__":
    main()
