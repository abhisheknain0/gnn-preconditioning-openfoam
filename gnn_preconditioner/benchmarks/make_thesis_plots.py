#!/usr/bin/env python3
"""Unified thesis line/bar plots — one shared palette and style.

Regenerates, into thesis/images/:
  learning_curve.pdf      training/validation loss + LR   (checkpoint history.json)
  noise_rollout_rmse.pdf  baseline vs noise rollout RMSE   (rollout_comparison.json)
  results_summary.pdf     solver-panel iteration deltas    (aggregate.json)
  domain_mesh_bc.pdf      example mesh + geometry/BC labels (case_011 geometry.msh)
  arch_study.pdf          pilot architecture sweep (4 configs) -- NOT used in the
                          thesis (dropped from 4.1.2; redundant with Table 4.1)

Regenerate just ONE figure without rerunning all of them: import this module,
call apply_style(), then the single function, e.g.:
  import make_thesis_plots as m; m.apply_style(); m.domain_mesh_bc()
Running this file as __main__ regenerates every figure above.

All four share PALETTE + apply_style() so the line/bar figures are visually
consistent across the thesis. (Flow-field figures use a separate sequential
colormap; drawio schematics share their own box palette.)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CK = REPO / "gnn_preconditioner/checkpoints"
RUNS = REPO / "gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521"
IMG = REPO / "thesis/images"

# --- shared palette + style --------------------------------------------------
PALETTE = {
    "primary": "#2166AC",   # deep blue  - training, baseline, total iterations
    "accent":  "#B2182B",   # deep red   - validation, noise, pressure iterations
    "muted":   "#6E6E6E",   # grey       - context (learning rate, unselected bars)
    "ok":      "#1B7837",   # deep green - positive accent
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "grid.color": "#222222",
        "grid.alpha": 0.12,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    out = IMG / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


# --- 1. learning curve -------------------------------------------------------
def learning_curve():
    h = json.loads((CK / "full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1"
                    / "history.json").read_text())
    ep = [e["epoch"] for e in h]
    train = [e["train"]["loss"] for e in h]
    lr = [e["lr"] for e in h]
    vep = [e["epoch"] for e in h if e.get("valid")]
    vloss = [e["valid"]["loss"] for e in h if e.get("valid")]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(ep, train, color=PALETTE["primary"], lw=1.7, label="Training loss")
    ax.plot(vep, vloss, color=PALETTE["accent"], lw=1.4, marker="o", ms=3.5,
            label="Validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_xlim(0, 100)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PALETTE["muted"])
    ax2.plot(ep, lr, color=PALETTE["muted"], lw=1.1, ls="--", label="Learning rate")
    ax2.set_ylabel("Learning rate", color=PALETTE["muted"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["muted"])
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    save(fig, "learning_curve")
    i = int(np.argmin(vloss))
    print(f"  val min {vloss[i]:.6f} @ epoch {vep[i]}; "
          f"train e5={train[4]:.3f} e100={train[-1]:.3f}")


# --- 2. architecture study ---------------------------------------------------
def arch_study():
    labels = ["96 / 8", "96 / 12", "128 / 8", "128 / 12"]
    loss = [0.05631, 0.05146, 0.03433, 0.02464]
    sel = 2  # 128/8 chosen for the mainline (inference cost)
    colors = [PALETTE["muted"]] * 4
    colors[sel] = PALETTE["primary"]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    bars = ax.bar(labels, loss, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
    bars[sel].set_edgecolor(PALETTE["accent"])
    bars[sel].set_linewidth(1.8)
    ax.annotate("selected", (sel, loss[sel]), ha="center", va="bottom",
                xytext=(0, 15), textcoords="offset points",
                fontsize=9, fontweight="bold", color=PALETTE["accent"])
    for i, v in enumerate(loss):
        ax.annotate(f"{v:.4f}", (i, v), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=8)
    ax.set_ylabel("Best pilot validation loss")
    ax.set_xlabel("Hidden dim / message-passing steps")
    ax.grid(axis="y")
    ax.set_ylim(0, max(loss) * 1.2)
    save(fig, "arch_study")


# --- 3. noise rollout RMSE ---------------------------------------------------
def noise_rollout_rmse():
    r = json.loads((CK / "full_surface_coarse_v1_t3p0_state_hd128_seed1337_noise0p05_run1"
                    / "comparison/rollout_comparison.json").read_text())
    n = len(r["baseline"]["vel_rmse_per_step_mean"])
    steps = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, key, title in [(axes[0], "vel", "Velocity RMSE"),
                           (axes[1], "p", "Pressure RMSE")]:
        for src, color, lab in [("baseline", PALETTE["primary"], "Baseline"),
                                ("noise", PALETTE["accent"], "Noise $\\sigma$=0.05")]:
            m = np.array(r[src][f"{key}_rmse_per_step_mean"])
            s = np.array(r[src][f"{key}_rmse_per_step_std"])
            ax.plot(steps, m, color=color, lw=1.6, label=lab)
            ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Rollout step")
        ax.set_xlim(1, n)
        ax.grid(True, axis="y", alpha=0.12)
    axes[0].set_ylabel("RMSE")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "noise_rollout_rmse")
    print(f"  wins vel {r['noise_wins_velocity']}/{r['total_trajectories']}, "
          f"p {r['noise_wins_pressure']}/{r['total_trajectories']}")


# --- 4. results summary ------------------------------------------------------
def results_summary():
    def pct(a, b):
        return 100.0 * (b - a) / a

    def trans(name):
        d = json.loads((RUNS / name / "aggregate.json").read_text())
        return pct(d["cold_total"], d["model_total"]), pct(d["cold_p"], d["model_p"])

    def steady(k):
        d = json.loads((RUNS / "steady_simplefoam_startup_rollout_r30_r100_30case"
                        / "aggregate.json").read_text())["rollouts"][k]
        return d["delta_pct"], d["delta_p_pct"]

    labels = ["Matched\ntransient\n($\\Delta t$=0.01)",
              "Fine dt-scaled\ntransient\n($\\Delta t$=0.002)",
              "Steady\nwarm-start\n(rollout 30)",
              "Steady\nwarm-start\n(rollout 100)"]
    mt, mp = trans("transient_matched_dt0p01_every_step")
    ft, fp = trans("transient_fine_dt0p002_dtscaled_every_step")
    r30t, r30p = steady("30")
    r100t, r100p = steady("100")
    total = [mt, ft, r30t, r100t]
    press = [mp, fp, r30p, r100p]

    x = list(range(4))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    b1 = ax.bar([xi - w / 2 for xi in x], total, w, label="Total iterations",
                color=PALETTE["primary"], linewidth=0, zorder=3)
    b2 = ax.bar([xi + w / 2 for xi in x], press, w, label="Pressure iterations",
                color=PALETTE["accent"], linewidth=0, zorder=3)
    for bars in (b1, b2):
        for rr in bars:
            hh = rr.get_height()
            ax.annotate(f"{hh:+.1f}", (rr.get_x() + rr.get_width() / 2, hh),
                        ha="center", va="bottom" if hh >= 0 else "top",
                        xytext=(0, 3 if hh >= 0 else -3), textcoords="offset points",
                        fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="black", linewidth=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Iteration change vs cold baseline (%)")
    ax.set_ylim(-32, 30)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right")
    save(fig, "results_summary")


# --- 5. domain / mesh / boundary conditions ---------------------------------
def domain_mesh_bc():
    from matplotlib.tri import Triangulation
    case = ("gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0/"
            "case_011_U0.359_H0.0125_L0.0749/system/geometry.msh")
    lines = (REPO / case).read_text().splitlines()
    nodes, tris = {}, []
    i = 0
    while i < len(lines):
        if lines[i].startswith("$Nodes"):
            n = int(lines[i + 1]); i += 2
            for j in range(n):
                p = lines[i + j].split()
                nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
            i += n
        elif lines[i].startswith("$Elements"):
            m = int(lines[i + 1]); i += 2
            for j in range(m):
                p = lines[i + j].split()
                if int(p[1]) == 2:                      # 3-node triangle
                    nt = int(p[2])
                    tris.append(list(map(int, p[3 + nt:6 + nt])))
            i += m
        else:
            i += 1
    zmin = min(v[2] for v in nodes.values())
    front = [t for t in tris if all(abs(nodes[k][2] - zmin) < 1e-6 for k in t)]
    used = sorted({k for t in front for k in t})
    rm = {k: j for j, k in enumerate(used)}
    X = np.array([nodes[k][0] for k in used])
    Y = np.array([nodes[k][1] for k in used])
    T = np.array([[rm[k] for k in t] for t in front])
    triang = Triangulation(X, Y, T)

    xmid, ymid, ymax, xmax = 0.07487, 0.01254, 0.03, float(X.max())
    outline = np.array([(0, ymid), (0, ymax), (xmax, ymax), (xmax, 0),
                        (xmid, 0), (xmid, ymid), (0, ymid)])
    blue, red, grey = PALETTE["primary"], PALETTE["accent"], PALETTE["muted"]

    fig, ax = plt.subplots(figsize=(8.4, 1.8))
    ax.triplot(triang, color="#b9b9b9", lw=0.18)
    ax.plot(outline[:, 0], outline[:, 1], color="#333333", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.055, 0.42)
    ax.set_ylim(-0.014, 0.052)
    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_yticks([0, 0.03]); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(labelsize=8); ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.annotate("inlet", xy=(0.001, (ymid + ymax) / 2), xytext=(0.055, ymax + 0.008),
                ha="center", color=blue, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=blue, lw=1.2, mutation_scale=8))
    ax.annotate("Outlet", xy=(xmax - 0.001, ymax / 2), xytext=(xmax - 0.055, ymax + 0.008),
                ha="center", color=red, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=red, lw=1.2, mutation_scale=8))
    ax.annotate("fixedWalls", xy=(0.22, ymax), xytext=(0.22, ymax + 0.016),
                ha="center", va="bottom", color=grey, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=grey, lw=1.2, mutation_scale=8))
    ax.annotate("", xy=(-0.032, 0), xytext=(-0.032, ymax),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=0.8, mutation_scale=6))
    ax.text(-0.032, ymax + 0.003, "$y_{\\max}$", ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(-0.016, 0), xytext=(-0.016, ymid),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=0.8, mutation_scale=6))
    ax.text(-0.016, ymid + 0.003, "$y_{\\mathrm{mid}}$", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("", xy=(0, -0.0075), xytext=(xmid, -0.0075),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=0.8, mutation_scale=6))
    ax.text(xmid / 2, -0.013, "$x_{\\mathrm{mid}}$", ha="center", va="top", fontsize=9)
    save(fig, "domain_mesh_bc")


if __name__ == "__main__":
    apply_style()
    learning_curve()
    arch_study()
    noise_rollout_rmse()
    results_summary()
    domain_mesh_bc()
