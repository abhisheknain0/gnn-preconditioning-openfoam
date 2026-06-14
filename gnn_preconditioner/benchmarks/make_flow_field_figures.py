#!/usr/bin/env python3
"""Flow-field figures for the thesis (true aspect ratio, no vertical stretch).

Source data: the stored rollout frames for test trajectory 188
(`.../noise0p05_run1/comparison/frames_trajectory_188.npz`), which contain the
node positions, the fluid-domain polygon, and ground-truth / baseline / noise
velocity and pressure fields over 101 rollout steps.

Outputs:
  thesis/images/flow_triptych.pdf      -- hero: cold init -> GNN warm -> converged
  thesis/images/noise_frame_strip.pdf  -- corrected GT/baseline/noise strip
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "gnn_preconditioner/benchmarks/frames_trajectory_188_r200.npz"
IMG = REPO / "thesis/images"

d = np.load(NPZ)
pos = d["positions"].astype(float)
x, y = pos[:, 0], pos[:, 1]
poly = d["polygon"].astype(float)
poly_closed = np.vstack([poly, poly[0]])

# --- masked triangulation (exclude the solid step + long bridging tris) ------
triang = mtri.Triangulation(x, y)
tri_pts = pos[triang.triangles]                      # (ntri, 3, 2)
centroids = tri_pts.mean(axis=1)
inside = MplPath(poly).contains_points(centroids)
edges = np.stack([
    np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 1], axis=1),
    np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 2], axis=1),
    np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 0], axis=1),
], axis=1)
max_edge = edges.max(axis=1)
edge_thresh = np.percentile(max_edge, 95) * 1.8
triang.set_mask((~inside) | (max_edge > edge_thresh))

# --- shared style + document-locked figure type scale -----------------------
# Nimbus Sans (Helvetica clone) to match every other thesis figure, and text
# authored at target/scale so it renders at the {11/10/9}pt type scale once the
# figure is placed at `frac` of the 451pt column (see thesis figure design system).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
})
_COL = 451.0
RENDER = {"title": 11.0, "label": 10.0, "small": 9.0}
_SCALE = 1.0


def fs(pt):
    return pt / _SCALE


def set_scale(native_width_pt, frac=1.0):
    global _SCALE
    _SCALE = (frac * _COL) / native_width_pt


def umag(vel):
    return np.linalg.norm(vel, axis=-1)


def draw(ax, field, vmin, vmax, cmap):
    tcf = ax.tricontourf(triang, field, levels=np.linspace(vmin, vmax, 24),
                         cmap=cmap, extend="max")
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="#222222", lw=0.7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.01, y=0.05)
    return tcf


# ============================ HERO TRIPTYCH ==================================
def make_triptych(cmap="turbo"):
    gt, bl = d["gt_vel"], d["baseline_vel"]
    fields = [umag(gt[1]), umag(bl[100]), umag(gt[100])]
    labels = ["Cold start  (early field)",
              "GNN warm start  (100-step rollout)",
              "Converged solution  (reference)"]
    vmax = float(np.percentile(umag(gt[100]), 99.5))
    set_scale(463.6, 1.0)  # placed at width=100%
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.05))
    for ax, f, lab in zip(axes, fields, labels):
        tcf = draw(ax, f, 0.0, vmax, cmap)
        ax.set_title(lab, fontsize=fs(RENDER["title"]), loc="left", pad=2)
    cbar = fig.colorbar(tcf, ax=axes, fraction=0.025, pad=0.012,
                        ticks=np.linspace(0, vmax, 5))
    cbar.set_label("velocity magnitude  $|\\mathbf{u}|$  (m/s)", fontsize=fs(RENDER["label"]))
    cbar.ax.tick_params(labelsize=fs(RENDER["small"]))
    out = IMG / "flow_triptych.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", out, "vmax=%.3f" % vmax)


# ===================== CORRECTED NOISE FRAME STRIP ===========================
def make_frame_strip(cmap="turbo", steps=(100, 200)):
    rows = [("OpenFOAM", d["gt_vel"]),
            ("Baseline", d["baseline_vel"]),
            ("Noise  $\\sigma$=0.05", d["noise_vel"])]
    vmax = float(np.percentile(umag(d["gt_vel"][100]), 99.5))
    set_scale(558.6, 1.0)  # placed at width=100%
    fig, axes = plt.subplots(3, len(steps), figsize=(7.6, 2.6), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.0, wspace=0.0, h_pad=0.015, w_pad=0.02)
    for r, (rlabel, vel) in enumerate(rows):
        for c, s in enumerate(steps):
            tcf = draw(axes[r, c], umag(vel[s]), 0.0, vmax, cmap)
        axes[r, 0].set_title(rlabel, loc="left", fontsize=fs(RENDER["label"]), pad=1)
    for c, s in enumerate(steps):
        axes[2, c].set_xlabel(f"step {s}", fontsize=fs(RENDER["small"]), labelpad=1)
    cbar = fig.colorbar(tcf, ax=axes, fraction=0.015, pad=0.012, aspect=28,
                        ticks=np.linspace(0, vmax, 5))
    cbar.set_label("$|\\mathbf{u}|$  (m/s)", fontsize=fs(RENDER["small"]))
    cbar.ax.tick_params(labelsize=fs(RENDER["small"]))
    out = IMG / "noise_frame_strip.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    make_triptych()
    make_frame_strip()
