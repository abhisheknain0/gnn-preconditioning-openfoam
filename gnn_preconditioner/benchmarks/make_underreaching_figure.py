#!/usr/bin/env python3
"""Figure for the receptive-field under-reaching evidence (Exp A).

Two panels: (a) radially-averaged, normalised error power spectrum for u_x, u_y, p
on log-log axes; (b) cumulative error-power fraction vs wavenumber. Pressure error
is concentrated at the largest scales (lowest wavenumber); velocity, especially
u_y, carries comparatively more local (high-k) content. Spectral centroids are
annotated.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "gnn_preconditioner" / "benchmarks" / "underreaching_results.json"
OUT = ROOT / "thesis" / "images" / "error_spectrum.pdf"

COL = {"p": "#C44E52", "ux": "#4C72B0", "uy": "#55A868"}
LAB = {"p": r"pressure $p$", "ux": r"velocity $u_x$", "uy": r"velocity $u_y$"}

# Nimbus Sans (Helvetica clone) to match every other thesis figure, and a
# document-locked type scale: text authored at target/scale renders at {11/10/9}pt
# once placed at `frac` of the 451pt column (see thesis figure design system).
_COL, RENDER = 451.0, {"title": 11.0, "label": 10.0, "small": 9.0}
_SCALE = (0.98 * _COL) / 556.5  # error_spectrum placed at width=98%, native 556.5pt


def fs(pt):
    return pt / _SCALE


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": fs(RENDER["label"]),
    "axes.labelsize": fs(RENDER["label"]),
    "axes.titlesize": fs(RENDER["title"]),
    "legend.fontsize": fs(RENDER["small"]),
    "xtick.labelsize": fs(RENDER["small"]),
    "ytick.labelsize": fs(RENDER["small"]),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def main():
    a = json.load(open(RES))["exp_a"]
    spec = {c: np.asarray(a["spectra"][c], float) for c in a["spectra"]}
    cen = a["summary"]
    # drop DC (k=0), normalise to unit total power
    norm = {c: spec[c][1:] / max(spec[c][1:].sum(), 1e-30) for c in spec}
    kk = np.arange(1, 1 + len(norm["p"]))

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.1))
    for c in ["ux", "uy", "p"]:
        ax[0].loglog(kk, norm[c], color=COL[c], lw=2.0, label=LAB[c])
    ax[0].set_xlabel(r"wavenumber $k$ (radial bin)")
    ax[0].set_ylabel("normalised error power")
    ax[0].set_title("(a) error power spectrum", loc="left", fontsize=fs(RENDER["title"]))
    ax[0].legend(frameon=False, fontsize=fs(RENDER["small"]))

    for c in ["ux", "uy", "p"]:
        ax[1].semilogx(kk, np.cumsum(norm[c]), color=COL[c], lw=2.0, label=LAB[c])
    ax[1].set_xlabel(r"wavenumber $k$")
    ax[1].set_ylabel("cumulative power fraction")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title("(b) cumulative error power", loc="left", fontsize=fs(RENDER["title"]))
    ctext = "spectral centroid   " + "    ".join(
        f"{LAB[c].split()[-1]} $k_c$={cen[c]['spectral_centroid']:.1f}" for c in ["p", "ux", "uy"]
    )
    ax[1].text(0.03, 0.06, ctext, transform=ax[1].transAxes, fontsize=fs(RENDER["small"]), color="#555555")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")
    for c in ["p", "ux", "uy"]:
        print(f"  {c}: centroid={cen[c]['spectral_centroid']:.2f}  large-scale-frac={cen[c]['large_scale_frac']:.3f}")


if __name__ == "__main__":
    main()
