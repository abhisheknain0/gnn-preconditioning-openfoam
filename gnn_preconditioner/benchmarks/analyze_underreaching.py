#!/usr/bin/env python3
"""Empirical test of the receptive-field under-reaching hypothesis.

Experiment A (primary): spatial-scale decomposition of the one-step prediction
error. If a fixed-depth local GNN under-reaches the globally-coupled elliptic
pressure, the *pressure* error should be concentrated at large spatial scales
(low wavenumber) while the *velocity* error stays comparatively local. We
compute a radially-averaged power spectrum of the per-node error for Ux, Uy, p,
averaged over the held-out test trajectories, and report the fraction of error
power at large scales plus a spectral centroid.

Experiment B (secondary): per-component one-step RMSE for the four pilot
checkpoints (96/8, 96/12, 128/8, 128/12) plus the full-data 128/8 model, to see
whether more message-passing depth helps pressure relatively more than velocity.

Both are inference-only on existing checkpoints + the full test split. No
training, no external models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training.normalizer import Normalizer
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.dataset import (
    make_dataset,
    load_split_keys,
    load_static_graph_from_group,
)
from torch_geometric.loader import DataLoader

CKPT_ROOT = PROJECT_ROOT / "gnn_preconditioner" / "checkpoints"
DATASET = PROJECT_ROOT / "gnn_preconditioner" / "data_generation" / "datasets" / "iterreduce_full_surface_coarse_v1_t3p0.h5"
SPLIT = PROJECT_ROOT / "gnn_preconditioner" / "manifests" / "full_split_240_30_30_v1.json"

# (checkpoint dir, hidden_dim, message_passing_steps)
CKPTS = {
    "full 128/8":  ("full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1", 128, 8),
    "pilot 96/8":  ("pilot_surface_coarse_v1_stateonly_run1", 96, 8),
    "pilot 96/12": ("pilot_surface_coarse_v1_stateonly_mp12_run1", 96, 12),
    "pilot 128/8": ("pilot_surface_coarse_v1_stateonly_hd128_run1", 128, 8),
    "pilot 128/12":("pilot_surface_coarse_v1_stateonly_hd128_mp12_run1", 128, 12),
}


def _load_model(ckpt_dir: str, hd: int, mp: int, node_dim: int, edge_dim: int, device):
    norm = Normalizer()
    norm.load(CKPT_ROOT / ckpt_dir / "stats.json")
    model = MeshGraphNet(input_dim_nodes=node_dim, input_dim_edges=edge_dim,
                         hidden_dim=hd, num_mp_steps=mp).to(device)
    ckpt = torch.load(CKPT_ROOT / ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, norm


# --------------------------------------------------------------------------- #
# Experiment B: per-component one-step RMSE by architecture                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def exp_b(device) -> dict:
    rows = {}
    for name, (cdir, hd, mp) in CKPTS.items():
        norm = Normalizer()
        norm.load(CKPT_ROOT / cdir / "stats.json")
        ds = make_dataset(DATASET, SPLIT, norm, split="test", in_memory=True,
                          representation="surface_prism")
        model = MeshGraphNet(input_dim_nodes=int(ds.node_input_dim),
                             input_dim_edges=int(ds.edge_input_dim),
                             hidden_dim=hd, num_mp_steps=mp).to(device)
        ckpt = torch.load(CKPT_ROOT / cdir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        se = {"ux": 0.0, "uy": 0.0, "p": 0.0}
        n = 0
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_attr, batch.edge_index)
            pdu = norm.denormalize(pred[:, :2], Normalizer.KEY_TARGET_DU)
            pdp = norm.denormalize(pred[:, 2:3], Normalizer.KEY_TARGET_DP)
            tdu = norm.denormalize(batch.y[:, :2], Normalizer.KEY_TARGET_DU)
            tdp = norm.denormalize(batch.y[:, 2:3], Normalizer.KEY_TARGET_DP)
            se["ux"] += float(((pdu[:, 0] - tdu[:, 0]) ** 2).sum())
            se["uy"] += float(((pdu[:, 1] - tdu[:, 1]) ** 2).sum())
            se["p"]  += float(((pdp[:, 0] - tdp[:, 0]) ** 2).sum())
            n += batch.x.shape[0]
        rows[name] = {k: (v / n) ** 0.5 for k, v in se.items()}
        rows[name]["hidden_dim"] = hd
        rows[name]["mp_steps"] = mp
        print(f"[B] {name:13s}  RMSE  Ux={rows[name]['ux']:.3e}  Uy={rows[name]['uy']:.3e}  p={rows[name]['p']:.3e}")
    return rows


# --------------------------------------------------------------------------- #
# Experiment A: radial power spectrum of the one-step error (p vs U)           #
# --------------------------------------------------------------------------- #
def _radial_spectrum(field2d: np.ndarray, win: np.ndarray):
    """Radially-averaged power spectrum of a 2D field (already on a grid)."""
    f = field2d - np.nanmean(field2d)
    f = np.nan_to_num(f, nan=0.0) * win
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F) ** 2
    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    nbins = min(cx, cy)
    tbin = np.bincount(r.ravel(), P.ravel(), minlength=nbins + 1)[:nbins + 1]
    cnt = np.bincount(r.ravel(), minlength=nbins + 1)[:nbins + 1]
    return tbin / np.maximum(cnt, 1)  # radial power vs wavenumber bin


@torch.no_grad()
def exp_a(device, frames_per_case: int, grid: int) -> dict:
    cdir, hd, mp = CKPTS["full 128/8"]
    norm = Normalizer()
    norm.load(CKPT_ROOT / cdir / "stats.json")
    test_keys = load_split_keys(SPLIT)["test"]

    # spectra accumulators (per component); a fixed radial length set on first sample
    acc = {"ux": None, "uy": None, "p": None}
    nacc = 0
    model = None
    with h5py.File(DATASET, "r") as f:
        for key in test_keys:
            g = f[key]
            static = load_static_graph_from_group(g, norm, device=device)
            if model is None:
                node_dim = int(static["node_input_dim"]); edge_dim = int(static["edge_input_dim"])
                model = MeshGraphNet(input_dim_nodes=node_dim, input_dim_edges=edge_dim,
                                     hidden_dim=hd, num_mp_steps=mp).to(device)
                ckpt = torch.load(CKPT_ROOT / cdir / "best_model.pt", map_location=device)
                model.load_state_dict(ckpt["model_state"]); model.eval()
            edge_index = static["edge_index"]; edge_attr = static["edge_attr"]
            snf = static["static_node_feat"]
            pos = static["positions"].detach().cpu().numpy()[:, :2]
            vxy = torch.tensor(g["velocity_xy"][()], dtype=torch.float32, device=device)
            pg = torch.tensor(g["pressure_gauge"][()], dtype=torch.float32, device=device)
            nframes = vxy.shape[0]
            # evenly spaced developed-flow frames
            idxs = np.linspace(int(0.25 * nframes), nframes - 2, frames_per_case).astype(int)
            idxs = np.unique(np.clip(idxs, 0, nframes - 2))
            # grid setup for this case (bounding box)
            xmin, ymin = pos.min(0); xmax, ymax = pos.max(0)
            gx = np.linspace(xmin, xmax, grid); gy = np.linspace(ymin, ymax, grid)
            GX, GY = np.meshgrid(gx, gy)
            try:
                from scipy.interpolate import griddata
                def to_grid(vals):
                    return griddata(pos, vals, (GX, GY), method="linear")
            except Exception:
                # fallback: nearest-bin averaging
                ix = np.clip(((pos[:, 0] - xmin) / (xmax - xmin + 1e-12) * (grid - 1)).astype(int), 0, grid - 1)
                iy = np.clip(((pos[:, 1] - ymin) / (ymax - ymin + 1e-12) * (grid - 1)).astype(int), 0, grid - 1)
                def to_grid(vals):
                    acc2 = np.zeros((grid, grid)); cnt2 = np.zeros((grid, grid))
                    np.add.at(acc2, (iy, ix), vals); np.add.at(cnt2, (iy, ix), 1.0)
                    out = np.divide(acc2, cnt2, out=np.full_like(acc2, np.nan), where=cnt2 > 0)
                    return out
            win = np.outer(np.hanning(grid), np.hanning(grid))
            for i in idxs:
                u = vxy[i]; p = pg[i].unsqueeze(-1)
                x = torch.cat([norm.normalize(u, Normalizer.KEY_U_XY),
                               norm.normalize(p, Normalizer.KEY_P_GAUGE), snf], dim=-1)
                pred = model(x, edge_attr, edge_index)
                du = norm.denormalize(pred[:, :2], Normalizer.KEY_TARGET_DU)
                dp = norm.denormalize(pred[:, 2:3], Normalizer.KEY_TARGET_DP)
                tdu = vxy[i + 1] - u
                tdp = (pg[i + 1].unsqueeze(-1)) - p
                err = {
                    "ux": (du[:, 0] - tdu[:, 0]).detach().cpu().numpy(),
                    "uy": (du[:, 1] - tdu[:, 1]).detach().cpu().numpy(),
                    "p":  (dp[:, 0] - tdp[:, 0]).detach().cpu().numpy(),
                }
                for comp, vals in err.items():
                    spec = _radial_spectrum(to_grid(vals), win)
                    if acc[comp] is None:
                        acc[comp] = np.zeros_like(spec)
                    L = min(len(acc[comp]), len(spec))
                    acc[comp][:L] += spec[:L]
                nacc += 1
            print(f"[A] {key:16s} frames={len(idxs)} nodes={pos.shape[0]}")
    spectra = {c: (acc[c] / max(nacc, 1)) for c in acc}
    # summary: normalize each spectrum to unit total power; large-scale fraction
    k = np.arange(len(spectra["p"]))
    summary = {}
    klo = max(2, int(0.15 * len(k)))  # "large scale" = lowest ~15% of wavenumbers (skip k=0 DC)
    for c, s in spectra.items():
        sn = s[1:] / max(s[1:].sum(), 1e-30)  # drop DC bin
        kk = k[1:]
        large_scale_frac = float(sn[: klo].sum())
        centroid = float((kk * sn).sum())
        summary[c] = {"large_scale_frac": large_scale_frac, "spectral_centroid": centroid}
        print(f"[A] {c}: large-scale power frac={large_scale_frac:.3f}  centroid_k={centroid:.2f}")
    return {"k": k.tolist(),
            "spectra": {c: spectra[c].tolist() for c in spectra},
            "summary": summary,
            "n_samples": nacc, "klo_bins": klo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames-per-case", type=int, default=6)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "gnn_preconditioner" / "benchmarks" / "underreaching_results.json")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device={device}  frames/case={args.frames_per_case}  grid={args.grid}")

    res = {}
    res["exp_a"] = exp_a(device, args.frames_per_case, args.grid)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"wrote exp_a -> {args.out}")
    if not args.skip_b:
        try:
            res["exp_b"] = exp_b(device)
            args.out.write_text(json.dumps(res, indent=2))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[B] FAILED (exp_a preserved): {e}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
