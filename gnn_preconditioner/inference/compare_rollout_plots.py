#!/usr/bin/env python3
"""Per-step rollout RMSE comparison between two checkpoints across a split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training import config
from gnn_preconditioner.training.dataset import load_split_keys, load_static_graph_from_group
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer


def _resolve_arch(checkpoint_path: Path) -> tuple[int, int]:
    hd, mp = int(config.HIDDEN_DIM), int(config.MESSAGE_PASSING_STEPS)
    summary_path = checkpoint_path.parent / "train_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        hd = int(s.get("hidden_dim", hd))
        mp = int(s.get("message_passing_steps", mp))
    return hd, mp


def _load_model(checkpoint_path: Path, stats_path: Path, device: torch.device) -> tuple[MeshGraphNet, Normalizer]:
    normalizer = Normalizer()
    normalizer.load(stats_path)
    hd, mp = _resolve_arch(checkpoint_path)
    model = MeshGraphNet(
        input_dim_nodes=int(config.NUM_NODE_INPUT),
        input_dim_edges=int(config.NUM_EDGE_INPUT),
        hidden_dim=hd,
        num_mp_steps=mp,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, normalizer


@torch.no_grad()
def _rollout_per_step(
    model: MeshGraphNet,
    normalizer: Normalizer,
    dataset_path: Path,
    traj_keys: list[str],
    rollout_steps: int,
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(dataset_path, "r") as f:
        for key in traj_keys:
            g = f[key]
            velocity_xy = torch.tensor(g["velocity_xy"][()], dtype=torch.float32, device=device)
            pressure = torch.tensor(g["pressure_gauge"][()], dtype=torch.float32, device=device)
            static = load_static_graph_from_group(g, normalizer, device=device)
            u_xy = velocity_xy[0]
            p_g = pressure[0].unsqueeze(-1)
            max_step = min(rollout_steps, velocity_xy.shape[0] - 1)
            internal = (static["boundary_mask"].sum(dim=-1) == 0)
            vel_rmse = np.zeros(max_step, dtype=np.float32)
            p_rmse = np.zeros(max_step, dtype=np.float32)
            for step in range(max_step):
                x = torch.cat([
                    normalizer.normalize(u_xy, Normalizer.KEY_U_XY),
                    normalizer.normalize(p_g, Normalizer.KEY_P_GAUGE),
                    static["static_node_feat"],
                ], dim=-1)
                pred = model(x, static["edge_attr"], static["edge_index"])
                du = normalizer.denormalize(pred[:, :2], Normalizer.KEY_TARGET_DU)
                dp = normalizer.denormalize(pred[:, 2:3], Normalizer.KEY_TARGET_DP)
                u_xy = u_xy + du
                p_g = p_g + dp
                tg_u = velocity_xy[step + 1]
                tg_p = pressure[step + 1].unsqueeze(-1)
                vel_rmse[step] = float(torch.sqrt(torch.mean((u_xy[internal] - tg_u[internal]) ** 2)).item())
                p_rmse[step] = float(torch.sqrt(torch.mean((p_g[internal] - tg_p[internal]) ** 2)).item())
            results[key] = (vel_rmse, p_rmse)
    return results


def _plot_mean_band(ax, steps, data, label, color):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    ax.plot(steps, mean, label=label, color=color, lw=1.8)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ckpt", type=Path, required=True)
    ap.add_argument("--baseline-stats", type=Path, required=True)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--noise-ckpt", type=Path, required=True)
    ap.add_argument("--noise-stats", type=Path, required=True)
    ap.add_argument("--noise-label", default="noise σ=0.05")
    ap.add_argument("--dataset-path", type=Path, default=config.DATASET_PATH)
    ap.add_argument("--split-path", type=Path, default=config.SPLIT_PATH)
    ap.add_argument("--split", default="test")
    ap.add_argument("--rollout-steps", type=int, default=100)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    keys = load_split_keys(args.split_path)[args.split]
    print(f"[compare] {args.split} split: {len(keys)} trajectories, {args.rollout_steps} rollout steps")

    print(f"[compare] rolling out {args.baseline_label}...")
    b_model, b_norm = _load_model(args.baseline_ckpt, args.baseline_stats, device)
    b_results = _rollout_per_step(b_model, b_norm, args.dataset_path, keys, args.rollout_steps, device)
    del b_model

    print(f"[compare] rolling out {args.noise_label}...")
    n_model, n_norm = _load_model(args.noise_ckpt, args.noise_stats, device)
    n_results = _rollout_per_step(n_model, n_norm, args.dataset_path, keys, args.rollout_steps, device)
    del n_model

    b_vel = np.stack([b_results[k][0] for k in keys])
    b_p = np.stack([b_results[k][1] for k in keys])
    n_vel = np.stack([n_results[k][0] for k in keys])
    n_p = np.stack([n_results[k][1] for k in keys])
    steps = np.arange(1, args.rollout_steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, b, n, ylabel in [
        (ax1, b_vel, n_vel, "velocity RMSE (fluid-only)"),
        (ax2, b_p, n_p, "pressure RMSE (fluid-only)"),
    ]:
        _plot_mean_band(ax, steps, b, args.baseline_label, "C0")
        _plot_mean_band(ax, steps, n, args.noise_label, "C3")
        ax.set_xlabel("rollout step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Rollout error vs step ({args.split} split, mean ± std over {len(keys)} trajectories)")
    fig.savefig(args.output_dir / "rollout_rmse_vs_step.png", dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, b, n, title in [
        (ax1, b_vel[:, -1], n_vel[:, -1], "Final-step velocity RMSE"),
        (ax2, b_p[:, -1], n_p[:, -1], "Final-step pressure RMSE"),
    ]:
        ax.scatter(b, n, s=22, color="C3", alpha=0.8, edgecolor="black", linewidth=0.4)
        lo = float(min(b.min(), n.min())) * 0.95
        hi = float(max(b.max(), n.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{args.baseline_label} RMSE")
        ax.set_ylabel(f"{args.noise_label} RMSE")
        ax.set_title(f"{title}\n(below y=x → noise wins)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(args.output_dir / "final_step_scatter.png", dpi=150)
    plt.close(fig)

    summary = {
        "split": args.split,
        "keys": list(keys),
        "rollout_steps": int(args.rollout_steps),
        "baseline_label": args.baseline_label,
        "noise_label": args.noise_label,
        "baseline": {
            "vel_rmse_per_step_mean": b_vel.mean(axis=0).tolist(),
            "vel_rmse_per_step_std": b_vel.std(axis=0).tolist(),
            "p_rmse_per_step_mean": b_p.mean(axis=0).tolist(),
            "p_rmse_per_step_std": b_p.std(axis=0).tolist(),
            "vel_rmse_final_per_traj": b_vel[:, -1].tolist(),
            "p_rmse_final_per_traj": b_p[:, -1].tolist(),
        },
        "noise": {
            "vel_rmse_per_step_mean": n_vel.mean(axis=0).tolist(),
            "vel_rmse_per_step_std": n_vel.std(axis=0).tolist(),
            "p_rmse_per_step_mean": n_p.mean(axis=0).tolist(),
            "p_rmse_per_step_std": n_p.std(axis=0).tolist(),
            "vel_rmse_final_per_traj": n_vel[:, -1].tolist(),
            "p_rmse_final_per_traj": n_p[:, -1].tolist(),
        },
        "noise_wins_velocity": int(np.sum(n_vel[:, -1] < b_vel[:, -1])),
        "noise_wins_pressure": int(np.sum(n_p[:, -1] < b_p[:, -1])),
        "total_trajectories": int(len(keys)),
    }
    (args.output_dir / "rollout_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[compare] velocity wins: {summary['noise_wins_velocity']}/{summary['total_trajectories']}")
    print(f"[compare] pressure wins: {summary['noise_wins_pressure']}/{summary['total_trajectories']}")
    print(f"[compare] outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
