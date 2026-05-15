#!/usr/bin/env python3
"""Dump baseline+noise rollout arrays for one trajectory to an NPZ for thesis figures."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training import config
from gnn_preconditioner.training.dataset import load_static_graph_from_group
from gnn_preconditioner.inference.case_paths import resolve_case_dir
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer

_POINT_RE = re.compile(r"Point\((\d+)\)\s*=\s*\{([^,]+),\s*([^,]+),\s*[^,]+,\s*[^}]+\};")


def _bfs_polygon(geo_path: Path) -> np.ndarray:
    pts: dict[int, tuple[float, float]] = {}
    for line in geo_path.read_text(encoding="utf-8").splitlines():
        m = _POINT_RE.match(line.strip())
        if m is not None:
            pts[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return np.asarray([pts[1], pts[2], pts[5], pts[6], pts[8], pts[4]], dtype=np.float32)


def _resolve_arch(ckpt_path: Path) -> tuple[int, int]:
    hd, mp = int(config.HIDDEN_DIM), int(config.MESSAGE_PASSING_STEPS)
    summary_path = ckpt_path.parent / "train_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        hd = int(s.get("hidden_dim", hd))
        mp = int(s.get("message_passing_steps", mp))
    return hd, mp


@torch.no_grad()
def _rollout(ckpt_path: Path, stats_path: Path, dataset_path: Path, traj_key: str,
             rollout_steps: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    normalizer = Normalizer()
    normalizer.load(stats_path)
    hd, mp = _resolve_arch(ckpt_path)
    with h5py.File(dataset_path, "r") as f:
        g = f[traj_key]
        velocity_xy = torch.tensor(g["velocity_xy"][()], dtype=torch.float32, device=device)
        pressure = torch.tensor(g["pressure_gauge"][()], dtype=torch.float32, device=device)
        static = load_static_graph_from_group(g, normalizer, device=device)
    model = MeshGraphNet(
        input_dim_nodes=int(static["node_input_dim"]),
        input_dim_edges=int(static["edge_input_dim"]),
        hidden_dim=hd, num_mp_steps=mp,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    u_xy = velocity_xy[0]
    p_g = pressure[0].unsqueeze(-1)
    max_step = min(rollout_steps, velocity_xy.shape[0] - 1)
    pred_vel = [u_xy.detach().cpu().numpy()]
    pred_p = [p_g.detach().cpu().squeeze(-1).numpy()]
    for _ in range(max_step):
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
        pred_vel.append(u_xy.detach().cpu().numpy())
        pred_p.append(p_g.detach().cpu().squeeze(-1).numpy())
    return np.stack(pred_vel), np.stack(pred_p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ckpt", type=Path, required=True)
    ap.add_argument("--baseline-stats", type=Path, required=True)
    ap.add_argument("--noise-ckpt", type=Path, required=True)
    ap.add_argument("--noise-stats", type=Path, required=True)
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--trajectory-key", type=str, required=True)
    ap.add_argument("--rollout-steps", type=int, default=100)
    ap.add_argument("--output-npz", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    with h5py.File(args.dataset_path, "r") as f:
        g = f[args.trajectory_key]
        gt_vel = np.asarray(g["velocity_xy"][: args.rollout_steps + 1], dtype=np.float32)
        gt_p = np.asarray(g["pressure_gauge"][: args.rollout_steps + 1], dtype=np.float32)
        positions = np.asarray(g["surface_positions"] if "surface_positions" in g else g["cell_centers"],
                               dtype=np.float32)[:, :2]
        case_id = g.attrs.get("case_id")
        source_case_path = resolve_case_dir(
            source_case_path=g.attrs.get("source_case_path"),
            case_id=int(case_id) if case_id is not None else None,
            required_files=("system/geometry.geo",),
        )
        if source_case_path is None:
            raise FileNotFoundError(
                "Could not resolve active case directory from HDF5 source_case_path/case_id"
            )
        polygon = _bfs_polygon(source_case_path / "system" / "geometry.geo")

    b_vel, b_p = _rollout(args.baseline_ckpt, args.baseline_stats, args.dataset_path,
                          args.trajectory_key, args.rollout_steps, device)
    n_vel, n_p = _rollout(args.noise_ckpt, args.noise_stats, args.dataset_path,
                          args.trajectory_key, args.rollout_steps, device)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        positions=positions,
        polygon=polygon,
        gt_vel=gt_vel, gt_p=gt_p,
        baseline_vel=b_vel, baseline_p=b_p,
        noise_vel=n_vel, noise_p=n_p,
        trajectory_key=np.asarray(args.trajectory_key),
    )
    print(f"[dump] wrote {args.output_npz} (positions {positions.shape}, gt_vel {gt_vel.shape})")


if __name__ == "__main__":
    main()
