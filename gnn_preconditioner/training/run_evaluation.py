#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training import config
from gnn_preconditioner.training.boundary_constraints import apply_default_boundary_constraints as apply_boundary_constraints
from gnn_preconditioner.training.dataset import load_split_keys, load_static_graph_from_group, make_dataset
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer
from gnn_preconditioner.inference.render_diagnostics import render_diagnostics


@torch.no_grad()
def _evaluate_one_step(
    model: MeshGraphNet,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = {"rmse_ux": 0.0, "rmse_uy": 0.0, "rmse_p": 0.0}
    count = 0
    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch.x, batch.edge_attr, batch.edge_index)
        pred_du = normalizer.denormalize(pred_norm[:, :2], Normalizer.KEY_TARGET_DU)
        pred_dp = normalizer.denormalize(pred_norm[:, 2:3], Normalizer.KEY_TARGET_DP)
        target_du = normalizer.denormalize(batch.y[:, :2], Normalizer.KEY_TARGET_DU)
        target_dp = normalizer.denormalize(batch.y[:, 2:3], Normalizer.KEY_TARGET_DP)
        total["rmse_ux"] += float(torch.sqrt(torch.mean((pred_du[:, 0] - target_du[:, 0]) ** 2)).item())
        total["rmse_uy"] += float(torch.sqrt(torch.mean((pred_du[:, 1] - target_du[:, 1]) ** 2)).item())
        total["rmse_p"] += float(torch.sqrt(torch.mean((pred_dp[:, 0] - target_dp[:, 0]) ** 2)).item())
        count += 1
    return {k: v / max(count, 1) for k, v in total.items()}


@torch.no_grad()
def _evaluate_rollout(
    model: MeshGraphNet,
    dataset_path: Path,
    split_keys: list[str],
    normalizer: Normalizer,
    device: torch.device,
    rollout_steps: int,
    apply_boundary_constraints_in_rollout: bool,
) -> dict[str, float]:
    model.eval()
    vel_rmse: list[float] = []
    p_rmse: list[float] = []
    vel_rmse_fluid: list[float] = []
    p_rmse_fluid: list[float] = []
    with h5py.File(dataset_path, "r") as f:
        for key in split_keys:
            g = f[key]
            velocity_xy = torch.tensor(g["velocity_xy"][()], dtype=torch.float32, device=device)
            pressure = torch.tensor(g["pressure_gauge"][()], dtype=torch.float32, device=device)
            static = load_static_graph_from_group(g, normalizer, device=device)
            edge_index = static["edge_index"]
            edge_attr = static["edge_attr"]
            static_node_feat = static["static_node_feat"]

            u_xy = velocity_xy[0]
            p_g = pressure[0].unsqueeze(-1)
            max_step = min(rollout_steps, velocity_xy.shape[0] - 1)
            boundary_mask = static["boundary_mask"]
            internal = (boundary_mask.sum(dim=-1) == 0)
            for step in range(max_step):
                x = torch.cat([
                    normalizer.normalize(u_xy, Normalizer.KEY_U_XY),
                    normalizer.normalize(p_g, Normalizer.KEY_P_GAUGE),
                    static_node_feat,
                ], dim=-1)
                pred = model(x, edge_attr, edge_index)
                du = normalizer.denormalize(pred[:, :2], Normalizer.KEY_TARGET_DU)
                dp = normalizer.denormalize(pred[:, 2:3], Normalizer.KEY_TARGET_DP)
                u_next = u_xy + du
                p_next = p_g + dp
                if apply_boundary_constraints_in_rollout:
                    u_xy, p_g = apply_boundary_constraints(
                        u_next=u_next,
                        p_next=p_next,
                        u_current=u_xy,
                        p_current=p_g,
                        boundary_mask=boundary_mask,
                    )
                else:
                    u_xy, p_g = u_next, p_next
                target_u = velocity_xy[step + 1]
                target_p = pressure[step + 1].unsqueeze(-1)
                vel_rmse.append(float(torch.sqrt(torch.mean((u_xy - target_u) ** 2)).item()))
                p_rmse.append(float(torch.sqrt(torch.mean((p_g - target_p) ** 2)).item()))
                if internal.any():
                    vel_rmse_fluid.append(
                        float(torch.sqrt(torch.mean((u_xy[internal] - target_u[internal]) ** 2)).item())
                    )
                    p_rmse_fluid.append(
                        float(torch.sqrt(torch.mean((p_g[internal] - target_p[internal]) ** 2)).item())
                    )
    return {
        "rollout_steps": rollout_steps,
        "velocity_rmse_mean": float(sum(vel_rmse) / max(len(vel_rmse), 1)),
        "pressure_rmse_mean": float(sum(p_rmse) / max(len(p_rmse), 1)),
        "velocity_rmse_fluid_mean": float(sum(vel_rmse_fluid) / max(len(vel_rmse_fluid), 1)),
        "pressure_rmse_fluid_mean": float(sum(p_rmse_fluid) / max(len(p_rmse_fluid), 1)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=config.DATASET_PATH)
    parser.add_argument("--split-path", type=Path, default=config.SPLIT_PATH)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--representation",
        choices=["auto", "surface_prism", "volume"],
        default=config.REPRESENTATION,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diagnostic-trajectory", type=str, default=None)
    parser.add_argument("--no-render-diagnostics", action="store_true")
    parser.add_argument(
        "--apply-default-boundary-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy mode: hold velocity on inlet/wall and pressure on outlet during rollout evaluation.",
    )
    return parser.parse_args()


def _resolve_architecture(checkpoint_path: Path) -> tuple[int, int]:
    hidden_dim = int(config.HIDDEN_DIM)
    message_passing_steps = int(config.MESSAGE_PASSING_STEPS)
    summary_path = checkpoint_path.parent / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        hidden_dim = int(summary.get("hidden_dim", hidden_dim))
        message_passing_steps = int(summary.get("message_passing_steps", message_passing_steps))
    return hidden_dim, message_passing_steps


def main() -> None:
    args = _parse_args()
    normalizer = Normalizer()
    normalizer.load(args.stats_path)
    device = torch.device(args.device)
    hidden_dim, message_passing_steps = _resolve_architecture(args.checkpoint_path)
    dataset = make_dataset(
        args.dataset_path,
        args.split_path,
        normalizer,
        split=args.split,
        in_memory=True,
        representation=args.representation,
    )
    model = MeshGraphNet(
        input_dim_nodes=int(dataset.node_input_dim),
        input_dim_edges=int(dataset.edge_input_dim),
        hidden_dim=hidden_dim,
        num_mp_steps=message_passing_steps,
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    one_step = _evaluate_one_step(model, loader, normalizer, device)
    split_keys = load_split_keys(args.split_path)[args.split]
    rollout = _evaluate_rollout(
        model,
        args.dataset_path,
        split_keys,
        normalizer,
        device,
        args.rollout_steps,
        apply_boundary_constraints_in_rollout=bool(args.apply_default_boundary_constraints),
    )
    output_dir = args.output_dir if args.output_dir is not None else args.checkpoint_path.parent / "eval"
    diagnostics = None
    if not args.no_render_diagnostics and split_keys:
            trajectory_key = (
            args.diagnostic_trajectory
            if args.diagnostic_trajectory is not None
            else ("trajectory_51" if "trajectory_51" in split_keys else split_keys[0])
        )
            diagnostics = render_diagnostics(
            checkpoint_path=args.checkpoint_path,
            stats_path=args.stats_path,
            dataset_path=args.dataset_path,
            trajectory_key=trajectory_key,
            output_dir=output_dir,
                rollout_steps=args.rollout_steps,
                device=args.device,
                apply_default_boundary_constraints=bool(args.apply_default_boundary_constraints),
            )
    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "stats_path": str(args.stats_path),
        "split": args.split,
        "apply_default_boundary_constraints": bool(args.apply_default_boundary_constraints),
        "one_step": one_step,
        "rollout": rollout,
        "diagnostics": diagnostics,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
