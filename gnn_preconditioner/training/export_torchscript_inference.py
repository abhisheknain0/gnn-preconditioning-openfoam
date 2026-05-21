#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training import config
from gnn_preconditioner.training.boundary_constraints import boundary_hold_masks
from gnn_preconditioner.inference.openfoam_fields import (
    read_openfoam_scalar_internal_field,
    read_openfoam_vector_internal_field,
)
from gnn_preconditioner.training.dataset import load_static_graph_from_case_dir, load_static_graph_from_group
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer


class IterReduceInferenceModule(nn.Module):
    rollout_steps: torch.jit.Final[int]
    apply_default_boundary_constraints: torch.jit.Final[bool]

    def __init__(
        self,
        model: MeshGraphNet,
        normalizer: Normalizer,
        node_positions: torch.Tensor,
        cell_volumes: torch.Tensor,
        boundary_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_raw: torch.Tensor,
        apply_default_boundary_constraints: bool = True,
        rollout_steps: int = 1,
    ):
        super().__init__()
        self.model = model.eval()
        self.apply_default_boundary_constraints = bool(apply_default_boundary_constraints)
        self.rollout_steps = int(rollout_steps)
        self.register_buffer("boundary_mask", boundary_mask)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", normalizer.normalize(edge_attr_raw, Normalizer.KEY_EDGE))
        self.register_buffer("vol_norm", normalizer.normalize(cell_volumes, Normalizer.KEY_CELL_VOLUME))
        self.register_buffer("pos_norm", normalizer.normalize(node_positions, Normalizer.KEY_POS))
        self.register_buffer("u_mean", torch.tensor(normalizer.stats[Normalizer.KEY_U_XY]["mean"], dtype=torch.float32))
        self.register_buffer("u_std", torch.tensor(normalizer.stats[Normalizer.KEY_U_XY]["std"], dtype=torch.float32))
        self.register_buffer("p_mean", torch.tensor(normalizer.stats[Normalizer.KEY_P_GAUGE]["mean"], dtype=torch.float32))
        self.register_buffer("p_std", torch.tensor(normalizer.stats[Normalizer.KEY_P_GAUGE]["std"], dtype=torch.float32))
        self.register_buffer("uin_mean", torch.tensor(normalizer.stats[Normalizer.KEY_UINLET]["mean"], dtype=torch.float32))
        self.register_buffer("uin_std", torch.tensor(normalizer.stats[Normalizer.KEY_UINLET]["std"], dtype=torch.float32))
        self.register_buffer("du_mean", torch.tensor(normalizer.stats[Normalizer.KEY_TARGET_DU]["mean"], dtype=torch.float32))
        self.register_buffer("du_std", torch.tensor(normalizer.stats[Normalizer.KEY_TARGET_DU]["std"], dtype=torch.float32))
        self.register_buffer("dp_mean", torch.tensor(normalizer.stats[Normalizer.KEY_TARGET_DP]["mean"], dtype=torch.float32))
        self.register_buffer("dp_std", torch.tensor(normalizer.stats[Normalizer.KEY_TARGET_DP]["std"], dtype=torch.float32))
        velocity_hold_mask, pressure_hold_mask = boundary_hold_masks(boundary_mask)
        self.register_buffer("velocity_hold_mask", velocity_hold_mask.unsqueeze(-1))
        self.register_buffer("pressure_hold_mask", pressure_hold_mask.unsqueeze(-1))

    def forward(self, u_xy_t: torch.Tensor, p_gauge_t: torch.Tensor, u_inlet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inlet = u_inlet.reshape(1).expand(self.boundary_mask.shape[0], 1)
        inlet_norm = (inlet - self.uin_mean) / self.uin_std
        static_node_feat = torch.cat([self.boundary_mask, self.vol_norm, inlet_norm, self.pos_norm], dim=-1)
        u_cur = u_xy_t
        p_cur = p_gauge_t
        for _ in range(self.rollout_steps):
            u_norm = (u_cur - self.u_mean) / self.u_std
            p_norm = (p_cur - self.p_mean) / self.p_std
            x = torch.cat([u_norm, p_norm, static_node_feat], dim=-1)
            pred = self.model(x, self.edge_attr, self.edge_index)
            du = pred[:, :2] * self.du_std + self.du_mean
            dp = pred[:, 2:3] * self.dp_std + self.dp_mean
            u_next = u_cur + du
            p_next = p_cur + dp
            if self.apply_default_boundary_constraints:
                u_next = torch.where(self.velocity_hold_mask, u_cur, u_next)
                p_next = torch.where(self.pressure_hold_mask, p_cur, p_next)
            u_cur = u_next
            p_cur = p_next
        return u_cur, p_cur


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Build static graph buffers directly from this OpenFOAM case instead of an HDF5 trajectory.",
    )
    parser.add_argument("--dataset-path", type=Path, default=config.DATASET_PATH)
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default=None,
        help="HDF5 trajectory key used by the legacy dataset-based export path.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--message-passing-steps", type=int, default=None)
    parser.add_argument(
        "--representation",
        choices=["auto", "surface_prism", "volume"],
        default=config.REPRESENTATION,
    )
    parser.add_argument(
        "--apply-default-boundary-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy mode: hold velocity on inlet/wall and pressure on outlet during exported rollout.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="Number of internal model steps per exported forward() call. Default 1 (one-step forecast).",
    )
    return parser.parse_args()


def _resolve_checkpoint_config(args: argparse.Namespace) -> tuple[int, int]:
    hidden_dim = args.hidden_dim
    message_passing_steps = args.message_passing_steps
    summary_path = args.checkpoint_path.parent / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if hidden_dim is None:
            hidden_dim = int(summary.get("hidden_dim", config.HIDDEN_DIM))
        if message_passing_steps is None:
            message_passing_steps = int(summary.get("message_passing_steps", config.MESSAGE_PASSING_STEPS))
    if hidden_dim is None:
        hidden_dim = int(config.HIDDEN_DIM)
    if message_passing_steps is None:
        message_passing_steps = int(config.MESSAGE_PASSING_STEPS)
    return hidden_dim, message_passing_steps


def _load_mesh_graph_net_state(model: MeshGraphNet, state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("gnn.") for key in state_dict):
        gnn_state = {
            key.removeprefix("gnn."): value
            for key, value in state_dict.items()
            if key.startswith("gnn.")
        }
        model.load_state_dict(gnn_state)
        return "gnn_branch"
    model.load_state_dict(state_dict)
    return "mesh_graph_net"


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    hidden_dim, message_passing_steps = _resolve_checkpoint_config(args)
    normalizer = Normalizer()
    normalizer.load(args.stats_path)
    if args.case_dir is not None:
        static = load_static_graph_from_case_dir(
            args.case_dir,
            normalizer,
            representation=args.representation,
            device=device,
        )
        n_nodes = int(static["positions"].shape[0])  # type: ignore[index,union-attr]
        u0 = read_openfoam_vector_internal_field(args.case_dir / "0" / "U", expected_size=n_nodes)
        p0 = read_openfoam_scalar_internal_field(args.case_dir / "0" / "p", expected_size=n_nodes)
        velocity0 = u0[:, :2].to(device)
        pressure0 = (p0 - p0.mean()).unsqueeze(-1).to(device)
        uinlet = 0.0
        source = "case"
    else:
        if args.trajectory_key is None:
            raise ValueError("--trajectory-key is required unless --case-dir is provided")
        with h5py.File(args.dataset_path, "r") as f:
            g = f[args.trajectory_key]
            static = load_static_graph_from_group(g, normalizer, device=device)
            velocity0 = torch.tensor(g["velocity_xy"][0], dtype=torch.float32, device=device)
            pressure0 = torch.tensor(g["pressure_gauge"][0], dtype=torch.float32, device=device).unsqueeze(-1)
            uinlet = float(g["Uinlet"][()][0])
        source = "dataset"
    model = MeshGraphNet(
        input_dim_nodes=int(static["node_input_dim"]),
        input_dim_edges=int(static["edge_input_dim"]),
        hidden_dim=hidden_dim,
        num_mp_steps=message_passing_steps,
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    exported_state_source = _load_mesh_graph_net_state(model, ckpt["model_state"])
    wrapper = IterReduceInferenceModule(
        model,
        normalizer,
        static["positions"],
        static["cell_volumes"],
        static["boundary_mask"],
        static["edge_index"],
        static["edge_attr_raw"],
        apply_default_boundary_constraints=bool(args.apply_default_boundary_constraints),
        rollout_steps=int(args.rollout_steps),
    ).to(device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    example_inputs = (velocity0, pressure0, torch.tensor([uinlet], dtype=torch.float32, device=device))
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output_path))
    meta = {
        "source": source,
        "case_dir": str(args.case_dir.resolve()) if args.case_dir is not None else None,
        "dataset_path": str(args.dataset_path.resolve()) if args.case_dir is None else None,
        "trajectory_key": args.trajectory_key,
        "output_path": str(args.output_path.resolve()),
        "device": str(device),
        "representation": str(static["representation"]),
        "hidden_dim": int(hidden_dim),
        "message_passing_steps": int(message_passing_steps),
        "exported_state_source": exported_state_source,
        "surface_edge_feature_mode": str(config.SURFACE_EDGE_FEATURE_MODE),
        "apply_default_boundary_constraints": bool(args.apply_default_boundary_constraints),
        "rollout_steps": int(args.rollout_steps),
        "n_nodes": int(static["positions"].shape[0]),
        "n_edges_directed": int(static["edge_index"].shape[1]),
    }
    args.output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
