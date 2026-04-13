#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.inference.openfoam_fields import (
    read_openfoam_scalar_internal_field,
    read_openfoam_vector_internal_field,
    write_openfoam_initial_fields,
)
from gnn_preconditioner.inference.trajectory_selection import read_case_inlet_u
from gnn_preconditioner.data_generation.mesh_utils import load_case_static_geometry
from gnn_preconditioner.data_generation.surface_prism import (
    SURFACE_PRISM_REPRESENTATION,
    build_surface_directed_edge_graph,
    build_surface_prism_mapping,
    collapse_boundary_mask_to_surface,
)
from gnn_preconditioner.training import config
from gnn_preconditioner.training.boundary_constraints import apply_default_boundary_constraints
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer


def _device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default=_device_default())
    parser.add_argument(
        "--apply-default-boundary-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Old warm-start style: hold velocity on inlet/wall and pressure on outlet during startup rollout.",
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
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


def _build_case_static_graph(
    *,
    case_dir: Path,
    normalizer: Normalizer,
    device: torch.device,
) -> dict[str, torch.Tensor | float | str]:
    static_case = load_case_static_geometry(case_dir)
    mapping = build_surface_prism_mapping(static_case)

    positions = mapping.surface_positions_canonical[:, :2].to(device)
    n_cells = int(static_case["cell_centers"].shape[0])
    if positions.shape[0] != n_cells:
        raise ValueError(
            f"Surface-prism mapping mismatch for {case_dir}: "
            f"{positions.shape[0]} front faces vs {n_cells} cells"
        )

    boundary_mask_np = collapse_boundary_mask_to_surface(static_case["boundary_mask"])
    boundary_mask = torch.tensor(boundary_mask_np, dtype=torch.float32, device=device)
    cell_volumes = static_case["cell_volumes"].to(device).unsqueeze(-1)
    owner = torch.tensor(static_case["owner"], dtype=torch.long, device=device)
    neighbour = torch.tensor(static_case["neighbour"], dtype=torch.long, device=device)
    sf_internal = static_case["Sf_internal"].to(device)

    edge_index, edge_attr_raw = build_surface_directed_edge_graph(
        positions,
        owner,
        neighbour,
        sf_internal,
    )
    edge_attr = normalizer.normalize(edge_attr_raw, Normalizer.KEY_EDGE)

    case_uinlet = read_case_inlet_u(case_dir)
    if case_uinlet is None:
        raise ValueError(f"Could not parse inlet velocity from {case_dir / '0' / 'U'}")

    pos_norm = normalizer.normalize(positions, Normalizer.KEY_POS)
    vol_norm = normalizer.normalize(cell_volumes, Normalizer.KEY_CELL_VOLUME)
    inlet_norm = normalizer.normalize(
        torch.full((positions.shape[0], 1), float(case_uinlet), dtype=torch.float32, device=device),
        Normalizer.KEY_UINLET,
    )
    static_node_feat = torch.cat([boundary_mask, vol_norm, inlet_norm, pos_norm], dim=-1)

    return {
        "representation": SURFACE_PRISM_REPRESENTATION,
        "positions": positions,
        "boundary_mask": boundary_mask,
        "cell_volumes": cell_volumes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_raw": edge_attr_raw,
        "static_node_feat": static_node_feat,
        "uinlet": float(case_uinlet),
        "front_owner_patch_order": mapping.front_owner_patch_order.to(device),
        "surface_patch_to_cell_order_perm": mapping.patch_to_cell_order_perm.to(device),
        "surface_cell_to_patch_order_perm": mapping.cell_to_patch_order_perm.to(device),
    }


@torch.no_grad()
def _rollout_from_openfoam_state(
    *,
    model: MeshGraphNet,
    normalizer: Normalizer,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    static_node_feat: torch.Tensor,
    boundary_mask: torch.Tensor,
    u_xy0: torch.Tensor,
    p0: torch.Tensor,
    rollout_steps: int,
    apply_boundary_constraints: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    u_xy = u_xy0
    p_g = p0
    step_summaries: list[dict[str, float]] = []
    for step in range(int(rollout_steps)):
        x = torch.cat(
            [
                normalizer.normalize(u_xy, Normalizer.KEY_U_XY),
                normalizer.normalize(p_g, Normalizer.KEY_P_GAUGE),
                static_node_feat,
            ],
            dim=-1,
        )
        pred = model(x, edge_attr, edge_index)
        du = normalizer.denormalize(pred[:, :2], Normalizer.KEY_TARGET_DU)
        dp = normalizer.denormalize(pred[:, 2:3], Normalizer.KEY_TARGET_DP)
        u_next = u_xy + du
        p_next = p_g + dp
        if apply_boundary_constraints:
            u_xy, p_g = apply_default_boundary_constraints(
                u_next=u_next,
                p_next=p_next,
                u_current=u_xy,
                p_current=p_g,
                boundary_mask=boundary_mask,
            )
        else:
            u_xy, p_g = u_next, p_next
        step_summaries.append(
            {
                "step": step + 1,
                "du_l2": float(torch.linalg.norm(du).item()),
                "dp_l2": float(torch.linalg.norm(dp).item()),
                "u_l2": float(torch.linalg.norm(u_xy).item()),
                "p_l2": float(torch.linalg.norm(p_g).item()),
            }
        )
    return u_xy, p_g, step_summaries


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    stats_path = args.stats_path.expanduser().resolve()
    case_dir = args.case_dir.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case dir not found: {case_dir}")
    if int(args.rollout_steps) < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {args.rollout_steps}")

    device = torch.device(args.device)
    normalizer = Normalizer()
    normalizer.load(stats_path)
    hidden_dim, message_passing_steps = _resolve_architecture(checkpoint_path)

    static = _build_case_static_graph(case_dir=case_dir, normalizer=normalizer, device=device)

    model = MeshGraphNet(
        input_dim_nodes=int(static["static_node_feat"].shape[1] + 3),
        input_dim_edges=int(static["edge_attr"].shape[1]),
        hidden_dim=hidden_dim,
        num_mp_steps=message_passing_steps,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    positions = static["positions"]
    u_template_path = case_dir / "0" / "U"
    p_template_path = case_dir / "0" / "p"
    expected_internal_size = int(positions.shape[0])
    u_case = read_openfoam_vector_internal_field(u_template_path, expected_size=expected_internal_size).to(device)
    p_case = read_openfoam_scalar_internal_field(p_template_path, expected_size=expected_internal_size).to(device)

    if int(positions.shape[0]) != int(expected_internal_size):
        raise ValueError(
            f"Case graph has {positions.shape[0]} surface nodes but fields have {expected_internal_size} cells"
        )

    u_xy0 = u_case[:, :2]
    p0 = p_case.unsqueeze(-1)

    u_xy_roll, p_roll, step_summaries = _rollout_from_openfoam_state(
        model=model,
        normalizer=normalizer,
        edge_index=static["edge_index"],
        edge_attr=static["edge_attr"],
        static_node_feat=static["static_node_feat"],
        boundary_mask=static["boundary_mask"],
        u_xy0=u_xy0,
        p0=p0,
        rollout_steps=int(args.rollout_steps),
        apply_boundary_constraints=bool(args.apply_default_boundary_constraints),
    )

    vel_out = torch.cat([u_xy_roll, u_case[:, 2:3]], dim=-1).detach().cpu()
    p_out = p_roll.squeeze(-1).detach().cpu()

    u_out_path, p_out_path = write_openfoam_initial_fields(
        output_dir=case_dir / "0",
        velocity=vel_out,
        pressure=p_out,
        u_template_path=u_template_path,
        p_template_path=p_template_path,
        expected_internal_size=expected_internal_size,
    )

    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path is not None
        else (case_dir / "startup_rollout_manifest.json")
    )
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "stats_path": str(stats_path),
        "case_dir": str(case_dir),
        "device": str(device),
        "rollout_steps": int(args.rollout_steps),
        "apply_default_boundary_constraints": bool(args.apply_default_boundary_constraints),
        "case_uinlet": float(static["uinlet"]),
        "surface_node_count": int(positions.shape[0]),
        "expected_internal_size": int(expected_internal_size),
        "input_mapping_mode": "identity_case_cell_order",
        "output_mapping_mode": "identity_case_cell_order",
        "u_out_path": str(u_out_path),
        "p_out_path": str(p_out_path),
        "step_summaries": step_summaries,
        "surface_edge_feature_mode": str(config.SURFACE_EDGE_FEATURE_MODE),
        "representation": str(static["representation"]),
        "standalone_case_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
