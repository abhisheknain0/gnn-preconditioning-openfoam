#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.data_generation.mesh_utils import load_case_static_geometry
from gnn_preconditioner.data_generation.surface_prism import (
    SURFACE_PRISM_REPRESENTATION,
    build_surface_prism_mapping,
    collapse_boundary_mask_to_surface,
    infer_representation_from_group,
)
from gnn_preconditioner.training import config
from gnn_preconditioner.training.dataset import (
    build_directed_edge_graph,
    load_static_graph_from_case_dir,
    load_static_graph_from_group,
)
from gnn_preconditioner.training.normalizer import Normalizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--trajectory-key", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=["auto", SURFACE_PRISM_REPRESENTATION, "volume"],
        default=config.REPRESENTATION,
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    return parser.parse_args()


def _static_only_normalizer(representation: str) -> Normalizer:
    resolved = config.REPRESENTATION if representation == "auto" else representation
    edge_dim = config.NUM_EDGE_INPUT if resolved == SURFACE_PRISM_REPRESENTATION else 8
    pos_dim = 2 if resolved == SURFACE_PRISM_REPRESENTATION else 3
    normalizer = Normalizer()
    normalizer.stats = {
        Normalizer.KEY_EDGE: {"mean": [0.0] * edge_dim, "std": [1.0] * edge_dim},
        Normalizer.KEY_POS: {"mean": [0.0] * pos_dim, "std": [1.0] * pos_dim},
        Normalizer.KEY_CELL_VOLUME: {"mean": [0.0], "std": [1.0]},
        Normalizer.KEY_UINLET: {"mean": [0.0], "std": [1.0]},
    }
    return normalizer


def _write_case_only_context(args: argparse.Namespace, static_case: dict[str, object], output_dir: Path) -> None:
    representation = config.REPRESENTATION if args.representation == "auto" else args.representation
    normalizer = _static_only_normalizer(representation)
    static_graph = load_static_graph_from_case_dir(args.case_dir.expanduser().resolve(), normalizer, representation)
    n_cells = int(static_case["cell_centers"].shape[0])  # type: ignore[index,union-attr]
    meta = {
        "case_dir": str(args.case_dir.expanduser().resolve()),
        "dataset_path": None,
        "trajectory_key": None,
        "case_id": None,
        "representation": representation,
        "n_cells": n_cells,
        "n_internal_faces": int(len(static_case["owner"])),  # type: ignore[arg-type]
        "n_directed_edges": int(static_graph["edge_index"].shape[1]),  # type: ignore[index,union-attr]
        "source": "case",
        "tolerance": args.tolerance,
    }

    if representation == SURFACE_PRISM_REPRESENTATION:
        torch.save(static_graph["positions"].cpu(), output_dir / "surface_positions.pt")  # type: ignore[union-attr]
        torch.save(static_graph["boundary_mask"].cpu(), output_dir / "surface_boundary_mask.pt")  # type: ignore[union-attr]
        torch.save(static_graph["front_owner_patch_order"].cpu(), output_dir / "front_owner_patch_order.pt")  # type: ignore[union-attr]
        torch.save(static_graph["surface_patch_to_cell_order_perm"].cpu(), output_dir / "surface_patch_to_cell_order_perm.pt")  # type: ignore[union-attr]
        torch.save(static_graph["surface_cell_to_patch_order_perm"].cpu(), output_dir / "surface_cell_to_patch_order_perm.pt")  # type: ignore[union-attr]
        meta.update(
            {
                "front_face_count": int(static_graph["front_owner_patch_order"].numel()),  # type: ignore[union-attr]
                "front_owner_unique_count": int(torch.unique(static_graph["front_owner_patch_order"]).numel()),  # type: ignore[arg-type]
            }
        )
    else:
        torch.save(static_graph["positions"].cpu(), output_dir / "cell_centers.pt")  # type: ignore[union-attr]
        torch.save(static_graph["boundary_mask"].cpu(), output_dir / "boundary_mask.pt")  # type: ignore[union-attr]

    torch.save(static_graph["edge_index"].cpu(), output_dir / "edge_index.pt")  # type: ignore[union-attr]
    torch.save(static_graph["edge_attr_raw"].cpu(), output_dir / "edge_attr.pt")  # type: ignore[union-attr]
    torch.save(torch.arange(n_cells, dtype=torch.long), output_dir / "cell_ids.pt")
    torch.save(static_graph["cell_volumes"].cpu(), output_dir / "cell_volumes.pt")  # type: ignore[union-attr]
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    args = _parse_args()
    case_dir = args.case_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    static_case = load_case_static_geometry(case_dir)
    if args.dataset_path is None or args.trajectory_key is None:
        if args.dataset_path is not None or args.trajectory_key is not None:
            raise ValueError("--dataset-path and --trajectory-key must be provided together")
        _write_case_only_context(args, static_case, output_dir)
        return

    with h5py.File(args.dataset_path.expanduser().resolve(), "r") as f:
        g = f[args.trajectory_key]
        representation = infer_representation_from_group(g)
        ds_centers = torch.tensor(g["cell_centers"][()], dtype=torch.float32)
        ds_volumes = torch.tensor(g["cell_volumes"][()], dtype=torch.float32).unsqueeze(-1)
        ds_owner = torch.tensor(g["owner"][()], dtype=torch.long)
        ds_neighbour = torch.tensor(g["neighbour"][()], dtype=torch.long)
        ds_sf = torch.tensor(g["Sf_internal"][()], dtype=torch.float32)
        ds_cell_ids = torch.tensor(g["cell_ids"][()], dtype=torch.long)
        velocity_xy = torch.tensor(g["velocity_xy"][0], dtype=torch.float32)
        pressure_gauge = torch.tensor(g["pressure_gauge"][0], dtype=torch.float32)
        if representation == SURFACE_PRISM_REPRESENTATION:
            ds_surface = torch.tensor(g["surface_positions"][()], dtype=torch.float32)
            ds_surface_boundary = torch.tensor(g["surface_boundary_mask"][()], dtype=torch.float32)
            ds_front_owner_patch = torch.tensor(g["front_owner_patch_order"][()], dtype=torch.long)
            ds_patch_to_cell = torch.tensor(g["surface_patch_to_cell_order_perm"][()], dtype=torch.long)
            ds_cell_to_patch = torch.tensor(g["surface_cell_to_patch_order_perm"][()], dtype=torch.long)
        else:
            ds_boundary = torch.tensor(g["boundary_mask"][()], dtype=torch.float32)
        case_id = int(g.attrs["case_id"])

    case_centers = static_case["cell_centers"]
    case_volumes = static_case["cell_volumes"].unsqueeze(-1)
    case_owner = torch.tensor(static_case["owner"], dtype=torch.long)
    case_neighbour = torch.tensor(static_case["neighbour"], dtype=torch.long)
    case_sf = static_case["Sf_internal"]

    if case_centers.shape != ds_centers.shape:
        raise ValueError(f"Cell count mismatch: case {tuple(case_centers.shape)} vs dataset {tuple(ds_centers.shape)}")
    max_center_abs = float((case_centers - ds_centers).abs().max().item())
    max_volume_abs = float((case_volumes - ds_volumes).abs().max().item())
    if max_center_abs > args.tolerance:
        raise ValueError(f"Cell-center mismatch exceeds tolerance: {max_center_abs} > {args.tolerance}")
    if max_volume_abs > 1e-8:
        raise ValueError(f"Cell-volume mismatch exceeds tolerance: {max_volume_abs}")
    if not torch.equal(case_owner, ds_owner):
        raise ValueError("Owner list mismatch between case and dataset")
    if not torch.equal(case_neighbour, ds_neighbour):
        raise ValueError("Neighbour list mismatch between case and dataset")
    if float((case_sf - ds_sf).abs().max().item()) > 1e-7:
        raise ValueError("Internal-face area vectors mismatch between case and dataset")
    if not torch.equal(ds_cell_ids, torch.arange(ds_cell_ids.numel(), dtype=torch.long)):
        raise ValueError("Dataset cell_ids are not identity-ordered")

    normalizer = Normalizer()
    edge_dim = config.NUM_EDGE_INPUT if representation == SURFACE_PRISM_REPRESENTATION else 8
    pos_dim = 2 if representation == SURFACE_PRISM_REPRESENTATION else 3
    normalizer.stats = {
        Normalizer.KEY_EDGE: {"mean": [0.0] * edge_dim, "std": [1.0] * edge_dim},
        Normalizer.KEY_POS: {"mean": [0.0] * pos_dim, "std": [1.0] * pos_dim},
        Normalizer.KEY_CELL_VOLUME: {"mean": [0.0], "std": [1.0]},
        Normalizer.KEY_UINLET: {"mean": [0.0], "std": [1.0]},
    }
    with h5py.File(args.dataset_path.expanduser().resolve(), "r") as f:
        static_graph = load_static_graph_from_group(f[args.trajectory_key], normalizer)

    meta = {
        "case_dir": str(case_dir),
        "dataset_path": str(args.dataset_path.expanduser().resolve()),
        "trajectory_key": args.trajectory_key,
        "case_id": case_id,
        "representation": representation,
        "n_cells": int(ds_centers.shape[0]),
        "n_internal_faces": int(ds_owner.numel()),
        "n_directed_edges": int(static_graph["edge_index"].shape[1]),
        "max_cell_center_abs_error": max_center_abs,
        "max_cell_volume_abs_error": max_volume_abs,
        "tolerance": args.tolerance,
    }

    if representation == SURFACE_PRISM_REPRESENTATION:
        mapping = build_surface_prism_mapping(static_case)
        case_surface = mapping.surface_positions_canonical[:, :2]
        max_surface_abs = float((case_surface - ds_surface).abs().max().item())
        case_surface_boundary = torch.tensor(
            collapse_boundary_mask_to_surface(static_case["boundary_mask"]),
            dtype=torch.float32,
        )
        if max_surface_abs > args.tolerance:
            raise ValueError(f"Surface-position mismatch exceeds tolerance: {max_surface_abs} > {args.tolerance}")
        if not torch.equal(mapping.front_owner_patch_order, ds_front_owner_patch):
            raise ValueError("Front owner patch ordering mismatch between case and dataset")
        if not torch.equal(mapping.patch_to_cell_order_perm, ds_patch_to_cell):
            raise ValueError("Surface patch->cell permutation mismatch")
        if not torch.equal(mapping.cell_to_patch_order_perm, ds_cell_to_patch):
            raise ValueError("Surface cell->patch permutation mismatch")
        if not torch.equal(case_surface_boundary, ds_surface_boundary):
            raise ValueError("Surface boundary-mask mismatch")

        surface_u_patch = velocity_xy.index_select(0, ds_front_owner_patch)
        surface_p_patch = pressure_gauge.index_select(0, ds_front_owner_patch)
        lifted_u = torch.empty_like(velocity_xy)
        lifted_p = torch.empty_like(pressure_gauge)
        lifted_u.index_copy_(0, ds_front_owner_patch, surface_u_patch)
        lifted_p.index_copy_(0, ds_front_owner_patch, surface_p_patch)
        roundtrip_u_abs = float((lifted_u - velocity_xy).abs().max().item())
        roundtrip_p_abs = float((lifted_p - pressure_gauge).abs().max().item())
        if roundtrip_u_abs > 1e-7 or roundtrip_p_abs > 1e-7:
            raise ValueError(
                f"Surface round-trip mismatch too large: u={roundtrip_u_abs}, p={roundtrip_p_abs}"
            )

        torch.save(static_graph["positions"].cpu(), output_dir / "surface_positions.pt")
        torch.save(ds_surface_boundary.cpu(), output_dir / "surface_boundary_mask.pt")
        torch.save(ds_front_owner_patch.cpu(), output_dir / "front_owner_patch_order.pt")
        torch.save(ds_patch_to_cell.cpu(), output_dir / "surface_patch_to_cell_order_perm.pt")
        torch.save(ds_cell_to_patch.cpu(), output_dir / "surface_cell_to_patch_order_perm.pt")

        meta.update({
            "front_face_count": int(ds_front_owner_patch.numel()),
            "front_owner_unique_count": int(torch.unique(ds_front_owner_patch).numel()),
            "sorted_front_owner_is_identity": bool(
                torch.equal(
                    ds_front_owner_patch.index_select(0, ds_patch_to_cell),
                    torch.arange(ds_front_owner_patch.numel(), dtype=torch.long),
                )
            ),
            "max_surface_position_abs_error": max_surface_abs,
            "roundtrip_u_xy_abs_error": roundtrip_u_abs,
            "roundtrip_p_gauge_abs_error": roundtrip_p_abs,
        })
    else:
        torch.save(ds_centers.cpu(), output_dir / "cell_centers.pt")
        torch.save(ds_boundary.cpu(), output_dir / "boundary_mask.pt")

    torch.save(static_graph["edge_index"].cpu(), output_dir / "edge_index.pt")
    torch.save(static_graph["edge_attr_raw"].cpu(), output_dir / "edge_attr.pt")
    torch.save(ds_cell_ids.cpu(), output_dir / "cell_ids.pt")
    torch.save(ds_volumes.cpu(), output_dir / "cell_volumes.pt")
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
