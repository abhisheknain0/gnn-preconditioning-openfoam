from __future__ import annotations

import bisect
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from . import config
from gnn_preconditioner.data_generation.surface_prism import (
    SURFACE_PRISM_REPRESENTATION,
    VOLUME_REPRESENTATION,
    build_surface_directed_edge_graph,
    build_surface_prism_mapping,
    collapse_boundary_mask_to_surface,
    infer_representation_from_group,
)
from gnn_preconditioner.data_generation.mesh_utils import load_case_static_geometry

from .normalizer import Normalizer


def load_split_keys(split_path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(split_path).read_text(encoding="utf-8"))


def build_directed_edge_graph(
    cell_centers: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    sf_internal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    disp = cell_centers.index_select(0, neighbour) - cell_centers.index_select(0, owner)
    dist = torch.linalg.norm(disp, dim=1, keepdim=True)
    sf_mag = torch.linalg.norm(sf_internal, dim=1, keepdim=True)
    edge_attr_fwd = torch.cat([disp, dist, sf_mag, sf_internal], dim=1)
    edge_attr_rev = torch.cat([-disp, dist, sf_mag, -sf_internal], dim=1)
    edge_index_fwd = torch.stack([owner, neighbour], dim=0)
    edge_index_rev = torch.stack([neighbour, owner], dim=0)
    edge_index = torch.cat([edge_index_fwd, edge_index_rev], dim=1)
    edge_attr = torch.cat([edge_attr_fwd, edge_attr_rev], dim=0)
    return edge_index, edge_attr


def _to_tensor(array, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(array[()], dtype=dtype, device=device)


def _scalar_attr(group, key: str) -> float:
    return float(np.asarray(group[key]).reshape(-1)[0])


def load_static_graph_from_group(
    group: h5py.Group,
    normalizer: Normalizer,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor | float | str]:
    representation = infer_representation_from_group(group)
    cell_volumes = _to_tensor(group["cell_volumes"], torch.float32, device=device).unsqueeze(-1)
    owner = _to_tensor(group["owner"], torch.long, device=device)
    neighbour = _to_tensor(group["neighbour"], torch.long, device=device)
    sf_internal = _to_tensor(group["Sf_internal"], torch.float32, device=device)
    uinlet = _scalar_attr(group, "Uinlet")

    if representation == SURFACE_PRISM_REPRESENTATION:
        positions = _to_tensor(group["surface_positions"], torch.float32, device=device)
        boundary_mask = _to_tensor(group["surface_boundary_mask"], torch.float32, device=device)
        edge_index, edge_attr_raw = build_surface_directed_edge_graph(positions, owner, neighbour, sf_internal)
    else:
        positions = _to_tensor(group["cell_centers"], torch.float32, device=device)
        boundary_mask = _to_tensor(group["boundary_mask"], torch.float32, device=device)
        edge_index, edge_attr_raw = build_directed_edge_graph(positions, owner, neighbour, sf_internal)

    edge_attr = normalizer.normalize(edge_attr_raw, Normalizer.KEY_EDGE)
    pos_norm = normalizer.normalize(positions, Normalizer.KEY_POS)
    vol_norm = normalizer.normalize(cell_volumes, Normalizer.KEY_CELL_VOLUME)
    inlet_norm = normalizer.normalize(
        torch.full((positions.shape[0], 1), uinlet, dtype=torch.float32, device=device),
        Normalizer.KEY_UINLET,
    )
    static_node_feat = torch.cat([boundary_mask, vol_norm, inlet_norm, pos_norm], dim=-1)

    out: dict[str, torch.Tensor | float | str] = {
        "representation": representation,
        "positions": positions,
        "boundary_mask": boundary_mask,
        "cell_volumes": cell_volumes,
        "owner": owner,
        "neighbour": neighbour,
        "sf_internal": sf_internal,
        "edge_index": edge_index,
        "edge_attr_raw": edge_attr_raw,
        "edge_attr": edge_attr,
        "static_node_feat": static_node_feat,
        "uinlet": uinlet,
        "node_input_dim": int(static_node_feat.shape[1] + 3),
        "edge_input_dim": int(edge_attr.shape[1]),
    }
    if "cell_centers" in group:
        out["cell_centers"] = _to_tensor(group["cell_centers"], torch.float32, device=device)
    if "surface_positions" in group:
        out["surface_positions"] = _to_tensor(group["surface_positions"], torch.float32, device=device)
    if "surface_patch_to_cell_order_perm" in group:
        out["surface_patch_to_cell_order_perm"] = _to_tensor(group["surface_patch_to_cell_order_perm"], torch.long, device=device)
        out["surface_cell_to_patch_order_perm"] = _to_tensor(group["surface_cell_to_patch_order_perm"], torch.long, device=device)
        out["front_owner_patch_order"] = _to_tensor(group["front_owner_patch_order"], torch.long, device=device)
    return out


def load_static_graph_from_case_dir(
    case_dir: str | Path,
    normalizer: Normalizer,
    representation: str = "auto",
    device: torch.device | None = None,
) -> dict[str, torch.Tensor | float | str]:
    """Build inference static graph features directly from an OpenFOAM case.

    This is the standalone inference counterpart to ``load_static_graph_from_group``:
    no HDF5 trajectory is needed. The returned tensor contract intentionally
    matches the HDF5 loader so TorchScript export can use either source.
    """
    requested = config.REPRESENTATION if representation == "auto" else representation
    if requested not in {SURFACE_PRISM_REPRESENTATION, VOLUME_REPRESENTATION}:
        raise ValueError(f"Unsupported representation: {representation!r}")

    static_case = load_case_static_geometry(Path(case_dir))
    cell_volumes = static_case["cell_volumes"].to(torch.float32).unsqueeze(-1)  # type: ignore[union-attr]
    owner = torch.tensor(static_case["owner"], dtype=torch.long)  # type: ignore[arg-type]
    neighbour = torch.tensor(static_case["neighbour"], dtype=torch.long)  # type: ignore[arg-type]
    sf_internal = static_case["Sf_internal"].to(torch.float32)  # type: ignore[union-attr]

    out: dict[str, torch.Tensor | float | str]
    if requested == SURFACE_PRISM_REPRESENTATION:
        mapping = build_surface_prism_mapping(static_case)
        positions = mapping.surface_positions_canonical[:, :2].to(torch.float32)
        boundary_mask_np = collapse_boundary_mask_to_surface(static_case["boundary_mask"])  # type: ignore[arg-type]
        boundary_mask = torch.tensor(boundary_mask_np, dtype=torch.float32)
        edge_index, edge_attr_raw = build_surface_directed_edge_graph(positions, owner, neighbour, sf_internal)
        out = {
            "representation": SURFACE_PRISM_REPRESENTATION,
            "surface_positions": positions,
            "positions": positions,
            "surface_boundary_mask": boundary_mask,
            "boundary_mask": boundary_mask,
            "front_owner_patch_order": mapping.front_owner_patch_order,
            "surface_patch_to_cell_order_perm": mapping.patch_to_cell_order_perm,
            "surface_cell_to_patch_order_perm": mapping.cell_to_patch_order_perm,
        }
    else:
        positions = static_case["cell_centers"].to(torch.float32)  # type: ignore[union-attr]
        boundary_mask = torch.tensor(static_case["boundary_mask"], dtype=torch.float32)  # type: ignore[arg-type]
        edge_index, edge_attr_raw = build_directed_edge_graph(positions, owner, neighbour, sf_internal)
        out = {
            "representation": VOLUME_REPRESENTATION,
            "cell_centers": positions,
            "positions": positions,
            "boundary_mask": boundary_mask,
        }

    edge_attr = normalizer.normalize(edge_attr_raw, Normalizer.KEY_EDGE)
    pos_norm = normalizer.normalize(positions, Normalizer.KEY_POS)
    vol_norm = normalizer.normalize(cell_volumes, Normalizer.KEY_CELL_VOLUME)

    inlet_norm = normalizer.normalize(
        torch.zeros((positions.shape[0], 1), dtype=torch.float32),
        Normalizer.KEY_UINLET,
    )
    static_node_feat = torch.cat([boundary_mask, vol_norm, inlet_norm, pos_norm], dim=-1)

    out.update(
        {
            "cell_volumes": cell_volumes,
            "owner": owner,
            "neighbour": neighbour,
            "sf_internal": sf_internal,
            "edge_index": edge_index,
            "edge_attr_raw": edge_attr_raw,
            "edge_attr": edge_attr,
            "static_node_feat": static_node_feat,
            "uinlet": 0.0,
            "node_input_dim": int(static_node_feat.shape[1] + 3),
            "edge_input_dim": int(edge_attr.shape[1]),
        }
    )

    if device is not None:
        moved: dict[str, torch.Tensor | float | str] = {}
        for key, value in out.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        out = moved
    return out


class CFDDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        split_path: str | Path,
        normalizer: Normalizer,
        split: str = "train",
        in_memory: bool = True,
        representation: str = "auto",
        noise_std: float = 0.0,
    ):
        self.h5_path = str(h5_path)
        self.normalizer = normalizer
        self.split = split
        self.in_memory = in_memory
        self.requested_representation = representation
        # noise_std is a dimensionless fraction of each feature's natural std (per-component
        # for u_xy, scalar for p_gauge), only applied to internal nodes during training.
        self.noise_fraction = float(noise_std) if split == "train" else 0.0
        self._noise_u_scale: torch.Tensor | None = None
        self._noise_p_scale: torch.Tensor | None = None
        if self.noise_fraction > 0.0:
            cpu = torch.device("cpu")
            self._noise_u_scale = normalizer.get_std(Normalizer.KEY_U_XY, cpu).detach().clone()
            self._noise_p_scale = normalizer.get_std(Normalizer.KEY_P_GAUGE, cpu).detach().clone()
        self._h5 = None
        split_keys = load_split_keys(split_path)
        if split not in split_keys:
            raise KeyError(f"Split {split!r} not found in {split_path}")
        self.keys = list(split_keys[split])
        self.traj_lengths: list[int] = []
        self.cache: dict[str, dict[str, torch.Tensor | int | float | str]] = {}
        self.representation = None
        self.node_input_dim = None
        self.edge_input_dim = None
        with h5py.File(self.h5_path, "r") as f:
            if self.keys:
                group = f[self.keys[0]]
                self.representation = self._resolve_representation(group)
            for key in self.keys:
                n_frames = int(f[key]["velocity_xy"].shape[0])
                self.traj_lengths.append(n_frames - 1)
                if in_memory:
                    self.cache[key] = self._load_trajectory_to_memory(f[key])
            if self.node_input_dim is None and self.keys:
                sample = self.cache[self.keys[0]] if in_memory else self._load_trajectory_to_memory(f[self.keys[0]])
                self.node_input_dim = int(sample["node_input_dim"])  # type: ignore[arg-type]
                self.edge_input_dim = int(sample["edge_input_dim"])  # type: ignore[arg-type]
        self.cumulative = np.cumsum(self.traj_lengths)
        self.total = int(self.cumulative[-1]) if len(self.cumulative) else 0

    def __len__(self) -> int:
        return self.total

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _resolve_representation(self, group: h5py.Group) -> str:
        actual = infer_representation_from_group(group)
        if self.requested_representation not in {"auto", actual}:
            raise ValueError(
                f"Dataset representation mismatch: requested {self.requested_representation!r}, found {actual!r}"
            )
        return actual

    def _load_trajectory_to_memory(self, g: h5py.Group) -> dict[str, torch.Tensor | int | float | str]:
        representation = self._resolve_representation(g)
        static = load_static_graph_from_group(
            g,
            self.normalizer,
        )
        velocity_xy = _to_tensor(g["velocity_xy"], torch.float32)
        pressure_gauge = _to_tensor(g["pressure_gauge"], torch.float32)
        times = _to_tensor(g["times"], torch.float32)
        case_id = int(g.attrs["case_id"])
        out = {
            **static,
            "representation": representation,
            "velocity_xy": velocity_xy,
            "pressure_gauge": pressure_gauge,
            "times": times,
            "case_id": case_id,
        }
        return out

    def _load_trajectory(self, key: str) -> dict[str, torch.Tensor | int | float | str]:
        if self.in_memory:
            return self.cache[key]
        return self._load_trajectory_to_memory(self._get_h5()[key])

    def __getitem__(self, idx: int) -> Data:
        traj_idx = bisect.bisect_right(self.cumulative, idx)
        frame_idx = idx if traj_idx == 0 else idx - int(self.cumulative[traj_idx - 1])
        key = self.keys[traj_idx]
        traj = self._load_trajectory(key)

        current_u_xy = traj["velocity_xy"][frame_idx]  # type: ignore[index]
        next_u_xy = traj["velocity_xy"][frame_idx + 1]  # type: ignore[index]
        current_p = traj["pressure_gauge"][frame_idx].unsqueeze(-1)  # type: ignore[index]
        next_p = traj["pressure_gauge"][frame_idx + 1].unsqueeze(-1)  # type: ignore[index]
        target_du = next_u_xy - current_u_xy
        target_dp = next_p - current_p

        if self.noise_fraction > 0.0:
            boundary_mask = traj["boundary_mask"]  # type: ignore[assignment]
            internal_mask = (boundary_mask.sum(dim=-1, keepdim=True) == 0)
            u_noise = torch.randn_like(current_u_xy) * (self.noise_fraction * self._noise_u_scale)
            current_u_xy = torch.where(internal_mask, current_u_xy + u_noise, current_u_xy)
            target_du = torch.where(internal_mask, target_du - u_noise, target_du)
            p_noise = torch.randn_like(current_p) * (self.noise_fraction * self._noise_p_scale)
            current_p = torch.where(internal_mask, current_p + p_noise, current_p)
            target_dp = torch.where(internal_mask, target_dp - p_noise, target_dp)

        x = torch.cat([
            self.normalizer.normalize(current_u_xy, Normalizer.KEY_U_XY),
            self.normalizer.normalize(current_p, Normalizer.KEY_P_GAUGE),
            traj["static_node_feat"],  # type: ignore[list-item]
        ], dim=-1)
        y = torch.cat([
            self.normalizer.normalize(target_du, Normalizer.KEY_TARGET_DU),
            self.normalizer.normalize(target_dp, Normalizer.KEY_TARGET_DP),
        ], dim=-1)

        data = Data(
            x=x,
            edge_index=traj["edge_index"],  # type: ignore[arg-type]
            edge_attr=traj["edge_attr"],  # type: ignore[arg-type]
            y=y,
            pos=traj["positions"],  # type: ignore[arg-type]
        )
        data.boundary_mask = traj["boundary_mask"]  # type: ignore[assignment]
        data.frame_idx = torch.tensor([frame_idx], dtype=torch.long)
        data.time_t = traj["times"][frame_idx].unsqueeze(0)  # type: ignore[index]
        data.time_tp1 = traj["times"][frame_idx + 1].unsqueeze(0)  # type: ignore[index]
        data.uinlet = torch.tensor([traj["uinlet"]], dtype=torch.float32)
        data.case_id = torch.tensor([traj["case_id"]], dtype=torch.long)
        return data


class VolumeCFDDataset(CFDDataset):
    def __init__(self, *args, **kwargs):
        kwargs["representation"] = VOLUME_REPRESENTATION
        super().__init__(*args, **kwargs)


class SurfacePrismCFDDataset(CFDDataset):
    def __init__(self, *args, **kwargs):
        kwargs["representation"] = SURFACE_PRISM_REPRESENTATION
        super().__init__(*args, **kwargs)


def make_dataset(
    h5_path: str | Path,
    split_path: str | Path,
    normalizer: Normalizer,
    split: str = "train",
    in_memory: bool = True,
    representation: str = "auto",
    noise_std: float = 0.0,
) -> CFDDataset:
    return CFDDataset(
        h5_path=h5_path,
        split_path=split_path,
        normalizer=normalizer,
        split=split,
        in_memory=in_memory,
        representation=representation,
        noise_std=noise_std,
    )
