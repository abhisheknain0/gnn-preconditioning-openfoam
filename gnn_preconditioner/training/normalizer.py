from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from . import config
from .h5_keys import sorted_trajectory_keys
from gnn_preconditioner.data_generation.surface_prism import (
    SURFACE_PRISM_REPRESENTATION,
    build_surface_directed_edge_graph,
    infer_representation_from_group,
)


class Normalizer:
    KEY_U_XY = "u_xy"
    KEY_P_GAUGE = "p_gauge"
    KEY_CELL_VOLUME = "cell_volume"
    KEY_UINLET = "Uinlet"
    KEY_POS = "position"
    KEY_EDGE = "edge"
    KEY_TARGET_DU = "target_du"
    KEY_TARGET_DP = "target_dp"

    def __init__(self):
        self.stats: dict[str, dict[str, list[float]]] = {}
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _std_floor(self, key: str, dim: int) -> np.ndarray:
        if key == self.KEY_TARGET_DU:
            floor = np.asarray(config.TARGET_DU_MIN_STD, dtype=np.float64)
        elif key == self.KEY_TARGET_DP:
            floor = np.full((dim,), config.TARGET_DP_MIN_STD, dtype=np.float64)
        elif key == self.KEY_CELL_VOLUME:
            floor = np.full((dim,), config.CELL_VOLUME_MIN_STD, dtype=np.float64)
        else:
            floor = np.full((dim,), config.MIN_STD_DEFAULT, dtype=np.float64)
        if floor.size == 1:
            floor = np.full((dim,), float(floor[0]), dtype=np.float64)
        return floor

    def compute_stats(
        self,
        h5_path: str | Path,
        trajectory_keys: list[str] | None = None,
        required_keys: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        required = set(required_keys or [
            self.KEY_U_XY,
            self.KEY_P_GAUGE,
            self.KEY_CELL_VOLUME,
            self.KEY_UINLET,
            self.KEY_POS,
            self.KEY_EDGE,
            self.KEY_TARGET_DU,
            self.KEY_TARGET_DP,
        ])
        accum = None
        with h5py.File(h5_path, "r") as f:
            keys = sorted_trajectory_keys(f.keys()) if trajectory_keys is None else trajectory_keys
            for key in keys:
                g = f[key]
                representation = infer_representation_from_group(g)
                if representation == SURFACE_PRISM_REPRESENTATION:
                    positions = np.asarray(g["surface_positions"], dtype=np.float64)
                else:
                    positions = np.asarray(g["cell_centers"], dtype=np.float64)
                if accum is None:
                    dims = {
                        self.KEY_U_XY: 2,
                        self.KEY_P_GAUGE: 1,
                        self.KEY_CELL_VOLUME: 1,
                        self.KEY_UINLET: 1,
                        self.KEY_POS: int(positions.shape[1]),
                        self.KEY_EDGE: 3 if representation == SURFACE_PRISM_REPRESENTATION else 8,
                        self.KEY_TARGET_DU: 2,
                        self.KEY_TARGET_DP: 1,
                    }
                    accum = {name: _RunningStats(dims[name]) for name in required}

                cell_volumes = np.asarray(g["cell_volumes"], dtype=np.float64).reshape(-1, 1)
                uinlet = float(np.asarray(g["Uinlet"]).reshape(-1)[0])
                velocity_xy = np.asarray(g["velocity_xy"], dtype=np.float64)
                pressure_gauge = np.asarray(g["pressure_gauge"], dtype=np.float64)
                owner = np.asarray(g["owner"], dtype=np.int64)
                neighbour = np.asarray(g["neighbour"], dtype=np.int64)
                sf_internal = np.asarray(g["Sf_internal"], dtype=np.float64)

                if self.KEY_POS in accum:
                    accum[self.KEY_POS].update_batch(positions)
                if self.KEY_CELL_VOLUME in accum:
                    accum[self.KEY_CELL_VOLUME].update_batch(cell_volumes)
                if self.KEY_U_XY in accum:
                    accum[self.KEY_U_XY].update_batch(velocity_xy.reshape(-1, 2))
                if self.KEY_P_GAUGE in accum:
                    accum[self.KEY_P_GAUGE].update_batch(pressure_gauge.reshape(-1, 1))
                if self.KEY_UINLET in accum:
                    accum[self.KEY_UINLET].update_batch(
                        np.full((positions.shape[0], 1), uinlet, dtype=np.float64)
                    )
                if self.KEY_TARGET_DU in accum:
                    du = velocity_xy[1:] - velocity_xy[:-1]
                    accum[self.KEY_TARGET_DU].update_batch(du.reshape(-1, 2))
                if self.KEY_TARGET_DP in accum:
                    dp = pressure_gauge[1:] - pressure_gauge[:-1]
                    accum[self.KEY_TARGET_DP].update_batch(dp.reshape(-1, 1))

                if self.KEY_EDGE in accum and representation == SURFACE_PRISM_REPRESENTATION:
                    edge_attr = build_surface_directed_edge_graph(
                        torch.from_numpy(positions).to(torch.float32),
                        torch.from_numpy(owner).to(torch.long),
                        torch.from_numpy(neighbour).to(torch.long),
                        torch.from_numpy(sf_internal).to(torch.float32),
                    )[1].numpy().astype(np.float64)
                elif self.KEY_EDGE in accum:
                    disp = positions[neighbour] - positions[owner]
                    dist = np.linalg.norm(disp, axis=1, keepdims=True)
                    sf_mag = np.linalg.norm(sf_internal, axis=1, keepdims=True)
                    edge_fwd = np.concatenate([disp, dist, sf_mag, sf_internal], axis=1)
                    edge_rev = np.concatenate([-disp, dist, sf_mag, -sf_internal], axis=1)
                    edge_attr = np.concatenate([edge_fwd, edge_rev], axis=0)
                else:
                    edge_attr = None
                if edge_attr is not None:
                    accum[self.KEY_EDGE].update_batch(edge_attr)

        if accum is None:
            raise RuntimeError(f"No trajectories found in {h5_path}")
        for key, rs in accum.items():
            mean, std = rs.finalize()
            std = np.maximum(std, self._std_floor(key, std.shape[0]))
            self.stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
        self._cache = {}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.stats = json.loads(Path(path).read_text(encoding="utf-8"))
        self._cache = {}

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        mean, std = self._get_tensors(key, x.device)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        mean, std = self._get_tensors(key, x.device)
        return x * std + mean

    def get_std(self, key: str, device: torch.device) -> torch.Tensor:
        _, std = self._get_tensors(key, device)
        return std

    def has_keys(self, *keys: str) -> bool:
        return all(key in self.stats for key in keys)

    def _get_tensors(self, key: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if key not in self._cache:
            stats = self.stats[key]
            self._cache[key] = (
                torch.tensor(stats["mean"], dtype=torch.float32),
                torch.tensor(stats["std"], dtype=torch.float32),
            )
        mean, std = self._cache[key]
        if mean.device != device:
            mean = mean.to(device)
            std = std.to(device)
        return mean, std


class _RunningStats:
    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros((dim,), dtype=np.float64)
        self.m2 = np.zeros((dim,), dtype=np.float64)

    def update_batch(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        x = x.reshape(-1, self.dim)
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_var * batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_var * batch_count + delta * delta * (self.count * batch_count / total)
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("No samples observed")
        var = self.m2 / max(self.count, 1)
        std = np.sqrt(np.maximum(var, 0.0))
        return self.mean, std

