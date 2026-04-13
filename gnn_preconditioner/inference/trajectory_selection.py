#!/usr/bin/env python3
"""Trajectory selection helpers for case-consistent OpenFOAM/GNN inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py

from gnn_preconditioner.training.h5_keys import sorted_trajectory_keys
from gnn_preconditioner.training.h5_keys import trajectory_numeric_id

# Match case ids in names like:
#   case_043
#   case_043_abg_1s_...
# while avoiding partial numeric tails.
_CASE_ID_RE = re.compile(r"case_(\d+)(?!\d)", flags=re.IGNORECASE)
_INLET_RE = re.compile(r"\binlet\s*\{(.*?)\}", flags=re.DOTALL | re.IGNORECASE)
_INLET_VAL_RE = re.compile(
    r"value\s+uniform\s+\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TrajectorySelection:
    key: str
    index: int
    mode: str
    case_id: int | None
    case_uinlet: float | None
    dataset_uinlet: float | None
    inlet_abs_diff: float | None


def _resolve_trajectory_index(index: int, n_total: int) -> int:
    resolved = index if index >= 0 else n_total + index
    if resolved < 0 or resolved >= n_total:
        raise IndexError(f"trajectory index {index} out of range [0, {n_total - 1}]")
    return resolved


def _to_scalar_float(v) -> float:
    try:
        # Handles scalars and numpy/h5 arrays.
        return float(v[0])  # type: ignore[index]
    except Exception:
        return float(v)


def infer_case_id_from_path(path: Path | None, max_parents: int = 8) -> int | None:
    if path is None:
        return None
    cur = path
    for _ in range(max_parents + 1):
        m = _CASE_ID_RE.search(cur.name)
        if m is not None:
            return int(m.group(1))
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def read_case_inlet_u(case_dir: Path | None) -> float | None:
    if case_dir is None:
        return None
    u_path = case_dir / "0" / "U"
    if not u_path.exists():
        return None
    text = u_path.read_text(encoding="utf-8", errors="ignore")
    inlet_block = _INLET_RE.search(text)
    if inlet_block is None:
        return None
    value_match = _INLET_VAL_RE.search(inlet_block.group(1))
    if value_match is None:
        return None
    return float(value_match.group(1))


def select_trajectory_for_case(
    *,
    dataset_path: Path,
    trajectory_key: str | None,
    trajectory_index: int,
    case_dir: Path | None,
) -> TrajectorySelection:
    """
    Select a trajectory with robust defaults:
    1) explicit key if provided,
    2) case id from path (case_<id>),
    3) nearest inlet velocity,
    4) index fallback.
    """
    with h5py.File(dataset_path, "r") as f:
        keys = sorted_trajectory_keys(f.keys())
        if not keys:
            raise RuntimeError(f"No trajectories found in {dataset_path}")

        u_by_key: dict[str, float] = {}
        for k in keys:
            u_by_key[k] = _to_scalar_float(f[k]["Uinlet"][()])

        case_id = infer_case_id_from_path(case_dir)
        case_uinlet = read_case_inlet_u(case_dir)

        if trajectory_key is not None:
            if trajectory_key not in f:
                available = ", ".join(keys[:8])
                raise KeyError(
                    f"Trajectory key {trajectory_key!r} not found in {dataset_path}. "
                    f"Example available keys: {available}"
                )
            idx = keys.index(trajectory_key)
            ds_u = u_by_key[trajectory_key]
            abs_diff = (
                abs(ds_u - case_uinlet) if case_uinlet is not None else None
            )
            return TrajectorySelection(
                key=trajectory_key,
                index=idx,
                mode="explicit_key",
                case_id=case_id,
                case_uinlet=case_uinlet,
                dataset_uinlet=ds_u,
                inlet_abs_diff=abs_diff,
            )

        if case_id is not None:
            expected = f"trajectory_{case_id}"
            if expected in f:
                return TrajectorySelection(
                    key=expected,
                    index=keys.index(expected),
                    mode="case_id",
                    case_id=case_id,
                    case_uinlet=case_uinlet,
                    dataset_uinlet=u_by_key[expected],
                    inlet_abs_diff=(
                        abs(u_by_key[expected] - case_uinlet)
                        if case_uinlet is not None
                        else None
                    ),
                )
            for i, k in enumerate(keys):
                if trajectory_numeric_id(k) == case_id:
                    return TrajectorySelection(
                        key=k,
                        index=i,
                        mode="case_id_numeric_match",
                        case_id=case_id,
                        case_uinlet=case_uinlet,
                        dataset_uinlet=u_by_key[k],
                        inlet_abs_diff=(
                            abs(u_by_key[k] - case_uinlet)
                            if case_uinlet is not None
                            else None
                        ),
                    )

        if case_uinlet is not None:
            best_key = min(keys, key=lambda k: abs(u_by_key[k] - case_uinlet))
            best_diff = abs(u_by_key[best_key] - case_uinlet)
            return TrajectorySelection(
                key=best_key,
                index=keys.index(best_key),
                mode="nearest_uinlet",
                case_id=case_id,
                case_uinlet=case_uinlet,
                dataset_uinlet=u_by_key[best_key],
                inlet_abs_diff=best_diff,
            )

        idx = _resolve_trajectory_index(trajectory_index, len(keys))
        key = keys[idx]
        return TrajectorySelection(
            key=key,
            index=idx,
            mode="trajectory_index",
            case_id=case_id,
            case_uinlet=case_uinlet,
            dataset_uinlet=u_by_key[key],
            inlet_abs_diff=(
                abs(u_by_key[key] - case_uinlet)
                if case_uinlet is not None
                else None
            ),
        )
