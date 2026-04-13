from __future__ import annotations

import torch


def boundary_hold_masks(boundary_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match the legacy rollout/export behavior:
    - hold velocity on inlet and wall nodes
    - hold pressure on outlet nodes
    """
    if boundary_mask.dim() != 2 or boundary_mask.size(1) < 3:
        raise ValueError(
            f"boundary_mask must have shape (N, >=3) with [inlet, outlet, wall], got {tuple(boundary_mask.shape)}"
        )
    inlet = boundary_mask[:, 0] > 0.5
    outlet = boundary_mask[:, 1] > 0.5
    wall = boundary_mask[:, 2] > 0.5
    velocity_hold = inlet | wall
    pressure_hold = outlet
    return velocity_hold, pressure_hold


def apply_default_boundary_constraints(
    u_next: torch.Tensor,
    p_next: torch.Tensor,
    u_current: torch.Tensor,
    p_current: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity_hold, pressure_hold = boundary_hold_masks(boundary_mask)
    u_next = torch.where(velocity_hold.unsqueeze(-1), u_current, u_next)
    p_next = torch.where(pressure_hold.unsqueeze(-1), p_current, p_next)
    return u_next, p_next
