from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .mesh_utils import face_center


SURFACE_PRISM_REPRESENTATION = "surface_prism"
VOLUME_REPRESENTATION = "volume"


@dataclass(frozen=True)
class SurfacePrismMapping:
    front_face_indices_patch_order: torch.Tensor
    front_owner_patch_order: torch.Tensor
    surface_positions_patch_order: torch.Tensor
    surface_positions_canonical: torch.Tensor
    patch_to_cell_order_perm: torch.Tensor
    cell_to_patch_order_perm: torch.Tensor


def infer_representation_from_group(group) -> str:
    if "surface_positions" in group:
        return SURFACE_PRISM_REPRESENTATION
    return VOLUME_REPRESENTATION


def collapse_boundary_mask_to_surface(boundary_mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return boundary_mask[..., :3]


def build_surface_prism_mapping(static: dict[str, object]) -> SurfacePrismMapping:
    boundary = static["boundary"]
    points = static["points"]
    faces = static["faces"]
    owner_all = torch.tensor(static["owner_all"], dtype=torch.long)

    front = next((entry for entry in boundary if entry.name.lower() == "front"), None)
    if front is None:
        raise ValueError("Could not find a `front` patch for surface-prism mapping")

    start = int(front.start_face)
    end = start + int(front.n_faces)
    face_indices = torch.arange(start, end, dtype=torch.long)
    front_owner = owner_all[start:end]
    surface_patch = torch.stack(
        [face_center(points, faces[int(face_i)]) for face_i in face_indices.tolist()],
        dim=0,
    )
    patch_to_cell = torch.argsort(front_owner)
    sorted_owner = front_owner.index_select(0, patch_to_cell)
    expected = torch.arange(front_owner.numel(), dtype=torch.long)
    if not torch.equal(sorted_owner, expected):
        raise ValueError("Front owners do not become identity after owner sorting")

    cell_to_patch = torch.empty_like(patch_to_cell)
    cell_to_patch[patch_to_cell] = torch.arange(patch_to_cell.numel(), dtype=torch.long)
    surface_canonical = surface_patch.index_select(0, patch_to_cell)

    return SurfacePrismMapping(
        front_face_indices_patch_order=face_indices,
        front_owner_patch_order=front_owner,
        surface_positions_patch_order=surface_patch,
        surface_positions_canonical=surface_canonical,
        patch_to_cell_order_perm=patch_to_cell,
        cell_to_patch_order_perm=cell_to_patch,
    )


def build_surface_directed_edge_graph(
    surface_positions_xy: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    sf_internal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    disp_xy = surface_positions_xy.index_select(0, neighbour) - surface_positions_xy.index_select(0, owner)
    dist_xy = torch.linalg.norm(disp_xy, dim=1, keepdim=True)
    edge_attr_fwd = torch.cat([disp_xy, dist_xy], dim=1)
    edge_attr_rev = torch.cat([-disp_xy, dist_xy], dim=1)
    edge_index_fwd = torch.stack([owner, neighbour], dim=0)
    edge_index_rev = torch.stack([neighbour, owner], dim=0)
    edge_index = torch.cat([edge_index_fwd, edge_index_rev], dim=1)
    edge_attr = torch.cat([edge_attr_fwd, edge_attr_rev], dim=0)
    return edge_index, edge_attr
