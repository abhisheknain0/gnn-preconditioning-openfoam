# Mesh Profile Schema

Mesh profiles are stored as JSON files under `gnn_preconditioner/data_generation/mesh_profiles/`.

Required fields:
- `profile_name: str`
- `mesh_size_coarse: float`
- `mesh_size_fine: float`
- `dist_min: float`
- `dist_max: float`
- `n_nodes_by_edge: int`

Canonical v1 profile:
- `coarse_v1`
- target cell band: `8k-12k`
