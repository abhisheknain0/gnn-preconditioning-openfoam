# IterReduce v1 Dataset Schema

Per-trajectory datasets are stored in `gnn_preconditioner/data_generation/datasets/*.h5`.

Required datasets per trajectory group:
- `times: (T,) float64`
- `Uinlet: (1,) float32`
- `nu: (1,) float32`
- `cell_centers: (Nc, 3) float32`
- `cell_volumes: (Nc,) float32`
- `boundary_mask: (Nc, 4) uint8`
- `owner: (Nf_internal,) int64`
- `neighbour: (Nf_internal,) int64`
- `Sf_internal: (Nf_internal, 3) float32`
- `Cf_internal: (Nf_internal, 3) float32`
- `velocity_xy: (T, Nc, 2) float32`
- `pressure_raw: (T, Nc) float32`
- `pressure_gauge: (T, Nc) float32`
- `phi_xy_next_internal: (T, Nf_internal) float32`
- `gradp_jump_next_internal: (T, Nf_internal) float32`
- `cell_ids: (Nc,) int64`

Boundary-mask channel order:
- `0 = touches_inlet`
- `1 = touches_outlet`
- `2 = touches_wall`
- `3 = touches_empty`

Attributes:
- `case_id`
- `source_case_path`
- `pressure_gauge_mode = "mean_subtracted"`
- `front_back_mode = "empty"`
- `mesh_profile`

Notes:
- `Uz` is intentionally not stored for the canonical `empty` stack.
- Runtime should reconstruct the OpenFOAM vector field later as `(Ux, Uy, 0)`.
