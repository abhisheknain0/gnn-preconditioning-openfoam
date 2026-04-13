# incompressibleFluidIterReduce

Base solver-module fork used by the `gnn_preconditioner` runtime.

This layer keeps the OpenFOAM incompressible solver logic in a dedicated module
that can be built in user space. The iterreduce-specific wrapper is added in
`../iterReduceIncompressibleFluid/`.

Built by:

- `openfoam_solver/build_helpers/build_user_dir.sh`

Installed to:

- `${WM_PROJECT_USER_DIR}/modules/incompressibleFluidIterReduce`
