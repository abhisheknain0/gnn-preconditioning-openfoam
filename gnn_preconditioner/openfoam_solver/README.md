# gnn_preconditioner OpenFOAM runtime

This directory contains the active OpenFOAM-side runtime for the
`gnn_preconditioner` workflow.

Runtime components:

- `libIterReduceCoupling/`
  - Coupling library that loads the exported TorchScript model, reads the
    prepared runtime context, and injects predicted `U/p` fields.
- `custom/incompressibleFluidIterReduce/`
  - Base solver-module fork derived from the OpenFOAM incompressible fluid
    solver stack.
- `custom/iterReduceIncompressibleFluid/`
  - Wrapper solver module that adds the iterreduce hook on top of the base
    module.
- `custom/iterReduceIncompressibleFluidApp/`
  - Standalone application binary entrypoint. This is the executable invoked by
    the inference benchmark scripts.
- `build_helpers/`
  - Sync and build helpers that copy these sources into
    `${WM_PROJECT_USER_DIR}` and build the user-space libraries/app.

Build flow:

1. `build_helpers/sync_to_user_dir.sh`
2. `build_helpers/build_user_dir.sh`
3. `iterReduceIncompressibleFluid` becomes available in `FOAM_USER_APPBIN`

The Python-side preparation stack lives in:

- `gnn_preconditioner/data_generation`
- `gnn_preconditioner/training`
- `gnn_preconditioner/inference`
