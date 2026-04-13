# iterReduceIncompressibleFluidApp

Standalone application entrypoint for the `gnn_preconditioner` runtime.

This target builds the `iterReduceIncompressibleFluid` executable that is used
by the benchmark and case-runner scripts. The app instantiates the wrapped
solver module and runs the normal OpenFOAM time loop with the iterreduce hook
enabled.
