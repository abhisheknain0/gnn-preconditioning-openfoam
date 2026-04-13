# libIterReduceCoupling

OpenFOAM-to-model coupling library for `gnn_preconditioner`.

Responsibilities:

- read `iterReduceCouplingDict`
- load the exported TorchScript model
- load graph/context tensors prepared by the Python pipeline
- decide whether the current time step should receive an injection
- run model inference and write the predicted `U/p` guess back into OpenFOAM

This library is linked into `iterReduceIncompressibleFluid`.
