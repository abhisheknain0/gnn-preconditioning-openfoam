# iterReduceIncompressibleFluid

Wrapper solver module for `gnn_preconditioner`.

This layer extends `incompressibleFluidIterReduce` and places the custom field
injection hook into the solver loop:

- `prePredictor()`
- `injectPredictorFields()`
- `momentumPredictor()`

It owns the `IterReduceCoupler` instance and is the C++ layer that connects the
prepared runtime context and TorchScript model to the OpenFOAM solver state.
