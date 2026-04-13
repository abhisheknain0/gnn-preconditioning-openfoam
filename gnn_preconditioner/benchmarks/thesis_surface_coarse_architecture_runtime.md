# Surface-Coarse Architecture Runtime Comparison

## Scope

Controlled comparison on the **surface-on-prism coarse mesh** using the same pilot case family and runtime setup:

- dataset: `gnn_preconditioner/data_generation/datasets/iterreduce_pilot_surface_coarse_v1.h5`
- benchmark horizon: `0.1 s`
- runtime cases:
  - `case_011` (`trajectory_11`, pilot split: `valid`)
  - `case_051` (`trajectory_51`, pilot split: `test`)

Models compared:

1. `solver_96x8`
   - loss: `state + gradp + mass`
   - `hidden_dim = 96`
   - `message_passing_steps = 8`
2. `state_128x8`
   - loss: `state` only
   - `hidden_dim = 128`
   - `message_passing_steps = 8`
3. `state_128x12`
   - loss: `state` only
   - `hidden_dim = 128`
   - `message_passing_steps = 12`

## Training Summary

| Model | Best epoch | Best valid state | Best valid gradp_raw | Best valid mass_raw | Best epoch wall [s] |
|---|---:|---:|---:|---:|---:|
| `solver_96x8` | 29 | 0.06051 | 5.98e-09 | 0.15453 | 19.61 |
| `state_128x8` | 30 | 0.03433 | 1.99e-08 | 0.11840 | 20.83 |
| `state_128x12` | 30 | 0.02464 | 1.56e-08 | 0.11196 | 25.78 |

Training-side read:

- Pure state regression improves monotonically from `solver_96x8` to `state_128x8` to `state_128x12`.
- The `128/12` model is the strongest state regressor.
- The `128/8` model is the better cost/performance point among the state-only variants.
- The solver-loss model remains the only one explicitly optimized for solver-facing residual structure.

## Runtime Summary

### `case_011`

Cold baseline:

- total linear iterations: `7604`

| Model | Total iterations | Delta vs cold | Delta [%] | Pressure delta | Exec. time delta [s] |
|---|---:|---:|---:|---:|---:|
| `solver_96x8` | 7335 | -269 | -3.54 | -271 | +2.641 |
| `state_128x8` | 7379 | -225 | -2.96 | -227 | +2.105 |
| `state_128x12` | 7442 | -162 | -2.13 | -164 | +1.669 |

### `case_051`

Cold baseline:

- total linear iterations: `7366`

| Model | Total iterations | Delta vs cold | Delta [%] | Pressure delta | Exec. time delta [s] |
|---|---:|---:|---:|---:|---:|
| `solver_96x8` | 7361 | -5 | -0.07 | -4 | +1.891 |
| `state_128x8` | 7278 | -88 | -1.19 | -87 | +2.074 |
| `state_128x12` | 7314 | -52 | -0.71 | -53 | +1.042 |

## Interpretation

Main observations:

- `solver_96x8` is still the strongest model on `case_011`.
- `state_128x8` is the strongest model on `case_051`.
- `state_128x12` improves training loss further than `state_128x8`, but that extra training gain does **not** translate into better runtime iteration reduction on either benchmark case.
- The runtime benefit continues to come almost entirely from the pressure solves.
- All three injected models remain slower than cold start in wall-clock time because runtime load and repeated inference calls dominate the saved linear iterations.

Current practical takeaway:

- If the goal is the **best pure regression model**, `state_128x12` is the winner.
- If the goal is the **best cost/performance state-only architecture**, `state_128x8` is the better choice.
- If the goal is the **most reliable solver-oriented model across the current benchmarks**, `solver_96x8` remains the safer baseline because its loss design is runtime-aware and it performs best on the holdout-style `case_011`.

## Source Artifacts

Training histories:

- `gnn_preconditioner/checkpoints/pilot_surface_coarse_v1_run1/history.json`
- `gnn_preconditioner/checkpoints/pilot_surface_coarse_v1_stateonly_hd128_run1/history.json`
- `gnn_preconditioner/checkpoints/pilot_surface_coarse_v1_stateonly_hd128_mp12_run1/history.json`

Runtime benchmark JSON:

- `gnn_preconditioner/benchmarks/runs/case_011_surface_coarse_solver_0p1/abg_metrics_iterreduce_v1.json`
- `gnn_preconditioner/benchmarks/runs/case_051_surface_coarse_solver_0p1/abg_metrics_iterreduce_v1.json`
- `gnn_preconditioner/benchmarks/runs/case_011_surface_coarse_stateonly_hd128_0p1_v2/abg_metrics_iterreduce_v1.json`
- `gnn_preconditioner/benchmarks/runs/case_051_surface_coarse_stateonly_hd128_0p1_v2/abg_metrics_iterreduce_v1.json`
- `gnn_preconditioner/benchmarks/runs/case_011_surface_coarse_stateonly_hd128_mp12_0p1_v2/abg_metrics_iterreduce_v1.json`
- `gnn_preconditioner/benchmarks/runs/case_051_surface_coarse_stateonly_hd128_mp12_0p1_v2/abg_metrics_iterreduce_v1.json`
