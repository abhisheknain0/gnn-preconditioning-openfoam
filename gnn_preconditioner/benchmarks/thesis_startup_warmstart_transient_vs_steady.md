# Startup Warm-Start: Transient vs Steady

## Summary

The startup-rollout model should be discussed in two separate modes:

1. transient startup acceleration with `pimpleFoam`
2. steady-state initialization with `simpleFoam`

These should not be mixed into a single claim, because the transient benchmark benefits from skipping part of the cold startup transient, while the steady benchmark measures improved initialization for convergence to a steady solution.

## Case 011: Transient vs Steady

| Mode | Solver | Cold Total | Warm Total | Delta | Delta % | Pressure Delta | Execution Time Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Transient startup | `pimpleFoam` | 7604 | 6836 | -768 | -10.10% | -762 | -0.01373 s |
| Steady startup | `simpleFoam` | 2249 | 2138 | -111 | -4.94% | -97 | -0.02196 s |

Interpretation:

- The transient startup experiment shows a much larger improvement because the rollout gives the solver a more developed initial field and effectively skips part of the zero-field startup transient.
- The steady experiment is the cleaner benchmark for a steady-state convergence claim.
- On `case_011`, the steady startup rollout still improves convergence meaningfully, but the effect is smaller than the transient startup effect.

## Steady Results

| Case | Solver | Cold Total | Warm Total | Delta | Delta % | Pressure Delta | Execution Time Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `case_011` | `simpleFoam` | 2249 | 2138 | -111 | -4.94% | -97 | -0.02196 s |
| `case_018` | `simpleFoam` | 1711 | 1653 | -58 | -3.39% | -58 | +0.00133 s |

Interpretation:

- The steady startup warm-start effect holds on both tested cases.
- The gain is smaller than in transient startup mode, but it remains positive on both a high-Re and a highest-Re case.
- The pressure solve accounts for most of the improvement.

## Thesis Claim

A careful thesis claim that fits the evidence is:

> The GNN startup rollout is effective as an initialization strategy. In transient `pimpleFoam` startup it yields large iteration-count reductions by replacing part of the cold startup transient with a learned initial guess. When evaluated in a clean steady-state setting with `simpleFoam`, it still reduces linear iterations, but with smaller gains. Therefore, transient and steady startup results should be reported separately rather than combined into a single convergence claim.

## Source Artifacts

- Transient `case_011`:
  - [abg_metrics_startup_rollout_iterreduce_v1.json](/work/<hpc-user>/Abhishek/Thesis/Code/Simulator/ma_gnn_pc/gnn_preconditioner/benchmarks/runs/case_011_startup_rollout_full_surface_state_hd128_0p1/abg_metrics_startup_rollout_iterreduce_v1.json)
- Steady `case_011`:
  - [abg_metrics_startup_rollout_iterreduce_v1.json](/work/<hpc-user>/Abhishek/Thesis/Code/Simulator/ma_gnn_pc/gnn_preconditioner/benchmarks/runs/case_011_startup_rollout_full_surface_state_hd128_steady/abg_metrics_startup_rollout_iterreduce_v1.json)
- Steady `case_018`:
  - [abg_metrics_startup_rollout_iterreduce_v1.json](/work/<hpc-user>/Abhishek/Thesis/Code/Simulator/ma_gnn_pc/gnn_preconditioner/benchmarks/runs/case_018_startup_rollout_full_surface_state_hd128_steady/abg_metrics_startup_rollout_iterreduce_v1.json)
