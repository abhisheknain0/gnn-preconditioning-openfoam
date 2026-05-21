# Fresh Preserve-Flux Benchmark Plan And Status

This document records the completed benchmark pass after the solver-state fix
that preserves OpenFOAM `phi` and `Uf` by default after model injection.

Status: completed for the current thesis-facing benchmark set. Promoted results
are indexed in `EVIDENCE_MAP.md` and summarized in
`BENCHMARK_PRESENTATION_TABLE.md`.

## Common Configuration

- Environment: `source gnn_preconditioner/setup_env.sh --runtime-only`
- Checkpoint:
  `gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/best_model.pt`
- Stats:
  `gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/stats.json`
- Dataset:
  `gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5`
- Transient case root:
  `gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0`
- Steady case root:
  `gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_steady`
- Runtime graph/context: build from the OpenFOAM case mesh with
  `--case-static-context`.
- Runtime device: explicit CUDA outside the sandbox.
- Default injection policy:
  `correctBoundariesAfterInjection=true`,
  `refreshPhiAfterInjection=false`, `refreshUfAfterInjection=false`.

## Test Sets

- Sanity smoke: cases `011 094 188`.
- Full transient test set: all 30 cases from
  `gnn_preconditioner/manifests/full_split_240_30_30_v1.json`.
- Steady warm-start set: all 30 full-test cases. Missing steady templates were
  generated from the current case manifest with `--skip-solve`.

## Stage 1: Sanity Controls

Source:
`gnn_preconditioner/benchmarks/runs/preserve_flux_campaign_20260521T133930Z/sanity_controls/summary.csv`

| Control | Cases | Horizon | Result |
|---|---:|---:|---|
| `identityNoop` | 3 | `endTime=2.0`, `deltaT=0.01` | exact cold match |
| `identity_model` | 3 | `endTime=2.0`, `deltaT=0.01` | exact cold match through TorchScript model path |
| `bad_zero` | 3 | `endTime=0.1`, `deltaT=0.01` | total `+13.58%`, pressure `+12.52%` |

Interpretation: the machinery can be a true no-op, and bad field quality
increases solver work.

## Stage 2: Matched-Delta Transient

Source:
`gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_matched_dt0p01_every_step/aggregate.json`

Configuration:

- Cases: full 30-case test set.
- `endTime=2.0`
- `deltaT=0.01`
- `adjustTimeStep=no`
- `predictorMode=model`
- `injectEveryTimeStep=true`
- `macroDt=0.01`
- `blendMode=manual`
- `guessBlendAlpha=1.0`
- Rollout/export: one-step model.

Result:

- Aggregate total iterations: `+21.59%`.
- Aggregate pressure iterations: `+14.80%`.
- Aggregate runtime: `+26.88%`.
- Improved cases: `1/30`.

Interpretation: negative result for repeated local transient injection.

## Stage 3: Fine-Delta Blend Formula

Source:
`gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_fine_dt0p002_dtscaled_every_step/aggregate.json`

Configuration:

- Cases: full 30-case test set.
- `endTime=2.0`
- `deltaT=0.002`
- `adjustTimeStep=no`
- One-step GNN, no exported rollout.
- `injectEveryTimeStep=true`
- `macroDt=0.01`
- `blendMode=dtScaled`
- `gnnTrainingDt=0.01`
- `dtScaledAlphaMax=1.0`, so `alphaEff=0.002 / 0.01 = 0.2`.

Result:

- Aggregate total iterations: `+18.04%`.
- Aggregate pressure iterations: `+23.47%`.
- Aggregate runtime: `+22.76%`.
- Improved cases: `0/30`.

Interpretation: also negative for repeated transient injection, even with the
dt-scaled small-step blend.

## Stage 4: Transient Startup Warm-Start

Source:
`gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_startup_rollout100_initial_dt0p002_t2p0/aggregate.json`

Configuration:

- Cases: full 30-case test set.
- `endTime=2.0`
- `deltaT=0.002`
- `adjustTimeStep=no`
- Startup injection at `t=0`.
- `injectEveryTimeStep=false`
- Rollout depth: `100`.

Result:

- Aggregate total iterations: `-19.24%`.
- Aggregate pressure iterations: `-24.79%`.
- Improved cases: `30/30`.
- Aggregate runtime: `+38.20%`.

Interpretation: positive iteration result, but runtime cost is high because the
rollout is generated inside the run.

## Stage 5: Steady Warm-Start

Source:
`gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/steady_simplefoam_startup_rollout_r30_r100_30case/aggregate.json`

Configuration:

- Cases: full 30-case test set.
- Solver: `simpleFoam`.
- Rollout depths: `30` and `100`.

Result:

- Rollout 30: total `-11.22%`, pressure `-15.24%`, improved `30/30`.
- Rollout 100: total `-17.92%`, pressure `-21.72%`, improved `30/30`.

Interpretation: strongest positive solver result. Learned initialization
accelerates steady-state convergence across all tested cases.
