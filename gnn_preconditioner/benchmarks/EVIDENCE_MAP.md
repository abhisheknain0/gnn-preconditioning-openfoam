# Benchmark Evidence Map

This is the curated source map for thesis and supervisor-facing benchmark
claims. It does not replace raw run logs; it identifies which artifacts should
be used first and how older runs should be interpreted.

## Current Evidence Set

Use this evidence set as the current benchmark story. The thesis groups it into
three groups: sanity controls, repeated transient injection (negative), and
steady warm-start (positive). The framing is **two-approach**: transient
`pimpleFoam` in-loop injection (does not help) vs steady `simpleFoam`
initialization (helps).

| Group | Status | Main source | Headline | What it supports |
|---|---|---|---|---|
| Sanity controls | Primary baseline | `gnn_preconditioner/benchmarks/runs/preserve_flux_campaign_20260521T133930Z/sanity_controls/summary.csv` | `identityNoop` and TorchScript `identity_model` both match cold exactly on 3 cases over `endTime=2.0`; `bad_zero` increases total iterations by `+13.58%`. | The custom solver and TorchScript model-call path can be a true no-op when the output equals the input; bad fields are measurably harmful. |
| Matched transient injection | Primary negative transient result | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_matched_dt0p01_every_step/aggregate.json` | 30 cases, `deltaT=0.01`, every-step one-step injection: total `+21.59%`, pressure `+14.80%`, runtime `+26.88%`, improved `1/30`. | Exact timestep matching does not make repeated one-step GNN injection a useful local transient preconditioner. |
| Fine-step dt-scaled transient injection | Primary negative transient result | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_fine_dt0p002_dtscaled_every_step/aggregate.json` | 30 cases, `deltaT=0.002`, `alpha=0.2`: total `+18.04%`, pressure `+23.47%`, runtime `+22.76%`, improved `0/30`. | The supervisor-style dt-scaled blend preserves the intended small-step setup, but still does not reduce iterations. |
| Steady warm-start | Primary positive steady result | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/steady_simplefoam_startup_rollout_r30_r100_30case/aggregate.json` | Rollout 30: total `-11.22%`, pressure `-15.24%`; rollout 100: total `-17.92%`, pressure `-21.72%`; both improve `30/30` cases. | The strongest solver result is steady-state startup acceleration from learned initial fields. |
| Noise-checkpoint steady control | Checkpoint-choice justification (thesis §4.1.3) | `gnn_preconditioner/benchmarks/runs/steady_noise0p05_r30_r100_30case_20260604T204110Z/summary.csv` | Noise σ=0.05 checkpoint, same 30 cases / cold `56848`: total `+1.76%` (r30) / `+2.31%` (r100), improved `10/30` and `8/30`; vs mainline `-11.22%`/`-17.92%`, `30/30`. | The noise model's lower rollout RMSE does NOT transfer to a better solver warm-start; the non-noise checkpoint is the mainline. |

## Current Thesis Interpretation

Start with the controls. The no-op and identity-model rows show that the
current injection machinery itself is not damaging the solver state. The
`bad_zero` control shows that OpenFOAM iteration counts react to field quality.

Then contrast the two coupling approaches:

- Transient (`pimpleFoam`) repeated injection is negative in both clean forms
  tested: matched `deltaT=0.01` and fine-step `deltaT=0.002` with dt-scaled alpha.
- Steady (`simpleFoam`) warm-start is the positive result: both rollout 30 and
  rollout 100 reduce total and pressure iterations on every test case.

This supports a solver-first, two-approach thesis story: the current one-step GNN
is not a reliable per-timestep transient preconditioner, but a learned rollout is
a useful steady-state initializer.

**Transient startup warm-start — dropped from the thesis (2026-06-03).**
Injecting a developed rollout-100 field at `t=0` of a transient run reduces
iterations (total `-19.24%`, pressure `-24.79%`, `30/30`; runtime `+38.20%`) but
discards the startup trajectory the transient run is meant to compute, so it is
not a valid transient intervention. The
`thesis_fresh_30case_20260521/transient_startup_rollout100_initial_dt0p002_t2p0/aggregate.json`
artifact is retained for provenance but is **no longer current thesis evidence**.
Do not re-introduce it into the thesis narrative.

## Active Raw Layout

Keep current evidence in place:

- `gnn_preconditioner/benchmarks/runs/preserve_flux_campaign_20260521T133930Z/`
- `gnn_preconditioner/benchmarks/runs/preserve_flux_full300_case011_dt0p01_t2p0_20260521T131441Z/`
- `gnn_preconditioner/benchmarks/runs/preserve_flux_identity_model_case011_dt0p01_t2p0_20260521T131823Z/`
- `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/`

Older raw run directories are archived under:

- `gnn_preconditioner/benchmarks/runs/archive_pre_fresh_20260521/`

Those archived runs are still useful for provenance and old-meeting
reconstruction, but they should not be used as current thesis evidence unless
explicitly labeled as pre-fix or historical diagnostics.

## Historical Supporting Files

Historical CSV/Markdown summaries are archived under:

- `gnn_preconditioner/benchmarks/archive_pre_fresh_20260521/summaries/`

They are kept for continuity with older slides and notes. They are secondary
evidence unless a current table explicitly cites them. The current benchmark
tables should cite the evidence set above first.
