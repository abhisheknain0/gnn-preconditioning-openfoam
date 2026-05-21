# Benchmark Presentation Table

This is the compact table for supervisor slides and thesis-result planning.
Use it together with `EVIDENCE_MAP.md`; raw run directories are only for audit.

| Slide block | Experiment | Scope / setting | Iteration result | Runtime / overhead | Presentation takeaway | Source |
|---|---|---|---|---|---|---|
| Sanity baseline | No-op and identity injection controls | 3 cases: `011`, `094`, `188`; `endTime=2.0`; 600 scheduled injections per mode | `identityNoop`: total `0.00%`, pressure `0.00%`; TorchScript `identity_model`: total `0.00%`, pressure `0.00%` | `identity_model` forward wall `0.152 s` aggregate | The current custom solver and TorchScript call path can be an exact solver no-op when output equals input. | `gnn_preconditioner/benchmarks/runs/preserve_flux_campaign_20260521T133930Z/sanity_controls/summary.csv` |
| Sanity baseline | Deliberately bad guess | Same 3 cases, `endTime=0.1`, `bad_zero` | Total `+13.58%`; pressure `+12.52%` | Forward wall `0.207 s` aggregate | Bad fields increase OpenFOAM work, so iteration counts are sensitive to predictor quality. | `gnn_preconditioner/benchmarks/runs/preserve_flux_campaign_20260521T133930Z/sanity_controls/summary.csv` |
| Transient, matched timestep | One-step GNN injected every OpenFOAM step | 30 full-test cases; `deltaT=0.01`; `macroDt=0.01`; alpha `1.0`; 200 injections per case | Total `+21.59%`; pressure `+14.80%`; improved `1/30`; near-neutral within `+5%`: `7/30` | Runtime `+26.88%`; forward wall `35.09 s`; injection wall `57.15 s` aggregate | Exact timestep matching is not enough; repeated local transient injection worsens the solve in aggregate. | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_matched_dt0p01_every_step/aggregate.json` |
| Transient, fine timestep | One-step GNN with supervisor dt-scaled alpha | 30 full-test cases; `deltaT=0.002`; `macroDt=0.01`; `alpha=0.2`; 1000 injections per case | Total `+18.04%`; pressure `+23.47%`; improved `0/30`; near-neutral within `+5%`: `7/30` | Runtime `+22.76%`; forward wall `84.43 s`; injection wall `124.14 s` aggregate | The physically cleaner small-step blend still does not reduce linear iterations. | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_fine_dt0p002_dtscaled_every_step/aggregate.json` |
| Transient startup | Initial rollout-100 warm-start at `t=0` | 30 full-test cases; `deltaT=0.002`; `endTime=2.0`; one startup injection per case | Total `-19.24%`; pressure `-24.79%`; improved `30/30` | Runtime `+38.20%`; forward wall `150.76 s`; injection wall `297.16 s` aggregate | Learned fields are useful transient initializers, but current in-run rollout generation is too expensive for wall-clock claims. | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/transient_startup_rollout100_initial_dt0p002_t2p0/aggregate.json` |
| Steady warm-start | `simpleFoam` initialized from rollout 30 | 30 full-test cases; steady templates; rollout 30 | Total `-11.22%`; pressure `-15.24%`; improved `30/30` | Execution time `-9.19%` aggregate | Short learned rollout already gives a robust steady-state iteration reduction. | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/steady_simplefoam_startup_rollout_r30_r100_30case/aggregate.json` |
| Steady warm-start | `simpleFoam` initialized from rollout 100 | 30 full-test cases; steady templates; rollout 100 | Total `-17.92%`; pressure `-21.72%`; improved `30/30` | Execution time `-19.86%` aggregate | Strongest positive solver result: learned initialization accelerates steady-state convergence across all test cases. | `gnn_preconditioner/benchmarks/runs/thesis_fresh_30case_20260521/steady_simplefoam_startup_rollout_r30_r100_30case/aggregate.json` |

## Slide Order

1. Start with sanity controls: the machinery can be exact no-op, and bad fields
   increase work.
2. Show repeated transient injection as a negative result in both clean
   formulations: matched timestep and dt-scaled fine timestep.
3. Show startup warm-start as the useful transient result, with runtime cost
   stated honestly.
4. End with steady warm-start as the strongest positive solver result.

Do not merge these into one claim. The thesis distinction is important:
per-timestep transient preconditioning fails here, while learned initialization
works.
