# IterReduce v1

Solver-first stack for reducing OpenFOAM linear iterations via a GNN
preconditioner. This is the only active stack in the repository.

Key decisions:
- canonical BFS setup uses `front/back = empty`
- no front-patch extraction
- train on cell-volume `Ux, Uy, p`
- no stored `Uz` in the dataset for the canonical `empty` stack
- include `t=0`
- model predicts `delta Ux, delta Uy, delta p_gauge`
- runtime uses predictions as initial guesses only
- moderate mesh coarsening with canonical target band `8k-12k` cells

Top-level layout:
- `data_generation/`: manifests, base cases, mesh profiles, case generation, HDF5 build
- `training/`: volume-state training stack
- `inference/`: context build, benchmark runner, analysis
- `openfoam_solver/`: isolated runtime stack for the new method
- `docs/`: schemas and design notes

Environment prerequisite (path is relative — pick the one that matches your cwd):

```bash
# From the repo root (ma_gnn_pc/):
source gnn_preconditioner/setup_env.sh --runtime-only

# From inside this directory (gnn_preconditioner/):
source setup_env.sh --runtime-only
```

Bash-tool calls do not persist env between invocations — the source command
must run in the same Bash call as the Python/solver invocation that follows.

Current implementation status:
- implemented:
  - canonical empty-based base case
  - mesh profiles and mesh-calibration CLI
  - pilot manifest builder and split generator
  - empty-based case generation CLI
  - volume HDF5 dataset builder
  - cell-volume training/evaluation/export pipeline
  - exact cell-order runtime context builder
  - isolated `iterReduceIncompressibleFluid` C++ runtime binary under `gnn_preconditioner/openfoam_solver`
  - benchmark runner with automatic cold vs injected metric parsing
  - case-id based test wrapper for running arbitrary pilot cases

Typical flow:

```bash
python3 gnn_preconditioner/data_generation/build_pilot_manifest.py \
  --output-manifest gnn_preconditioner/manifests/pilot_cases_v1.json \
  --output-split gnn_preconditioner/manifests/pilot_split_v1.json \
  --seed 42 \
  --num-cases 64 \
  --force-case-id 51

python3 gnn_preconditioner/data_generation/calibrate_mesh_profile.py \
  --profile gnn_preconditioner/data_generation/mesh_profiles/coarse_v1.json \
  --case-ids 0 51 100 200 299 \
  --target-min-cells 8000 \
  --target-max-cells 12000

python3 gnn_preconditioner/data_generation/generate_cases_iterreduce.py \
  --manifest gnn_preconditioner/manifests/pilot_cases_v1.json \
  --mesh-profile gnn_preconditioner/data_generation/mesh_profiles/coarse_v1.json \
  --end-time 2.0 \
  --write-interval 0.01 \
  --front-back-mode empty

python3 gnn_preconditioner/data_generation/build_volume_dataset.py \
  --manifest gnn_preconditioner/manifests/pilot_cases_v1.json \
  --output-h5 gnn_preconditioner/data_generation/datasets/iterreduce_pilot_v1.h5

bash gnn_preconditioner/training/run_full_surface_coarse_v1_t3p0_state_hd128_seed1337.sh

python3 gnn_preconditioner/training/run_evaluation.py \
  --checkpoint-path gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/best_model.pt \
  --stats-path gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/stats.json \
  --dataset-path gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5 \
  --split-path gnn_preconditioner/manifests/full_split_240_30_30_v1.json \
  --split test \
  --rollout-steps 100 \
  --device cuda

bash gnn_preconditioner/inference/test_case_iterreduce_v1.sh \
  --case-id 51 \
  --end-time 0.1
```

`run_evaluation.py` now also writes a geometry-faithful masked rollout GIF and a
boundary-assignment plot into `checkpoint_dir/eval/` by default.

The case test wrapper accepts either `--case-id` or `--case-dir` and writes
parsed benchmark metrics to `run_root/abg_metrics_iterreduce_v1.json`.
