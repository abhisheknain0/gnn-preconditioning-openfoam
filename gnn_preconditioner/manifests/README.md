# Manifests

This directory keeps only load-bearing dataset manifests and split files.

Retained files:

- `full_cases_v1.json`: all 300 full-surface coarse cases.
- `full_split_240_30_30_v1.json`: main thesis split with 240 train, 30
  validation, and 30 test trajectories.
- `pilot_cases_v1.json`: 64-case pilot manifest used by older pilot studies and
  isolated PINN/GNN experiments.
- `pilot_split_v1.json`: pilot split used by the experimental PINN/GNN scripts.

Smoke-only manifests and derived holdout case-list files were removed during the
2026-05 cleanup. The test trajectory lists are already encoded in the split
files, so separate holdout JSON files are redundant.
