#!/bin/bash
# Accuracy probe: cold vs warm on 3 cases at md=1.0/r=100, writeInterval=0.1.
# Time dirs preserved. Injection times (t=1.0, 2.0) excluded downstream.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
source gnn_preconditioner/setup_env.sh --runtime-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$REPO_ROOT/gnn_preconditioner/benchmarks/runs/accuracy_probe_${STAMP}"
mkdir -p "$RUN_ROOT"

CKPT="$REPO_ROOT/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/best_model.pt"
STATS="$REPO_ROOT/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/stats.json"
DATASET="$REPO_ROOT/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
CASES_ROOT="$REPO_ROOT/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"

declare -A CASE_DIR=(
  [011]="case_011_U0.359_H0.0125_L0.0749"
  [018]="case_018_U0.393_H0.0088_L0.0963"
  [094]="case_094_U0.328_H0.0118_L0.0908"
)

# Stage a tmp base-case per case with writeInterval=0.1, endTime=3.0.
TMP_ROOT="$RUN_ROOT/staged_cases"
mkdir -p "$TMP_ROOT"

for CASE_ID in 011 018 094; do
  SRC="$CASES_ROOT/${CASE_DIR[$CASE_ID]}"
  DST="$TMP_ROOT/${CASE_DIR[$CASE_ID]}"
  cp -r "$SRC" "$DST"
  foamDictionary -case "$DST" system/controlDict -entry writeInterval -set 0.1 >/dev/null
  foamDictionary -case "$DST" system/controlDict -entry endTime       -set 3.0 >/dev/null

  TAG="case${CASE_ID}_md1p0_r100"
  OUT="$RUN_ROOT/$TAG"
  echo "[$(date '+%H:%M:%S')] === $TAG start ==="
  bash "$REPO_ROOT/gnn_preconditioner/inference/run_case_iterreduce_v1.sh" \
    --case-dir "$DST" \
    --checkpoint-path "$CKPT" \
    --stats-path "$STATS" \
    --dataset-path "$DATASET" \
    --trajectory-key "trajectory_$((10#$CASE_ID))" \
    --run-root "$OUT" \
    --end-time 3.0 \
    --predictor-mode model \
    --start-inject-time 0.0 \
    --inject-every-time-step false \
    --macro-dt 1.0 \
    --rollout-steps 100 \
    --export-device cuda:0 \
    >"$OUT.stdout" 2>&1
  echo "[$(date '+%H:%M:%S')] === $TAG done ==="
done

echo "RUN_ROOT=$RUN_ROOT"
