#!/bin/bash
#SBATCH --job-name=dtscaled-fixed
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=gnn_preconditioner/benchmarks/runs/%x_%j.out
#SBATCH --error=gnn_preconditioner/benchmarks/runs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

source gnn_preconditioner/setup_env.sh --runtime-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

RUN_ROOT="${BENCH_RESULTS_DIR:-$REPO_ROOT/gnn_preconditioner/benchmarks/runs/dtscaled_10case_fixed_dt0p002_optimized_20260505Tfull}"

echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "=== $(date) ==="
echo "RUN_ROOT=$RUN_ROOT"
nvidia-smi -L || true

# The path is intentionally stable because the supervisor deck reads this run.
# Clear partial login-node output before the Slurm rerun.
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

bash gnn_preconditioner/inference/sweep_dtscaled_iterreduce_v1.sh \
  --cases "6 10 11 13 18 30 37 78 94 97" \
  --variants "manual_r100 dtscaled_train_r100 dtscaled_horizon_r100 dtscaled_train_r10 dtscaled_train_r1" \
  --end-time 3.0 \
  --adjust-time-step no \
  --delta-t 0.002 \
  --results-dir "$RUN_ROOT"

# Keep logs, summary.csv, metrics JSON, and time-zero fields; remove generated
# OpenFOAM time directories and prepared-case copies to reduce inode pressure.
find "$RUN_ROOT" -type d -name prepared_case -prune -exec rm -rf {} +
for sub in coldstart iterreduce; do
  find "$RUN_ROOT" -path "*/$sub/*" -type d \
    -regextype posix-extended \
    -regex '.*/[0-9]+([.][0-9]+)?' \
    ! -name '0' -prune -exec rm -rf {} +
done

echo "=== DONE $(date) ==="
echo "Summary: $RUN_ROOT/summary.csv"
