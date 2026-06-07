#!/bin/bash
# Dedicated wall-clock / inference timing run for the full 30-case test split,
# on an EXCLUSIVE H100 node (clean CPU+GPU, no login-node contention). Produces:
#   - transient matched (deltaT=0.01, alpha=1): per-case load-once + per-injection
#     forward (inference) + inject (full hook), and the solver-step time.
#   - steady simpleFoam startup-rollout (r=30, r=100): solve time + iteration deltas
#     (offline rollout-generation cost is derived from the transient per-call forward).
# Submit from the repo root:  sbatch gnn_preconditioner/benchmarks/slurm_timing_full_30case.sh
#SBATCH --job-name=timing-full30
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=gnn_preconditioner/benchmarks/runs/%x_%j.out
#SBATCH --error=gnn_preconditioner/benchmarks/runs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source gnn_preconditioner/setup_env.sh --runtime-only
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

TESTCASES="6 10 11 13 18 30 37 78 94 97 101 107 113 140 150 155 171 174 181 188 193 208 214 219 232 237 246 251 254 257"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${REPO_ROOT}/gnn_preconditioner/benchmarks/runs/timing_full_30case_${STAMP}"
mkdir -p "${OUT}"

echo "=== timing-full30 job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
nvidia-smi -L || true
echo "OUT=${OUT}"

# (1) Transient matched: dt=0.01 -> dt-scaled alpha = dt/T_GNN = 1 (full overwrite),
#     one-step injection every step. 200 injections/case.
echo "### TRANSIENT matched (dt=0.01, alpha=1, every step) ###"
bash gnn_preconditioner/inference/sweep_dtscaled_iterreduce_v1.sh \
  --cases "${TESTCASES}" \
  --variants "dtscaled_train_r1" \
  --delta-t 0.01 --end-time 2.0 \
  --results-dir "${OUT}/transient_matched"

# (2) Steady simpleFoam startup-rollout at r=30 and r=100.
echo "### STEADY startup-rollout (r=30, r=100) ###"
bash gnn_preconditioner/inference/sweep_simplefoam_startup_rollout.sh \
  --cases "${TESTCASES}" \
  --rollouts "30 100" \
  --results-dir "${OUT}/steady"

# inode hygiene: drop prepared-case copies and OpenFOAM time-step dirs; keep 0/,
# logs, metrics JSON, and summary.csv (everything the timing analysis needs).
echo "### cleanup (time-step dirs + prepared cases) ###"
find "${OUT}" -type d -name prepared_case -prune -exec rm -rf {} + 2>/dev/null || true
for sub in coldstart iterreduce cold warm; do
  find "${OUT}" -path "*/${sub}/*" -type d -regextype posix-extended \
    -regex '.*/[0-9]+([.][0-9]+)?' ! -name '0' -prune -exec rm -rf {} + 2>/dev/null || true
done

echo "=== DONE $(date) ==="
echo "transient summary dir: ${OUT}/transient_matched"
echo "steady    summary:     ${OUT}/steady/summary.csv"
