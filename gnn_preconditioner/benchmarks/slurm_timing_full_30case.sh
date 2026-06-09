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

# Under SLURM the batch script is copied to a spool dir, so BASH_SOURCE/$0 point
# there, not at the repo. sbatch is run from the repo root, so prefer SLURM_SUBMIT_DIR
# (fall back to BASH_SOURCE for a plain local `bash ...` run).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f gnn_preconditioner/setup_env.sh ]]; then
  echo "ERROR: gnn_preconditioner/setup_env.sh not found under REPO_ROOT=${REPO_ROOT}; submit from the repo root." >&2
  exit 1
fi

source gnn_preconditioner/setup_env.sh --runtime-only
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Pin CPU threads. --exclusive allocates the whole 96-core c23g node, and
# libtorch's default intra-op pool = all cores, which oversubscribes the tiny
# per-call GNN forward: the CPU-side dispatch/sync/OpenMP spin-up dominates the
# ~6 ms of useful work, inflating measured load/forward/inject ~1.3-1.6x even
# though the solver itself runs faster on the uncontended node. Pin to the 8
# allocated cores so the reported inference overhead is reproducible and not a
# thread-oversubscription artifact.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

TESTCASES="6 10 11 13 18 30 37 78 94 97 101 107 113 140 150 155 171 174 181 188 193 208 214 219 232 237 246 251 254 257"
SANITY_CASES="11 94 188"
CKPT_DIR="${REPO_ROOT}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1"
CASES_ROOT_TRANSIENT="${REPO_ROOT}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
DATASET_PATH="${REPO_ROOT}/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${REPO_ROOT}/gnn_preconditioner/benchmarks/runs/timing_full_30case_${STAMP}"
mkdir -p "${OUT}"

echo "=== timing-full30 job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
nvidia-smi -L || true
echo "OUT=${OUT}"

# (A) Transient matched: dt=0.01 -> dt-scaled alpha = dt/T_GNN = 1 (full overwrite),
#     one-step injection every step. 200 injections/case (6000 calls total).
#     --adjust-time-step no is REQUIRED: the case template defaults to
#     adjustTimeStep yes (maxCo 5, maxDeltaT 0.01, deltaT 1e-6), so without it the
#     COLD baseline runs adaptive (Courant-limited below 0.01 on high-velocity
#     cases) while the injected model run is pinned to a fixed 0.01 cadence -- an
#     invalid adaptive-vs-fixed comparison that inflates cold and halves the
#     apparent penalty. Fixed dt for both reproduces the published +21.59%.
echo "### (A) TRANSIENT matched (dt=0.01, alpha=1, every step, FIXED dt) ###"
bash gnn_preconditioner/inference/sweep_dtscaled_iterreduce_v1.sh \
  --cases "${TESTCASES}" \
  --variants "dtscaled_train_r1" \
  --adjust-time-step no --delta-t 0.01 --end-time 2.0 \
  --results-dir "${OUT}/transient_matched"

# (B) Transient fine-step: dt=0.002 with gnnTrainingDt=0.01 -> alpha=0.2 (dt-scaled
#     blend), one-step injection every step. 1000 injections/case (30000 calls).
echo "### (B) TRANSIENT fine-step (dt=0.002, alpha=0.2, every step) ###"
bash gnn_preconditioner/inference/sweep_dtscaled_iterreduce_v1.sh \
  --cases "${TESTCASES}" \
  --variants "dtscaled_train_r1" \
  --adjust-time-step no --delta-t 0.002 --end-time 2.0 \
  --results-dir "${OUT}/transient_finestep"

# (C) Steady simpleFoam startup-rollout at r=30 and r=100. The instrumented
#     write_startup_rollout_fields.py records offline rollout-generation wall time
#     (model load + r-step forward) into each warm case's manifest.
echo "### (C) STEADY startup-rollout (r=30, r=100) ###"
bash gnn_preconditioner/inference/sweep_simplefoam_startup_rollout.sh \
  --cases "${TESTCASES}" \
  --rollouts "30 100" \
  --results-dir "${OUT}/steady"

# (D) Sanity controls: identityNoop (no model call) and identity_model (model
#     loaded, returns input) must match cold exactly; bad_zero (writes zeros)
#     must hurt. Same dtscaled_train_r1 regime as panel (A); bad_zero diverges
#     fast so it runs only to t=0.1 (10 injections).
echo "### (D) SANITY controls (identityNoop, identity_model, bad_zero) ###"
for MODE in identityNoop identity_model bad_zero; do
  if [[ "${MODE}" == "bad_zero" ]]; then SAN_END="0.1"; else SAN_END="2.0"; fi
  for CASE_ID in ${SANITY_CASES}; do
    CASE_TAG="$(printf "%03d" "${CASE_ID}")"
    SAN_ROOT="${OUT}/sanity/${MODE}/case_${CASE_TAG}"
    mkdir -p "$(dirname "${SAN_ROOT}")"
    echo "--- sanity ${MODE} case ${CASE_TAG} (end=${SAN_END}) ---"
    bash gnn_preconditioner/inference/test_case_iterreduce_v1.sh \
      --case-id "${CASE_ID}" \
      --cases-root "${CASES_ROOT_TRANSIENT}" \
      --dataset-path "${DATASET_PATH}" \
      --checkpoint-path "${CKPT_DIR}/best_model.pt" \
      --stats-path "${CKPT_DIR}/stats.json" \
      --end-time "${SAN_END}" \
      --predictor-mode "${MODE}" \
      --macro-dt 0.01 --rollout-steps 1 --blend-mode dtScaled --gnn-training-dt 0.01 \
      --adjust-time-step no --delta-t 0.01 \
      --run-root "${SAN_ROOT}" 2>&1 | tee "${SAN_ROOT}.log" \
      || echo "[sanity FAIL] ${MODE} case ${CASE_TAG}"
  done
done

# inode hygiene: drop prepared-case copies and OpenFOAM time-step dirs; keep 0/,
# logs, metrics JSON, and summary.csv (everything the timing analysis needs).
echo "### cleanup (time-step dirs + prepared cases) ###"
find "${OUT}" -type d -name prepared_case -prune -exec rm -rf {} + 2>/dev/null || true
for sub in coldstart iterreduce cold warm; do
  find "${OUT}" -path "*/${sub}/*" -type d -regextype posix-extended \
    -regex '.*/[0-9]+([.][0-9]+)?' ! -name '0' -prune -exec rm -rf {} + 2>/dev/null || true
done

echo "=== DONE $(date) ==="
echo "transient matched dir:  ${OUT}/transient_matched"
echo "transient finestep dir: ${OUT}/transient_finestep"
echo "steady    summary:      ${OUT}/steady/summary.csv"
echo "sanity    dir:          ${OUT}/sanity"
echo ""
echo "Summarize all timing from this one run with:"
echo "  python3 gnn_preconditioner/benchmarks/summarize_timing_run.py ${OUT}"
