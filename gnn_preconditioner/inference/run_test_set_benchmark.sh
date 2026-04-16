#!/usr/bin/env bash
set -euo pipefail
#
# Run loop-coupled (first-iter injection) benchmark across all 30 test-set cases.
# Uses the full 300-case checkpoint (hd128, seed1337, run1).
#
# Usage:
#   bash run_test_set_benchmark.sh [--end-time <scalar>] [--dry-run]
#
# Requires: OpenFOAM env loaded, GPU available.
# Expected runtime: ~3-5 min per case, ~90-150 min total.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Full 300-case checkpoint paths (override CHECKPOINT_DIR env to swap model) ---
CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
DATASET_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_model.pt"
STATS_PATH="${CHECKPOINT_DIR}/stats.json"
TAG="${TAG:-$(basename "${CHECKPOINT_DIR}")}"

END_TIME="0.1"
DRY_RUN="false"
RESULTS_DIR=""

# --- Test set case IDs (from full_split_240_30_30_v1.json) ---
# Full 30: 6 10 11 13 18 30 37 78 94 97 101 107 113 140 150 155 171 174 181 188 193 208 214 219 232 237 246 251 254 257
# Default: first 10 for a quick sweep; override with TEST_CASES env var
TEST_CASES=(
  ${TEST_CASES:-6 10 11 13 18 30 37 78 94 97}
)

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options]

Options:
  --end-time <scalar>   Benchmark end time (default: ${END_TIME})
  --results-dir <path>  Override results output directory
  --dry-run             Print commands without executing
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --end-time) END_TIME="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/test_set_bench_t${END_TIME//./p}_${TAG}_${TIMESTAMP}"
fi

echo "=== Test Set Benchmark ==="
echo "  cases:      ${#TEST_CASES[@]}"
echo "  end time:   ${END_TIME}"
echo "  checkpoint: ${CHECKPOINT_PATH}"
echo "  results:    ${RESULTS_DIR}"
echo ""

# Verify critical paths
for P in "${CHECKPOINT_PATH}" "${STATS_PATH}" "${DATASET_PATH}"; do
  if [[ ! -f "${P}" ]]; then
    echo "FATAL: Missing file: ${P}" >&2
    exit 1
  fi
done

PASSED=0
FAILED=0
SKIPPED=0
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"

mkdir -p "${RESULTS_DIR}"
echo "case_id,cold_total,model_total,delta,delta_pct,cold_p,model_p,delta_p,delta_p_pct,status" > "${SUMMARY_FILE}"

for CASE_ID in "${TEST_CASES[@]}"; do
  CASE_TAG=$(printf "%03d" "${CASE_ID}")
  RUN_ROOT="${RESULTS_DIR}/case_${CASE_TAG}"

  echo ""
  echo "--- Case ${CASE_TAG} ($(( PASSED + FAILED + SKIPPED + 1 ))/${#TEST_CASES[@]}) ---"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] Would run: test_case_iterreduce_v1.sh --case-id ${CASE_ID} --end-time ${END_TIME} --run-root ${RUN_ROOT}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if bash "${SCRIPT_DIR}/test_case_iterreduce_v1.sh" \
    --case-id "${CASE_ID}" \
    --cases-root "${CASES_ROOT}" \
    --dataset-path "${DATASET_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --stats-path "${STATS_PATH}" \
    --end-time "${END_TIME}" \
    --predictor-mode model \
    --run-root "${RUN_ROOT}" 2>&1 | tee "${RUN_ROOT}.log"; then

    # Extract results from the metrics JSON
    METRICS="${RUN_ROOT}/abg_metrics_iterreduce_v1.json"
    if [[ -f "${METRICS}" ]]; then
      ROW=$(python3 -c "
import json, sys
m = json.load(open('${METRICS}'))
comp = m.get('comparison', {})
cold_t = m['coldstart']['total_linear_iterations']
model_t = m['iterreduce']['total_linear_iterations']
delta = comp['total_iteration_delta']
pct = comp['total_iteration_delta_pct']
cold_p = m['coldstart']['counts']['p']
model_p = m['iterreduce']['counts']['p']
delta_p = comp['pressure_iteration_delta']
pct_p = 100.0 * delta_p / cold_p if cold_p else 0
print(f'${CASE_TAG},{cold_t},{model_t},{delta},{pct:.2f},{cold_p},{model_p},{delta_p},{pct_p:.2f},ok')
" 2>/dev/null || echo "${CASE_TAG},,,,,,,,,metrics_parse_error")
      echo "${ROW}" >> "${SUMMARY_FILE}"
      echo "[OK] Case ${CASE_TAG}: ${ROW}"
      PASSED=$((PASSED + 1))
    else
      echo "${CASE_TAG},,,,,,,,,no_metrics" >> "${SUMMARY_FILE}"
      echo "[WARN] Case ${CASE_TAG}: ran but no abg_metrics_iterreduce_v1.json found"
      FAILED=$((FAILED + 1))
    fi
  else
    echo "${CASE_TAG},,,,,,,,,run_failed" >> "${SUMMARY_FILE}"
    echo "[FAIL] Case ${CASE_TAG}: benchmark failed"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=== Test Set Benchmark Complete ==="
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo "  Summary: ${SUMMARY_FILE}"

if [[ "${FAILED}" -gt 0 ]]; then
  echo ""
  echo "  WARNING: ${FAILED} case(s) failed. Check individual logs."
fi
