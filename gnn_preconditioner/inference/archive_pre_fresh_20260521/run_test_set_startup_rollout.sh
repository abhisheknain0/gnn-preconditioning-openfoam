#!/usr/bin/env bash
set -euo pipefail
#
# Run startup warm-start benchmark across a subset of test-set cases.
# Calls test_case_startup_rollout_iterreduce_v1.sh per case and aggregates
# a summary.csv.
#
# Usage:
#   CHECKPOINT_DIR=... TAG=... bash run_test_set_startup_rollout.sh \
#     [--solver pimpleFoam|simpleFoam] [--end-time 0.1] [--dry-run]
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_model.pt"
STATS_PATH="${CHECKPOINT_DIR}/stats.json"
TAG="${TAG:-$(basename "${CHECKPOINT_DIR}")}"

SOLVER="pimpleFoam"
END_TIME="0.1"
ROLLOUT_STEPS="10"
DRY_RUN="false"
RESULTS_DIR=""

TEST_CASES=(
  ${TEST_CASES:-6 10 11 13 18 30 37 78 94 97}
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --solver) SOLVER="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) echo "See header"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/startup_rollout_${SOLVER}_t${END_TIME//./p}_${TAG}_${TIMESTAMP}"
fi

echo "=== Startup Warm-Start Batch ==="
echo "  solver:     ${SOLVER}"
echo "  cases:      ${#TEST_CASES[@]}"
echo "  end time:   ${END_TIME}"
echo "  rollout:    ${ROLLOUT_STEPS} steps"
echo "  checkpoint: ${CHECKPOINT_PATH}"
echo "  results:    ${RESULTS_DIR}"
echo ""

for P in "${CHECKPOINT_PATH}" "${STATS_PATH}"; do
  [[ -f "${P}" ]] || { echo "FATAL: Missing file: ${P}" >&2; exit 1; }
done

mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary.csv"
echo "case_id,cold_total,model_total,delta,delta_pct,cold_p,model_p,delta_p,delta_p_pct,status" > "${SUMMARY}"

PASSED=0; FAILED=0; SKIPPED=0
for CASE_ID in "${TEST_CASES[@]}"; do
  TAG_ID=$(printf "%03d" "${CASE_ID}")
  RUN_ROOT="${RESULTS_DIR}/case_${TAG_ID}"
  echo ""
  echo "--- Case ${TAG_ID} ($(( PASSED + FAILED + SKIPPED + 1 ))/${#TEST_CASES[@]}) ---"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] Would run startup rollout on case ${CASE_ID}"
    SKIPPED=$((SKIPPED + 1)); continue
  fi

  if bash "${SCRIPT_DIR}/test_case_startup_rollout_iterreduce_v1.sh" \
       --case-id "${CASE_ID}" \
       --cases-root "${CASES_ROOT}" \
       --checkpoint-path "${CHECKPOINT_PATH}" \
       --stats-path "${STATS_PATH}" \
       --solver "${SOLVER}" \
       --end-time "${END_TIME}" \
       --rollout-steps "${ROLLOUT_STEPS}" \
       --run-root "${RUN_ROOT}" 2>&1 | tee "${RUN_ROOT}.log"; then
    METRICS="${RUN_ROOT}/abg_metrics_startup_rollout_iterreduce_v1.json"
    if [[ -f "${METRICS}" ]]; then
      ROW=$(python3 -c "
import json
m = json.load(open('${METRICS}'))
cold = m['coldstart']; mdl = m['startup_rollout']; comp = m['comparison']
cold_t = cold['total_linear_iterations']; model_t = mdl['total_linear_iterations']
delta = comp['total_iteration_delta']; pct = comp['total_iteration_delta_pct']
cold_p = cold['counts']['p']; model_p = mdl['counts']['p']
delta_p = comp['pressure_iteration_delta']
pct_p = 100.0 * delta_p / cold_p if cold_p else 0
print(f'${TAG_ID},{cold_t},{model_t},{delta},{pct:.2f},{cold_p},{model_p},{delta_p},{pct_p:.2f},ok')
" 2>/dev/null || echo "${TAG_ID},,,,,,,,,metrics_parse_error")
      echo "${ROW}" >> "${SUMMARY}"
      echo "[OK] Case ${TAG_ID}: ${ROW}"
      PASSED=$((PASSED + 1))
    else
      echo "${TAG_ID},,,,,,,,,no_metrics" >> "${SUMMARY}"
      FAILED=$((FAILED + 1))
    fi
  else
    echo "${TAG_ID},,,,,,,,,run_failed" >> "${SUMMARY}"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=== Complete: ${PASSED} passed / ${FAILED} failed / ${SKIPPED} skipped ==="
echo "Summary: ${SUMMARY}"
