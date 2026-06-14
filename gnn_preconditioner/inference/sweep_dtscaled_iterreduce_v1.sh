#!/usr/bin/env bash
set -euo pipefail
#
# Sweep loop-coupled iterReduce variants for the dtScaled interpretation check.
#
# Default panel:
#   manual_r100             macroDt=1.0, rollout=100, full overwrite
#   dtscaled_train_r100     macroDt=1.0, rollout=100, gnnTrainingDt=0.01
#   dtscaled_horizon_r100   macroDt=1.0, rollout=100, gnnTrainingDt=1.0
#   dtscaled_train_r10      macroDt=0.1, rollout=10,  gnnTrainingDt=0.01
#   dtscaled_train_r1       macroDt=0.01, rollout=1,  gnnTrainingDt=0.01
#
# Fixed-step ablations can be run with:
#   --adjust-time-step no --delta-t 0.002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
DATASET_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_model.pt"
STATS_PATH="${CHECKPOINT_DIR}/stats.json"

CASES="6 10 11 13 18 30 37 78 94 97"
VARIANTS="manual_r100 dtscaled_train_r100 dtscaled_horizon_r100 dtscaled_train_r10 dtscaled_train_r1"
END_TIME="3.0"
ADJUST_TIME_STEP=""
INJECT_EVERY_TIME_STEP=""
DELTA_T=""
RESULTS_DIR=""
DRY_RUN="false"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options]

Options:
  --cases "<ids>"             Case ids (default: ${CASES})
  --variants "<names>"        Variant names (default: ${VARIANTS})
  --end-time <scalar>         Benchmark end time (default: ${END_TIME})
  --adjust-time-step <yes|no> Override adjustTimeStep
  --delta-t <scalar>          Override deltaT
  --results-dir <path>        Output directory
  --dry-run                   Print commands without running
  -h, --help                  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases) CASES="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --adjust-time-step) ADJUST_TIME_STEP="$2"; shift 2 ;;
    --inject-every-time-step) INJECT_EVERY_TIME_STEP="$2"; shift 2 ;;
    --delta-t) DELTA_T="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${RESULTS_DIR}" ]]; then
  tag="adaptive"
  if [[ -n "${ADJUST_TIME_STEP}" || -n "${DELTA_T}" ]]; then
    tag="fixed_dt${DELTA_T//./p}"
  fi
  RESULTS_DIR="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/dtscaled_panel_${tag}_${TIMESTAMP}"
fi
mkdir -p "${RESULTS_DIR}"

SUMMARY="${RESULTS_DIR}/summary.csv"
echo "variant,case_id,macro_dt,rollout_steps,blend_mode,gnn_training_dt,adjust_time_step,delta_t,cold_total,model_total,delta,delta_pct,cold_p,model_p,delta_p,delta_p_pct,cold_Ux,model_Ux,cold_Uy,model_Uy,injections,total_forward_wall_s,status,run_root" > "${SUMMARY}"

variant_config() {
  local variant="$1"
  case "${variant}" in
    manual_r100)
      MACRO_DT="1.0"; ROLLOUT_STEPS="100"; BLEND_MODE="manual"; GNN_TRAINING_DT=""; START_INJECT_TIME="" ;;
    manual_r100_skip0p5)
      MACRO_DT="1.0"; ROLLOUT_STEPS="100"; BLEND_MODE="manual"; GNN_TRAINING_DT=""; START_INJECT_TIME="0.5" ;;
    dtscaled_train_r100)
      MACRO_DT="1.0"; ROLLOUT_STEPS="100"; BLEND_MODE="dtScaled"; GNN_TRAINING_DT="0.01"; START_INJECT_TIME="" ;;
    dtscaled_horizon_r100)
      MACRO_DT="1.0"; ROLLOUT_STEPS="100"; BLEND_MODE="dtScaled"; GNN_TRAINING_DT="1.0"; START_INJECT_TIME="" ;;
    dtscaled_train_r10)
      MACRO_DT="0.1"; ROLLOUT_STEPS="10"; BLEND_MODE="dtScaled"; GNN_TRAINING_DT="0.01"; START_INJECT_TIME="" ;;
    dtscaled_train_r1)
      MACRO_DT="0.01"; ROLLOUT_STEPS="1"; BLEND_MODE="dtScaled"; GNN_TRAINING_DT="0.01"; START_INJECT_TIME="" ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 1 ;;
  esac
}

echo "=== dtScaled iterReduce panel ==="
echo "  cases:      ${CASES}"
echo "  variants:   ${VARIANTS}"
echo "  endTime:    ${END_TIME}"
echo "  adjustTime: ${ADJUST_TIME_STEP:-template}"
echo "  deltaT:     ${DELTA_T:-template}"
echo "  results:    ${RESULTS_DIR}"
echo ""

PASSED=0
FAILED=0

for VARIANT in ${VARIANTS}; do
  variant_config "${VARIANT}"

  for CASE_ID in ${CASES}; do
    CASE_TAG="$(printf "%03d" "${CASE_ID}")"
    RUN_ROOT="${RESULTS_DIR}/${VARIANT}/case_${CASE_TAG}"
    mkdir -p "$(dirname "${RUN_ROOT}")"

    CMD=(
      bash "${SCRIPT_DIR}/test_case_iterreduce_v1.sh"
      --case-id "${CASE_ID}"
      --cases-root "${CASES_ROOT}"
      --dataset-path "${DATASET_PATH}"
      --checkpoint-path "${CHECKPOINT_PATH}"
      --stats-path "${STATS_PATH}"
      --end-time "${END_TIME}"
      --predictor-mode model
      --macro-dt "${MACRO_DT}"
      --rollout-steps "${ROLLOUT_STEPS}"
      --blend-mode "${BLEND_MODE}"
      --dt-scaled-alpha-max 1.0
      --run-root "${RUN_ROOT}"
    )
    if [[ -n "${GNN_TRAINING_DT}" ]]; then
      CMD+=(--gnn-training-dt "${GNN_TRAINING_DT}")
    fi
    if [[ -n "${START_INJECT_TIME}" ]]; then
      CMD+=(--start-inject-time "${START_INJECT_TIME}")
    fi
    if [[ -n "${INJECT_EVERY_TIME_STEP}" ]]; then
      CMD+=(--inject-every-time-step "${INJECT_EVERY_TIME_STEP}")
    fi
    if [[ -n "${ADJUST_TIME_STEP}" ]]; then
      CMD+=(--adjust-time-step "${ADJUST_TIME_STEP}")
    fi
    if [[ -n "${DELTA_T}" ]]; then
      CMD+=(--delta-t "${DELTA_T}")
    fi

    echo ""
    echo "--- ${VARIANT} case ${CASE_TAG} ---"
    if [[ "${DRY_RUN}" == "true" ]]; then
      printf '[dry-run]'
      printf ' %q' "${CMD[@]}"
      printf '\n'
      continue
    fi

    if "${CMD[@]}" 2>&1 | tee "${RUN_ROOT}.log"; then
      METRICS="${RUN_ROOT}/abg_metrics_iterreduce_v1.json"
      if [[ -f "${METRICS}" ]]; then
        ROW=$(python3 - "${METRICS}" "${VARIANT}" "${CASE_TAG}" "${MACRO_DT}" "${ROLLOUT_STEPS}" "${BLEND_MODE}" "${GNN_TRAINING_DT}" "${ADJUST_TIME_STEP}" "${DELTA_T}" "${RUN_ROOT}" <<'PY'
import json
import sys

path, variant, case_tag, macro_dt, rollout, blend, gnn_dt, adjust, delta_t, run_root = sys.argv[1:]
m = json.load(open(path))
cold = m["coldstart"]
model = m["iterreduce"]
comp = m["comparison"]
cold_p = cold["counts"]["p"]
model_p = model["counts"]["p"]
delta_p = comp["pressure_iteration_delta"]
pct_p = 100.0 * delta_p / cold_p if cold_p else 0.0
fields = [
    variant,
    case_tag,
    macro_dt,
    rollout,
    blend,
    gnn_dt,
    adjust,
    delta_t,
    cold["total_linear_iterations"],
    model["total_linear_iterations"],
    comp["total_iteration_delta"],
    f"{comp['total_iteration_delta_pct']:.4f}",
    cold_p,
    model_p,
    delta_p,
    f"{pct_p:.4f}",
    cold["counts"]["Ux"],
    model["counts"]["Ux"],
    cold["counts"]["Uy"],
    model["counts"]["Uy"],
    model.get("injections"),
    model.get("total_forward_wall_s"),
    "ok",
    run_root,
]
print(",".join("" if v is None else str(v) for v in fields))
PY
)
        echo "${ROW}" >> "${SUMMARY}"
        PASSED=$((PASSED + 1))
      else
        echo "${VARIANT},${CASE_TAG},${MACRO_DT},${ROLLOUT_STEPS},${BLEND_MODE},${GNN_TRAINING_DT},${ADJUST_TIME_STEP},${DELTA_T},,,,,,,,,,,,,,no_metrics,${RUN_ROOT}" >> "${SUMMARY}"
        FAILED=$((FAILED + 1))
      fi
    else
      echo "${VARIANT},${CASE_TAG},${MACRO_DT},${ROLLOUT_STEPS},${BLEND_MODE},${GNN_TRAINING_DT},${ADJUST_TIME_STEP},${DELTA_T},,,,,,,,,,,,,,run_failed,${RUN_ROOT}" >> "${SUMMARY}"
      FAILED=$((FAILED + 1))
    fi
  done
done

echo ""
echo "=== dtScaled panel complete ==="
echo "  passed:  ${PASSED}"
echo "  failed:  ${FAILED}"
echo "  summary: ${SUMMARY}"
