#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults point at the current (full-data, surface_prism/legacy3) configuration.
CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
DATASET_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
CHECKPOINT_PATH="${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/best_model.pt"
STATS_PATH=""   # default: sibling stats.json of the checkpoint (see below)
END_TIME="0.1"
ADJUST_TIME_STEP=""
DELTA_T=""
CASE_ID=""
CASE_DIR=""
RUN_ROOT=""
PREPARE_ONLY="false"
SKIP_COLD="false"
LIST_CASES="false"
PREDICTOR_MODE="model"
START_INJECT_TIME=""
INJECT_AT_INITIAL_TIME=""
INJECT_EVERY_TIME_STEP=""
MACRO_DT=""
ROLLOUT_STEPS=""
BLEND_MODE=""
GNN_TRAINING_DT=""
DT_SCALED_ALPHA_MAX=""
RANDOM_SCALE="${ITERREDUCE_RANDOM_SCALE:-1.0}"
RANDOM_SEED="${ITERREDUCE_RANDOM_SEED:-1234}"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --case-id <id> [options]
  $(basename "$0") --case-dir <path> [options]

Options:
  --case-id <id>            Pilot case id, e.g. 51
  --case-dir <path>         Full case path instead of case id
  --cases-root <path>       Override case root (default: ${CASES_ROOT})
  --dataset-path <path>     Override HDF5 dataset
  --checkpoint-path <path>  Override best_model.pt
  --stats-path <path>       Override stats.json
  --end-time <scalar>       Benchmark end time (default: ${END_TIME})
  --adjust-time-step <yes|no>
                            Override adjustTimeStep in system/controlDict
  --delta-t <scalar>        Override deltaT in system/controlDict
  --predictor-mode <mode>   IterReduce predictor mode: model, identity, random, bad_zero, or bad_reverse (default: ${PREDICTOR_MODE})
  --start-inject-time <t>   Override startInjectTime in iterReduceCouplingDict
  --inject-at-initial-time <true|false>
                            Inject before OpenFOAM advances to first timestep
  --inject-every-time-step <true|false>
                            Override injectEveryTimeStep in iterReduceCouplingDict
  --macro-dt <scalar>       Override macroDt in iterReduceCouplingDict
  --rollout-steps <int>     Export TorchScript with internal rollout steps
  --blend-mode <mode>       manual or dtScaled
  --gnn-training-dt <t>     Denominator for dtScaled alphaEff
  --dt-scaled-alpha-max <a> Cap for dtScaled alphaEff
  --random-scale <scalar>   Random mode noise scale in target-std units (default: ${RANDOM_SCALE})
  --random-seed <int>       Random mode seed (default: ${RANDOM_SEED})
  --run-root <path>         Explicit output run root
  --prepare-only            Only export/build context
  --skip-cold               Skip cold baseline
  --list-cases              Print available cases and exit
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-id) CASE_ID="$2"; shift 2 ;;
    --case-dir) CASE_DIR="$2"; shift 2 ;;
    --cases-root) CASES_ROOT="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --stats-path) STATS_PATH="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --adjust-time-step) ADJUST_TIME_STEP="$2"; shift 2 ;;
    --delta-t) DELTA_T="$2"; shift 2 ;;
    --predictor-mode) PREDICTOR_MODE="$2"; shift 2 ;;
    --start-inject-time) START_INJECT_TIME="$2"; shift 2 ;;
    --inject-at-initial-time) INJECT_AT_INITIAL_TIME="$2"; shift 2 ;;
    --inject-every-time-step) INJECT_EVERY_TIME_STEP="$2"; shift 2 ;;
    --macro-dt) MACRO_DT="$2"; shift 2 ;;
    --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
    --blend-mode) BLEND_MODE="$2"; shift 2 ;;
    --gnn-training-dt) GNN_TRAINING_DT="$2"; shift 2 ;;
    --dt-scaled-alpha-max) DT_SCALED_ALPHA_MAX="$2"; shift 2 ;;
    --random-scale) RANDOM_SCALE="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY="true"; shift ;;
    --skip-cold) SKIP_COLD="true"; shift ;;
    --list-cases) LIST_CASES="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Default the stats file to the checkpoint's own directory when not overridden,
# so a single --checkpoint-path override "just works" for any model.
if [[ -z "${STATS_PATH}" ]]; then
  STATS_PATH="$(dirname "${CHECKPOINT_PATH}")/stats.json"
fi

if [[ "${LIST_CASES}" == "true" ]]; then
  ls "${CASES_ROOT}"
  exit 0
fi

if [[ -n "${CASE_ID}" && -n "${CASE_DIR}" ]]; then
  echo "Use either --case-id or --case-dir, not both." >&2
  exit 2
fi

resolve_case_from_id() {
  local raw_id="$1"
  local case_num=$((10#${raw_id}))
  local case_tag
  printf -v case_tag "%03d" "${case_num}"
  local matches=( "${CASES_ROOT}"/case_"${case_tag}"_* )
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Could not resolve unique case for id ${raw_id} under ${CASES_ROOT}" >&2
    exit 2
  fi
  CASE_DIR="${matches[0]}"
  CASE_ID="${case_num}"
}

resolve_case_from_dir() {
  local base
  base="$(basename "${CASE_DIR}")"
  if [[ ! "${base}" =~ ^case_([0-9]+)_ ]]; then
    echo "Could not derive case id from case dir basename: ${base}" >&2
    exit 2
  fi
  CASE_ID="$((10#${BASH_REMATCH[1]}))"
}

if [[ -n "${CASE_ID}" ]]; then
  resolve_case_from_id "${CASE_ID}"
elif [[ -n "${CASE_DIR}" ]]; then
  resolve_case_from_dir
else
  usage
  exit 2
fi

if [[ ! -d "${CASE_DIR}" ]]; then
  echo "Case dir does not exist: ${CASE_DIR}" >&2
  exit 2
fi

TRAJECTORY_KEY="trajectory_${CASE_ID}"

if [[ -z "${RUN_ROOT}" ]]; then
  local_tag="$(basename "${CASE_DIR}")"
  time_tag="${END_TIME//./p}"
  RUN_ROOT="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/${local_tag}_t${time_tag}_$(date -u +%Y%m%dT%H%M%SZ)"
fi

echo "[gnn_preconditioner] resolved case: ${CASE_DIR}"
echo "[gnn_preconditioner] trajectory key: ${TRAJECTORY_KEY}"
echo "[gnn_preconditioner] checkpoint: ${CHECKPOINT_PATH}"
echo "[gnn_preconditioner] stats: ${STATS_PATH}"
echo "[gnn_preconditioner] endTime: ${END_TIME}"
if [[ -n "${ADJUST_TIME_STEP}" ]]; then
  echo "[gnn_preconditioner] adjustTimeStep: ${ADJUST_TIME_STEP}"
fi
if [[ -n "${DELTA_T}" ]]; then
  echo "[gnn_preconditioner] deltaT: ${DELTA_T}"
fi
echo "[gnn_preconditioner] predictorMode: ${PREDICTOR_MODE}"
if [[ -n "${START_INJECT_TIME}" ]]; then
  echo "[gnn_preconditioner] startInjectTime: ${START_INJECT_TIME}"
fi
if [[ -n "${INJECT_AT_INITIAL_TIME}" ]]; then
  echo "[gnn_preconditioner] injectAtInitialTime: ${INJECT_AT_INITIAL_TIME}"
fi
if [[ -n "${INJECT_EVERY_TIME_STEP}" ]]; then
  echo "[gnn_preconditioner] injectEveryTimeStep: ${INJECT_EVERY_TIME_STEP}"
fi
if [[ "${PREDICTOR_MODE}" == "random" ]]; then
  echo "[gnn_preconditioner] randomScale: ${RANDOM_SCALE}"
  echo "[gnn_preconditioner] randomSeed: ${RANDOM_SEED}"
fi
echo "[gnn_preconditioner] run root: ${RUN_ROOT}"

CMD=(
  bash "${SCRIPT_DIR}/run_case_iterreduce_v1.sh"
  --case-dir "${CASE_DIR}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --stats-path "${STATS_PATH}"
  --dataset-path "${DATASET_PATH}"
  --trajectory-key "${TRAJECTORY_KEY}"
  --run-root "${RUN_ROOT}"
  --end-time "${END_TIME}"
  --predictor-mode "${PREDICTOR_MODE}"
)

if [[ -n "${ADJUST_TIME_STEP}" ]]; then
  CMD+=(--adjust-time-step "${ADJUST_TIME_STEP}")
fi
if [[ -n "${DELTA_T}" ]]; then
  CMD+=(--delta-t "${DELTA_T}")
fi
if [[ -n "${START_INJECT_TIME}" ]]; then
  CMD+=(--start-inject-time "${START_INJECT_TIME}")
fi
if [[ -n "${INJECT_AT_INITIAL_TIME}" ]]; then
  CMD+=(--inject-at-initial-time "${INJECT_AT_INITIAL_TIME}")
fi
if [[ -n "${INJECT_EVERY_TIME_STEP}" ]]; then
  CMD+=(--inject-every-time-step "${INJECT_EVERY_TIME_STEP}")
fi
if [[ -n "${MACRO_DT}" ]]; then
  CMD+=(--macro-dt "${MACRO_DT}")
fi
if [[ -n "${ROLLOUT_STEPS}" ]]; then
  CMD+=(--rollout-steps "${ROLLOUT_STEPS}")
fi
if [[ -n "${BLEND_MODE}" ]]; then
  CMD+=(--blend-mode "${BLEND_MODE}")
fi
if [[ -n "${GNN_TRAINING_DT}" ]]; then
  CMD+=(--gnn-training-dt "${GNN_TRAINING_DT}")
fi
if [[ -n "${DT_SCALED_ALPHA_MAX}" ]]; then
  CMD+=(--dt-scaled-alpha-max "${DT_SCALED_ALPHA_MAX}")
fi
if [[ "${PREDICTOR_MODE}" == "random" ]]; then
  CMD+=(--random-scale "${RANDOM_SCALE}" --random-seed "${RANDOM_SEED}")
fi

if [[ "${PREPARE_ONLY}" == "true" ]]; then
  CMD+=(--prepare-only)
fi
if [[ "${SKIP_COLD}" == "true" ]]; then
  CMD+=(--skip-cold)
fi

"${CMD[@]}"
