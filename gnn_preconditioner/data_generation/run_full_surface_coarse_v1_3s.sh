#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MANIFEST_PATH="${ROOT_DIR}/gnn_preconditioner/manifests/full_cases_v1.json"
SPLIT_PATH="${ROOT_DIR}/gnn_preconditioner/manifests/full_split_240_30_30_v1.json"
MESH_PROFILE_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/mesh_profiles/coarse_v1.json"

CASE_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
DATASET_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"

END_TIME="3.0"
WRITE_INTERVAL="0.01"
WORKERS="${ITERREDUCE_FULL_WORKERS:-4}"
OVERWRITE="0"
RUN_GENERATE="1"
RUN_BUILD="1"
MAX_CASES=""

usage() {
  cat <<EOF
Usage:
  bash gnn_preconditioner/data_generation/run_full_surface_coarse_v1_3s.sh [options]

Options:
  --workers N         Number of worker processes for generation/build. Default: ${WORKERS}
  --overwrite         Recreate case directories if they already exist.
  --generate-only     Only generate the OpenFOAM case directories.
  --build-only        Only build the surface HDF5 from existing cases.
  --max-cases N       Limit to first N manifest entries for a smoke/batch run.
  -h, --help          Show this help.

Fixed full-run settings:
  manifest:      ${MANIFEST_PATH}
  split:         ${SPLIT_PATH}
  mesh profile:  ${MESH_PROFILE_PATH}
  case root:     ${CASE_ROOT}
  dataset:       ${DATASET_PATH}
  end time:      ${END_TIME}
  write interval:${WRITE_INTERVAL}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --overwrite) OVERWRITE="1"; shift ;;
    --generate-only) RUN_BUILD="0"; shift ;;
    --build-only) RUN_GENERATE="0"; shift ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ "${RUN_GENERATE}" == "1" || "${RUN_BUILD}" == "1" ]] || {
  echo "Nothing to do: both generation and build are disabled." >&2
  exit 1
}

mkdir -p "$(dirname "${DATASET_PATH}")" "${CASE_ROOT}"

echo "=========================================="
echo "Full Surface Dataset Preparation"
echo "=========================================="
echo "manifest:      ${MANIFEST_PATH}"
echo "split:         ${SPLIT_PATH}"
echo "mesh profile:  ${MESH_PROFILE_PATH}"
echo "case root:     ${CASE_ROOT}"
echo "dataset:       ${DATASET_PATH}"
echo "end time:      ${END_TIME}"
echo "write interval:${WRITE_INTERVAL}"
echo "workers:       ${WORKERS}"
echo "overwrite:     ${OVERWRITE}"
if [[ -n "${MAX_CASES}" ]]; then
  echo "max cases:     ${MAX_CASES}"
fi
echo "=========================================="

source "${ROOT_DIR}/gnn_preconditioner/setup_env.sh" --runtime-only

if [[ "${RUN_GENERATE}" == "1" ]]; then
  GENERATE_CMD=(
    python3 gnn_preconditioner/data_generation/generate_cases_iterreduce.py
    --manifest "${MANIFEST_PATH}"
    --mesh-profile "${MESH_PROFILE_PATH}"
    --output-root "${CASE_ROOT}"
    --end-time "${END_TIME}"
    --write-interval "${WRITE_INTERVAL}"
    --front-back-mode empty
    --workers "${WORKERS}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    GENERATE_CMD+=(--overwrite)
  fi
  if [[ -n "${MAX_CASES}" ]]; then
    GENERATE_CMD+=(--max-cases "${MAX_CASES}")
  fi
  echo
  echo "[1/2] Generating full 300-case coarse surface cases"
  printf '  %q ' "${GENERATE_CMD[@]}"
  echo
  "${GENERATE_CMD[@]}"
fi

if [[ "${RUN_BUILD}" == "1" ]]; then
  BUILD_CMD=(
    python3 gnn_preconditioner/data_generation/build_surface_dataset.py
    --manifest "${MANIFEST_PATH}"
    --case-root "${CASE_ROOT}"
    --output-h5 "${DATASET_PATH}"
    --mesh-profile-name coarse_v1
    --workers "${WORKERS}"
  )
  if [[ -n "${MAX_CASES}" ]]; then
    BUILD_CMD+=(--max-cases "${MAX_CASES}")
  fi
  echo
  echo "[2/2] Building full surface-prism HDF5"
  printf '  %q ' "${BUILD_CMD[@]}"
  echo
  "${BUILD_CMD[@]}"
fi

echo
echo "Prepared full dataset configuration:"
echo "  split:   ${SPLIT_PATH}"
echo "  cases:   ${CASE_ROOT}"
echo "  dataset: ${DATASET_PATH}"
