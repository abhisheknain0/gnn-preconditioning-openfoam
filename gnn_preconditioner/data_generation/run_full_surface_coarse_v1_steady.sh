#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MANIFEST_PATH="${ROOT_DIR}/gnn_preconditioner/manifests/full_cases_v1.json"
MESH_PROFILE_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/mesh_profiles/coarse_v1.json"
BASE_CASE_DIR="${ROOT_DIR}/gnn_preconditioner/data_generation/base_steady"
CASE_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_steady"

END_TIME="${ITERREDUCE_STEADY_END_TIME:-5000}"
WRITE_INTERVAL="${ITERREDUCE_STEADY_WRITE_INTERVAL:-5000}"
WORKERS="${ITERREDUCE_FULL_WORKERS:-2}"
OVERWRITE="0"
MAX_CASES=""

usage() {
  cat <<EOF
Usage:
  bash gnn_preconditioner/data_generation/run_full_surface_coarse_v1_steady.sh [options]

Options:
  --workers N         Number of worker processes. Default: ${WORKERS}
  --overwrite         Recreate case directories if they already exist.
  --max-cases N       Limit to first N manifest entries for a smoke/batch run.
  -h, --help          Show this help.

Fixed steady-run settings:
  manifest:      ${MANIFEST_PATH}
  mesh profile:  ${MESH_PROFILE_PATH}
  base case:     ${BASE_CASE_DIR}
  case root:     ${CASE_ROOT}
  solver:        simpleFoam
  end time:      ${END_TIME}
  write interval:${WRITE_INTERVAL}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --overwrite) OVERWRITE="1"; shift ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${CASE_ROOT}"

echo "=========================================="
echo "Full Steady Surface Case Preparation"
echo "=========================================="
echo "manifest:      ${MANIFEST_PATH}"
echo "mesh profile:  ${MESH_PROFILE_PATH}"
echo "base case:     ${BASE_CASE_DIR}"
echo "case root:     ${CASE_ROOT}"
echo "solver:        simpleFoam"
echo "end time:      ${END_TIME}"
echo "write interval:${WRITE_INTERVAL}"
echo "workers:       ${WORKERS}"
echo "overwrite:     ${OVERWRITE}"
if [[ -n "${MAX_CASES}" ]]; then
  echo "max cases:     ${MAX_CASES}"
fi
echo "=========================================="

source "${ROOT_DIR}/setup_env.sh" --runtime-only

CMD=(
  python3 gnn_preconditioner/data_generation/generate_cases_iterreduce.py
  --manifest "${MANIFEST_PATH}"
  --base-case-dir "${BASE_CASE_DIR}"
  --mesh-profile "${MESH_PROFILE_PATH}"
  --output-root "${CASE_ROOT}"
  --end-time "${END_TIME}"
  --write-interval "${WRITE_INTERVAL}"
  --front-back-mode empty
  --solver simpleFoam
  --workers "${WORKERS}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ -n "${MAX_CASES}" ]]; then
  CMD+=(--max-cases "${MAX_CASES}")
fi

printf '  %q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Prepared steady case configuration:"
echo "  cases: ${CASE_ROOT}"
