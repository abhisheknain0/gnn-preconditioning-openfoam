#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
CHECKPOINT_DIR_DEFAULT="${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1"
CHECKPOINT_PATH="${CHECKPOINT_DIR_DEFAULT}/best_model.pt"
STATS_PATH="${CHECKPOINT_DIR_DEFAULT}/stats.json"
END_TIME="0.1"
SOLVER="pimpleFoam"
CASE_ID=""
CASE_DIR=""
RUN_ROOT=""
ROLLOUT_STEPS="10"
EXPORT_DEVICE="${ITERREDUCE_EXPORT_DEVICE:-cuda:0}"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --case-id <id> [options]
  $(basename "$0") --case-dir <path> [options]

Options:
  --case-id <id>
  --case-dir <path>
  --cases-root <path>
  --checkpoint-path <path>
  --stats-path <path>
  --solver <name>
  --end-time <scalar>
  --rollout-steps <int>
  --run-root <path>
  --export-device <cuda:N|cpu>
  -h, --help
USAGE
}

ensure_runtime_env() {
  if command -v foamDictionary >/dev/null 2>&1 && command -v "${SOLVER}" >/dev/null 2>&1; then
    return
  fi
  if [[ -f "${ROOT_DIR}/gnn_preconditioner/setup_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/gnn_preconditioner/setup_env.sh" --runtime-only >/dev/null 2>&1 || true
  fi
  if ! command -v foamDictionary >/dev/null 2>&1 || ! command -v "${SOLVER}" >/dev/null 2>&1; then
    echo "OpenFOAM runtime environment is not available. Source gnn_preconditioner/setup_env.sh --runtime-only first." >&2
    exit 3
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-id) CASE_ID="$2"; shift 2 ;;
    --case-dir) CASE_DIR="$2"; shift 2 ;;
    --cases-root) CASES_ROOT="$2"; shift 2 ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --stats-path) STATS_PATH="$2"; shift 2 ;;
    --solver) SOLVER="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --export-device) EXPORT_DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

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

ensure_runtime_env

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/startup_rollout_case_${CASE_ID}_t${END_TIME//./p}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
BENCHMARK_MODE="startup_rollout_${SOLVER}"
mkdir -p "${RUN_ROOT}"

clean_case_for_run() {
  local case_path="$1"
  find "${case_path}" -maxdepth 1 -mindepth 1 -type d \
    -regextype posix-extended \
    -regex '.*/[0-9]+([.][0-9]+)?' \
    ! -name '0' -exec rm -rf {} +
  rm -f "${case_path}"/log.*
}

copy_case_minimal() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  cp -r "${src}/0" "${dst}/0"
  cp -r "${src}/constant" "${dst}/constant"
  cp -r "${src}/system" "${dst}/system"
  if [[ -f "${src}/case.foam" ]]; then
    cp "${src}/case.foam" "${dst}/case.foam"
  fi
}

apply_run_controls() {
  local case_path="$1"
  foamDictionary -case "${case_path}" system/controlDict -entry endTime -set "${END_TIME}" >/dev/null
}

COLD_CASE="${RUN_ROOT}/coldstart"
WARM_CASE="${RUN_ROOT}/startup_rollout"
rm -rf "${COLD_CASE}" "${WARM_CASE}"
copy_case_minimal "${CASE_DIR}" "${COLD_CASE}"
copy_case_minimal "${CASE_DIR}" "${WARM_CASE}"
clean_case_for_run "${COLD_CASE}"
clean_case_for_run "${WARM_CASE}"
apply_run_controls "${COLD_CASE}"
apply_run_controls "${WARM_CASE}"

echo "[gnn_preconditioner/startup] writing startup rollout fields"
python3 "${ROOT_DIR}/gnn_preconditioner/inference/write_startup_rollout_fields.py" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --stats-path "${STATS_PATH}" \
  --case-dir "${WARM_CASE}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --device "${EXPORT_DEVICE}" \
  --apply-default-boundary-constraints \
  --manifest-path "${RUN_ROOT}/startup_rollout_manifest.json"

echo "[gnn_preconditioner/startup] running cold baseline"
(cd "${COLD_CASE}" && "${SOLVER}" > log.coldstart 2>&1)

echo "[gnn_preconditioner/startup] running startup-rollout case"
(cd "${WARM_CASE}" && "${SOLVER}" > log.startup_rollout 2>&1)

echo "[gnn_preconditioner/startup] parsing metrics"
python3 - <<PY
import json
import re
from pathlib import Path
run_root = Path(${RUN_ROOT@Q})
rollout_steps = int(${ROLLOUT_STEPS@Q})
end_time = ${END_TIME@Q}

def parse_log(path: Path):
    text = path.read_text()
    counts = {"Ux": 0, "Uy": 0, "p": 0}
    exec_time = None
    clock_time = None
    for line in text.splitlines():
        m = re.search(r"Solving for (Ux|Uy|p), .* No Iterations (\\d+)", line)
        if m:
            counts[m.group(1)] += int(m.group(2))
        m = re.search(r"ExecutionTime = ([0-9.eE+-]+) s\\s+ClockTime = ([0-9.eE+-]+) s", line)
        if m:
            exec_time = float(m.group(1))
            clock_time = float(m.group(2))
    return {
        "log_path": str(path),
        "counts": counts,
        "total_linear_iterations": counts["Ux"] + counts["Uy"] + counts["p"],
        "execution_time_s": exec_time,
        "clock_time_s": clock_time,
    }

cold = parse_log(run_root / "coldstart" / "log.coldstart")
startup = parse_log(run_root / "startup_rollout" / "log.startup_rollout")
manifest_path = run_root / "startup_rollout_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
comparison = {
    "total_iteration_delta": startup["total_linear_iterations"] - cold["total_linear_iterations"],
    "pressure_iteration_delta": startup["counts"]["p"] - cold["counts"]["p"],
    "ux_iteration_delta": startup["counts"]["Ux"] - cold["counts"]["Ux"],
    "uy_iteration_delta": startup["counts"]["Uy"] - cold["counts"]["Uy"],
    "execution_time_delta_s": None if cold["execution_time_s"] is None or startup["execution_time_s"] is None else startup["execution_time_s"] - cold["execution_time_s"],
}
if cold["total_linear_iterations"]:
    comparison["total_iteration_delta_pct"] = 100.0 * comparison["total_iteration_delta"] / cold["total_linear_iterations"]
else:
    comparison["total_iteration_delta_pct"] = None
if comparison["execution_time_delta_s"] is not None and cold["execution_time_s"]:
    comparison["execution_time_delta_pct"] = 100.0 * comparison["execution_time_delta_s"] / cold["execution_time_s"]
else:
    comparison["execution_time_delta_pct"] = None
metrics = {
    "case_dir": ${CASE_DIR@Q},
    "checkpoint_path": ${CHECKPOINT_PATH@Q},
    "stats_path": ${STATS_PATH@Q},
    "solver": ${SOLVER@Q},
    "run_root": str(run_root),
    "benchmark_mode": ${BENCHMARK_MODE@Q},
    "rollout_steps": rollout_steps,
    "end_time": end_time,
    "coldstart": cold,
    "startup_rollout": startup,
    "manifest": manifest,
    "comparison": comparison,
}
out_path = run_root / "abg_metrics_startup_rollout_iterreduce_v1.json"
out_path.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
PY


echo "[gnn_preconditioner/startup] completed run root: ${RUN_ROOT}"
