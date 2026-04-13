#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CASE_DIR=""
CHECKPOINT_PATH=""
STATS_PATH=""
DATASET_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/datasets/iterreduce_pilot_v1.h5"
TRAJECTORY_KEY=""
RUN_ROOT="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/$(date -u +%Y%m%dT%H%M%SZ)"
PREPARE_ONLY="false"
SKIP_COLD="false"
END_TIME=""
PREDICTOR_MODE=""
START_INJECT_TIME=""
INJECT_EVERY_TIME_STEP=""
EXPORT_DEVICE="${ITERREDUCE_EXPORT_DEVICE:-cuda:0}"
RANDOM_SCALE="${ITERREDUCE_RANDOM_SCALE:-1.0}"
RANDOM_SEED="${ITERREDUCE_RANDOM_SEED:-1234}"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") \
    --case-dir <case_dir> \
    --checkpoint-path <best_model.pt> \
    --stats-path <stats.json> \
    --trajectory-key <trajectory_...> \
    [--dataset-path <iterreduce_pilot_v1.h5>] \
    [--run-root <dir>] \
    [--end-time <scalar>] \
    [--predictor-mode <model|identity|random|bad_zero|bad_reverse>] \
    [--start-inject-time <scalar>] \
    [--inject-every-time-step <true|false>] \
    [--random-scale <scalar>] \
    [--random-seed <int>] \
    [--export-device <cuda:N|cpu>] \
    [--prepare-only] \
    [--skip-cold]
USAGE
}

ensure_runtime_env() {
  if command -v foamDictionary >/dev/null 2>&1 && command -v pimpleFoam >/dev/null 2>&1; then
    return
  fi

  if [[ -f "${ROOT_DIR}/setup_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/setup_env.sh" --runtime-only >/dev/null 2>&1 || true
  fi

  if ! command -v foamDictionary >/dev/null 2>&1 || ! command -v pimpleFoam >/dev/null 2>&1; then
    echo "OpenFOAM runtime environment is not available. Source setup_env.sh --runtime-only first." >&2
    exit 3
  fi
}

ensure_torch_runtime_libs() {
  local -a extra_libs=()
  local base_py="${HOME}/.local/lib/python3.12/site-packages"
  local preferred_libtorch="${HOME}/.local/libtorch-2.5.1-cu121-cxx11/libtorch/lib"
  local torch_lib="${base_py}/torch/lib"
  local nvrtc_lib="${base_py}/nvidia/cuda_nvrtc/lib"

  if [[ -d "${preferred_libtorch}" ]]; then
    extra_libs+=("${preferred_libtorch}")
  elif [[ -d "${torch_lib}" ]]; then
    extra_libs+=("${torch_lib}")
  fi
  if [[ -d "${nvrtc_lib}" ]]; then
    extra_libs+=("${nvrtc_lib}")
  fi

  if [[ ${#extra_libs[@]} -eq 0 ]]; then
    return
  fi

  local joined
  joined="$(IFS=:; printf '%s' "${extra_libs[*]}")"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${joined}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${joined}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-dir) CASE_DIR="$2"; shift 2 ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --stats-path) STATS_PATH="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --trajectory-key) TRAJECTORY_KEY="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --predictor-mode) PREDICTOR_MODE="$2"; shift 2 ;;
    --start-inject-time) START_INJECT_TIME="$2"; shift 2 ;;
    --inject-every-time-step) INJECT_EVERY_TIME_STEP="$2"; shift 2 ;;
    --random-scale) RANDOM_SCALE="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --export-device) EXPORT_DEVICE="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY="true"; shift ;;
    --skip-cold) SKIP_COLD="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${CASE_DIR}" || -z "${CHECKPOINT_PATH}" || -z "${STATS_PATH}" || -z "${TRAJECTORY_KEY}" ]]; then
  usage
  exit 2
fi

ensure_runtime_env
ensure_torch_runtime_libs

mkdir -p "${RUN_ROOT}"
PREP_CASE="${RUN_ROOT}/prepared_case"
rm -rf "${PREP_CASE}"
mkdir -p "${PREP_CASE}"
cp -r "${CASE_DIR}/0" "${PREP_CASE}/0"
cp -r "${CASE_DIR}/constant" "${PREP_CASE}/constant"
cp -r "${CASE_DIR}/system" "${PREP_CASE}/system"
mkdir -p "${PREP_CASE}/constant/iterreduce/context"

clean_case_for_run() {
  local case_path="$1"
  find "${case_path}" -maxdepth 1 -mindepth 1 -type d \
    -regextype posix-extended \
    -regex '.*/[0-9]+([.][0-9]+)?' \
    ! -name '0' -exec rm -rf {} +
  rm -f "${case_path}"/log.*
}

apply_run_controls() {
  local case_path="$1"
  local runtime_predictor_mode="${PREDICTOR_MODE}"
  if [[ "${runtime_predictor_mode}" == "random" || "${runtime_predictor_mode}" == bad_* ]]; then
    runtime_predictor_mode="model"
  fi
  if [[ -n "${END_TIME}" ]]; then
    foamDictionary -case "${case_path}" system/controlDict -entry endTime -set "${END_TIME}" >/dev/null
  fi
  if [[ -n "${runtime_predictor_mode}" ]]; then
    foamDictionary -case "${case_path}" system/iterReduceCouplingDict -entry predictorMode -set "${runtime_predictor_mode}" >/dev/null
  fi
  if [[ -n "${START_INJECT_TIME}" ]]; then
    foamDictionary -case "${case_path}" system/iterReduceCouplingDict -entry startInjectTime -set "${START_INJECT_TIME}" >/dev/null
  fi
  if [[ -n "${INJECT_EVERY_TIME_STEP}" ]]; then
    foamDictionary -case "${case_path}" system/iterReduceCouplingDict -entry injectEveryTimeStep -set "${INJECT_EVERY_TIME_STEP}" >/dev/null
  fi
}

clean_case_for_run "${PREP_CASE}"
apply_run_controls "${PREP_CASE}"

echo "[gnn_preconditioner] exporting TorchScript model"
echo "[gnn_preconditioner] export device: ${EXPORT_DEVICE}"
if [[ "${PREDICTOR_MODE}" == "random" ]]; then
  python3 "${ROOT_DIR}/gnn_preconditioner/training/export_random_torchscript_inference.py" \
    --stats-path "${STATS_PATH}" \
    --device "${EXPORT_DEVICE}" \
    --random-scale "${RANDOM_SCALE}" \
    --seed "${RANDOM_SEED}" \
    --output-path "${PREP_CASE}/constant/iterreduce/model_inference.ts"
elif [[ "${PREDICTOR_MODE}" == bad_* ]]; then
  python3 "${ROOT_DIR}/gnn_preconditioner/training/export_synthetic_torchscript_inference.py" \
    --device "${EXPORT_DEVICE}" \
    --mode "${PREDICTOR_MODE#bad_}" \
    --output-path "${PREP_CASE}/constant/iterreduce/model_inference.ts"
else
  python3 "${ROOT_DIR}/gnn_preconditioner/training/export_torchscript_inference.py" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --stats-path "${STATS_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --trajectory-key "${TRAJECTORY_KEY}" \
    --device "${EXPORT_DEVICE}" \
    --output-path "${PREP_CASE}/constant/iterreduce/model_inference.ts"
fi

echo "[gnn_preconditioner] building runtime context"
python3 "${ROOT_DIR}/gnn_preconditioner/inference/build_case_context.py" \
  --case-dir "${PREP_CASE}" \
  --dataset-path "${DATASET_PATH}" \
  --trajectory-key "${TRAJECTORY_KEY}" \
  --output-dir "${PREP_CASE}/constant/iterreduce/context"

if [[ "${SKIP_COLD}" != "true" ]]; then
  echo "[gnn_preconditioner] running cold baseline"
  COLD_CASE="${RUN_ROOT}/coldstart"
  rm -rf "${COLD_CASE}"
  cp -r "${PREP_CASE}" "${COLD_CASE}"
  clean_case_for_run "${COLD_CASE}"
  apply_run_controls "${COLD_CASE}"
  (cd "${COLD_CASE}" && pimpleFoam > log.coldstart 2>&1)
fi

if [[ "${PREPARE_ONLY}" == "true" ]]; then
  echo "[gnn_preconditioner] prepare-only mode complete"
  exit 0
fi

if ! command -v iterReduceIncompressibleFluid >/dev/null 2>&1; then
  echo "iterReduceIncompressibleFluid binary not found in PATH." >&2
  echo "The Python-side stack is prepared, but the OpenFOAM runtime binary is not currently available." >&2
  echo "Build it with gnn_preconditioner/openfoam_solver/build_helpers/build_user_dir.sh after loading the OpenFOAM environment." >&2
  exit 3
fi

ITER_CASE="${RUN_ROOT}/iterreduce"
rm -rf "${ITER_CASE}"
cp -r "${PREP_CASE}" "${ITER_CASE}"
clean_case_for_run "${ITER_CASE}"
apply_run_controls "${ITER_CASE}"
(cd "${ITER_CASE}" && iterReduceIncompressibleFluid > log.iterreduce 2>&1)

if [[ "${SKIP_COLD}" != "true" ]]; then
  echo "[gnn_preconditioner] parsing benchmark metrics"
  python3 - <<PY
import json
import re
from pathlib import Path

run_root = Path(${RUN_ROOT@Q})
end_time = ${END_TIME@Q}
predictor_mode = ${PREDICTOR_MODE@Q}
start_inject_time = ${START_INJECT_TIME@Q}
inject_every_time_step = ${INJECT_EVERY_TIME_STEP@Q}

def parse_log(path: Path):
    text = path.read_text()
    counts = {"Ux": 0, "Uy": 0, "p": 0}
    exec_time = None
    clock_time = None
    injections = None
    inject_wall = None
    forward_wall = None
    runtime_load_wall = None
    runtime_device = None

    for line in text.splitlines():
        m = re.search(r"Solving for (Ux|Uy|p), .* No Iterations (\\d+)", line)
        if m:
            counts[m.group(1)] += int(m.group(2))

        m = re.search(r"ExecutionTime = ([0-9.eE+-]+) s\\s+ClockTime = ([0-9.eE+-]+) s", line)
        if m:
            exec_time = float(m.group(1))
            clock_time = float(m.group(2))

        m = re.search(r"runtime loaded device=([^ ]+) nCells=(\\d+) loadWall=([0-9.eE+-]+) s", line)
        if m:
            runtime_device = m.group(1)
            runtime_load_wall = float(m.group(3))

        m = re.search(r"injections=(\\d+) totalInjectWall=([0-9.eE+-]+) totalForwardWall=([0-9.eE+-]+)", line)
        if m:
            injections = int(m.group(1))
            inject_wall = float(m.group(2))
            forward_wall = float(m.group(3))

    return {
        "log_path": str(path),
        "counts": counts,
        "total_linear_iterations": counts["Ux"] + counts["Uy"] + counts["p"],
        "execution_time_s": exec_time,
        "clock_time_s": clock_time,
        "runtime_device": runtime_device,
        "runtime_load_wall_s": runtime_load_wall,
        "injections": injections,
        "total_inject_wall_s": inject_wall,
        "total_forward_wall_s": forward_wall,
    }

cold = parse_log(run_root / "coldstart" / "log.coldstart")
iterreduce = parse_log(run_root / "iterreduce" / "log.iterreduce")

comparison = {
    "total_iteration_delta": iterreduce["total_linear_iterations"] - cold["total_linear_iterations"],
    "pressure_iteration_delta": iterreduce["counts"]["p"] - cold["counts"]["p"],
    "ux_iteration_delta": iterreduce["counts"]["Ux"] - cold["counts"]["Ux"],
    "uy_iteration_delta": iterreduce["counts"]["Uy"] - cold["counts"]["Uy"],
    "execution_time_delta_s": (
        None if cold["execution_time_s"] is None or iterreduce["execution_time_s"] is None
        else iterreduce["execution_time_s"] - cold["execution_time_s"]
    ),
}

if cold["total_linear_iterations"]:
    comparison["total_iteration_delta_pct"] = 100.0 * comparison["total_iteration_delta"] / cold["total_linear_iterations"]
else:
    comparison["total_iteration_delta_pct"] = None

if comparison["execution_time_delta_s"] is not None and cold["execution_time_s"]:
    comparison["execution_time_delta_pct"] = 100.0 * comparison["execution_time_delta_s"] / cold["execution_time_s"]
else:
    comparison["execution_time_delta_pct"] = None

if iterreduce["execution_time_s"] is not None and iterreduce["runtime_load_wall_s"] is not None:
    comparison["execution_time_minus_load_s"] = iterreduce["execution_time_s"] - iterreduce["runtime_load_wall_s"]
else:
    comparison["execution_time_minus_load_s"] = None

metrics = {
    "case_dir": ${CASE_DIR@Q},
    "trajectory_key": ${TRAJECTORY_KEY@Q},
    "checkpoint_path": ${CHECKPOINT_PATH@Q},
    "stats_path": ${STATS_PATH@Q},
    "dataset_path": ${DATASET_PATH@Q},
    "run_root": str(run_root),
    "end_time": end_time or None,
    "predictor_mode": predictor_mode or "model",
    "start_inject_time": start_inject_time or None,
    "inject_every_time_step": inject_every_time_step or None,
    "coldstart": cold,
    "iterreduce": iterreduce,
    "comparison": comparison,
}

out_path = run_root / "abg_metrics_iterreduce_v1.json"
out_path.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
PY
fi

echo "[gnn_preconditioner] completed run root: ${RUN_ROOT}"
