#!/usr/bin/env bash
set -euo pipefail
#
# Steady-state consistency check:
#   Run loop-coupled transient (cold vs model) to a long end time,
#   then compare iteration counts in the steady portion against the
#   direct steady warm-start result.
#
# With adaptive time stepping, cold and model runs will have different
# step counts and deltaTs.  We compare total iterations within the same
# simulation-time window.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- defaults (full-data checkpoint, case_011) ---
CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"
DATASET_PATH="${ROOT_DIR}/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
CHECKPOINT_PATH="${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/best_model.pt"
STATS_PATH="${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/stats.json"
END_TIME="1.0"
STEADY_START="0.5"
CASE_ID="011"
RUN_ROOT=""

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options]

Options:
  --case-id <id>            Case id (default: ${CASE_ID})
  --end-time <scalar>       Simulation end time (default: ${END_TIME})
  --steady-start <scalar>   Start of steady window (default: ${STEADY_START})
  --cases-root <path>       Override cases root
  --checkpoint-path <path>  Override checkpoint
  --stats-path <path>       Override stats
  --dataset-path <path>     Override dataset
  --run-root <path>         Override output directory
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-id) CASE_ID="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --steady-start) STEADY_START="$2"; shift 2 ;;
    --cases-root) CASES_ROOT="$2"; shift 2 ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --stats-path) STATS_PATH="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/steady_consistency_case_${CASE_ID}_t${END_TIME//./p}_$(date -u +%Y%m%dT%H%M%SZ)"
fi

echo "=== Steady-State Consistency Check ==="
echo "  case:         ${CASE_ID}"
echo "  end time:     ${END_TIME}"
echo "  steady start: ${STEADY_START}"
echo "  run root:     ${RUN_ROOT}"
echo ""

# --- run the benchmark via existing test wrapper ---
echo "[1/3] Running cold + model benchmark to endTime=${END_TIME}"
bash "${SCRIPT_DIR}/test_case_iterreduce_v1.sh" \
  --case-id "${CASE_ID}" \
  --cases-root "${CASES_ROOT}" \
  --dataset-path "${DATASET_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --stats-path "${STATS_PATH}" \
  --end-time "${END_TIME}" \
  --predictor-mode model \
  --run-root "${RUN_ROOT}"

# --- locate logs ---
COLD_LOG="${RUN_ROOT}/coldstart/log.coldstart"
MODEL_LOG="${RUN_ROOT}/iterreduce/log.iterreduce"

if [[ ! -f "${COLD_LOG}" ]]; then
  echo "Cold log not found: ${COLD_LOG}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_LOG}" ]]; then
  echo "Model log not found: ${MODEL_LOG}" >&2
  exit 1
fi

PARSER="${SCRIPT_DIR}/parse_per_timestep_iterations.py"

# --- per-time-step dump ---
echo ""
echo "[2/3] Per-time-step iteration counts"
echo ""
echo "--- Cold baseline ---"
python3 "${PARSER}" "${COLD_LOG}" --steady-start "${STEADY_START}"
echo ""
echo "--- Model (loop-coupled) ---"
python3 "${PARSER}" "${MODEL_LOG}" --steady-start "${STEADY_START}"

# --- structured comparison ---
echo ""
echo "[3/3] Steady-portion comparison"
python3 - <<PY
import json, sys
sys.path.insert(0, "${SCRIPT_DIR}")
from parse_per_timestep_iterations import parse_log, summarize_window
from pathlib import Path

steady_start = ${STEADY_START}

cold_steps = parse_log(Path("${COLD_LOG}"))
model_steps = parse_log(Path("${MODEL_LOG}"))

cold_full = summarize_window(cold_steps)
model_full = summarize_window(model_steps)
cold_steady = summarize_window(cold_steps, t_start=steady_start)
model_steady = summarize_window(model_steps, t_start=steady_start)
cold_trans = summarize_window(cold_steps, t_end=steady_start)
model_trans = summarize_window(model_steps, t_end=steady_start)

def delta_str(model_val, cold_val):
    d = model_val - cold_val
    pct = 100.0 * d / cold_val if cold_val else 0
    return f"{d:+d} ({pct:+.2f}%)"

print("=" * 70)
print(f"{'':30s}  {'Cold':>10s}  {'Model':>10s}  {'Delta':>18s}")
print("-" * 70)

print(f"{'Full run total':30s}  {cold_full['total']:10d}  {model_full['total']:10d}  {delta_str(model_full['total'], cold_full['total']):>18s}")
print(f"{'Full run p':30s}  {cold_full['p']:10d}  {model_full['p']:10d}  {delta_str(model_full['p'], cold_full['p']):>18s}")
print(f"{'Full run steps':30s}  {cold_full['n_steps']:10d}  {model_full['n_steps']:10d}")
print()

if cold_trans and model_trans:
    print(f"{'Transient (t<' + str(steady_start) + ') total':30s}  {cold_trans['total']:10d}  {model_trans['total']:10d}  {delta_str(model_trans['total'], cold_trans['total']):>18s}")
    print(f"{'Transient p':30s}  {cold_trans['p']:10d}  {model_trans['p']:10d}  {delta_str(model_trans['p'], cold_trans['p']):>18s}")
    print(f"{'Transient steps':30s}  {cold_trans['n_steps']:10d}  {model_trans['n_steps']:10d}")
    print()

if cold_steady and model_steady:
    print(f"{'Steady (t>=' + str(steady_start) + ') total':30s}  {cold_steady['total']:10d}  {model_steady['total']:10d}  {delta_str(model_steady['total'], cold_steady['total']):>18s}")
    print(f"{'Steady p':30s}  {cold_steady['p']:10d}  {model_steady['p']:10d}  {delta_str(model_steady['p'], cold_steady['p']):>18s}")
    print(f"{'Steady Ux':30s}  {cold_steady['Ux']:10d}  {model_steady['Ux']:10d}  {delta_str(model_steady['Ux'], cold_steady['Ux']):>18s}")
    print(f"{'Steady Uy':30s}  {cold_steady['Uy']:10d}  {model_steady['Uy']:10d}  {delta_str(model_steady['Uy'], cold_steady['Uy']):>18s}")
    print(f"{'Steady steps':30s}  {cold_steady['n_steps']:10d}  {model_steady['n_steps']:10d}")
    print()
    print(f"Steady-portion iteration delta: {delta_str(model_steady['total'], cold_steady['total'])}")
    print(f"Reference: direct steady warm-start on case_011 was -4.94%")
print("=" * 70)

# save structured result
result = {
    "case_id": "${CASE_ID}",
    "end_time": ${END_TIME},
    "steady_start": steady_start,
    "cold_full": cold_full,
    "model_full": model_full,
    "cold_steady": cold_steady,
    "model_steady": model_steady,
    "cold_transient": cold_trans,
    "model_transient": model_trans,
}
out = Path("${RUN_ROOT}") / "steady_consistency_result.json"
out.write_text(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
PY
