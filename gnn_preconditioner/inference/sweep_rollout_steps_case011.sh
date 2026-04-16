#!/usr/bin/env bash
set -euo pipefail
#
# Sweep --rollout-steps over {1, 3, 5, 10, 20} for case 011, both checkpoints
# (clean vs noise0p05). Aggregates a single sweep_summary.csv.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_ROOT="${ROOT_DIR}/gnn_preconditioner/checkpoints"

CASE_ID="${CASE_ID:-11}"
STEPS=(1 3 5 10 20)
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SWEEP_DIR="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/sweep_rollout_steps_case$(printf "%03d" "${CASE_ID}")_${TIMESTAMP}"
mkdir -p "${SWEEP_DIR}"

SUMMARY="${SWEEP_DIR}/sweep_summary.csv"
echo "checkpoint,rollout_steps,cold_total,model_total,delta,delta_pct,cold_p,model_p,delta_p,delta_p_pct,status" > "${SUMMARY}"

declare -A CKPTS=(
  [clean]="${CKPT_ROOT}/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1"
  [noise0p05]="${CKPT_ROOT}/full_surface_coarse_v1_t3p0_state_hd128_seed1337_noise0p05_run1"
)

for TAG in clean noise0p05; do
  CKPT_DIR="${CKPTS[${TAG}]}"
  for STEPS_N in "${STEPS[@]}"; do
    LABEL="${TAG}_steps${STEPS_N}"
    RUN_ROOT="${SWEEP_DIR}/${LABEL}"
    echo ""
    echo "=== ${LABEL} ==="
    if bash "${SCRIPT_DIR}/test_case_startup_rollout_iterreduce_v1.sh" \
         --case-id "${CASE_ID}" \
         --checkpoint-path "${CKPT_DIR}/best_model.pt" \
         --stats-path "${CKPT_DIR}/stats.json" \
         --rollout-steps "${STEPS_N}" \
         --run-root "${RUN_ROOT}" > "${RUN_ROOT}.log" 2>&1; then
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
print(f'${TAG},${STEPS_N},{cold_t},{model_t},{delta},{pct:.2f},{cold_p},{model_p},{delta_p},{pct_p:.2f},ok')
" 2>/dev/null || echo "${TAG},${STEPS_N},,,,,,,,,metrics_parse_error")
        echo "${ROW}" >> "${SUMMARY}"
        echo "[OK] ${ROW}"
      else
        echo "${TAG},${STEPS_N},,,,,,,,,no_metrics" >> "${SUMMARY}"
      fi
    else
      echo "${TAG},${STEPS_N},,,,,,,,,run_failed" >> "${SUMMARY}"
      echo "[FAIL] ${LABEL} -- see ${RUN_ROOT}.log"
    fi
  done
done

echo ""
echo "=== Sweep complete ==="
echo "Summary: ${SUMMARY}"
column -s, -t "${SUMMARY}"
