#!/usr/bin/env bash
set -euo pipefail
#
# Sweep startup-rollout warm-start with simpleFoam across cases × rollout depths.
# Cold simpleFoam baseline is run ONCE per case and reused across all rollout
# values (single shared cold result per case). Each warm run lives in its own
# standalone case directory under RUN_ROOT/case_XXX/r<rollout>/.
#
# Usage:
#   bash sweep_simplefoam_startup_rollout.sh \
#     [--cases "6 10 11 13 18 30 37 78 94 97"] \
#     [--rollouts "1 10 30 100 300"] \
#     [--end-time 5000] \
#     [--results-dir <path>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CASES_ROOT="${ROOT_DIR}/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_steady"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_model.pt"
STATS_PATH="${CHECKPOINT_DIR}/stats.json"

CASES="6 10 11 13 18 30 37 78 94 97"
ROLLOUTS="1 10 30 100 300"
END_TIME="5000"
RESULTS_DIR=""
EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases) CASES="$2"; shift 2 ;;
    --rollouts) ROLLOUTS="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --cases-root) CASES_ROOT="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_model.pt"; STATS_PATH="${CHECKPOINT_DIR}/stats.json"; shift 2 ;;
    -h|--help) echo "See header"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="${ROOT_DIR}/gnn_preconditioner/benchmarks/runs/sweep_simpleFoam_startup_rollout_${TIMESTAMP}"
fi
mkdir -p "${RESULTS_DIR}"

SUMMARY="${RESULTS_DIR}/summary.csv"
echo "case_id,rollout,cold_total,warm_total,delta,delta_pct,cold_p,warm_p,delta_p,delta_p_pct,cold_Ux,warm_Ux,cold_Uy,warm_Uy,status" > "${SUMMARY}"

echo "=== simpleFoam startup-rollout sweep ==="
echo "  cases:      ${CASES}"
echo "  rollouts:   ${ROLLOUTS}"
echo "  end-time:   ${END_TIME}"
echo "  cases-root: ${CASES_ROOT}"
echo "  results:    ${RESULTS_DIR}"
echo ""

resolve_case_dir() {
  local cid="$1"
  local tag
  tag="$(printf "case_%03d" "${cid}")"
  local match
  match="$(find "${CASES_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${tag}_*" | head -n1)"
  if [[ -z "${match}" ]]; then
    echo ""; return 1
  fi
  echo "${match}"
}

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

parse_log_py() {
  local log_path="$1"
  python3 - "${log_path}" <<'PY'
import re, sys, json
text = open(sys.argv[1]).read()
counts = {"Ux": 0, "Uy": 0, "p": 0}
for line in text.splitlines():
    m = re.search(r"Solving for (Ux|Uy|p), .* No Iterations (\d+)", line)
    if m:
        counts[m.group(1)] += int(m.group(2))
total = counts["Ux"] + counts["Uy"] + counts["p"]
print(json.dumps({"counts": counts, "total": total}))
PY
}

PASSED=0; FAILED=0
for CID in ${CASES}; do
  TAG_ID="$(printf "%03d" "${CID}")"
  SRC_CASE="$(resolve_case_dir "${CID}")"
  if [[ -z "${SRC_CASE}" || ! -d "${SRC_CASE}" ]]; then
    echo "[SKIP] case ${TAG_ID}: no template under ${CASES_ROOT}"
    echo "${TAG_ID},,,,,,,,,,,,,,no_template" >> "${SUMMARY}"
    FAILED=$((FAILED + 1))
    continue
  fi

  CASE_RUN_DIR="${RESULTS_DIR}/case_${TAG_ID}"
  mkdir -p "${CASE_RUN_DIR}"

  # --- COLD baseline (once per case) ---
  COLD_CASE="${CASE_RUN_DIR}/coldstart"
  if [[ ! -f "${CASE_RUN_DIR}/cold_metrics.json" ]]; then
    echo ""
    echo "=== case ${TAG_ID}: cold simpleFoam ==="
    rm -rf "${COLD_CASE}"
    copy_case_minimal "${SRC_CASE}" "${COLD_CASE}"
    clean_case_for_run "${COLD_CASE}"
    apply_run_controls "${COLD_CASE}"
    if (cd "${COLD_CASE}" && simpleFoam > log.coldstart 2>&1); then
      parse_log_py "${COLD_CASE}/log.coldstart" > "${CASE_RUN_DIR}/cold_metrics.json"
    else
      echo "[FAIL] case ${TAG_ID} cold run failed"
      echo "${TAG_ID},,,,,,,,,,,,,,cold_failed" >> "${SUMMARY}"
      FAILED=$((FAILED + 1))
      continue
    fi
  else
    echo "[reuse] case ${TAG_ID} cold metrics already cached"
  fi
  COLD_TOT=$(python3 -c "import json; m=json.load(open('${CASE_RUN_DIR}/cold_metrics.json')); print(m['total'])")
  COLD_P=$(python3  -c "import json; m=json.load(open('${CASE_RUN_DIR}/cold_metrics.json')); print(m['counts']['p'])")
  COLD_UX=$(python3 -c "import json; m=json.load(open('${CASE_RUN_DIR}/cold_metrics.json')); print(m['counts']['Ux'])")
  COLD_UY=$(python3 -c "import json; m=json.load(open('${CASE_RUN_DIR}/cold_metrics.json')); print(m['counts']['Uy'])")

  # --- WARM runs per rollout ---
  for R in ${ROLLOUTS}; do
    WARM_CASE="${CASE_RUN_DIR}/r${R}"
    echo ""
    echo "=== case ${TAG_ID}: warm simpleFoam, rollout=${R} ==="
    rm -rf "${WARM_CASE}"
    copy_case_minimal "${SRC_CASE}" "${WARM_CASE}"
    clean_case_for_run "${WARM_CASE}"
    apply_run_controls "${WARM_CASE}"

    if ! python3 "${ROOT_DIR}/gnn_preconditioner/inference/write_startup_rollout_fields.py" \
          --checkpoint-path "${CHECKPOINT_PATH}" \
          --stats-path "${STATS_PATH}" \
          --case-dir "${WARM_CASE}" \
          --rollout-steps "${R}" \
          --device "${EXPORT_DEVICE}" \
          --apply-default-boundary-constraints \
          --manifest-path "${WARM_CASE}/startup_rollout_manifest.json" \
          > "${WARM_CASE}/log.startup_fields" 2>&1; then
      echo "[FAIL] case ${TAG_ID} r=${R}: rollout-field write failed"
      echo "${TAG_ID},${R},${COLD_TOT},,,,${COLD_P},,,,${COLD_UX},,${COLD_UY},,write_failed" >> "${SUMMARY}"
      FAILED=$((FAILED + 1))
      continue
    fi

    if ! (cd "${WARM_CASE}" && simpleFoam > log.startup_rollout 2>&1); then
      echo "[FAIL] case ${TAG_ID} r=${R}: warm simpleFoam failed"
      echo "${TAG_ID},${R},${COLD_TOT},,,,${COLD_P},,,,${COLD_UX},,${COLD_UY},,warm_failed" >> "${SUMMARY}"
      FAILED=$((FAILED + 1))
      continue
    fi

    WARM_JSON="${WARM_CASE}/warm_metrics.json"
    parse_log_py "${WARM_CASE}/log.startup_rollout" > "${WARM_JSON}"
    WARM_TOT=$(python3 -c "import json; m=json.load(open('${WARM_JSON}')); print(m['total'])")
    WARM_P=$(python3  -c "import json; m=json.load(open('${WARM_JSON}')); print(m['counts']['p'])")
    WARM_UX=$(python3 -c "import json; m=json.load(open('${WARM_JSON}')); print(m['counts']['Ux'])")
    WARM_UY=$(python3 -c "import json; m=json.load(open('${WARM_JSON}')); print(m['counts']['Uy'])")

    DELTA=$((WARM_TOT - COLD_TOT))
    DELTA_P=$((WARM_P - COLD_P))
    PCT=$(python3 -c "print(f'{100.0 * (${WARM_TOT} - ${COLD_TOT}) / ${COLD_TOT}:.2f}')" )
    PCT_P=$(python3 -c "print(f'{100.0 * (${WARM_P} - ${COLD_P}) / ${COLD_P}:.2f}' if ${COLD_P}>0 else '0.00')" )

    echo "${TAG_ID},${R},${COLD_TOT},${WARM_TOT},${DELTA},${PCT},${COLD_P},${WARM_P},${DELTA_P},${PCT_P},${COLD_UX},${WARM_UX},${COLD_UY},${WARM_UY},ok" >> "${SUMMARY}"
    echo "[OK] case ${TAG_ID} r=${R}: total ${COLD_TOT}->${WARM_TOT} (${PCT}%), p ${COLD_P}->${WARM_P} (${PCT_P}%)"
    PASSED=$((PASSED + 1))
  done
done

echo ""
echo "=== Sweep complete: ${PASSED} ok / ${FAILED} failed ==="
echo "Summary: ${SUMMARY}"
