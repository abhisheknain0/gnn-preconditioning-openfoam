#!/bin/bash
#SBATCH --job-name=macrodt-rerun
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

# Re-run macroDt benchmarks on an exclusive GPU.
# Primary target: 10 cases × md=1p0 r=100 (clean wallclock at sweet spot).
# Secondary:      011/018/094 × md=1p5 r=150 and md=3p0 r=300
#                 (previously contaminated by GPU contention).
# Total: 16 runs.

set -euo pipefail

REPO_ROOT="<cluster-workdir>/Abhishek/Thesis/Code/Simulator/ma_gnn_pc"
cd "$REPO_ROOT"
source gnn_preconditioner/setup_env.sh --runtime-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

STAMP="${BENCH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$REPO_ROOT/gnn_preconditioner/benchmarks/runs/macrodt_rerun_clean_${STAMP}"
mkdir -p "$RUN_ROOT"
PROGRESS="$RUN_ROOT/progress.log"
SUMMARY="$RUN_ROOT/summary.csv"

CKPT="$REPO_ROOT/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/best_model.pt"
STATS="$REPO_ROOT/gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1/stats.json"
DATASET="$REPO_ROOT/gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5"
CASES_ROOT="$REPO_ROOT/gnn_preconditioner/data_generation/cases/full_surface_coarse_v1_t3p0"

echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "=== $(date) ==="
nvidia-smi -L || true
echo "RUN_ROOT=$RUN_ROOT"

echo "case,macro_dt,rollout,status,cold_exec_s,model_exec_s,fwd_wall_s" > "$SUMMARY"

# case_id => case_dir suffix
declare -A CASE_DIR=(
  [006]="case_006_U0.145_H0.0089_L0.0918"
  [010]="case_010_U0.220_H0.0112_L0.0693"
  [011]="case_011_U0.359_H0.0125_L0.0749"
  [013]="case_013_U0.331_H0.0100_L0.0953"
  [018]="case_018_U0.393_H0.0088_L0.0963"
  [030]="case_030_U0.341_H0.0110_L0.0891"
  [037]="case_037_U0.213_H0.0109_L0.0946"
  [078]="case_078_U0.383_H0.0107_L0.0684"
  [094]="case_094_U0.328_H0.0118_L0.0908"
  [097]="case_097_U0.304_H0.0142_L0.0700"
)

# Run list: (case, macro_dt, rollout) triples
RUNS=(
  "006 1.0 100" "010 1.0 100" "011 1.0 100" "013 1.0 100" "018 1.0 100"
  "030 1.0 100" "037 1.0 100" "078 1.0 100" "094 1.0 100" "097 1.0 100"
  "011 1.5 150" "018 1.5 150" "094 1.5 150"
  "011 3.0 300" "018 3.0 300" "094 3.0 300"
)

# Allow caller to restrict to a single run for smoke testing.
# Pass SMOKE_RUN="011 1.0 100" to run only that triple.
if [[ -n "${SMOKE_RUN:-}" ]]; then
  echo "SMOKE mode: restricting to '${SMOKE_RUN}'"
  RUNS=("$SMOKE_RUN")
fi

log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$PROGRESS"; }

for entry in "${RUNS[@]}"; do
  read -r CASE_ID MD ROLL <<<"$entry"
  TAG="case${CASE_ID}_md$(printf '%s' "$MD" | tr '.' 'p')_r${ROLL}"
  log "=== $TAG start ==="

  CASE_PATH="$CASES_ROOT/${CASE_DIR[$CASE_ID]}"
  if [[ ! -d "$CASE_PATH" ]]; then
    log "  MISSING case dir: $CASE_PATH"
    echo "$CASE_ID,$MD,$ROLL,missing_case,,," >> "$SUMMARY"
    continue
  fi

  OUT="$RUN_ROOT/$TAG"

  set +e
  bash "$REPO_ROOT/gnn_preconditioner/inference/run_case_iterreduce_v1.sh" \
    --case-dir "$CASE_PATH" \
    --checkpoint-path "$CKPT" \
    --stats-path "$STATS" \
    --dataset-path "$DATASET" \
    --trajectory-key "trajectory_$((10#$CASE_ID))" \
    --run-root "$OUT" \
    --end-time 3.0 \
    --predictor-mode model \
    --start-inject-time 0.0 \
    --inject-every-time-step false \
    --macro-dt "$MD" \
    --rollout-steps "$ROLL" \
    --export-device cuda:0 \
    >"$OUT.stdout" 2>&1
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    log "  $TAG FAILED rc=$RC (see $OUT.stdout)"
    echo "$CASE_ID,$MD,$ROLL,failed,,," >> "$SUMMARY"
    continue
  fi

  J="$OUT/abg_metrics_iterreduce_v1.json"
  if [[ -f "$J" ]]; then
    CS=$(python3 -c "import json; d=json.load(open('$J')); print(d.get('coldstart',{}).get('execution_time_s',''))")
    MS=$(python3 -c "import json; d=json.load(open('$J')); print(d.get('iterreduce',{}).get('execution_time_s',''))")
    FW=$(python3 -c "import json; d=json.load(open('$J')); print(d.get('iterreduce',{}).get('total_forward_wall_s',''))")
    echo "$CASE_ID,$MD,$ROLL,ok,$CS,$MS,$FW" >> "$SUMMARY"
    log "  $TAG OK  cold=${CS}s warm=${MS}s fwd=${FW}s"
  else
    echo "$CASE_ID,$MD,$ROLL,no_metrics,,," >> "$SUMMARY"
    log "  $TAG ran but no metrics json"
  fi
done

log "=== DONE $(date) ==="
echo "Summary: $SUMMARY"
echo "Progress: $PROGRESS"

# Strip bulky OpenFOAM time directories to save inodes; keep logs + metrics.
for d in "$RUN_ROOT"/*/; do
  [[ -d "$d" ]] || continue
  [[ "$(basename "$d")" == "prepared_case" ]] && { rm -rf "$d"; continue; }
  for sub in "$d/coldstart" "$d/iterreduce"; do
    [[ -d "$sub" ]] || continue
    find "$sub" -maxdepth 1 -mindepth 1 -type d \
      -regextype posix-extended -regex '.*/[0-9]+([.][0-9]+)?' \
      ! -name '0' -exec rm -rf {} +
  done
done
log "=== cleanup of time dirs complete ==="
