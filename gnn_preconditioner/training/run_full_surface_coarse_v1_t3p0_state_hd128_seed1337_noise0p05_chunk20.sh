#!/usr/bin/env bash
set -euo pipefail

cd /work/<hpc-user>/Abhishek/Thesis/Code/Simulator/ma_gnn_pc/gnn_preconditioner
source setup_env.sh --runtime-only

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

DATASET_PATH="${DATASET_PATH:-gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5}"
SPLIT_PATH="${SPLIT_PATH:-gnn_preconditioner/manifests/full_split_240_30_30_v1.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_noise0p05_run1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5}"
SEED="${SEED:-1337}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
NOISE_STD="${NOISE_STD:-0.05}"

mkdir -p "$CHECKPOINT_DIR"
exec > >(tee -a "$CHECKPOINT_DIR/train.log") 2>&1

CURRENT_EPOCH=$(python3 - <<PY
import json
from pathlib import Path
p = Path(r"$CHECKPOINT_DIR") / 'history.json'
if not p.exists():
    print(0)
else:
    hist = json.loads(p.read_text())
    print(int(hist[-1]['epoch']) if hist else 0)
PY
)
DEFAULT_STOP_AFTER_EPOCH=$(( ((CURRENT_EPOCH / CHUNK_SIZE) + 1) * CHUNK_SIZE ))
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-$DEFAULT_STOP_AFTER_EPOCH}"
if (( STOP_AFTER_EPOCH > TOTAL_EPOCHS )); then
  STOP_AFTER_EPOCH="$TOTAL_EPOCHS"
fi

ARGS=(
  --dataset-path "$DATASET_PATH"
  --split-path "$SPLIT_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --representation surface_prism
  --epochs "$TOTAL_EPOCHS"
  --stop-after-epoch "$STOP_AFTER_EPOCH"
  --batch-size "$BATCH_SIZE"
  --validation-interval "$VALIDATION_INTERVAL"
  --device cuda
  --h100-fast
  --seed "$SEED"
  --noise-std "$NOISE_STD"
)

if [[ -f "$CHECKPOINT_DIR/last_model.pt" ]]; then
  echo "Resuming from $CHECKPOINT_DIR/last_model.pt"
  echo "Current completed epoch: $CURRENT_EPOCH"
  echo "Stopping after epoch: $STOP_AFTER_EPOCH"
  echo "Total planned epochs: $TOTAL_EPOCHS"
  echo "Noise std (fraction of feature std): $NOISE_STD"
  ARGS+=(--resume)
else
  echo "Starting fresh run"
  echo "Stopping after epoch: $STOP_AFTER_EPOCH"
  echo "Total planned epochs: $TOTAL_EPOCHS"
  echo "Noise std (fraction of feature std): $NOISE_STD"
fi

python3 gnn_preconditioner/training/train.py "${ARGS[@]}"
