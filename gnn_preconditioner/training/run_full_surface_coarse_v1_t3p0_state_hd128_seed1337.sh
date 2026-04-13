#!/usr/bin/env bash
set -euo pipefail

cd /work/<hpc-user>/Abhishek/Thesis/Code/Simulator/ma_gnn_pc/gnn_preconditioner
source setup_env.sh --runtime-only

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

DATASET_PATH="${DATASET_PATH:-gnn_preconditioner/data_generation/datasets/iterreduce_full_surface_coarse_v1_t3p0.h5}"
SPLIT_PATH="${SPLIT_PATH:-gnn_preconditioner/manifests/full_split_240_30_30_v1.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-gnn_preconditioner/checkpoints/full_surface_coarse_v1_t3p0_state_hd128_seed1337_run1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5}"
SEED="${SEED:-1337}"

mkdir -p "$CHECKPOINT_DIR"
exec > >(tee -a "$CHECKPOINT_DIR/train.log") 2>&1

ARGS=(
  --dataset-path "$DATASET_PATH"
  --split-path "$SPLIT_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --representation surface_prism
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --validation-interval "$VALIDATION_INTERVAL"
  --device cuda
  --h100-fast
  --seed "$SEED"
)

if [[ -f "$CHECKPOINT_DIR/last_model.pt" ]]; then
  echo "Resuming from $CHECKPOINT_DIR/last_model.pt"
  ARGS+=(--resume)
fi

python3 gnn_preconditioner/training/train.py "${ARGS[@]}"
