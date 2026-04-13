from __future__ import annotations

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ITERREDUCE_ROOT = PROJECT_ROOT / "gnn_preconditioner"

DATASET_PATH = Path(
    os.environ.get(
        "ITERREDUCE_DATASET_PATH",
        ITERREDUCE_ROOT / "data_generation" / "datasets" / "iterreduce_pilot_surface_coarse_v1.h5",
    )
).expanduser()
SPLIT_PATH = Path(
    os.environ.get(
        "ITERREDUCE_SPLIT_PATH",
        ITERREDUCE_ROOT / "manifests" / "pilot_split_v1.json",
    )
).expanduser()
CHECKPOINT_DIR = Path(
    os.environ.get("ITERREDUCE_CHECKPOINT_DIR", ITERREDUCE_ROOT / "checkpoints")
).expanduser()
REPRESENTATION = "surface_prism"
SURFACE_EDGE_FEATURE_MODE = "legacy3"

NUM_NODE_INPUT = 10
NUM_EDGE_INPUT = 3
NUM_OUTPUT = 3

HIDDEN_DIM = 128
MESSAGE_PASSING_STEPS = 8
MLP_LAYERS = 2
AGGREGATION = "sum"

BATCH_SIZE = int(os.environ.get("ITERREDUCE_BATCH_SIZE", "16"))
LR = float(os.environ.get("ITERREDUCE_LR", "1e-4"))
LR_MIN = float(os.environ.get("ITERREDUCE_LR_MIN", "1e-6"))
EPOCHS = int(os.environ.get("ITERREDUCE_EPOCHS", "100"))
NOISE_STD = float(os.environ.get("ITERREDUCE_NOISE_STD", "0.0"))
LOSS_MASK_MODE = os.environ.get("ITERREDUCE_LOSS_MASK_MODE", "all")
TEMPORAL_WEIGHTING = os.environ.get("ITERREDUCE_TEMPORAL_WEIGHTING", "0").lower() in {"1", "true", "yes"}
NUM_WORKERS = int(os.environ.get("ITERREDUCE_NUM_WORKERS", "2"))
PERSISTENT_WORKERS = os.environ.get("ITERREDUCE_PERSISTENT_WORKERS", "1").lower() not in {"0", "false", "no"}
PREFETCH_FACTOR = int(os.environ.get("ITERREDUCE_PREFETCH_FACTOR", "2"))
IN_MEMORY = os.environ.get("ITERREDUCE_IN_MEMORY", "1").lower() not in {"0", "false", "no"}
GRAD_CLIP = float(os.environ.get("ITERREDUCE_GRAD_CLIP", "1.0"))
WEIGHT_DECAY = float(os.environ.get("ITERREDUCE_WEIGHT_DECAY", "0.0"))
VALIDATION_INTERVAL = int(os.environ.get("ITERREDUCE_VALIDATION_INTERVAL", "1"))

MIN_STD_DEFAULT = 1e-8
TARGET_DU_MIN_STD = (5e-4, 5e-4)
TARGET_DP_MIN_STD = 1e-4
CELL_VOLUME_MIN_STD = 1e-11
