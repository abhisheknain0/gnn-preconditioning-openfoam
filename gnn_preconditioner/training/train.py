#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training import config
from gnn_preconditioner.training.dataset import load_split_keys, make_dataset
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer

REQUIRED_STATS_KEYS = (
    Normalizer.KEY_U_XY,
    Normalizer.KEY_P_GAUGE,
    Normalizer.KEY_CELL_VOLUME,
    Normalizer.KEY_UINLET,
    Normalizer.KEY_POS,
    Normalizer.KEY_EDGE,
    Normalizer.KEY_TARGET_DU,
    Normalizer.KEY_TARGET_DP,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=config.DATASET_PATH)
    parser.add_argument("--split-path", type=Path, default=config.SPLIT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--lr-min", type=float, default=config.LR_MIN)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=config.NOISE_STD,
        help=(
            "Dimensionless fraction of each feature's natural std, applied as Gaussian "
            "noise to u_xy (per-component) and p_gauge on internal nodes during training. "
            "Target increments are shifted by the same draw (MGN-style compensation)."
        ),
    )
    parser.add_argument(
        "--loss-mask-mode",
        choices=["all", "fluid_only", "fluid_outlet"],
        default=config.LOSS_MASK_MODE,
    )
    parser.add_argument(
        "--temporal-weighting",
        action=argparse.BooleanOptionalAction,
        default=config.TEMPORAL_WEIGHTING,
        help="Up-weight early transitions with 1/sqrt(frame_idx+1), normalized to mean 1.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed Python/NumPy/Torch and DataLoader shuffling")
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=config.VALIDATION_INTERVAL,
        help="Run validation every N epochs and always on the final epoch",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Stop after this completed epoch while keeping the full schedule set by --epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <checkpoint-dir>/last_model.pt",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a specific checkpoint path instead of <checkpoint-dir>/last_model.pt",
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument(
        "--in-memory",
        action=argparse.BooleanOptionalAction,
        default=config.IN_MEMORY,
        help="Load split trajectories into RAM before training",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=config.PERSISTENT_WORKERS,
        help="Keep DataLoader workers alive across epochs",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=config.PREFETCH_FACTOR,
        help="DataLoader prefetch factor per worker (only when num_workers > 0)",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--representation",
        choices=["auto", "surface_prism", "volume"],
        default=config.REPRESENTATION,
        help="Dataset representation to use. `auto` detects from the HDF5 schema.",
    )
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-tf32", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--no-fused-adam", action="store_true", default=False)
    parser.add_argument(
        "--h100-fast",
        action="store_true",
        default=False,
        help="Apply conservative H100 speed defaults (bf16, tf32, compile, fused Adam, RAM loading, worker tuning)",
    )
    parser.add_argument("--recompute-stats", action="store_true")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _load_checkpoint_model_state(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> None:
    target_model = getattr(model, "_orig_mod", model)
    target_model.load_state_dict(checkpoint_state)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    random_state = state.get("python")
    if random_state is not None:
        random.setstate(random_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _state_metrics(loss: torch.Tensor) -> dict[str, float]:
    value = float(loss.item())
    return {"loss": value, "state": value}


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(sum(m[key] for m in metrics) / len(metrics)) for key in keys}


def _build_loss_mask(boundary_mask: torch.Tensor, mask_mode: str) -> torch.Tensor:
    internal = boundary_mask.sum(dim=-1) == 0
    if mask_mode == "fluid_only":
        return internal
    if mask_mode == "fluid_outlet":
        outlet = boundary_mask[:, 1] > 0.5
        return internal | outlet
    if mask_mode == "all":
        return torch.ones_like(internal, dtype=torch.bool)
    raise ValueError(f"Unknown loss mask mode: {mask_mode!r}")


def _compute_temporal_weights(frame_idx: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    per_graph = 1.0 / torch.sqrt(frame_idx.float() + 1.0)
    per_node = per_graph[batch_idx]
    return per_node / (per_node.mean() + 1e-8)


def _masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_mask: torch.Tensor,
    mask_mode: str,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = _build_loss_mask(boundary_mask, mask_mode).to(dtype=pred.dtype).unsqueeze(-1)
    sq_err = (pred - target).pow(2)
    if sample_weight is not None:
        w = sample_weight.to(dtype=pred.dtype).unsqueeze(-1)
        weighted = sq_err * mask * w
        denom = (mask * w).sum()
    else:
        weighted = sq_err * mask
        denom = mask.sum()
    if float(denom.item()) <= 0.0:
        raise RuntimeError(f"Loss mask {mask_mode!r} selected zero nodes in batch")
    return weighted.sum(dim=0).mean() / denom


def evaluate(
    model: MeshGraphNet,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_torch_dtype: torch.dtype,
    loss_mask_mode: str,
    temporal_weighting: bool,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=amp_torch_dtype)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                pred = model(batch.x, batch.edge_attr, batch.edge_index)
                sample_weight = (
                    _compute_temporal_weights(batch.frame_idx, batch.batch)
                    if temporal_weighting else None
                )
                loss = _masked_mse_loss(
                    pred,
                    batch.y,
                    batch.boundary_mask,
                    loss_mask_mode,
                    sample_weight=sample_weight,
                )
            metrics.append(_state_metrics(loss))
    return _mean_metrics(metrics) if metrics else {"loss": float("nan"), "state": float("nan")}


def main() -> None:
    args = _parse_args()
    if args.validation_interval <= 0:
        raise ValueError(f"--validation-interval must be > 0, got {args.validation_interval}")
    if args.stop_after_epoch is not None and args.stop_after_epoch <= 0:
        raise ValueError(f"--stop-after-epoch must be > 0, got {args.stop_after_epoch}")
    if args.h100_fast:
        args.in_memory = True
        args.no_amp = False
        args.amp_dtype = "bf16"
        args.no_tf32 = False
        args.no_compile = False
        args.no_fused_adam = False
        args.persistent_workers = True
        if args.num_workers < 4:
            args.num_workers = 4
        if args.prefetch_factor < 2:
            args.prefetch_factor = 2

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_path = checkpoint_dir / "stats.json"
    resume_path = None
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser().resolve()
    elif args.resume:
        resume_path = checkpoint_dir / "last_model.pt"
    resume_ckpt = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")

    if args.seed is not None:
        _set_seed(args.seed)
    final_epoch = args.epochs if args.stop_after_epoch is None else min(args.stop_after_epoch, args.epochs)

    split_keys_all = load_split_keys(args.split_path)
    if "train" not in split_keys_all:
        raise KeyError(f"Split 'train' not found in {args.split_path}")
    train_keys = list(split_keys_all["train"])
    if not train_keys:
        raise ValueError(f"Train split is empty in {args.split_path}")

    normalizer = Normalizer()
    if args.recompute_stats or not stats_path.exists():
        normalizer.compute_stats(
            args.dataset_path,
            trajectory_keys=train_keys,
            required_keys=list(REQUIRED_STATS_KEYS),
        )
        normalizer.save(stats_path)
    else:
        normalizer.load(stats_path)
        if not normalizer.has_keys(*REQUIRED_STATS_KEYS):
            print("Stats file is missing required keys; recomputing stats.")
            normalizer.compute_stats(
                args.dataset_path,
                trajectory_keys=train_keys,
                required_keys=list(REQUIRED_STATS_KEYS),
            )
            normalizer.save(stats_path)

    train_ds = make_dataset(
        args.dataset_path,
        args.split_path,
        normalizer,
        split="train",
        in_memory=args.in_memory,
        representation=args.representation,
        noise_std=args.noise_std,
    )
    valid_ds = make_dataset(
        args.dataset_path,
        args.split_path,
        normalizer,
        split="valid",
        in_memory=args.in_memory,
        representation=args.representation,
    )

    device = _resolve_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    valid_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    loader_generator = None
    if args.seed is not None or resume_ckpt is not None:
        loader_generator = torch.Generator()
        loader_seed = int(args.seed) if args.seed is not None else torch.initial_seed()
        loader_generator.manual_seed(loader_seed)
        train_loader_kwargs["generator"] = loader_generator
        if args.num_workers > 0:
            train_loader_kwargs["worker_init_fn"] = _worker_init_fn
            valid_loader_kwargs["worker_init_fn"] = _worker_init_fn
    if args.num_workers > 0:
        if args.prefetch_factor <= 0:
            raise ValueError(f"--prefetch-factor must be > 0 when using workers, got {args.prefetch_factor}")
        train_loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        valid_loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        train_loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        valid_loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    valid_loader = DataLoader(valid_ds, **valid_loader_kwargs)

    amp_enabled = bool(not args.no_amp and device.type == "cuda")
    amp_torch_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if amp_enabled and amp_torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Warning: BF16 not supported on this GPU. Falling back to FP16 AMP.")
        amp_torch_dtype = torch.float16
    if device.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = MeshGraphNet(
        input_dim_nodes=int(train_ds.node_input_dim),
        input_dim_edges=int(train_ds.edge_input_dim),
        hidden_dim=config.HIDDEN_DIM,
        num_mp_steps=config.MESSAGE_PASSING_STEPS,
    ).to(device)
    compile_active = False
    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            compile_active = True
        except Exception as exc:
            print(f"Warning: torch.compile failed, using eager mode. Reason: {exc}")

    optimizer = None
    if not args.no_fused_adam and device.type == "cuda":
        try:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=config.WEIGHT_DECAY,
                fused=True,
            )
        except TypeError:
            optimizer = None
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_torch_dtype == torch.float16)
    history = []
    best_valid = None
    best_path = checkpoint_dir / "best_model.pt"
    start_epoch = 1
    resumed_from_epoch = None
    if resume_ckpt is not None:
        _load_checkpoint_model_state(model, resume_ckpt["model_state"])
        optimizer_state = resume_ckpt.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = resume_ckpt.get("scheduler_state")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            if hasattr(scheduler, "T_max"):
                scheduler.T_max = args.epochs
        scaler_state = resume_ckpt.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        history = list(resume_ckpt.get("history", []))
        best_valid = resume_ckpt.get("best_valid")
        resumed_from_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = resumed_from_epoch + 1
        if start_epoch > final_epoch:
            raise ValueError(
                f"Resume checkpoint is already at epoch {resumed_from_epoch}, "
                f"but this run stops at epoch {final_epoch}. Increase --stop-after-epoch or --epochs to continue."
            )
        if loader_generator is not None and resume_ckpt.get("loader_generator_state") is not None:
            loader_generator.set_state(resume_ckpt["loader_generator_state"])
        _restore_rng_state(resume_ckpt.get("rng_state"))

    print("Training setup:")
    print(f"  dataset:            {args.dataset_path}")
    print(f"  checkpoint:         {checkpoint_dir}")
    print(f"  device:             {device}")
    print(f"  representation:     {train_ds.representation}")
    print(f"  training_mode:      state-only")
    print(f"  hidden_dim:         {config.HIDDEN_DIM}")
    print(f"  message_passing:    {config.MESSAGE_PASSING_STEPS}")
    print(f"  surface_edge_features: {config.SURFACE_EDGE_FEATURE_MODE}")
    print(f"  active_losses:      state")
    print(f"  loss_mask_mode:     {args.loss_mask_mode}")
    print(f"  temporal_weighting: {args.temporal_weighting}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  in_memory:          {args.in_memory}")
    print(f"  num_workers:        {args.num_workers}")
    if args.num_workers > 0:
        print(f"  persistent_workers: {args.persistent_workers}")
        print(f"  prefetch_factor:    {args.prefetch_factor}")
    print(f"  AMP:                {amp_enabled} ({amp_torch_dtype})")
    print(f"  TF32:               {device.type == 'cuda' and not args.no_tf32}")
    print(f"  compile:            {compile_active}")
    print(f"  fused_adam:         {device.type == 'cuda' and not args.no_fused_adam}")
    print(f"  h100_fast:          {args.h100_fast}")
    print(f"  validation_interval:{args.validation_interval}")
    print(f"  total_epochs:       {args.epochs}")
    print(f"  stop_after_epoch:   {final_epoch}")
    if args.seed is not None:
        print(f"  seed:               {args.seed}")
    if resume_path is not None:
        print(f"  resume_from:        {resume_path}")
        print(f"  start_epoch:        {start_epoch}")

    for epoch in range(start_epoch, final_epoch + 1):
        model.train()
        step_metrics: list[dict[str, float]] = []
        epoch_start = time.time()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=amp_torch_dtype)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                pred = model(batch.x, batch.edge_attr, batch.edge_index)
                sample_weight = (
                    _compute_temporal_weights(batch.frame_idx, batch.batch)
                    if args.temporal_weighting else None
                )
                loss = _masked_mse_loss(
                    pred,
                    batch.y,
                    batch.boundary_mask,
                    args.loss_mask_mode,
                    sample_weight=sample_weight,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            step_metrics.append(_state_metrics(loss))

        scheduler.step()
        train_metrics = _mean_metrics(step_metrics)
        should_validate = (epoch % args.validation_interval == 0) or (epoch == final_epoch)
        valid_metrics = (
            evaluate(
                model,
                valid_loader,
                device,
                amp_enabled,
                amp_torch_dtype,
                args.loss_mask_mode,
                args.temporal_weighting,
            )
            if should_validate
            else None
        )
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "valid": valid_metrics,
            "epoch_wall_s": float(time.time() - epoch_start),
        }
        history.append(record)
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if valid_metrics is not None and (best_valid is None or valid_metrics["loss"] < best_valid):
            best_valid = valid_metrics["loss"]
            torch.save(
                {
                    "model_state": _checkpoint_model_state(model),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "rng_state": _capture_rng_state(),
                    "loader_generator_state": loader_generator.get_state() if loader_generator is not None else None,
                    "epoch": epoch,
                    "planned_epochs": args.epochs,
                    "stop_after_epoch": final_epoch,
                    "history": history,
                    "best_valid": best_valid,
                },
                best_path,
            )
        torch.save(
            {
                "model_state": _checkpoint_model_state(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": _capture_rng_state(),
                "loader_generator_state": loader_generator.get_state() if loader_generator is not None else None,
                "epoch": epoch,
                "planned_epochs": args.epochs,
                "stop_after_epoch": final_epoch,
                "history": history,
                "best_valid": best_valid,
            },
            checkpoint_dir / "last_model.pt",
        )
        print(json.dumps(record, indent=2))

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "best_model": str(best_path),
        "best_valid_loss": best_valid,
        "epochs": args.epochs,
        "stop_after_epoch": final_epoch,
        "representation": str(train_ds.representation),
        "training_mode": "state-only",
        "hidden_dim": int(config.HIDDEN_DIM),
        "message_passing_steps": int(config.MESSAGE_PASSING_STEPS),
        "surface_edge_feature_mode": str(config.SURFACE_EDGE_FEATURE_MODE),
        "edge_input_dim": int(train_ds.edge_input_dim),
        "noise_std": float(args.noise_std),
        "noise_semantics": "fraction_of_feature_std" if args.noise_std > 0 else "off",
        "stats_scope": "train_split_only",
        "loss_mask_mode": str(args.loss_mask_mode),
        "temporal_weighting": bool(args.temporal_weighting),
        "active_losses": ["state"],
        "batch_size": args.batch_size,
        "in_memory": bool(args.in_memory),
        "num_workers": int(args.num_workers),
        "persistent_workers": bool(args.persistent_workers) if args.num_workers > 0 else False,
        "prefetch_factor": int(args.prefetch_factor) if args.num_workers > 0 else None,
        "amp": bool(amp_enabled),
        "amp_dtype": str(amp_torch_dtype),
        "tf32": bool(device.type == "cuda" and not args.no_tf32),
        "compile": bool(compile_active),
        "fused_adam": bool(device.type == "cuda" and not args.no_fused_adam),
        "h100_fast": bool(args.h100_fast),
        "validation_interval": int(args.validation_interval),
        "seed": int(args.seed) if args.seed is not None else None,
        "resume_from": str(resume_path) if resume_path is not None else None,
        "resumed_from_epoch": resumed_from_epoch,
    }
    (checkpoint_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
