#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training.normalizer import Normalizer


class RandomGuessInferenceModule(nn.Module):
    def __init__(
        self,
        normalizer: Normalizer,
        random_scale: float = 1.0,
        seed: int = 1234,
        apply_default_boundary_constraints: bool = False,
    ):
        super().__init__()
        self.random_scale = float(random_scale)
        self.seed = int(seed)
        self.apply_default_boundary_constraints = bool(apply_default_boundary_constraints)
        self.register_buffer(
            "du_std",
            torch.tensor(normalizer.stats[Normalizer.KEY_TARGET_DU]["std"], dtype=torch.float32),
        )
        self.register_buffer(
            "dp_std",
            torch.tensor(normalizer.stats[Normalizer.KEY_TARGET_DP]["std"], dtype=torch.float32),
        )

    def forward(
        self,
        u_xy_t: torch.Tensor,
        p_gauge_t: torch.Tensor,
        u_inlet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del u_inlet
        du = (
            torch.randn_like(u_xy_t)
            * self.du_std
            * self.random_scale
        )
        dp = (
            torch.randn_like(p_gauge_t)
            * self.dp_std
            * self.random_scale
        )
        return u_xy_t + du, p_gauge_t + dp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--random-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--apply-default-boundary-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    normalizer = Normalizer()
    normalizer.load(args.stats_path)
    wrapper = RandomGuessInferenceModule(
        normalizer=normalizer,
        random_scale=args.random_scale,
        seed=args.seed,
        apply_default_boundary_constraints=bool(args.apply_default_boundary_constraints),
    ).to(device)
    wrapper.eval()
    scripted = torch.jit.script(wrapper)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output_path))
    meta = {
        "output_path": str(args.output_path.resolve()),
        "device": str(device),
        "random_scale": float(args.random_scale),
        "seed": int(args.seed),
        "mode": "random",
        "apply_default_boundary_constraints": bool(args.apply_default_boundary_constraints),
    }
    args.output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
