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


class SyntheticGuessInferenceModule(nn.Module):
    def __init__(self, mode: str = "zero"):
        super().__init__()
        self.mode = str(mode)

    def forward(
        self,
        u_xy_t: torch.Tensor,
        p_gauge_t: torch.Tensor,
        u_inlet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del u_inlet
        if self.mode == "zero":
            return torch.zeros_like(u_xy_t), torch.zeros_like(p_gauge_t)
        if self.mode == "reverse":
            return -u_xy_t, -p_gauge_t
        raise RuntimeError(f"unsupported synthetic mode: {self.mode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", choices=["zero", "reverse"], required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    wrapper = SyntheticGuessInferenceModule(mode=args.mode).to(device)
    wrapper.eval()
    scripted = torch.jit.script(wrapper)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output_path))
    meta = {
        "output_path": str(args.output_path.resolve()),
        "device": str(device),
        "mode": str(args.mode),
    }
    args.output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
