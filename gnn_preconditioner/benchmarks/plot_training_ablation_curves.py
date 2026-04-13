#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = PROJECT_ROOT / "gnn_preconditioner" / "checkpoints"
OUT_DIR = PROJECT_ROOT / "gnn_preconditioner" / "benchmarks" / "figures"


RUNS = [
    {
        "name": "solver_96x8",
        "label": "Solver Loss, H96, MP8",
        "path": CHECKPOINTS / "pilot_surface_coarse_v1_run1" / "history.json",
        "group": "all",
    },
    {
        "name": "state_96x8",
        "label": "State Only, H96, MP8",
        "path": CHECKPOINTS / "pilot_surface_coarse_v1_stateonly_run1" / "history.json",
        "group": "state_only",
    },
    {
        "name": "state_96x12",
        "label": "State Only, H96, MP12",
        "path": CHECKPOINTS / "pilot_surface_coarse_v1_stateonly_mp12_run1" / "history.json",
        "group": "state_only",
    },
    {
        "name": "state_128x8",
        "label": "State Only, H128, MP8",
        "path": CHECKPOINTS / "pilot_surface_coarse_v1_stateonly_hd128_run1" / "history.json",
        "group": "state_only",
    },
    {
        "name": "state_128x12",
        "label": "State Only, H128, MP12",
        "path": CHECKPOINTS / "pilot_surface_coarse_v1_stateonly_hd128_mp12_run1" / "history.json",
        "group": "state_only",
    },
]


def _load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _epochs(hist: list[dict]) -> list[int]:
    return [row["epoch"] for row in hist]


def _metric(hist: list[dict], split: str, key: str) -> list[float]:
    return [float(row[split][key]) for row in hist]


def _wall(hist: list[dict]) -> list[float]:
    return [float(row["epoch_wall_s"]) for row in hist]


def _best_epoch(hist: list[dict], key: str = "loss") -> int:
    best = min(hist, key=lambda row: float(row["valid"][key]))
    return int(best["epoch"])


def _plot_curves(run_specs: list[dict], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_train, ax_valid = axes[0]
    ax_gradp, ax_mass = axes[1]

    for spec in run_specs:
        hist = _load_history(spec["path"])
        x = _epochs(hist)
        label = spec["label"]
        ax_train.plot(x, _metric(hist, "train", "state"), label=label, linewidth=2)
        ax_valid.plot(x, _metric(hist, "valid", "state"), label=label, linewidth=2)
        ax_gradp.plot(x, _metric(hist, "valid", "gradp_raw"), label=label, linewidth=2)
        ax_mass.plot(x, _metric(hist, "valid", "mass_raw"), label=label, linewidth=2)

        best_epoch = _best_epoch(hist, key="loss")
        ax_valid.axvline(best_epoch, color=ax_valid.lines[-1].get_color(), linestyle="--", alpha=0.25)

    ax_train.set_title("Train State Loss")
    ax_valid.set_title("Valid State Loss")
    ax_gradp.set_title("Valid gradp_raw")
    ax_mass.set_title("Valid mass_raw")

    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)

    ax_train.set_ylabel("MSE")
    ax_valid.set_ylabel("MSE")
    ax_gradp.set_ylabel("Raw MSE")
    ax_mass.set_ylabel("Raw MSE")
    ax_gradp.set_yscale("log")
    ax_mass.set_yscale("log")

    handles, labels = ax_valid.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_epoch_time(run_specs: list[dict], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for spec in run_specs:
        hist = _load_history(spec["path"])
        ax.plot(_epochs(hist), _wall(hist), label=spec["label"], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch Wall Time [s]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    available = [spec for spec in RUNS if spec["path"].exists()]
    if not available:
        raise SystemExit("No matching history.json files found.")

    state_only = [spec for spec in available if spec["group"] == "state_only"]
    all_runs = available

    _plot_curves(
        all_runs,
        OUT_DIR / "coarse_surface_all_models_curves.png",
        "Coarse Surface Models: Training and Solver-Facing Curves",
    )
    _plot_curves(
        state_only,
        OUT_DIR / "coarse_surface_stateonly_architecture_curves.png",
        "Coarse Surface State-Only Architecture Ablations",
    )
    _plot_epoch_time(
        state_only,
        OUT_DIR / "coarse_surface_stateonly_epoch_time.png",
        "Coarse Surface State-Only Epoch Time",
    )
    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "runs_plotted": [spec["name"] for spec in available],
    }, indent=2))


if __name__ == "__main__":
    main()
