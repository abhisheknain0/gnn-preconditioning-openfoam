#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Colormap
from matplotlib.path import Path as MplPath
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn_preconditioner.training import config
from gnn_preconditioner.training.boundary_constraints import apply_default_boundary_constraints as apply_boundary_constraints
from gnn_preconditioner.training.dataset import load_static_graph_from_group
from gnn_preconditioner.training.model import MeshGraphNet
from gnn_preconditioner.training.normalizer import Normalizer


_POINT_RE = re.compile(
    r"Point\((\d+)\)\s*=\s*\{([^,]+),\s*([^,]+),\s*[^,]+,\s*[^}]+\};"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--trajectory-key", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid-width", type=int, default=760)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--apply-default-boundary-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy mode: hold velocity on inlet/wall and pressure on outlet during rollout rendering.",
    )
    return parser.parse_args()


def _resolve_checkpoint_config(checkpoint_path: Path) -> tuple[int, int]:
    hidden_dim = int(config.HIDDEN_DIM)
    message_passing_steps = int(config.MESSAGE_PASSING_STEPS)
    summary_path = checkpoint_path.parent / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        hidden_dim = int(summary.get("hidden_dim", hidden_dim))
        message_passing_steps = int(summary.get("message_passing_steps", message_passing_steps))
    return hidden_dim, message_passing_steps


def _load_bfs_points(geo_path: Path) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for line in geo_path.read_text(encoding="utf-8").splitlines():
        match = _POINT_RE.match(line.strip())
        if match is None:
            continue
        pid = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        points[pid] = (x, y)
    required = {1, 2, 4, 5, 6, 8}
    missing = required.difference(points)
    if missing:
        raise ValueError(f"{geo_path} missing required BFS points: {sorted(missing)}")
    return points


def _bfs_polygon_and_segments(geo_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pts = _load_bfs_points(geo_path)
    polygon = np.asarray([pts[1], pts[2], pts[5], pts[6], pts[8], pts[4]], dtype=np.float32)
    segments = {
        "inlet": np.asarray([pts[1], pts[4]], dtype=np.float32),
        "outlet": np.asarray([pts[6], pts[8]], dtype=np.float32),
        "wall_top": np.asarray([pts[4], pts[8]], dtype=np.float32),
        "wall_lower_inlet": np.asarray([pts[1], pts[2]], dtype=np.float32),
        "wall_step_vertical": np.asarray([pts[2], pts[5]], dtype=np.float32),
        "wall_floor": np.asarray([pts[5], pts[6]], dtype=np.float32),
    }
    return polygon, segments


def _copy_cmap(name: str) -> Colormap:
    cmap = plt.get_cmap(name)
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color="white")
    return cmap


def _robust_limits(a: np.ndarray, b: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    stacked = np.concatenate([a.reshape(-1), b.reshape(-1)], axis=0)
    vmin = float(np.percentile(stacked, lo))
    vmax = float(np.percentile(stacked, hi))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax


def _masked_interpolator(
    pos_xy: np.ndarray,
    polygon: np.ndarray,
    grid_width: int,
) -> tuple[callable, tuple[float, float, float, float], np.ndarray, np.ndarray]:
    x = pos_xy[:, 0]
    y = pos_xy[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = max(1e-8, x_max - x_min)
    y_range = max(1e-8, y_max - y_min)
    grid_height = max(80, int(grid_width * (y_range / x_range)))
    gx = np.linspace(x_min, x_max, grid_width, dtype=np.float32)
    gy = np.linspace(y_min, y_max, grid_height, dtype=np.float32)
    gxx, gyy = np.meshgrid(gx, gy)
    mask = MplPath(polygon).contains_points(
        np.column_stack([gxx.reshape(-1), gyy.reshape(-1)]),
        radius=1e-9,
    ).reshape(gxx.shape)
    triangulation = mtri.Triangulation(x, y)

    def _interp(values: np.ndarray) -> np.ndarray:
        interp = mtri.LinearTriInterpolator(triangulation, values.astype(np.float32))
        z = interp(gxx, gyy)
        if np.ma.isMaskedArray(z):
            z = z.filled(np.nan)
        z = np.asarray(z, dtype=np.float32)
        z[~mask] = np.nan
        return z

    return _interp, (x_min, x_max, y_min, y_max), gxx, gyy


def _rollout_prediction(
    model: MeshGraphNet,
    normalizer: Normalizer,
    node_positions: torch.Tensor,
    cell_volumes: torch.Tensor,
    boundary_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    velocity_xy: torch.Tensor,
    pressure_gauge: torch.Tensor,
    uinlet: float,
    rollout_steps: int,
    apply_default_boundary_constraints: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_node_feat = torch.cat([
        boundary_mask,
        normalizer.normalize(cell_volumes, Normalizer.KEY_CELL_VOLUME),
        normalizer.normalize(
            torch.full((node_positions.shape[0], 1), uinlet, dtype=torch.float32, device=node_positions.device),
            Normalizer.KEY_UINLET,
        ),
        normalizer.normalize(node_positions, Normalizer.KEY_POS),
    ], dim=-1)

    u_xy = velocity_xy[0]
    p_g = pressure_gauge[0].unsqueeze(-1)
    pred_vel = [u_xy.detach().cpu()]
    pred_pres = [p_g.detach().cpu().squeeze(-1)]
    max_step = min(rollout_steps, velocity_xy.shape[0] - 1)
    with torch.no_grad():
        for _ in range(max_step):
            x = torch.cat([
                normalizer.normalize(u_xy, Normalizer.KEY_U_XY),
                normalizer.normalize(p_g, Normalizer.KEY_P_GAUGE),
                static_node_feat,
            ], dim=-1)
            pred = model(x, edge_attr, edge_index)
            du = normalizer.denormalize(pred[:, :2], Normalizer.KEY_TARGET_DU)
            dp = normalizer.denormalize(pred[:, 2:3], Normalizer.KEY_TARGET_DP)
            u_next = u_xy + du
            p_next = p_g + dp
            if apply_default_boundary_constraints:
                u_xy, p_g = apply_boundary_constraints(
                    u_next=u_next,
                    p_next=p_next,
                    u_current=u_xy,
                    p_current=p_g,
                    boundary_mask=boundary_mask,
                )
            else:
                u_xy, p_g = u_next, p_next
            pred_vel.append(u_xy.detach().cpu())
            pred_pres.append(p_g.detach().cpu().squeeze(-1))
    return torch.stack(pred_vel, dim=0), torch.stack(pred_pres, dim=0)


def _save_boundary_assignment_plot(
    pos_xy: np.ndarray,
    polygon: np.ndarray,
    segments: dict[str, np.ndarray],
    boundary_mask: np.ndarray,
    output_path: Path,
    trajectory_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
    ax.scatter(pos_xy[:, 0], pos_xy[:, 1], s=4, c="#d9d9d9", alpha=0.5, label="all cells")

    styles = [
        ("inlet", 0, "#d62728"),
        ("outlet", 1, "#1f77b4"),
        ("wall", 2, "#2ca02c"),
    ]
    for label, col, color in styles:
        pts = pos_xy[boundary_mask[:, col] > 0]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, alpha=0.9, label=f"{label} cells ({len(pts)})")

    ax.plot(*polygon.T[[0, 1]], alpha=0)  # ensure axes get data limits from polygon
    polygon_closed = np.vstack([polygon, polygon[0]])
    ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], c="black", lw=1.0, alpha=0.6)
    ax.plot(segments["inlet"][:, 0], segments["inlet"][:, 1], c="#d62728", lw=2.5, label="inlet boundary")
    ax.plot(segments["outlet"][:, 0], segments["outlet"][:, 1], c="#1f77b4", lw=2.5, label="outlet boundary")
    for key in ("wall_top", "wall_lower_inlet", "wall_step_vertical", "wall_floor"):
        ax.plot(segments[key][:, 0], segments[key][:, 1], c="#2ca02c", lw=2.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Boundary Assignment Check: {trajectory_key}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_masked_rollout_gif(
    pos_xy: np.ndarray,
    polygon: np.ndarray,
    segments: dict[str, np.ndarray],
    gt_velocity: np.ndarray,
    pred_velocity: np.ndarray,
    gt_pressure: np.ndarray,
    pred_pressure: np.ndarray,
    output_path: Path,
    title_prefix: str,
    fps: int,
    dpi: int,
    grid_width: int,
) -> None:
    interp, extent_raw, _, _ = _masked_interpolator(pos_xy, polygon, grid_width)
    x_min, x_max, y_min, y_max = extent_raw
    extent = (x_min, x_max, y_min, y_max)

    gt_vmag = np.linalg.norm(gt_velocity, axis=-1)
    pred_vmag = np.linalg.norm(pred_velocity, axis=-1)
    err_vmag = pred_vmag - gt_vmag
    err_p = pred_pressure - gt_pressure

    vel_min, vel_max = _robust_limits(gt_vmag, pred_vmag)
    p_min, p_max = _robust_limits(gt_pressure, pred_pressure)
    vel_err_abs = max(float(np.percentile(np.abs(err_vmag.reshape(-1)), 99.0)), 1e-12)
    p_err_abs = max(float(np.percentile(np.abs(err_p.reshape(-1)), 99.0)), 1e-12)

    vel_cmap = _copy_cmap("viridis")
    pres_cmap = _copy_cmap("viridis")
    err_cmap = _copy_cmap("RdBu_r")

    fig, axes = plt.subplots(2, 3, figsize=(16, 4.6), constrained_layout=True)
    plot_axes = list(axes.reshape(-1))
    for ax in plot_axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        poly_closed = np.vstack([polygon, polygon[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], c="black", lw=0.8, alpha=0.7)
        for key in ("wall_top", "wall_lower_inlet", "wall_step_vertical", "wall_floor"):
            ax.plot(segments[key][:, 0], segments[key][:, 1], c="black", lw=0.8, alpha=0.7)

    axes[0, 0].set_title("Velocity |U| (Ground Truth)")
    axes[0, 1].set_title("Velocity |U| (Prediction)")
    axes[0, 2].set_title("Velocity Error (Pred - GT)")
    axes[1, 0].set_title("Pressure (Ground Truth)")
    axes[1, 1].set_title("Pressure (Prediction)")
    axes[1, 2].set_title("Pressure Error (Pred - GT)")

    def _field_frame(t: int) -> tuple[np.ndarray, ...]:
        return (
            interp(gt_vmag[t]),
            interp(pred_vmag[t]),
            interp(err_vmag[t]),
            interp(gt_pressure[t]),
            interp(pred_pressure[t]),
            interp(err_p[t]),
        )

    frame0 = _field_frame(0)
    artists = [
        axes[0, 0].imshow(frame0[0], origin="lower", extent=extent, cmap=vel_cmap, vmin=vel_min, vmax=vel_max, interpolation="bilinear", aspect="equal"),
        axes[0, 1].imshow(frame0[1], origin="lower", extent=extent, cmap=vel_cmap, vmin=vel_min, vmax=vel_max, interpolation="bilinear", aspect="equal"),
        axes[0, 2].imshow(frame0[2], origin="lower", extent=extent, cmap=err_cmap, vmin=-vel_err_abs, vmax=vel_err_abs, interpolation="bilinear", aspect="equal"),
        axes[1, 0].imshow(frame0[3], origin="lower", extent=extent, cmap=pres_cmap, vmin=p_min, vmax=p_max, interpolation="bilinear", aspect="equal"),
        axes[1, 1].imshow(frame0[4], origin="lower", extent=extent, cmap=pres_cmap, vmin=p_min, vmax=p_max, interpolation="bilinear", aspect="equal"),
        axes[1, 2].imshow(frame0[5], origin="lower", extent=extent, cmap=err_cmap, vmin=-p_err_abs, vmax=p_err_abs, interpolation="bilinear", aspect="equal"),
    ]
    fig.colorbar(artists[0], ax=axes[0, :2], shrink=0.85)
    fig.colorbar(artists[2], ax=axes[0, 2], shrink=0.85)
    fig.colorbar(artists[3], ax=axes[1, :2], shrink=0.85)
    fig.colorbar(artists[5], ax=axes[1, 2], shrink=0.85)

    vel_rmse_t = np.sqrt(np.mean((pred_velocity - gt_velocity) ** 2, axis=(1, 2)))
    p_rmse_t = np.sqrt(np.mean((pred_pressure - gt_pressure) ** 2, axis=1))
    title = fig.suptitle("", fontsize=12)

    def _update(frame_idx: int):
        current = _field_frame(frame_idx)
        for artist, data in zip(artists, current):
            artist.set_data(data)
        title.set_text(
            f"{title_prefix} | frame {frame_idx}/{gt_velocity.shape[0]-1} | "
            f"vel_rmse={vel_rmse_t[frame_idx]:.3e} | p_rmse={p_rmse_t[frame_idx]:.3e}"
        )
        return artists + [title]

    anim = FuncAnimation(fig, _update, frames=range(gt_velocity.shape[0]), interval=max(1, int(1000 / fps)), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def render_diagnostics(
    checkpoint_path: Path,
    stats_path: Path,
    dataset_path: Path,
    trajectory_key: str,
    output_dir: Path,
    rollout_steps: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    grid_width: int = 760,
    fps: int = 10,
    dpi: int = 120,
    apply_default_boundary_constraints: bool = False,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    stats_path = stats_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    normalizer = Normalizer()
    normalizer.load(stats_path)
    hidden_dim, message_passing_steps = _resolve_checkpoint_config(checkpoint_path)

    with h5py.File(dataset_path, "r") as f:
        g = f[trajectory_key]
        source_case_path = Path(str(g.attrs["source_case_path"]))
        if not source_case_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            parts = source_case_path.parts
            if "cases" in parts:
                idx = parts.index("cases")
                candidate = repo_root / "gnn_preconditioner" / "data_generation" / Path(*parts[idx:])
                if candidate.exists():
                    source_case_path = candidate
        geo_path = source_case_path / "system" / "geometry.geo"
        polygon, segments = _bfs_polygon_and_segments(geo_path)
        static = load_static_graph_from_group(g, normalizer, device=device_obj)
        velocity_xy = torch.tensor(g["velocity_xy"][()], dtype=torch.float32, device=device_obj)
        pressure_gauge = torch.tensor(g["pressure_gauge"][()], dtype=torch.float32, device=device_obj)
        uinlet = float(g["Uinlet"][()][0])
    model = MeshGraphNet(
        input_dim_nodes=int(static["node_input_dim"]),
        input_dim_edges=int(static["edge_input_dim"]),
        hidden_dim=hidden_dim,
        num_mp_steps=message_passing_steps,
    ).to(device_obj)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_vel, pred_pres = _rollout_prediction(
        model=model,
        normalizer=normalizer,
        node_positions=static["positions"],
        cell_volumes=static["cell_volumes"],
        boundary_mask=static["boundary_mask"],
        edge_index=static["edge_index"],
        edge_attr=static["edge_attr"],
        velocity_xy=velocity_xy,
        pressure_gauge=pressure_gauge,
        uinlet=uinlet,
        rollout_steps=rollout_steps,
        apply_default_boundary_constraints=apply_default_boundary_constraints,
    )

    max_step = pred_vel.shape[0] - 1
    gt_vel = velocity_xy[: max_step + 1].detach().cpu().numpy()
    gt_pres = pressure_gauge[: max_step + 1].detach().cpu().numpy()
    pred_vel_np = pred_vel.numpy()
    pred_pres_np = pred_pres.numpy()
    pos_xy = static["positions"][:, :2].detach().cpu().numpy()
    boundary_mask_np = static["boundary_mask"].detach().cpu().numpy().astype(np.uint8)

    assignment_path = output_dir / f"boundary_assignment_{trajectory_key}.png"
    gif_path = output_dir / f"rollout_{trajectory_key}_steps{max_step}_masked.gif"
    summary_path = output_dir / f"rollout_{trajectory_key}_steps{max_step}_masked.json"

    _save_boundary_assignment_plot(
        pos_xy=pos_xy,
        polygon=polygon,
        segments=segments,
        boundary_mask=boundary_mask_np,
        output_path=assignment_path,
        trajectory_key=trajectory_key,
    )
    _save_masked_rollout_gif(
        pos_xy=pos_xy,
        polygon=polygon,
        segments=segments,
        gt_velocity=gt_vel,
        pred_velocity=pred_vel_np,
        gt_pressure=gt_pres,
        pred_pressure=pred_pres_np,
        output_path=gif_path,
        title_prefix=f"IterReduce v1 {trajectory_key}",
        fps=fps,
        dpi=dpi,
        grid_width=grid_width,
    )

    summary = {
        "trajectory_key": trajectory_key,
        "rollout_steps": int(max_step),
        "checkpoint_path": str(checkpoint_path),
        "surface_edge_feature_mode": str(config.SURFACE_EDGE_FEATURE_MODE),
        "apply_default_boundary_constraints": bool(apply_default_boundary_constraints),
        "assignment_plot": str(assignment_path),
        "gif_path": str(gif_path),
        "velocity_rmse_mean": float(np.sqrt(np.mean((pred_vel_np[1:] - gt_vel[1:]) ** 2, axis=(1, 2))).mean()),
        "pressure_rmse_mean": float(np.sqrt(np.mean((pred_pres_np[1:] - gt_pres[1:]) ** 2, axis=1)).mean()),
        "boundary_counts": {
            "inlet": int(boundary_mask_np[:, 0].sum()),
            "outlet": int(boundary_mask_np[:, 1].sum()),
            "wall": int(boundary_mask_np[:, 2].sum()),
        },
    }
    if boundary_mask_np.shape[1] > 3:
        summary["boundary_counts"]["empty"] = int(boundary_mask_np[:, 3].sum())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    summary = render_diagnostics(
        checkpoint_path=args.checkpoint_path,
        stats_path=args.stats_path,
        dataset_path=args.dataset_path,
        trajectory_key=args.trajectory_key,
        output_dir=args.output_dir,
        rollout_steps=args.rollout_steps,
        device=args.device,
        grid_width=args.grid_width,
        fps=args.fps,
        dpi=args.dpi,
        apply_default_boundary_constraints=bool(args.apply_default_boundary_constraints),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
