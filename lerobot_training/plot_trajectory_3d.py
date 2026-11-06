"""Render an evaluated robot trajectory in world coordinates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcm_scene_v9 import RCMSceneV9


TASK_TO_STATE = np.array([180.0 / math.pi, 180.0 / math.pi, 1000.0])


def state_to_task(state_abc_mm):
    return np.asarray(state_abc_mm, dtype=np.float64) / TASK_TO_STATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    source = Path(args.evaluation_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    states = np.asarray(episode["trajectory"]["states"], dtype=np.float64)
    errors_reported = np.asarray(episode["trajectory"]["errors"], dtype=np.float64)
    target = np.asarray(episode["target_world_mm"], dtype=np.float64)

    scene = RCMSceneV9()
    points = []
    try:
        for state in states:
            scene.set_task(state_to_task(state))
            points.append(scene.site_mm("needle_tip"))
    finally:
        scene.close()
    points = np.asarray(points, dtype=np.float64)
    errors = np.linalg.norm(points - target[None, :], axis=1)
    final_error_json = float(episode["final_error_mm"])
    final_tip_json = np.asarray(episode["final_tip_world_mm"], dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("Reconstructed trajectory contains non-finite points")
    if np.linalg.norm(points[-1] - final_tip_json) > 1e-3:
        raise ValueError(
            "Reconstructed final tip disagrees with evaluation JSON: "
            f"{points[-1].tolist()} vs {final_tip_json.tolist()}"
        )
    if abs(errors[-1] - final_error_json) > 1e-3:
        raise ValueError(
            f"Final error mismatch: {errors[-1]} vs {final_error_json}"
        )

    steps = np.arange(len(points))
    norm = colors.Normalize(vmin=0, vmax=max(1, len(points) - 1))
    cmap = plt.colormaps["viridis"]
    labels = ("X (mm)", "Y (mm)", "Z (mm)")
    spans = np.ptp(points, axis=0)
    margin = np.maximum(spans * 0.08, 0.25)
    lower = points.min(axis=0) - margin
    upper = points.max(axis=0) + margin
    lower = np.minimum(lower, target - margin)
    upper = np.maximum(upper, target + margin)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(10.0, 7.8), layout="constrained")
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    ax3d.plot(
        points[:, 0], points[:, 1], points[:, 2],
        color="#B0B0B0", linewidth=0.8, alpha=0.8, zorder=1,
    )
    scatter = ax3d.scatter(
        points[::3, 0], points[::3, 1], points[::3, 2],
        c=steps[::3], cmap=cmap, norm=norm, s=5, alpha=0.9, zorder=2,
    )
    ax3d.scatter(*points[0], marker="o", s=70, color="#0072B2",
                 edgecolor="white", linewidth=0.8, depthshade=False,
                 label="Start", zorder=5)
    ax3d.scatter(*points[-1], marker="s", s=65, color="#D55E00",
                 edgecolor="white", linewidth=0.8, depthshade=False,
                 label="Final tip", zorder=5)
    ax3d.scatter(*target, marker="*", s=180, color="#F0E442",
                 edgecolor="#333333", linewidth=0.7, depthshade=False,
                 label="Target", zorder=6)
    ax3d.set(
        xlabel=labels[0],
        ylabel=labels[1],
        zlabel=labels[2],
        title="A. 3D needle-tip trajectory",
    )
    ax3d.set_xlim(lower[0], upper[0])
    ax3d.set_ylim(lower[1], upper[1])
    ax3d.set_zlim(lower[2], upper[2])
    ax3d.set_box_aspect(spans / max(spans.max(), 1e-9))
    ax3d.view_init(elev=24, azim=-58)
    ax3d.grid(True, alpha=0.25)
    ax3d.legend(loc="upper right", frameon=True)

    projections = (
        (ax_xy, 0, 1, "X (mm)", "Y (mm)", "B. XY projection"),
        (ax_xz, 0, 2, "X (mm)", "Z (mm)", "C. XZ projection"),
        (ax_yz, 1, 2, "Y (mm)", "Z (mm)", "D. YZ projection"),
    )
    for axis, i, j, xlabel, ylabel, title in projections:
        axis.plot(points[:, i], points[:, j], color="#707070",
                  linewidth=0.9, alpha=0.8, zorder=1)
        axis.scatter(points[::3, i], points[::3, j], c=steps[::3],
                     cmap=cmap, norm=norm, s=4, alpha=0.9, zorder=2)
        axis.scatter(points[0, i], points[0, j], marker="o", s=55,
                     color="#0072B2", edgecolor="white", zorder=4)
        axis.scatter(points[-1, i], points[-1, j], marker="s", s=55,
                     color="#D55E00", edgecolor="white", zorder=4)
        axis.scatter(target[i], target[j], marker="*", s=150,
                     color="#F0E442", edgecolor="#333333", zorder=5)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axis.set_xlim(lower[i], upper[i])
        axis.set_ylim(lower[j], upper[j])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)

    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax3d, ax_xy, ax_xz, ax_yz],
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_label("Control step (50 ms)")
    fig.suptitle(
        "OCT-conditioned diffusion policy closed-loop trajectory\n"
        f"result={episode['success']}, min error={errors.min():.3f} mm, "
        f"final error={errors[-1]:.3f} mm",
        fontsize=12,
    )

    png_path = output_dir / "trajectory_3d.png"
    pdf_path = output_dir / "trajectory_3d.pdf"
    fig.savefig(png_path, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    csv_path = output_dir / "trajectory_points.csv"
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "step,A_deg,B_deg,C_mm,tip_x_mm,tip_y_mm,tip_z_mm,error_mm\n"
        )
        for step, (state, point, error) in enumerate(
            zip(states, points, errors)
        ):
            stream.write(
                f"{step},{state[0]:.9g},{state[1]:.9g},{state[2]:.9g},"
                f"{point[0]:.9g},{point[1]:.9g},{point[2]:.9g},"
                f"{error:.9g}\n"
            )

    manifest = {
        "source_json": str(source.resolve()),
        "raw_source_unchanged": True,
        "transformation": (
            "Replay recorded A/B/C states through RCMSceneV9 forward "
            "kinematics; C converted from mm to m, A/B from degrees to radians."
        ),
        "sample_count": int(len(points)),
        "control_period_s": 0.05,
        "target_world_mm": target.tolist(),
        "start_tip_world_mm": points[0].tolist(),
        "final_tip_world_mm": points[-1].tolist(),
        "min_error_mm": float(errors.min()),
        "min_error_step": int(np.argmin(errors)),
        "final_error_mm": float(errors[-1]),
        "outputs": [png_path.name, pdf_path.name, csv_path.name],
    }
    (output_dir / "trajectory_plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
