"""Compare the recorded I0 needle-tip trajectories without running the policy."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from lerobot_training.plot_trajectory_3d import plt, state_to_task
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from rcm.scene_v9 import RCMSceneV9


TARGET_COUNT = 20
STEPS = 1600
CONTROL_PERIOD_S = 0.05
TOLERANCE_MM = 1e-3
ZOOM_UM = 150
DETAIL_INDICES = (8, 0, 2, 6)
STYLES = (
    ("Frozen R0", "#555b66", "-", 1.7),
    ("Cleaned replay 2k", "#0072b2", "--", 1.5),
    ("Cleaned replay 4k", "#d55e00", "-", 1.0),
)


@dataclass
class TipPath:
    points_mm: np.ndarray
    time_s: np.ndarray
    errors_um: np.ndarray
    success: bool
    max_error_delta_mm: float
    final_position_delta_mm: float


def read_episodes(files):
    episodes = {}
    expected = {
        "version": 2,
        "execution_backend": "MotorBackend",
        "state_action_units": "A_deg,B_deg,C_mm",
        "control_period_s": CONTROL_PERIOD_S,
        "flow_integration_steps": 4,
        "num_action_samples": 1,
        "success_error_threshold_mm": 0.00035,
    }
    for file in files:
        payload = json.loads(file.read_text())
        contract = payload["evaluation_contract"]
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{file}: not the expected I0 evaluation contract")
        for row in payload["episodes"]:
            index = row["episode_index"]
            if index in episodes or row["seed"] != 42 + index:
                raise ValueError(f"{file}: duplicate target index or unexpected seed: {index}")
            if row["steps"] != STEPS or row["early_stop_reason"]:
                raise ValueError(f"{file}: E{index:02d} is not a complete {STEPS}-step rollout")
            episodes[index] = row
    if (set(episodes) != set(range(TARGET_COUNT))
            or len({row["target_id"] for row in episodes.values()}) != TARGET_COUNT):
        raise ValueError("Expected 20 distinct targets E00–E19 in each group")
    return [episodes[index] for index in range(TARGET_COUNT)]


def finite_array(values, shape, name):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite array of shape {shape}")
    return array


def reconstruct(scene, episode):
    trace = episode["trajectory"]
    states = finite_array(trace["states"], (STEPS, 3), "recorded poses")
    final_state = finite_array(episode["final_state_A_deg_B_deg_C_mm"], (3,), "final pose")
    target_mm = finite_array(episode["target_world_mm"], (3,), "target position")
    saved_tip_mm = finite_array(episode["final_tip_world_mm"], (3,), "final tip position")
    errors = finite_array(trace["errors"], (STEPS,), "recorded errors")
    final_error = finite_array(episode["final_error_mm"], (), "final error")
    time_s = finite_array(trace["simulation_time_s"], (STEPS,), "recorded times")
    time_s = time_s - time_s[0]
    if not np.allclose(time_s, np.arange(STEPS) * CONTROL_PERIOD_S, rtol=0, atol=1e-8):
        raise ValueError("Invalid sampling grid")
    time_s = np.append(time_s, time_s[-1] + CONTROL_PERIOD_S)
    points_mm = np.empty((STEPS + 1, 3))
    # Recorded states precede their actions; the post-budget pose is stored separately.
    for step, state in enumerate(np.vstack((states, final_state))):
        scene.set_task(state_to_task(state))
        points_mm[step] = scene.site_mm("needle_tip")
    if not np.isfinite(points_mm).all():
        raise ValueError("Non-finite reconstructed needle-tip positions")
    relative_mm = points_mm - target_mm
    recorded_mm = np.append(errors, final_error)
    error_delta_mm = float(np.max(np.abs(np.linalg.norm(relative_mm, axis=1) - recorded_mm)))
    final_delta_mm = float(np.linalg.norm(points_mm[-1] - saved_tip_mm))
    if max(error_delta_mm, final_delta_mm) >= TOLERANCE_MM:
        raise ValueError(f"Reconstruction mismatch: distance {error_delta_mm:g} mm, tip {final_delta_mm:g} mm")
    return TipPath(relative_mm, time_s, recorded_mm * 1000, episode["success"],
                   error_delta_mm, final_delta_mm)


def legend(fig, show_start):
    handles = [Line2D([], [], label=label, color=color, linestyle=style, linewidth=2)
               for label, color, style, _ in STYLES]
    handles.append(Line2D([], [], color="#181d24", marker="s", linestyle="None", label="Final pose"))
    handles.append(Line2D([], [], color="#181d24", marker="*", markersize=10, linestyle="None", label="Target"))
    if show_start:
        handles.insert(3, Line2D([], [], color="#181d24", marker="o", markerfacecolor="white",
                                 linestyle="None", label="Shared start"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.937),
               ncols=len(handles), frameon=False)


def save_figure(fig, output_dir, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def plot_overview(paths, out):
    fig, axes = plt.subplots(4, 5, figsize=(17.4, 13.3), subplot_kw={"projection": "3d"})
    # Use one spatial scale and view angle across all 20 panels.
    points = np.concatenate([np.zeros((1, 3)), *(path.points_mm for row in paths for path in row)])
    middle = (points.max(axis=0) + points.min(axis=0)) / 2
    half_width_mm = np.ptp(points, axis=0).max() * 0.55
    for index, (ax, row) in enumerate(zip(axes.flat, paths, strict=True)):
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 1), zoom=0.91)
        ax.view_init(elev=25, azim=-55)
        for path, (_, color, style, width) in zip(row, STYLES, strict=True):
            ax.plot(*path.points_mm.T, color=color, linestyle=style, linewidth=width)
            ax.scatter(*path.points_mm[-1], color=color, marker="s", s=23,
                       edgecolor="white", linewidth=0.4, depthshade=False)
        ax.scatter(*row[0].points_mm[0], facecolor="white", edgecolor="#181d24",
                   marker="o", s=26, depthshade=False)
        ax.scatter(0, 0, 0, color="#181d24", marker="*", s=65, depthshade=False)
        for set_lim, axis, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),
                                        (ax.xaxis, ax.yaxis, ax.zaxis), middle, strict=True):
            set_lim(center - half_width_mm, center + half_width_mm)
            axis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        ax.tick_params(labelsize=7, pad=0)
        finals = "  /  ".join(f"{path.errors_um[-1]:.0f}" for path in row)
        ax.set_title(f"E{index:02d}  |  {finals} µm", fontsize=10, fontweight="bold", y=0.95)
    fig.suptitle("Closed-loop needle-tip paths | Development targets E00–E19", fontsize=17, y=0.986)
    fig.text(0.5, 0.951, "Target-relative XYZ (mm)  •  Shared scale and view  •  Titles list final errors: R0 / 2k / 4k",
             ha="center", fontsize=10)
    legend(fig, show_start=True)
    fig.subplots_adjust(left=0.025, right=0.96, bottom=0.025, top=0.884, wspace=0.10, hspace=0.00)
    save_figure(fig, out, "needle_tip_trajectories_3d")


def plot_detail(paths, out):
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 12.0))
    final_window_s = STEPS * CONTROL_PERIOD_S - 20
    for row_index, index in enumerate(DETAIL_INDICES):
        row = paths[index]
        for column, (first, second) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axes[row_index, column]
            for path, (_, color, style, width) in zip(row, STYLES, strict=True):
                xy = path.points_mm[:, [first, second]] * 1000
                # Clip artists to the viewport instead of dropping outside samples.
                ax.plot(*xy.T, color=color, linestyle=style, linewidth=width, alpha=0.85)
                ax.plot(*xy[-1], color=color, marker="s", markersize=5, markeredgecolor="white", zorder=4)
            ax.plot(0, 0, color="#181d24", marker="*", markersize=9, zorder=5)
            ax.set(xlim=(-ZOOM_UM, ZOOM_UM), ylim=(-ZOOM_UM, ZOOM_UM), aspect="equal",
                   xlabel=f"{'XYZ'[first]} (µm)", ylabel=f"{'XYZ'[second]} (µm)",
                   xticks=(-100, 0, 100), yticks=(-100, 0, 100))
            if row_index == 0:
                ax.set_title(f"{'XYZ'[first]}{'XYZ'[second]} projection", fontweight="bold")
        ax = axes[row_index, -1]
        for path, (_, color, style, width) in zip(row, STYLES, strict=True):
            window = path.time_s >= final_window_s - 1e-8
            ax.plot(path.time_s[window], path.errors_um[window], color=color, linestyle=style, linewidth=width)
            ax.plot(path.time_s[-1], path.errors_um[-1], color=color, marker="s", markersize=5)
        ax.set(xlim=(final_window_s, STEPS * CONTROL_PERIOD_S + 1),
               xlabel="Elapsed simulation time (s)", ylabel="3D error (µm)", yscale="log")
        if row_index == 0:
            ax.set_title("Final 20 s", fontweight="bold")
        axes[row_index, 0].annotate(f"E{index:02d}", (-0.46, 0.5), xycoords="axes fraction",
                                    ha="center", va="center", rotation=90, fontweight="bold", fontsize=12)
    for ax in axes.flat:
        ax.grid(color="#e4e7eb", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Near-target needle-tip paths | E08, E00, E02, E06", fontsize=17, y=0.981)
    fig.text(0.5, 0.951, "First three columns show full trajectories clipped to ±150 µm per axis; off-screen endpoints are omitted.",
             ha="center", fontsize=10)
    legend(fig, show_start=False)
    fig.subplots_adjust(left=0.093, right=0.976, bottom=0.048, top=0.883, hspace=0.36, wspace=0.46)
    save_figure(fig, out, "needle_tip_near_target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-json", type=Path, required=True)
    parser.add_argument("--overnight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    sources = [[args.r0_json]] + [
        [args.overnight_dir / f"task2_step{step}/E{index:02d}/evaluation_results.json"
         for index in range(TARGET_COUNT)]
        for step in (2000, 4000)
    ]
    groups = [read_episodes(files) for files in sources]
    paths = []
    scene = RCMSceneV9()
    try:
        for index, group in enumerate(zip(*groups, strict=True)):
            baseline = group[0]
            for record in group[1:]:
                if (record["target_id"] != baseline["target_id"]
                        or record["target_world_mm"] != baseline["target_world_mm"]
                        or record["trajectory"]["states"][0] != baseline["trajectory"]["states"][0]):
                    raise ValueError(f"E{index:02d}: target or start mismatch")
            try:
                paths.append([reconstruct(scene, record) for record in group])
            except ValueError as exc:
                raise ValueError(f"E{index:02d}: {exc}") from exc
    finally:
        scene.close()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    plot_overview(paths, args.output_dir)
    plot_detail(paths, args.output_dir)
    report = {
        "method": "RCMSceneV9 CPU forward kinematics on recorded poses, including each final pose; no policy rollout",
        "protocol": {"targets": TARGET_COUNT, "seed_base": 42, "steps": STEPS,
                     "control_period_s": CONTROL_PERIOD_S, "arm": "I0"},
        "validated_pose_count": sum(len(path.points_mm) for row in paths for path in row),
        "max_distance_discrepancy_um": max(path.max_error_delta_mm for row in paths for path in row) * 1000,
        "max_final_position_discrepancy_um": max(path.final_position_delta_mm for row in paths for path in row) * 1000,
        "reconstruction_tolerance_um": TOLERANCE_MM * 1000,
        "groups": {
            label: {"source_jsons": [str(file) for file in files],
                    "final_mean_um": float(np.mean([row[column].errors_um[-1] for row in paths])),
                    "success_count": sum(row[column].success for row in paths)}
            for column, ((label, *_), files) in enumerate(zip(STYLES, sources, strict=True))
        },
        "targets": [
            {"episode": f"E{index:02d}", "target_id": groups[0][index]["target_id"],
             "seed": groups[0][index]["seed"],
             "final_error_um": {style[0]: float(path.errors_um[-1])
                                for style, path in zip(STYLES, row, strict=True)}}
            for index, row in enumerate(paths)
        ],
    }
    (args.output_dir / "trajectory_reconstruction.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validated {report['validated_pose_count']:,} poses; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
