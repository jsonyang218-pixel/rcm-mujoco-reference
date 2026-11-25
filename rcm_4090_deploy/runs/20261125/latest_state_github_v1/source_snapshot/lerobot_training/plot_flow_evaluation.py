"""Plot closed-loop error and needle-tip trajectories for flow evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcm.scene_v9 import RCMSceneV9


TASK_TO_DEG_DEG_MM = np.array(
    [180.0 / np.pi, 180.0 / np.pi, 1000.0],
    dtype=np.float64,
)
AXIS_NAMES = ("X", "Y", "Z")


def _load_episodes(root: Path):
    episodes = []
    for path in sorted(root.glob("g*/evaluation_results.json")):
        episodes.extend(
            json.loads(path.read_text(encoding="utf-8"))["episodes"]
        )
    if not episodes:
        raise FileNotFoundError(f"no evaluated episodes under {root}")
    return sorted(episodes, key=lambda row: int(row["episode_index"]))


def _reconstruct_tip_paths(episodes):
    paths = []
    for episode in episodes:
        scene = RCMSceneV9()
        path = []
        try:
            for state_abc in episode["trajectory"]["states"]:
                task = np.asarray(state_abc, dtype=np.float64)
                task /= TASK_TO_DEG_DEG_MM
                scene.set_task(task)
                path.append(scene.site_mm("needle_tip").copy())
        finally:
            scene.close()
        paths.append(np.asarray(path, dtype=np.float64))
    return paths


def _plot_all_error_curves(episodes, errors_um, destination):
    figure, axis = plt.subplots(figsize=(12, 7))
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(episodes)))
    for episode, errors, color in zip(episodes, errors_um, colors):
        episode_index = int(episode["episode_index"])
        linewidth = 2.4 if episode_index == 14 else 1.2
        zorder = 5 if episode_index == 14 else 2
        axis.plot(
            np.arange(len(errors)),
            np.maximum(errors, 0.05),
            color=color,
            linewidth=linewidth,
            alpha=0.82,
            zorder=zorder,
            label=f"E{episode_index:02d}",
        )
    axis.axhline(
        0.1,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="0.1 um gate",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Control step")
    axis.set_ylabel("Needle-tip error (um, log scale)")
    axis.set_title("Flow Temporal U-Net: development 20 error trajectories")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(ncol=5, fontsize=7, loc="upper right")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _plot_error_small_multiples(episodes, errors_um, destination):
    figure, axes = plt.subplots(
        4,
        5,
        figsize=(15, 10),
        sharex=True,
        sharey=True,
    )
    for axis, episode, errors in zip(
        axes.flat, episodes, errors_um
    ):
        episode_index = int(episode["episode_index"])
        axis.plot(np.arange(len(errors)), errors, color="#0057B8")
        axis.axhline(0.1, color="red", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_ylim(0.08, 30000.0)
        axis.grid(True, alpha=0.2)
        axis.set_title(
            f"E{episode_index:02d}: final {errors[-1]:.1f} um",
            fontsize=9,
        )
    for axis in axes[-1]:
        axis.set_xlabel("Step")
    for axis in axes[:, 0]:
        axis.set_ylabel("Error (um)")
    figure.suptitle(
        "Per-target error trajectory; red dashed line = 0.1 um",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_error_summary(
    episodes, errors_um, minimum_um, destination
):
    indices = np.arange(len(episodes))
    labels = [
        f"E{int(row['episode_index']):02d}" for row in episodes
    ]
    finals = np.asarray([row[-1] for row in errors_um])
    figure, axis = plt.subplots(figsize=(14, 6))
    width = 0.38
    axis.bar(
        indices - width / 2,
        minimum_um,
        width,
        label="best error in episode",
        color="#4C78A8",
    )
    axis.bar(
        indices + width / 2,
        finals,
        width,
        label="final error",
        color="#F58518",
    )
    axis.axhline(
        0.1,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="0.1 um gate",
    )
    axis.set_yscale("log")
    axis.set_xticks(indices, labels, rotation=45)
    axis.set_ylabel("Error (um, log scale)")
    axis.set_title("Best and final endpoint error by development target")
    axis.grid(True, axis="y", which="both", alpha=0.22)
    axis.legend()
    for index, value in enumerate(finals):
        axis.text(
            index + width / 2,
            value * 1.08,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _plot_residual_components(
    episodes, residual_um, destination
):
    indices = np.asarray([
        int(row["episode_index"]) for row in episodes
    ])
    typical = indices != 14
    colors = ("#E45756", "#54A24B", "#4C78A8")
    figure, axes = plt.subplots(3, 2, figsize=(14, 10))
    for component, name in enumerate(AXIS_NAMES):
        for column, mask in enumerate((np.ones_like(typical), typical)):
            axis = axes[component, column]
            values = residual_um[mask, component]
            x_values = indices[mask]
            axis.bar(
                x_values,
                values,
                color=colors[component],
                alpha=0.82,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(
                f"{name} residual: "
                + ("all targets" if column == 0 else "excluding E14")
            )
            axis.set_ylabel("Tip - target (um)")
            axis.grid(True, axis="y", alpha=0.22)
            if column == 0:
                axis.set_yscale("symlog", linthresh=100.0)
            else:
                axis.set_ylim(-1000.0, 1600.0)
    for axis in axes[-1]:
        axis.set_xlabel("Episode")
    figure.suptitle(
        "Final endpoint residual components "
        "(positive means tip beyond target)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(destination, dpi=190)
    plt.close(figure)


def _plot_tip_path_projections(
    episodes, tip_paths, destination
):
    projections = ((0, 1), (0, 2), (1, 2))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(episodes)))
    for axis, (horizontal, vertical) in zip(axes, projections):
        for episode, path, color in zip(
            episodes, tip_paths, colors
        ):
            episode_index = int(episode["episode_index"])
            if episode_index == 14:
                continue
            target = np.asarray(episode["target_world_mm"])
            relative = path - target
            axis.plot(
                relative[:, horizontal],
                relative[:, vertical],
                color=color,
                linewidth=2.0 if episode_index == 14 else 1.0,
                alpha=0.8,
            )
            final_residual = (
                np.asarray(episode["final_tip_world_mm"]) - target
            )
            axis.scatter(
                final_residual[horizontal],
                final_residual[vertical],
                color=color,
                s=20,
            )
        axis.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=170,
            color="black",
            label="target",
            zorder=10,
        )
        axis.set_xlabel(
            f"{AXIS_NAMES[horizontal]} error from target (mm)"
        )
        axis.set_ylabel(
            f"{AXIS_NAMES[vertical]} error from target (mm)"
        )
        axis.set_title(
            f"{AXIS_NAMES[horizontal]}-{AXIS_NAMES[vertical]} projection"
        )
        axis.grid(True, alpha=0.22)
        axis.axis("equal")
    figure.suptitle(
        "Needle-tip trajectories relative to each target "
        "(kinematically reconstructed from recorded A/B/C states; "
        "E14 outlier omitted)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _plot_control_states(episodes, destination):
    state_units = ("A (deg)", "B (deg)", "C (mm)")
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(episodes)))
    for episode, color in zip(episodes, colors):
        states = np.asarray(episode["trajectory"]["states"])
        episode_index = int(episode["episode_index"])
        for component, axis in enumerate(axes):
            axis.plot(
                np.arange(len(states)),
                states[:, component],
                color=color,
                linewidth=2.0 if episode_index == 14 else 0.9,
                alpha=0.8,
            )
    for component, axis in enumerate(axes):
        axis.set_ylabel(state_units[component])
        axis.grid(True, alpha=0.22)
    axes[-1].set_xlabel("Control step")
    axes[0].set_title(
        "Recorded A/B/C control-state trajectories"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    return np.asarray([
        np.nanmean(values[index:index + window])
        for index in range(len(values) - window + 1)
    ])


def _plot_phase_diagnostics(
    episodes, errors_um, actions, destination
):
    typical_mask = np.asarray([
        int(row["episode_index"]) != 14 for row in episodes
    ])
    max_steps = max(len(values) for values in errors_um)
    error_matrix = np.full(
        (len(errors_um), max_steps),
        np.nan,
        dtype=np.float64,
    )
    for row, values in enumerate(errors_um):
        error_matrix[row, :len(values)] = values
    typical_errors = error_matrix[typical_mask]
    action_norms = np.full(
        (len(actions), max_steps),
        np.nan,
        dtype=np.float64,
    )
    for row, values in enumerate(actions):
        action_norms[row, :len(values)] = np.linalg.norm(
            values,
            axis=1,
        )
    steps = np.arange(error_matrix.shape[1])
    window = 50
    rolling_centers = np.arange(
        window // 2, error_matrix.shape[1] - window // 2 + 1
    )
    error_deltas = np.diff(error_matrix, axis=1)
    valid_deltas = np.isfinite(error_deltas)
    improving = (error_deltas < 0.0) & valid_deltas
    improving_fraction = (
        improving.sum(axis=0)
        / np.maximum(valid_deltas.sum(axis=0), 1)
    )
    improving_rolling = _rolling_mean(
        improving_fraction, window
    )
    improving_centers = np.arange(
        window // 2,
        window // 2 + len(improving_rolling),
    )

    figure, axes = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True
    )
    axes[0].fill_between(
        steps,
        np.nanpercentile(typical_errors, 25, axis=0),
        np.nanpercentile(typical_errors, 75, axis=0),
        color="#4C78A8",
        alpha=0.22,
        label="typical IQR, E14 excluded",
    )
    axes[0].plot(
        steps,
        np.nanmedian(typical_errors, axis=0),
        color="#0057B8",
        linewidth=2.0,
        label="typical median",
    )
    outlier_index = np.where(~typical_mask)[0][0]
    axes[0].plot(
        steps,
        error_matrix[outlier_index],
        color="#E45756",
        linewidth=1.7,
        label="E14 outlier",
    )
    axes[0].axhline(
        0.1,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="0.1 um gate",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error (um)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        "Phase transition around step 700: "
        "error remains while policy output becomes small"
    )

    axes[1].plot(
        steps,
        np.nanmedian(action_norms, axis=0),
        color="#54A24B",
        linewidth=1.4,
        label="median action norm",
    )
    axes[1].plot(
        rolling_centers,
        _rolling_mean(np.nanmedian(action_norms, axis=0), window),
        color="#1B5E20",
        linewidth=2.4,
        label="50-step rolling median-of-episodes",
    )
    axes[1].set_ylabel("||action|| (A/B/C mixed units)")
    axes[1].legend(loc="upper right")

    axes[2].plot(
        improving_centers,
        improving_rolling,
        color="#F58518",
        linewidth=2.2,
    )
    axes[2].axhline(
        0.5,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="50% improving",
    )
    axes[2].set_ylabel("Fraction of steps reducing error")
    axes[2].set_xlabel("Control step")
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.axvline(700, color="#B22222", linestyle="--", alpha=0.8)
        axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(destination, dpi=190)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    episodes = _load_episodes(args.root)
    errors_um = [
        np.asarray(row["trajectory"]["errors"], dtype=np.float64)
        * 1000.0
        for row in episodes
    ]
    minimum_um = np.asarray(
        [values.min() for values in errors_um], dtype=np.float64
    )
    residual_um = np.asarray([
        (
            np.asarray(row["final_tip_world_mm"], dtype=np.float64)
            - np.asarray(row["target_world_mm"], dtype=np.float64)
        )
        * 1000.0
        for row in episodes
    ])
    actions = [
        np.asarray(row["trajectory"]["actions"], dtype=np.float64)
        for row in episodes
    ]
    _plot_phase_diagnostics(
        episodes,
        errors_um,
        actions,
        args.output_dir / "step_700_phase_diagnostics.png",
    )
    if args.phase_only:
        return 0
    tip_paths = _reconstruct_tip_paths(episodes)

    _plot_all_error_curves(
        episodes, errors_um, args.output_dir / "error_curves_all.png"
    )
    _plot_error_small_multiples(
        episodes,
        errors_um,
        args.output_dir / "error_curves_small_multiples.png",
    )
    _plot_error_summary(
        episodes,
        errors_um,
        minimum_um,
        args.output_dir / "best_and_final_error.png",
    )
    _plot_residual_components(
        episodes,
        residual_um,
        args.output_dir / "final_residual_components.png",
    )
    _plot_tip_path_projections(
        episodes,
        tip_paths,
        args.output_dir / "tip_trajectory_projections.png",
    )
    _plot_control_states(
        episodes,
        args.output_dir / "control_state_trajectories.png",
    )

    final_um = np.asarray([
        float(row["final_error_mm"]) * 1000.0 for row in episodes
    ])
    typical_residual = residual_um[
        np.asarray([
            int(row["episode_index"]) != 14 for row in episodes
        ])
    ]
    summary = {
        "episodes": len(episodes),
        "final_error_um": {
            "mean": float(final_um.mean()),
            "median": float(np.median(final_um)),
            "minimum": float(final_um.min()),
            "maximum": float(final_um.max()),
        },
        "minimum_error_um": {
            "mean": float(minimum_um.mean()),
            "median": float(np.median(minimum_um)),
            "minimum": float(minimum_um.min()),
            "maximum": float(minimum_um.max()),
        },
        "final_residual_um": {
            "mean": residual_um.mean(axis=0).tolist(),
            "median": np.median(residual_um, axis=0).tolist(),
            "mean_absolute": np.abs(residual_um).mean(
                axis=0
            ).tolist(),
        },
        "final_residual_excluding_episode_14_um": {
            "mean": typical_residual.mean(axis=0).tolist(),
            "median": np.median(
                typical_residual, axis=0
            ).tolist(),
            "mean_absolute": np.abs(
                typical_residual
            ).mean(axis=0).tolist(),
            "mean_absolute_axis_share": (
                np.abs(typical_residual).mean(axis=0)
                / np.abs(typical_residual).mean(axis=0).sum()
            ).tolist(),
            "positive_axis_count": (
                typical_residual > 0.0
            ).sum(axis=0).tolist(),
        },
        "trajectory_reconstruction": (
            "A/B/C states replayed through the unactuated MuJoCo scene; "
            "recorded final tip coordinates are plotted directly"
        ),
    }
    (args.output_dir / "trajectory_plot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
