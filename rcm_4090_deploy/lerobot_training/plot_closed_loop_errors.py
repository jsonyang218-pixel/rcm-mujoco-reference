"""Plot closed-loop error trajectories and paired baseline comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_episodes(root):
    episodes = []
    for path in sorted(Path(root).glob("g*/evaluation_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes.extend(payload.get("episodes", []))
    if not episodes:
        raise FileNotFoundError(f"no evaluated episodes under {root}")
    return sorted(episodes, key=lambda row: int(row["episode_index"]))


def _errors_um(episodes):
    return [
        np.asarray(episode["trajectory"]["errors"], dtype=np.float64) * 1000.0
        for episode in episodes
    ]


def _plot_error_curves(
    episodes,
    errors_um,
    threshold_um,
    title,
    destination,
):
    figure, axis = plt.subplots(figsize=(12, 7))
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(episodes)))
    for episode, errors, color in zip(episodes, errors_um, colors):
        episode_index = int(episode["episode_index"])
        axis.plot(
            np.arange(len(errors)),
            np.maximum(errors, threshold_um * 0.5),
            color=color,
            linewidth=1.2,
            alpha=0.82,
            label=f"E{episode_index:02d}",
        )
    axis.axhline(
        threshold_um,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold_um:g} um gate",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Control step")
    axis.set_ylabel("Needle-tip error (um, log scale)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(ncol=5, fontsize=7, loc="upper right")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _plot_small_multiples(
    episodes,
    errors_um,
    threshold_um,
    destination,
):
    figure, axes = plt.subplots(
        4,
        5,
        figsize=(15, 10),
        sharex=True,
        sharey=True,
    )
    for axis, episode, errors in zip(axes.flat, episodes, errors_um):
        episode_index = int(episode["episode_index"])
        axis.plot(np.arange(len(errors)), errors, color="#0057B8")
        axis.axhline(
            threshold_um,
            color="red",
            linestyle="--",
            linewidth=0.8,
        )
        axis.set_yscale("log")
        axis.grid(True, alpha=0.2)
        axis.set_title(
            f"E{episode_index:02d}: final "
            f"{float(episode['final_error_mm']) * 1000.0:.1f} um",
            fontsize=9,
        )
    for axis in axes[-1]:
        axis.set_xlabel("Step")
    for axis in axes[:, 0]:
        axis.set_ylabel("Error (um)")
    figure.suptitle(
        f"Per-target error trajectory; red dashed line = {threshold_um:g} um",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_best_and_final(
    episodes,
    errors_um,
    threshold_um,
    destination,
):
    indices = np.arange(len(episodes))
    labels = [
        f"E{int(row['episode_index']):02d}" for row in episodes
    ]
    finals = np.asarray([
        float(row["final_error_mm"]) * 1000.0 for row in episodes
    ])
    minimums = np.asarray([values.min() for values in errors_um])
    figure, axis = plt.subplots(figsize=(14, 6))
    width = 0.38
    axis.bar(
        indices - width / 2,
        minimums,
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
        threshold_um,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold_um:g} um gate",
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


def _plot_paired_baseline(
    episodes,
    baseline_episodes,
    threshold_um,
    destination,
    baseline_label,
    candidate_label,
):
    baseline = {
        int(row["episode_index"]): row for row in baseline_episodes
    }
    if set(baseline) != {
        int(row["episode_index"]) for row in episodes
    }:
        raise ValueError("candidate and baseline episode indexes differ")
    indices = np.arange(len(episodes))
    labels = [
        f"E{int(row['episode_index']):02d}" for row in episodes
    ]
    candidate = np.asarray([
        float(row["final_error_mm"]) * 1000.0 for row in episodes
    ])
    reference = np.asarray([
        float(baseline[int(row["episode_index"])]["final_error_mm"]) * 1000.0
        for row in episodes
    ])
    width = 0.38
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(
        indices - width / 2,
        reference,
        width,
        label=baseline_label,
        color="#8C8C8C",
    )
    axis.bar(
        indices + width / 2,
        candidate,
        width,
        label=candidate_label,
        color="#E45756",
    )
    axis.axhline(
        threshold_um,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold_um:g} um gate",
    )
    axis.set_yscale("log")
    axis.set_xticks(indices, labels, rotation=45)
    axis.set_ylabel("Final error (um, log scale)")
    axis.set_title(
        f"Paired final error: {candidate_label} vs {baseline_label}"
    )
    axis.grid(True, axis="y", which="both", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-um", type=float, default=0.35)
    parser.add_argument(
        "--title",
        default="acthist DiT DAgger2: development 20 error trajectories",
    )
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument(
        "--baseline-label",
        default="old DiT DAgger2",
    )
    parser.add_argument(
        "--candidate-label",
        default="acthist DiT DAgger2",
    )
    parser.add_argument(
        "--paired-filename",
        default="final_error_vs_old.png",
    )
    args = parser.parse_args()
    if args.threshold_um <= 0:
        raise ValueError("threshold must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    episodes = _load_episodes(args.root)
    errors_um = _errors_um(episodes)
    final_um = np.asarray([
        float(row["final_error_mm"]) * 1000.0 for row in episodes
    ])
    minimum_um = np.asarray([values.min() for values in errors_um])

    _plot_error_curves(
        episodes,
        errors_um,
        args.threshold_um,
        args.title,
        args.output_dir / "error_curves_all.png",
    )
    _plot_small_multiples(
        episodes,
        errors_um,
        args.threshold_um,
        args.output_dir / "error_curves_small_multiples.png",
    )
    _plot_best_and_final(
        episodes,
        errors_um,
        args.threshold_um,
        args.output_dir / "best_and_final_error.png",
    )
    paired = None
    if args.baseline_root is not None:
        baseline = _load_episodes(args.baseline_root)
        _plot_paired_baseline(
            episodes,
            baseline,
            args.threshold_um,
            args.output_dir / args.paired_filename,
            args.baseline_label,
            args.candidate_label,
        )
        baseline_map = {
            int(row["episode_index"]): float(row["final_error_mm"]) * 1000.0
            for row in baseline
        }
        paired = np.asarray([
            value - baseline_map[int(episode["episode_index"])]
            for episode, value in zip(episodes, final_um)
        ])

    summary = {
        "episodes": len(episodes),
        "threshold_um": float(args.threshold_um),
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
        "paired_change_um": (
            None if paired is None else {
                "improved": int(np.sum(paired < 0)),
                "worsened": int(np.sum(paired > 0)),
                "mean": float(paired.mean()),
                "median": float(np.median(paired)),
            }
        ),
    }
    (args.output_dir / "error_plot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
