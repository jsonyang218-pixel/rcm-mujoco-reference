#!/usr/bin/env python3
"""Summarize A/B/C experience-replay closed-loop experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


ARMS = {
    "A": "a_original",
    "B": "b_static",
    "C": "c_balanced",
}
THRESHOLDS_UM = (300.0, 100.0, 30.0, 0.35)


def load_episodes(root):
    episodes = []
    paths = sorted((root / "evaluation").glob("g*/evaluation_results.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes.extend(payload.get("episodes", []))
    if not episodes:
        raise FileNotFoundError(f"no episode results under {root / 'evaluation'}")
    return sorted(episodes, key=lambda row: int(row["episode_index"]))


def longest_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def episode_metrics(episode):
    errors_um = np.asarray(episode["trajectory"]["errors"], dtype=np.float64) * 1000.0
    metrics = {
        "episode_index": int(episode["episode_index"]),
        "success": bool(episode["success"]),
        "steps": int(episode["steps"]),
        "final_error_um": float(episode["final_error_mm"] * 1000.0),
        "minimum_error_um": float(errors_um.min()),
        "safety_limited_steps": int(episode.get("safety_limited_steps", 0)),
        "encoder_settled_at_exit": bool(
            episode.get("terminal_success", {}).get("encoder_settled_at_exit", False)
        ),
        "final_speed": float(
            np.linalg.norm(episode["final_velocity_deg_deg_mm_per_s"])
        ),
    }
    for threshold in THRESHOLDS_UM:
        label = f"{threshold:g}"
        inside = errors_um <= threshold
        indices = np.flatnonzero(inside)
        first = int(indices[0]) if len(indices) else None
        metrics[f"first_{label}_step"] = first
        metrics[f"longest_{label}_run"] = int(longest_true_run(inside))
        metrics[f"left_{label}_after_entry"] = bool(
            first is not None and not inside[first:].all()
        )
    return metrics


def aggregate(episodes):
    rows = [episode_metrics(row) for row in episodes]
    finals = np.asarray([row["final_error_um"] for row in rows])
    summary = {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": statistics.mean(row["success"] for row in rows),
        "final_error_um": {
            "mean": float(finals.mean()),
            "median": float(np.median(finals)),
            "p90": float(np.percentile(finals, 90)),
            "min": float(finals.min()),
            "max": float(finals.max()),
        },
        "minimum_error_um": {
            "mean": statistics.mean(row["minimum_error_um"] for row in rows),
            "median": statistics.median(row["minimum_error_um"] for row in rows),
        },
        "safety_limited_steps": sum(row["safety_limited_steps"] for row in rows),
        "encoder_settled_at_exit": sum(
            row["encoder_settled_at_exit"] for row in rows
        ),
        "episodes_detail": rows,
    }
    for threshold in THRESHOLDS_UM:
        label = f"{threshold:g}"
        entries = [
            row[f"first_{label}_step"]
            for row in rows
            if row[f"first_{label}_step"] is not None
        ]
        summary[f"threshold_{label}_um"] = {
            "entered": len(entries),
            "first_entry_step_mean": statistics.mean(entries) if entries else None,
            "first_entry_step_median": statistics.median(entries) if entries else None,
            "longest_run_mean_frames": statistics.mean(
                row[f"longest_{label}_run"] for row in rows
            ),
            "left_after_entry": sum(
                row[f"left_{label}_after_entry"] for row in rows
            ),
        }
    return summary


def paired(reference, candidate, reference_label, candidate_label):
    left = {int(row["episode_index"]): row for row in reference["episodes_detail"]}
    right = {int(row["episode_index"]): row for row in candidate["episodes_detail"]}
    if set(left) != set(right):
        raise ValueError("paired runs contain different episode indexes")
    changes = np.asarray([
        left[index]["final_error_um"] - right[index]["final_error_um"]
        for index in sorted(left)
    ])
    return {
        "reference": reference_label,
        "candidate": candidate_label,
        "improved": int(np.sum(changes > 0)),
        "worsened": int(np.sum(changes < 0)),
        "unchanged": int(np.sum(changes == 0)),
        "mean_improvement_um": float(changes.mean()),
        "median_improvement_um": float(np.median(changes)),
    }


def markdown(results):
    lines = [
        "# Experience Replay A/B/C",
        "",
        "| Metric | A original | B static | C balanced |",
        "|---|---:|---:|---:|",
    ]
    labels = (
        ("Successes", lambda row: f"{row['successes']}/{row['episodes']}"),
        ("Final mean um", lambda row: f"{row['final_error_um']['mean']:.3f}"),
        ("Final median um", lambda row: f"{row['final_error_um']['median']:.3f}"),
        ("Final P90 um", lambda row: f"{row['final_error_um']['p90']:.3f}"),
        ("Final maximum um", lambda row: f"{row['final_error_um']['max']:.3f}"),
        (
            "Minimum error mean um",
            lambda row: f"{row['minimum_error_um']['mean']:.3f}",
        ),
        ("Safety-limited steps", lambda row: str(row["safety_limited_steps"])),
        (
            "Encoder settled at exit",
            lambda row: f"{row['encoder_settled_at_exit']}/{row['episodes']}",
        ),
    )
    for label, getter in labels:
        lines.append(
            f"| {label} | {getter(results['A'])} | "
            f"{getter(results['B'])} | {getter(results['C'])} |"
        )
    lines.extend(["", "## Paired final-error improvements", ""])
    for item in results["paired"]:
        lines.append(
            f"- `{item['candidate']}` vs `{item['reference']}`: "
            f"{item['improved']} improved, {item['worsened']} worsened, "
            f"mean improvement {item['mean_improvement_um']:+.3f} um."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)

    episodes = {
        label: load_episodes(args.root / directory)
        for label, directory in ARMS.items()
    }
    results = {
        label: aggregate(rows)
        for label, rows in episodes.items()
    }
    results["paired"] = [
        paired(results["A"], results["B"], "A", "B"),
        paired(results["A"], results["C"], "A", "C"),
        paired(results["B"], results["C"], "B", "C"),
    ]
    (args.output_dir / "abc_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ABC_COMPARISON.md").write_text(
        markdown(results), encoding="utf-8"
    )
    print(json.dumps({
        label: {
            "successes": result["successes"],
            "final_error_um": result["final_error_um"],
            "safety_limited_steps": result["safety_limited_steps"],
        }
        for label, result in results.items()
        if label != "paired"
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
