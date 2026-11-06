"""Aggregate sharded closed-loop evaluation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode_minimum_error(episode):
    trajectory = episode.get("trajectory", {})
    errors = trajectory.get("errors")
    if errors:
        return float(np.min(errors))
    return float(episode["final_error_mm"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--uncontrolled-mm", type=float, default=30.0)
    parser.add_argument("--max-uncontrolled", type=int, default=5)
    parser.add_argument("--max-median-min-error-mm", type=float, default=1.0)
    args = parser.parse_args()
    paths = (
        [path.resolve() for path in args.result]
        if args.result
        else sorted(args.root.glob("g*/evaluation_results.json"))
    )
    if not paths:
        raise FileNotFoundError(
            f"no g*/evaluation_results.json under {args.root}"
        )
    episodes = []
    shards = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_episodes = payload.get("episodes", [])
        if int(payload.get("num_episodes", len(shard_episodes))) != len(
            shard_episodes
        ):
            raise ValueError(f"episode count mismatch in {path}")
        episodes.extend(shard_episodes)
        shards.append({
            "path": str(path),
            "episodes": len(shard_episodes),
            "success_rate": payload.get("success_rate"),
        })
    finals = np.asarray(
        [row["final_error_mm"] for row in episodes],
        dtype=np.float64,
    )
    minimums = np.asarray(
        [episode_minimum_error(row) for row in episodes],
        dtype=np.float64,
    )
    safety_limited = np.asarray(
        [int(row.get("safety_limited_steps", 0)) for row in episodes],
        dtype=np.int64,
    )
    successes = int(sum(bool(row.get("success")) for row in episodes))
    uncontrolled = int(np.sum(finals > args.uncontrolled_mm))
    median_minimum = float(np.median(minimums))
    gates = {
        "uncontrolled": {
            "threshold_count": args.max_uncontrolled,
            "actual_count": uncontrolled,
            "pass": uncontrolled < args.max_uncontrolled,
        },
        "median_per_episode_minimum_error": {
            "threshold_mm": args.max_median_min_error_mm,
            "actual_mm": median_minimum,
            "pass": median_minimum < args.max_median_min_error_mm,
        },
    }
    result = {
        "result": (
            "PASS" if all(row["pass"] for row in gates.values()) else "FAIL"
        ),
        "episodes": len(episodes),
        "shards": shards,
        "successes": successes,
        "success_rate": successes / len(episodes),
        "strict_all_episodes_success": successes == len(episodes),
        "acceptance_gates": gates,
        "uncontrolled_over_threshold": uncontrolled,
        "uncontrolled_threshold_mm": args.uncontrolled_mm,
        "final_error_mm": {
            "mean": float(finals.mean()),
            "median": float(np.median(finals)),
            "minimum": float(finals.min()),
            "maximum": float(finals.max()),
        },
        "per_episode_minimum_error_mm": {
            "mean": float(minimums.mean()),
            "median": median_minimum,
            "minimum": float(minimums.min()),
            "maximum": float(minimums.max()),
        },
        "safety_limited_steps": {
            "total": int(safety_limited.sum()),
            "maximum_per_episode": int(safety_limited.max()),
            "episodes_with_any": int(np.sum(safety_limited > 0)),
        },
    }
    output = args.output or args.root / "aggregate_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
