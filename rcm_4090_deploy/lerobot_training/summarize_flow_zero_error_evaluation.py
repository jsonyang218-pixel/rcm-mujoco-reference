"""Summarize strict 0.1 um flow-policy evaluations and paired baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STRICT_ERROR_MM = 0.0001


def _load_episodes(root: Path):
    episodes = []
    shards = []
    for path in sorted(root.glob("g*/evaluation_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("episodes", [])
        if not rows:
            continue
        episodes.extend(rows)
        shards.append({
            "path": str(path),
            "episodes": len(rows),
            "successes": int(sum(bool(row["success"]) for row in rows)),
        })
    if not episodes:
        raise FileNotFoundError(f"no evaluated episodes under {root}")
    indices = [int(row["episode_index"]) for row in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate episode indices across evaluation shards")
    return sorted(episodes, key=lambda row: int(row["episode_index"])), shards


def _minimum_error_mm(episode):
    errors = episode.get("trajectory", {}).get("errors", [])
    return float(min(errors)) if errors else float(episode["final_error_mm"])


def _distribution(values_um):
    values_um = np.asarray(values_um, dtype=np.float64)
    return {
        "values_um": values_um.tolist(),
        "mean_um": float(values_um.mean()),
        "median_um": float(np.median(values_um)),
        "minimum_um": float(values_um.min()),
        "maximum_um": float(values_um.max()),
        "p75_um": float(np.percentile(values_um, 75)),
        "p90_um": float(np.percentile(values_um, 90)),
        "p95_um": float(np.percentile(values_um, 95)),
    }


def _count_at_most(values_um, threshold_um):
    return int(np.sum(np.asarray(values_um, dtype=np.float64) <= threshold_um))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-um", type=float, default=0.1)
    args = parser.parse_args()
    if args.threshold_um <= 0:
        raise ValueError("threshold must be positive")

    episodes, shards = _load_episodes(args.root)
    reference, _ = _load_episodes(args.reference_root)
    reference_by_index = {
        int(row["episode_index"]): row for row in reference
    }
    if set(reference_by_index) != {
        int(row["episode_index"]) for row in episodes
    }:
        raise ValueError("candidate and reference target sets differ")

    final_um = [
        float(row["final_error_mm"]) * 1000.0 for row in episodes
    ]
    minimum_um = [
        _minimum_error_mm(row) * 1000.0 for row in episodes
    ]
    settled_successes = int(
        sum(bool(row["success"]) for row in episodes)
    )
    final_gate_successes = _count_at_most(
        final_um, args.threshold_um
    )
    ever_reached_gate = _count_at_most(
        minimum_um, args.threshold_um
    )

    paired = []
    for row in episodes:
        index = int(row["episode_index"])
        baseline_mm = float(
            reference_by_index[index]["final_error_mm"]
        )
        candidate_mm = float(row["final_error_mm"])
        paired.append(candidate_mm - baseline_mm)
    paired = np.asarray(paired, dtype=np.float64)

    safety_limited = np.asarray(
        [int(row.get("safety_limited_steps", 0)) for row in episodes],
        dtype=np.int64,
    )
    records = []
    for row, final_value, minimum_value in zip(
        episodes, final_um, minimum_um
    ):
        index = int(row["episode_index"])
        baseline_mm = float(
            reference_by_index[index]["final_error_mm"]
        )
        records.append({
            "episode_index": index,
            "target_id": row.get("target_id"),
            "success_at_0p1um": bool(row["success"]),
            "final_error_um": final_value,
            "minimum_error_um": minimum_value,
            "reference_final_error_um": baseline_mm * 1000.0,
            "final_error_change_um": final_value - baseline_mm * 1000.0,
            "steps": int(row["steps"]),
            "safety_limited_steps": int(
                row.get("safety_limited_steps", 0)
            ),
            "early_stop_reason": row.get("early_stop_reason"),
            "oct_valid_frames": int(sum(
                bool(value)
                for value in row["trajectory"][
                    "oct_navigation_valid"
                ]
            )),
            "frames": int(len(row["trajectory"]["errors"])),
        })

    result = {
        "result": "PASS" if settled_successes == len(episodes) else "FAIL",
        "execution_backend": episodes[0]["execution_backend"],
        "strict_success_error_threshold_um": args.threshold_um,
        "episodes": len(episodes),
        "shards": shards,
        "settled_successes_at_threshold": settled_successes,
        "settled_success_rate_at_threshold": (
            settled_successes / len(episodes)
        ),
        "episodes_with_final_at_most_threshold": final_gate_successes,
        "episodes_ever_at_most_threshold": ever_reached_gate,
        "final_error_threshold_gate_pass": (
            final_gate_successes == len(episodes)
        ),
        "final_error": _distribution(final_um),
        "per_episode_minimum_error": _distribution(minimum_um),
        "safety_limited_steps": {
            "total": int(safety_limited.sum()),
            "maximum_per_episode": int(safety_limited.max()),
            "episodes_with_any": int(
                np.sum(safety_limited > 0)
            ),
        },
        "paired_reference": {
            "path": str(args.reference_root),
            "improved": int(np.sum(paired < 0)),
            "worsened": int(np.sum(paired > 0)),
            "unchanged": int(np.sum(paired == 0)),
            "mean_improvement_um": float(-paired.mean() * 1000.0),
            "median_improvement_um": float(
                -np.median(paired) * 1000.0
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
