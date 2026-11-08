"""Summarize final-test Stage 4 metrics and paired median control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from restricted_perception.stage34 import load_stage34_protocol, sha256_file


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combine(paths: list[Path]) -> list[dict]:
    episodes = []
    for path in paths:
        episodes.extend(_load(path)["episodes"])
    return episodes


def _success(episode: dict, threshold: str) -> bool:
    return bool(
        episode.get("success_by_threshold_um", {}).get(
            threshold,
            False,
        )
    )


def _cluster_bootstrap_interval(
    episodes: list[dict],
    *,
    threshold: str,
    samples: int,
    seed: int,
) -> list[float]:
    target_indices = sorted(
        {int(episode["target_index"]) for episode in episodes}
    )
    by_target = {
        index: [
            _success(episode, threshold)
            for episode in episodes
            if int(episode["target_index"]) == index
        ]
        for index in target_indices
    }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for sample in range(samples):
        chosen = rng.choice(target_indices, size=len(target_indices), replace=True)
        values[sample] = np.mean(
            [value for index in chosen for value in by_target[int(index)]]
        )
    return np.percentile(values, [2.5, 97.5]).tolist()


def _metrics(
    episodes: list[dict],
    thresholds: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    final_errors = np.asarray(
        [episode["final_error_mm"] * 1000.0 for episode in episodes],
        dtype=float,
    )
    minimum_errors = np.asarray(
        [
            episode.get("minimum_error_mm", episode["final_error_mm"]) * 1000.0
            for episode in episodes
        ],
        dtype=float,
    )
    steps = np.asarray([episode.get("steps", 0) for episode in episodes])
    threshold_metrics = {}
    for offset, threshold_um in enumerate(thresholds):
        key = f"{threshold_um:g}"
        successes = np.asarray(
            [_success(episode, key) for episode in episodes],
            dtype=float,
        )
        threshold_metrics[key] = {
            "successes": int(successes.sum()),
            "episodes": len(successes),
            "success_rate": float(successes.mean()),
            "target_cluster_bootstrap_95": _cluster_bootstrap_interval(
                episodes,
                threshold=key,
                samples=bootstrap_samples,
                seed=seed + offset,
            ),
        }
    return {
        "episodes": len(episodes),
        "thresholds": threshold_metrics,
        "final_error_p50_um": float(np.percentile(final_errors, 50.0)),
        "final_error_p95_um": float(np.percentile(final_errors, 95.0)),
        "minimum_error_p50_um": float(np.percentile(minimum_errors, 50.0)),
        "minimum_error_p95_um": float(np.percentile(minimum_errors, 95.0)),
        "stopping_time_p50_s": float(np.percentile(steps * 0.05, 50.0)),
        "stopping_time_p95_s": float(np.percentile(steps * 0.05, 95.0)),
        "safety_limited_steps": int(
            sum(episode.get("safety_limited_steps", 0) for episode in episodes)
        ),
        "uncontrolled_over_30mm": int(
            sum(
                str(episode.get("early_stop_reason", "")).startswith(
                    "ERROR_EXCEEDED"
                )
                for episode in episodes
            )
        ),
        "failure_rate": float(
            np.mean([not _success(episode, "20") for episode in episodes])
        ),
    }


def _paired_comparison(
    candidate: list[dict],
    baseline: list[dict],
    *,
    thresholds: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    candidate_by_key = {episode["episode_key"]: episode for episode in candidate}
    baseline_by_key = {episode["episode_key"]: episode for episode in baseline}
    keys = sorted(set(candidate_by_key).intersection(baseline_by_key))
    if not keys:
        raise ValueError("no paired Stage 4 episodes")
    target_indices = sorted(
        {int(candidate_by_key[key]["target_index"]) for key in keys}
    )
    rng = np.random.default_rng(seed)
    result = {"paired_episodes": len(keys)}
    for offset, threshold_um in enumerate(thresholds):
        key = f"{threshold_um:g}"
        candidate_success = {
            episode_key: _success(candidate_by_key[episode_key], key)
            for episode_key in keys
        }
        baseline_success = {
            episode_key: _success(baseline_by_key[episode_key], key)
            for episode_key in keys
        }
        differences = np.asarray(
            [
                float(candidate_success[episode_key])
                - float(baseline_success[episode_key])
                for episode_key in keys
            ]
        )
        bootstrap = np.empty(bootstrap_samples)
        for sample in range(bootstrap_samples):
            chosen = rng.choice(
                target_indices,
                size=len(target_indices),
                replace=True,
            )
            chosen_keys = [
                episode_key
                for episode_key in keys
                if int(candidate_by_key[episode_key]["target_index"])
                in set(int(index) for index in chosen)
            ]
            bootstrap[sample] = np.mean(
                [
                    float(candidate_by_key[episode_key]["success_by_threshold_um"][key])
                    - float(baseline_by_key[episode_key]["success_by_threshold_um"][key])
                    for episode_key in chosen_keys
                ]
            )
        error_delta = np.asarray(
            [
                (
                    candidate_by_key[episode_key]["final_error_mm"]
                    - baseline_by_key[episode_key]["final_error_mm"]
                )
                * 1000.0
                for episode_key in keys
            ]
        )
        result[key] = {
            "candidate_success_rate": float(
                np.mean(list(candidate_success.values()))
            ),
            "baseline_success_rate": float(
                np.mean(list(baseline_success.values()))
            ),
            "success_difference": float(differences.mean()),
            "success_difference_target_cluster_95": np.percentile(
                bootstrap,
                [2.5, 97.5],
            ).tolist(),
            "paired_final_error_difference_mean_um": float(
                error_delta.mean()
            ),
            "candidate_only_successes": int(
                np.sum(differences > 0)
            ),
            "baseline_only_successes": int(
                np.sum(differences < 0)
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/restricted_stage4_summary_20261106.json",
    )
    args = parser.parse_args()
    protocol = load_stage34_protocol(
        ROOT / "configs/stage34_protocol.json"
    )
    thresholds = [
        float(value) for value in protocol["stage4"]["thresholds_um"]
    ]
    bootstrap_samples = int(protocol["stage4"]["bootstrap_samples"])
    stage4_dir = ROOT / "outputs/restricted_stage4_20261106"
    distilled_paths = sorted(stage4_dir.glob("distilled_shard*.json"))
    median_paths = sorted(stage4_dir.glob("median_shard*.json"))
    if len(distilled_paths) != 3 or not median_paths:
        raise FileNotFoundError(
            "Stage 4 requires three distilled shards and median shards"
        )
    distilled = _combine(distilled_paths)
    median_by_key = {
        episode["episode_key"]: episode
        for episode in _combine(median_paths)
    }
    median = list(median_by_key.values())
    if len(distilled) != 180:
        raise ValueError(f"expected 180 distilled episodes, got {len(distilled)}")
    if len(median) != 60:
        raise ValueError(f"expected 60 median episodes, got {len(median)}")
    if len({episode["episode_key"] for episode in distilled}) != 180:
        raise ValueError("duplicate distilled episode keys")
    if len({episode["episode_key"] for episode in median}) != 60:
        raise ValueError("duplicate median episode keys")
    subset_keys = {episode["episode_key"] for episode in median}
    distilled_subset = [
        episode
        for episode in distilled
        if episode["episode_key"] in subset_keys
    ]
    if len(distilled_subset) != 60:
        raise ValueError("median subset does not pair with distilled episodes")
    latency_path = (
        ROOT / "outputs/restricted_stage3_latency_20261106.json"
    )
    paired = _paired_comparison(
        distilled_subset,
        median,
        thresholds=thresholds,
        bootstrap_samples=bootstrap_samples,
        seed=int(protocol["seed"]) + 22000,
    )
    report = {
        "schema_version": 1,
        "stage": 4,
        "completion": "COMPLETE",
        "primary_policy": protocol["stage4"]["primary_policy"],
        "median_control_policy": protocol["stage4"]["median_control_policy"],
        "primary": _metrics(
            distilled,
            thresholds,
            bootstrap_samples=bootstrap_samples,
            seed=int(protocol["seed"]) + 20000,
        ),
        "median_control": _metrics(
            median,
            thresholds,
            bootstrap_samples=bootstrap_samples,
            seed=int(protocol["seed"]) + 21000,
        ),
        "paired_on_hash_selected_subset": paired,
        "latency": (
            _load(latency_path) if latency_path.is_file() else None
        ),
        "test_target_isolation_used": True,
        "median_subset_selection_used_results": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (*distilled_paths, *median_paths)
        },
    }
    report["acceptance"] = {
        "all_180_primary_episodes_present": len(distilled) == 180,
        "all_60_median_control_episodes_present": len(median) == 60,
        "same_target_same_seed_pairing": len(distilled_subset) == 60,
        "no_uncontrolled_over_30mm": (
            report["primary"]["uncontrolled_over_30mm"] == 0
        ),
        "no_safety_limited_steps": (
            report["primary"]["safety_limited_steps"] == 0
        ),
    }
    report["performance_gates"] = {
        "primary_20um_success_rate_at_least_0_95": (
            report["primary"]["thresholds"]["20"]["success_rate"] >= 0.95
        ),
        "distilled_vs_seven_median_noninferior_at_20um": (
            paired["20"]["success_difference_target_cluster_95"][0]
            >= -0.05
        ),
    }
    report["result"] = (
        "PASS"
        if all(report["acceptance"].values())
        and all(report["performance_gates"].values())
        else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
