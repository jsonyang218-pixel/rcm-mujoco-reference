"""Paired target-cluster statistics for restricted-policy comparisons."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from .target_split import canonical_point


def episode_key(episode: dict) -> tuple:
    target = canonical_point(episode["target_robot_mm"])
    seed = episode.get("seed", episode.get("target_seed", 0))
    return (*target, int(seed or 0))


def paired_comparison(
    candidate_episodes: list[dict],
    baseline_episodes: list[dict],
    *,
    bootstrap_samples: int = 10000,
    seed: int = 20261208,
) -> dict:
    candidate = {episode_key(episode): episode for episode in candidate_episodes}
    baseline = {episode_key(episode): episode for episode in baseline_episodes}
    keys = sorted(set(candidate).intersection(baseline))
    if not keys:
        raise ValueError("candidate and baseline have no paired target/seed episodes")
    candidate_success = np.asarray([bool(candidate[key]["success"]) for key in keys])
    baseline_success = np.asarray([bool(baseline[key]["success"]) for key in keys])
    candidate_error = np.asarray(
        [float(candidate[key]["final_error_mm"]) * 1000.0 for key in keys]
    )
    baseline_error = np.asarray(
        [float(baseline[key]["final_error_mm"]) * 1000.0 for key in keys]
    )
    success_delta = candidate_success.astype(float) - baseline_success.astype(float)
    error_delta = candidate_error - baseline_error
    targets = np.asarray(
        [episode_key(candidate[key])[:3] for key in keys],
        dtype=float,
    )
    unique_targets = np.unique(targets, axis=0)
    rng = np.random.default_rng(seed)
    success_bootstrap = np.empty(bootstrap_samples)
    error_bootstrap = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        selected = rng.choice(unique_targets, size=len(unique_targets), replace=True)
        row_mask = np.zeros(len(keys), dtype=bool)
        for target in selected:
            row_mask |= np.all(targets == target, axis=1)
        success_bootstrap[index] = success_delta[row_mask].mean()
        error_bootstrap[index] = error_delta[row_mask].mean()
    candidate_only = int(np.sum(candidate_success & ~baseline_success))
    baseline_only = int(np.sum(~candidate_success & baseline_success))
    mcnemar_p = (
        float(binomtest(candidate_only, candidate_only + baseline_only, 0.5).pvalue)
        if candidate_only + baseline_only
        else 1.0
    )
    success_gain = float(success_delta.mean())
    error_improvement = (
        float(1.0 - candidate_error.mean() / baseline_error.mean())
        if baseline_error.mean() > 0.0
        else 0.0
    )
    return {
        "paired_episodes": len(keys),
        "paired_targets": int(len(unique_targets)),
        "candidate_success_rate": float(candidate_success.mean()),
        "baseline_success_rate": float(baseline_success.mean()),
        "success_rate_difference": success_gain,
        "success_difference_bootstrap_95": np.percentile(
            success_bootstrap, [2.5, 97.5]
        ).tolist(),
        "exact_mcnemar_p": mcnemar_p,
        "mcnemar_candidate_only": candidate_only,
        "mcnemar_baseline_only": baseline_only,
        "candidate_final_error_p95_um": float(np.percentile(candidate_error, 95.0)),
        "baseline_final_error_p95_um": float(np.percentile(baseline_error, 95.0)),
        "paired_error_difference_mean_um": float(error_delta.mean()),
        "paired_error_difference_bootstrap_95_um": np.percentile(
            error_bootstrap, [2.5, 97.5]
        ).tolist(),
        "relative_p95_error_improvement": float(
            1.0
            - np.percentile(candidate_error, 95.0)
            / np.percentile(baseline_error, 95.0)
        )
        if np.percentile(baseline_error, 95.0) > 0.0
        else 0.0,
        "relative_mean_error_improvement": error_improvement,
    }
