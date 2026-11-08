"""Counterfactual image/vector checks for restricted policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 20261204,
) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1 or not np.isfinite(deltas).all():
        raise ValueError("deltas must be a finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def summarize_counterfactual(
    normal_success: np.ndarray,
    ablated_success: np.ndarray,
    normal_error_um: np.ndarray,
    ablated_error_um: np.ndarray,
    *,
    target_ids: Sequence[str] | None = None,
    minimum_success_drop: float = 0.10,
    minimum_error_increase: float = 0.10,
) -> dict:
    """Summarize paired normal-vs-degraded closed-loop outcomes."""
    normal_success = np.asarray(normal_success, dtype=bool)
    ablated_success = np.asarray(ablated_success, dtype=bool)
    normal_error_um = np.asarray(normal_error_um, dtype=float)
    ablated_error_um = np.asarray(ablated_error_um, dtype=float)
    if not (
        normal_success.shape
        == ablated_success.shape
        == normal_error_um.shape
        == ablated_error_um.shape
    ):
        raise ValueError("counterfactual arrays must be paired and aligned")
    if normal_success.ndim != 1 or len(normal_success) == 0:
        raise ValueError("counterfactual arrays must be nonempty vectors")
    if not np.isfinite(normal_error_um).all() or not np.isfinite(ablated_error_um).all():
        raise ValueError("counterfactual errors must be finite")
    if target_ids is None:
        target_ids = [str(index) for index in range(len(normal_success))]
    target_ids = np.asarray(target_ids, dtype=object)
    if target_ids.shape != normal_success.shape:
        raise ValueError("target_ids must align with counterfactual arrays")
    normal_rate = float(normal_success.mean())
    ablated_rate = float(ablated_success.mean())
    success_drop = normal_rate - ablated_rate
    error_increase = (
        float(ablated_error_um.mean() / normal_error_um.mean() - 1.0)
        if normal_error_um.mean() > 0.0
        else 0.0
    )
    success_delta = normal_success.astype(float) - ablated_success.astype(float)
    success_ci = _paired_bootstrap_mean_ci(success_delta)
    error_delta = ablated_error_um - normal_error_um
    error_ci = _paired_bootstrap_mean_ci(error_delta)
    bootstrap_samples = 5000
    rng = np.random.default_rng(20261208)
    p95_delta_bootstrap = np.empty(bootstrap_samples)
    unique_targets = list(dict.fromkeys(target_ids.tolist()))
    for index in range(bootstrap_samples):
        selected = rng.choice(unique_targets, size=len(unique_targets), replace=True)
        rows = np.concatenate(
            [np.flatnonzero(target_ids == target) for target in selected]
        )
        p95_delta_bootstrap[index] = np.percentile(
            ablated_error_um[rows], 95.0
        ) - np.percentile(normal_error_um[rows], 95.0)
    normal_p95 = float(np.percentile(normal_error_um, 95.0))
    ablated_p95 = float(np.percentile(ablated_error_um, 95.0))
    p95_increase = ablated_p95 - normal_p95
    p95_relative_increase = p95_increase / normal_p95 if normal_p95 > 0.0 else 0.0
    discordant_a = int(np.sum(normal_success & ~ablated_success))
    discordant_b = int(np.sum(~normal_success & ablated_success))
    from scipy.stats import binomtest

    mcnemar_p = (
        float(binomtest(discordant_a, discordant_a + discordant_b, 0.5).pvalue)
        if discordant_a + discordant_b
        else 1.0
    )
    representation_contribution = (
        success_drop >= minimum_success_drop
        and p95_relative_increase >= minimum_error_increase
        and success_ci[0] > 0.0
        and p95_delta_bootstrap[2] > 0.0
    )
    return {
        "episodes": int(len(normal_success)),
        "normal_success_rate": normal_rate,
        "ablated_success_rate": ablated_rate,
        "success_drop": success_drop,
        "success_drop_bootstrap_95": list(success_ci),
        "near_error_increase_fraction": error_increase,
        "near_error_delta_um_bootstrap_95": list(error_ci),
        "normal_final_error_p95_um": normal_p95,
        "ablated_final_error_p95_um": ablated_p95,
        "final_error_p95_difference_um": p95_increase,
        "final_error_p95_relative_increase": p95_relative_increase,
        "final_error_p95_difference_bootstrap_95_um": np.percentile(
            p95_delta_bootstrap, [2.5, 97.5]
        ).tolist(),
        "mcnemar_discordant_normal_only": discordant_a,
        "mcnemar_discordant_ablated_only": discordant_b,
        "exact_mcnemar_p": mcnemar_p,
        "representation_contribution": bool(representation_contribution),
        "required_success_drop": float(minimum_success_drop),
        "required_error_increase": float(minimum_error_increase),
    }


def image_leakage_gate(
    zero_vector: dict,
    mismatch_vector: dict,
) -> dict:
    """Ablating a real vector must hurt in both paired success and error."""
    passed = bool(
        zero_vector.get("representation_contribution", False)
        and mismatch_vector.get("representation_contribution", False)
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "zero_vector_contributes": bool(
            zero_vector.get("representation_contribution", False)
        ),
        "mismatch_vector_contributes": bool(
            mismatch_vector.get("representation_contribution", False)
        ),
        "interpretation": (
            "image-only observations do not replace the restricted vector"
            if passed
            else "image may leak near-target information; apply one degradation at a time"
        ),
    }
