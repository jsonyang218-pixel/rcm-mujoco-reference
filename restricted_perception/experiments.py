"""Deterministic Stage 0 switching and precision-floor experiments."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable

import numpy as np

from .config import RestrictedPerceptionConfig
from .fusion import CoarseFineFusion
from .reporting import precision_floor, percentile
from .simulator import CoarseFinePerceptionSimulator


def _switch_motion(
    config: RestrictedPerceptionConfig,
    *,
    seed: int,
    switch_distance_mm: float,
    hard_switch: bool,
) -> tuple[list[np.ndarray], list[float]]:
    simulator = CoarseFinePerceptionSimulator(config, seed=seed)
    fusion = CoarseFineFusion(
        config,
        hard_switch=hard_switch,
        precision_threshold_um=1000.0,
    )
    target = np.array([0.0, 0.0, 30.0])
    start_distance = min(4.5, switch_distance_mm + 1.5)
    end_distance = max(0.15, switch_distance_mm - 1.2)
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    frame_count = 120
    for index in range(frame_count):
        fraction = index / (frame_count - 1)
        distance = start_distance + fraction * (end_distance - start_distance)
        tip = target - np.array([0.0, 0.0, distance])
        frame = simulator.update(target, tip, index * config.control_period_s)
        observation = fusion.step(frame, state=np.zeros(3))
        vectors.append(
            np.zeros(3)
            if observation.selected_goal_vector_robot_mm is None
            else observation.selected_goal_vector_robot_mm
        )
        weights.append(float(observation.switch_weight))
    return vectors, weights


def run_switch_scan(
    config: RestrictedPerceptionConfig,
    *,
    switch_distances_mm: Iterable[float] = (1.0, 2.0, 3.0),
    episode_count: int = 100,
) -> dict:
    """Measure bias-induced switching steps with sensor noise isolated.

    This is intentionally a controlled switching test: noise and drift are
    reduced to numerical zero so the result measures the switch law rather
    than Monte Carlo sensor noise. Stochastic closed-loop behavior is reported
    separately by ``run_closed_loop``.
    """
    isolated = replace(
        config,
        coarse=replace(
            config.coarse,
            noise_floor_mm=1e-9,
            noise_slope_mm_per_mm=1e-12,
            drift_innovation_sigma_mm=np.full(3, 1e-9),
        ),
        fine=replace(
            config.fine,
            noise_floor_mm=1e-9,
            noise_slope_mm_per_mm=1e-12,
            dropout_probability=0.0,
        ),
    )
    rows = []
    passing_widths = []
    candidate_widths = sorted(
        {
            config.fusion.transition_half_width_mm,
            0.5,
            1.0,
        }
    )
    for half_width in candidate_widths:
        candidate = replace(
            isolated,
            fusion=replace(isolated.fusion, transition_half_width_mm=half_width),
        )
        distance_rows = []
        all_pass = True
        for switch_distance in switch_distances_mm:
            hard_events = []
            blended_events = []
            for seed in range(episode_count):
                hard_vectors, hard_weights = _switch_motion(
                    candidate,
                    seed=seed,
                    switch_distance_mm=float(switch_distance),
                    hard_switch=True,
                )
                blended_vectors, _ = _switch_motion(
                    candidate,
                    seed=seed,
                    switch_distance_mm=float(switch_distance),
                    hard_switch=False,
                )
                event_index = next(
                    (
                        index
                        for index, weight in enumerate(hard_weights)
                        if index > 0 and weight >= 1.0
                    ),
                    None,
                )
                if event_index is None:
                    continue
                hard_events.append(
                    float(
                        np.linalg.norm(
                            hard_vectors[event_index] - hard_vectors[event_index - 1]
                        )
                    )
                )
                blended_events.append(
                    float(
                        np.linalg.norm(
                            blended_vectors[event_index]
                            - blended_vectors[event_index - 1]
                        )
                    )
                )
            if not hard_events:
                raise RuntimeError(
                    f"no hard coarse-to-fine transition at {switch_distance} mm"
                )
            hard_p95 = percentile(hard_events, 95.0)
            blended_p95 = percentile(blended_events, 95.0)
            reduction = 1.0 - blended_p95 / hard_p95 if hard_p95 > 0.0 else 0.0
            distance_pass = reduction >= 0.8
            all_pass = all_pass and distance_pass
            distance_rows.append(
                {
                    "switch_distance_mm": float(switch_distance),
                    "episodes": len(hard_events),
                    "hard_switch_step_p95_mm": hard_p95,
                    "blended_step_p95_mm": blended_p95,
                    "relative_reduction": float(reduction),
                    "pass": bool(distance_pass),
                }
            )
        row = {
            "transition_half_width_mm": float(half_width),
            "distances": distance_rows,
            "pass": bool(all_pass),
        }
        rows.append(row)
        if all_pass:
            passing_widths.append(half_width)
    selected = min(passing_widths) if passing_widths else None
    return {
        "noise_isolation": {
            "coarse_noise": "numerical_zero",
            "fine_noise": "numerical_zero",
            "fine_dropout_probability": 0.0,
        },
        "candidate_transition_half_widths_mm": candidate_widths,
        "scan": rows,
        "selected_transition_half_width_mm": selected,
        "pass": selected is not None,
    }


def _target_grid() -> list[np.ndarray]:
    targets = []
    for x in (-1.5, -0.75, 0.0, 0.75, 1.5):
        for y in (1.0, 2.5, 4.0, 5.5):
            targets.append(np.array([x, y, 22.0 + 0.08 * abs(x)]))
    return targets


def run_closed_loop(
    config: RestrictedPerceptionConfig,
    *,
    threshold_um: float,
    hard_switch: bool = False,
    seeds_per_target: int = 10,
    max_frames: int = 500,
    gain: float = 0.05,
    max_step_mm: float = 0.08,
) -> dict:
    """Run a deterministic restricted-vector servo.

    Ground truth is used only after each rollout for acceptance metrics. The
    controller receives only ``selected_goal_vector_robot_mm`` and the explicit
    safety retraction command from the restricted observation.
    """
    if threshold_um <= 0.0 or not math.isfinite(float(threshold_um)):
        raise ValueError("threshold_um must be finite and positive")
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be in (0, 1]")
    errors_um = []
    arrival_frames = []
    selected_errors_um = []
    coarse_errors_um = []
    fine_errors_um = []
    states = {}
    for target_index, target in enumerate(_target_grid()):
        for seed_offset in range(seeds_per_target):
            seed = 610000 + target_index * 100 + seed_offset
            tip = target - np.array([0.0, 0.0, 3.2])
            simulator = CoarseFinePerceptionSimulator(config, seed=seed)
            fusion = CoarseFineFusion(
                config,
                hard_switch=hard_switch,
                precision_threshold_um=threshold_um,
            )
            last_precise_goal_norm_mm = None
            first_within = None
            final_frame = max_frames
            for index in range(max_frames):
                frame = simulator.update(
                    target,
                    tip,
                    index * config.control_period_s,
                )
                observation = fusion.step(frame, state=np.zeros(3))
                states[observation.fallback_state.value] = (
                    states.get(observation.fallback_state.value, 0) + 1
                )
                true_goal = target - tip
                truth_norm_mm = float(np.linalg.norm(true_goal))
                if first_within is None and truth_norm_mm * 1000.0 <= threshold_um:
                    first_within = index
                if observation.coarse_goal_vector_robot_mm is not None:
                    coarse_errors_um.append(
                        float(
                            np.linalg.norm(
                                observation.coarse_goal_vector_robot_mm - true_goal
                            )
                            * 1000.0
                        )
                    )
                if observation.fine_goal_vector_robot_mm is not None:
                    fine_errors_um.append(
                        float(
                            np.linalg.norm(
                                observation.fine_goal_vector_robot_mm - true_goal
                            )
                            * 1000.0
                        )
                    )
                if (
                    observation.selected_goal_vector_robot_mm is not None
                    and truth_norm_mm < 0.002
                ):
                    selected_errors_um.append(
                        float(
                            np.linalg.norm(
                                observation.selected_goal_vector_robot_mm - true_goal
                            )
                            * 1000.0
                        )
                    )
                if observation.precision_sufficient:
                    precise_goal = observation.fine_goal_vector_robot_mm
                    if precise_goal is not None:
                        last_precise_goal_norm_mm = float(np.linalg.norm(precise_goal))
                if (
                    observation.fallback_state.value == "PRECISION_HOLD"
                    and last_precise_goal_norm_mm is not None
                    and last_precise_goal_norm_mm * 1000.0 <= threshold_um
                ):
                    final_frame = index
                    break
                selected = observation.selected_goal_vector_robot_mm
                if selected is not None:
                    norm = float(np.linalg.norm(selected))
                    if norm > 1e-12:
                        step = min(max_step_mm, gain * norm)
                        tip = tip + (step / norm) * selected
                tip = tip + observation.retraction_command_robot_mm
            errors_um.append(float(np.linalg.norm(target - tip) * 1000.0))
            arrival_frames.append(final_frame)
    error_array = np.asarray(errors_um)
    return {
        "threshold_um": float(threshold_um),
        "hard_switch": bool(hard_switch),
        "episodes": len(errors_um),
        "successes": int(np.sum(error_array <= threshold_um)),
        "success_rate": float(np.mean(error_array <= threshold_um)),
        "final_error_p50_um": percentile(errors_um, 50.0),
        "final_error_p95_um": percentile(errors_um, 95.0),
        "arrival_or_stop_frames_p50": percentile(arrival_frames, 50.0),
        "arrival_or_stop_frames_p95": percentile(arrival_frames, 95.0),
        "perception_error_p95_um": {
            "coarse": percentile(coarse_errors_um, 95.0),
            "fine": percentile(fine_errors_um, 95.0),
            "fused_near_target": percentile(selected_errors_um, 95.0),
        },
        "fallback_state_frames": states,
    }


def run_precision_scan(
    config: RestrictedPerceptionConfig,
    *,
    seeds_per_target: int = 10,
) -> dict:
    runs = {}
    for threshold in config.fusion.precision_levels_um:
        runs[float(threshold)] = run_closed_loop(
            config,
            threshold_um=float(threshold),
            seeds_per_target=seeds_per_target,
        )
    fused_p95 = percentile(
        [
            run["perception_error_p95_um"]["fused_near_target"]
            for run in runs.values()
        ],
        95.0,
    )
    rows = []
    for threshold in config.fusion.precision_levels_um:
        run = runs[float(threshold)]
        floor = precision_floor(
            config,
            p95_perception_error_um=fused_p95,
            deterministic_closed_loop_success=run["success_rate"],
        )
        matching = next(
            row for row in floor["levels"] if row["threshold_um"] == float(threshold)
        )
        rows.append(
            {
                **matching,
                "closed_loop_success_rate": run["success_rate"],
                "final_error_p95_um": run["final_error_p95_um"],
            }
        )
    passing = [row for row in rows if row["pass"]]
    selected = min(row["threshold_um"] for row in passing) if passing else None
    return {
        "p95_perception_error_um": fused_p95,
        "sigma_multiplier": config.fusion.precision_sigma_multiplier,
        "required_closed_loop_success": config.fusion.required_closed_loop_success,
        "levels": rows,
        "runs": {str(key): value for key, value in runs.items()},
        "selected_threshold_um": selected,
        "pass": selected is not None,
    }


def compare_hard_blend_arrival(
    config: RestrictedPerceptionConfig,
    precision_scan: dict,
) -> dict:
    threshold = precision_scan["selected_threshold_um"]
    if threshold is None:
        return {"pass": False, "reason": "no precision threshold passed"}
    hard = run_closed_loop(config, threshold_um=threshold, hard_switch=True)
    blended = precision_scan["runs"][str(float(threshold))]
    hard_s = hard["arrival_or_stop_frames_p50"] * config.control_period_s
    blended_s = blended["arrival_or_stop_frames_p50"] * config.control_period_s
    increase = blended_s - hard_s
    return {
        "threshold_um": threshold,
        "hard_p50_arrival_or_stop_s": hard_s,
        "blended_p50_arrival_or_stop_s": blended_s,
        "increase_s": increase,
        "maximum_allowed_increase_s": 0.5,
        "pass": bool(increase <= 0.5),
    }
