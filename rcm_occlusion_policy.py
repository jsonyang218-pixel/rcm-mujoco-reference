"""Occlusion strategies shared by controller tests and experiment runners.

The strategy names describe controller behaviour only. Detection of a
tool-coupled occlusion remains a separate, privileged scenario scheduler and
is never exposed to the perception/control path.
"""
from __future__ import annotations

import math


S0A_STOP_INSTANT = "S0a_stop_instant"
S0B_STOP_BRAKE = "S0b_stop_brake"
S1_OPEN_LOOP = "S1_open_loop"
S2_TARGET_MEMORY = "S2_target_memory"
S3_RETRACT_REOBSERVE = "S3_retract_reobserve"

STRATEGIES = (
    S0A_STOP_INSTANT,
    S0B_STOP_BRAKE,
    S1_OPEN_LOOP,
    S2_TARGET_MEMORY,
    S3_RETRACT_REOBSERVE,
)

LEGACY_STOP = "stop"
LEGACY_STATIC_MEMORY_REOBSERVE = "static_memory_reobserve"

ALIASES = {
    LEGACY_STOP: S0A_STOP_INSTANT,
    "stop_instant": S0A_STOP_INSTANT,
    "stop_brake": S0B_STOP_BRAKE,
    "open_loop": S1_OPEN_LOOP,
    "target_memory": S2_TARGET_MEMORY,
    "retract_reobserve": S3_RETRACT_REOBSERVE,
}

LABELS = {
    S0A_STOP_INSTANT: "S0a: immediate simulator hold",
    S0B_STOP_BRAKE: "S0b: acceleration-limited stop to zero",
    S1_OPEN_LOOP: "S1: continue the last admissible command",
    S2_TARGET_MEMORY: "S2: continue toward a registered static target",
    S3_RETRACT_REOBSERVE: "S3: brake, retract, and require fresh observations",
}

POLICY_REQUIRES_GEOMETRY = frozenset(STRATEGIES)
POLICY_CAN_USE_MOTOR = frozenset(STRATEGIES)


def normalize_occlusion_mode(value):
    """Return the canonical strategy name while preserving legacy behaviour."""
    value = str(value or LEGACY_STOP)
    if value == LEGACY_STATIC_MEMORY_REOBSERVE:
        return value
    value = ALIASES.get(value, value)
    if value not in STRATEGIES:
        allowed = ", ".join((*STRATEGIES, LEGACY_STOP, LEGACY_STATIC_MEMORY_REOBSERVE))
        raise ValueError(f"visual_servo.occlusion_handling must be one of: {allowed}")
    return value


def strategy_settings(settings):
    """Validate and normalize strategy tuning values.

    Durations are controller limits, not injected occlusion durations.
    """
    values = {
        "reacquire_frames": int(settings.get("occlusion_reacquire_frames", 3)),
        "max_recovery_attempts": int(settings.get("occlusion_max_recovery_attempts", 3)),
        "max_open_loop_s": float(settings.get("occlusion_open_loop_max_s", 2.2)),
        "memory_max_age_s": float(settings.get("occlusion_memory_max_age_s", 2.2)),
        "memory_max_travel_mm": float(settings.get("occlusion_memory_max_travel_mm", .8)),
        "memory_max_shift_mm": float(settings.get("occlusion_memory_max_shift_mm", .04)),
        "memory_stability_mm": float(settings.get("occlusion_memory_stability_mm", .01)),
        "retract_mm": float(settings.get("occlusion_retract_mm", 1.)),
        "retract_speed_mm_s": float(settings.get("occlusion_retract_speed_mm_s", .3)),
        "retract_tolerance_mm": float(settings.get("occlusion_retract_tolerance_mm", .005)),
        "recovery_timeout_s": float(settings.get("occlusion_recovery_timeout_s", 5.)),
    }
    if values["reacquire_frames"] < 1:
        raise ValueError("visual_servo.occlusion_reacquire_frames must be positive")
    if values["max_recovery_attempts"] < 1:
        raise ValueError(
            "visual_servo.occlusion_max_recovery_attempts must be positive")
    for key in ("max_open_loop_s", "memory_max_age_s", "memory_max_travel_mm",
                "memory_max_shift_mm", "memory_stability_mm", "retract_mm",
                "retract_speed_mm_s", "retract_tolerance_mm",
                "recovery_timeout_s"):
        if not math.isfinite(values[key]) or values[key] <= 0.:
            raise ValueError(f"visual_servo.occlusion_{key} must be positive")
    return values


def is_new_strategy(mode):
    return mode in STRATEGIES
