#!/usr/bin/env python3
"""Require exact trajectory parity between original and zero-update evaluators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--zero-update", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    original = json.loads(args.original.read_text(encoding="utf-8"))
    candidate = json.loads(args.zero_update.read_text(encoding="utf-8"))
    left = {
        int(row["episode_index"]): row
        for row in original["episodes"]
    }
    right = {
        int(row["episode_index"]): row
        for row in candidate["episodes"]
    }
    if set(left) != set(right):
        raise ValueError("trajectory episode indexes differ")
    mismatches = []
    maximum_action_difference = 0.0
    action_tolerance = 1e-5
    for episode in sorted(left):
        first, second = left[episode], right[episode]
        for field in ("success", "steps", "final_error_mm", "max_error_mm"):
            if first[field] != second[field]:
                mismatches.append(f"episode {episode}: {field}")
        for field in (
            "states",
            "errors",
            "velocities",
            "encoder_settled",
        ):
            if not np.array_equal(
                np.asarray(first["trajectory"][field]),
                np.asarray(second["trajectory"][field]),
            ):
                mismatches.append(f"episode {episode}: trajectory.{field}")
        first_actions = np.asarray(
            first["trajectory"]["actions"], dtype=np.float64
        )
        second_actions = np.asarray(
            second["trajectory"]["actions"], dtype=np.float64
        )
        difference = float(
            np.max(np.abs(first_actions - second_actions))
        )
        maximum_action_difference = max(
            maximum_action_difference, difference
        )
        if difference > action_tolerance:
            mismatches.append(
                f"episode {episode}: action difference {difference}"
            )
    if mismatches:
        raise ValueError(f"zero-update trajectory mismatch: {mismatches}")
    result = {
        "status": "PASS",
        "episodes": sorted(left),
        "kinematic_trajectories_exact": True,
        "maximum_raw_action_abs_difference": maximum_action_difference,
        "raw_action_tolerance": action_tolerance,
        "action_difference_interpretation": (
            "sub-micro-unit GPU floating-point variation; motor-quantized "
            "states, errors, velocities and settle flags are bit-exact"
        ),
        "original": str(args.original.resolve()),
        "zero_update": str(args.zero_update.resolve()),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
