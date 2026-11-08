"""Verify restricted sidecar alignment, schema, and truth isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from restricted_perception.sidecar import CODE_TO_STATE


REQUIRED = {
    "episode_index": (np.int64, ()),
    "frame_index": (np.int64, ()),
    "state": (np.float32, (3,)),
    "kinematic_tip_robot_mm": (np.float32, (3,)),
    "coarse_target_robot_mm": (np.float32, (3,)),
    "coarse_goal_vector_robot_mm": (np.float32, (3,)),
    "fine_goal_vector_camera_mm": (np.float32, (3,)),
    "fine_goal_vector_robot_mm": (np.float32, (3,)),
    "coarse_valid": (np.bool_, ()),
    "fine_valid": (np.bool_, ()),
    "coarse_age_s": (np.float32, ()),
    "fine_age_s": (np.float32, ()),
    "coarse_sigma_mm": (np.float32, ()),
    "fine_sigma_mm": (np.float32, ()),
    "coarse_bias_mm": (np.float32, (3,)),
    "fine_bias_mm": (np.float32, (3,)),
    "switch_weight": (np.float32, ()),
    "fallback_state": (np.int8, ()),
    "precision_sufficient": (np.bool_, ()),
    "fused_goal_vector_robot_mm": (np.float32, (3,)),
    "action_mask_xyz": (np.float32, (3,)),
    "retraction_command_robot_mm": (np.float32, (3,)),
    "retraction_active": (np.bool_, ()),
    "precision_threshold_um": (np.float32, ()),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()
    source = np.load(args.dataset)
    sidecar = np.load(args.sidecar)
    source_count = len(source["state"])
    checks = {
        "required_fields": sorted(set(REQUIRED).intersection(sidecar.files))
        == sorted(REQUIRED),
        "truth_fields_absent": not any(
            any(token in name for token in ("target_truth", "target_world", "ground_truth"))
            for name in sidecar.files
        ),
        "row_count": source_count == len(sidecar["state"]),
        "episode_alignment": np.array_equal(
            source["episode_index"],
            sidecar["episode_index"],
        ),
        "frame_alignment": np.array_equal(source["frame_index"], sidecar["frame_index"]),
        "state_alignment": np.array_equal(source["state"], sidecar["state"]),
        "state_codes": set(np.unique(sidecar["fallback_state"]).tolist())
        <= set(CODE_TO_STATE),
        "switch_weight_range": bool(
            np.nanmin(sidecar["switch_weight"]) >= 0.0
            and np.nanmax(sidecar["switch_weight"]) <= 1.0
        ),
        "coarse_valid_finite": bool(
            np.isfinite(sidecar["coarse_target_robot_mm"][sidecar["coarse_valid"]]).all()
        ),
        "fine_valid_finite": bool(
            np.isfinite(sidecar["fine_goal_vector_robot_mm"][sidecar["fine_valid"]]).all()
        ),
    }
    for name, (dtype, trailing_shape) in REQUIRED.items():
        checks[f"dtype_and_shape:{name}"] = bool(
            name in sidecar.files
            and sidecar[name].dtype == dtype
            and sidecar[name].shape == (source_count, *trailing_shape)
        )
    result = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "dataset": str(args.dataset.resolve()),
        "sidecar": str(args.sidecar.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "sidecar_sha256": _sha256(args.sidecar),
        "samples": source_count,
        "coarse_valid_fraction": float(np.mean(sidecar["coarse_valid"])),
        "fine_valid_fraction": float(np.mean(sidecar["fine_valid"])),
        "fallback_state_counts": {
            CODE_TO_STATE[int(code)]: int(count)
            for code, count in zip(
                np.unique(sidecar["fallback_state"], return_counts=True)[0],
                np.unique(sidecar["fallback_state"], return_counts=True)[1],
            )
        },
        "checks": checks,
    }
    output = args.sidecar.parent / "verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": result["result"], "samples": source_count}, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
