"""Freeze nearest integer-encoder endpoints for evaluation targets."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcm.scene_v9 import RCMSceneV9


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split", default="development_targets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-radius", type=int, default=2)
    parser.add_argument("--maximum-target-shift-um", type=float, default=0.35)
    args = parser.parse_args()
    if args.search_radius < 0:
        raise ValueError("search radius must be nonnegative")

    source = json.loads(
        args.source_manifest.read_text(encoding="utf-8")
    )
    targets = source[args.split]
    frozen = []
    failures = []
    scene = RCMSceneV9()
    try:
        scene.set_task(np.zeros(3))
        zero_task = scene.current_task().copy()
        for target in targets:
            nominal_robot_mm = np.asarray(
                target["point_mm"],
                dtype=np.float64,
            )
            nominal_world_mm = scene.robot_to_world(nominal_robot_mm)
            scene.set_target_world(nominal_world_mm)
            target_task = scene.target_task(validate=True)
            base_counts = np.rint(
                np.asarray(
                    scene.encoders(target_task),
                    dtype=np.float64,
                )
            ).astype(np.int64)
            best = None
            for delta in product(
                range(-args.search_radius, args.search_radius + 1),
                repeat=3,
            ):
                counts = base_counts + np.asarray(delta, dtype=np.int64)
                try:
                    candidate_task = scene.task_from_encoders(
                        counts,
                        reference_task=zero_task,
                    )
                    scene.set_task(candidate_task)
                    candidate_tip_world_mm = scene.site_mm(
                        "needle_tip"
                    )
                    error_um = float(
                        np.linalg.norm(
                            candidate_tip_world_mm - nominal_world_mm
                        )
                        * 1000.0
                    )
                except ValueError:
                    continue
                if best is None or error_um < best["error_um"]:
                    best = {
                        "error_um": error_um,
                        "counts": counts.copy(),
                        "tip_world_mm": (
                            candidate_tip_world_mm.copy()
                        ),
                    }
            scene.set_task(zero_task)
            if best is None:
                failures.append({
                    "id": target.get("id"),
                    "reason": "NO_VALID_INTEGER_ENDPOINT",
                })
                continue
            if best["error_um"] > args.maximum_target_shift_um:
                failures.append({
                    "id": target.get("id"),
                    "reason": "TARGET_SHIFT_EXCEEDS_LIMIT",
                    "error_um": best["error_um"],
                })
            reachable_world_mm = best["tip_world_mm"]
            reachable_robot_mm = scene.world_to_robot(
                reachable_world_mm
            )
            frozen.append({
                **target,
                "nominal_target_robot_mm": (
                    nominal_robot_mm.tolist()
                ),
                "nominal_target_world_mm": (
                    nominal_world_mm.tolist()
                ),
                "reachable_target_robot_mm": (
                    reachable_robot_mm.tolist()
                ),
                "reachable_target_world_mm": (
                    reachable_world_mm.tolist()
                ),
                "reachable_encoder_counts_ABC": (
                    best["counts"].tolist()
                ),
                "target_shift_um": best["error_um"],
                "target_shift_vector_um": (
                    (reachable_world_mm - nominal_world_mm) * 1000.0
                ).tolist(),
                "quantization_search_radius_counts": (
                    args.search_radius
                ),
            })
    finally:
        scene.close()

    if failures:
        raise ValueError(
            "reachable target audit failed: "
            + json.dumps(failures, ensure_ascii=False)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "source_split": args.split,
        "frozen_before_model_evaluation": True,
        "maximum_target_shift_um": float(
            args.maximum_target_shift_um
        ),
        "search_radius_counts": int(args.search_radius),
        "targets": frozen,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest_path = args.output.with_name(
        args.output.name + ".sha256.json"
    )
    digest_path.write_text(
        json.dumps(
            {args.output.name: _sha256(args.output)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shifts = np.asarray(
        [target["target_shift_um"] for target in frozen],
        dtype=np.float64,
    )
    print(json.dumps({
        "result": "PASS",
        "targets": len(frozen),
        "maximum_shift_um": float(shifts.max()),
        "median_shift_um": float(np.median(shifts)),
        "minimum_shift_um": float(shifts.min()),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
