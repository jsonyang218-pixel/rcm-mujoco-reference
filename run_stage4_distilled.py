"""Run Stage 4 final-target evaluation for the distilled single-step policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_restricted_policy import LiveRestrictedPolicy
from restricted_perception.config import load_config
from restricted_perception.stage34 import (
    load_stage34_protocol,
    stage4_run_specs,
)
from rcm_reference import Reference
from run_stage3_robustness import _run_episode


ROOT = Path(__file__).resolve().parent


def _resource_failure(episode: dict) -> bool:
    reason = str(episode.get("early_stop_reason", ""))
    return reason.startswith("EXCEPTION:") and any(
        token in reason
        for token in ("MemoryError", "FatalError", "Could not allocate")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/stage34_protocol.json",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=ROOT / "configs/target_split_manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/restricted_stage3_residual_distilled_v7_20261106/best_model.pth",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=ROOT / "outputs/diffusion_policy_v6_guided_oct_finetune/best_model.pth",
    )
    parser.add_argument("--spec-start", type=int, default=0)
    parser.add_argument("--spec-end", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun-resource-failures", action="store_true")
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    stage4 = protocol["stage4"]
    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    specs = stage4_run_specs(manifest, protocol)
    spec_start = int(args.spec_start)
    spec_end = min(int(args.spec_end), len(specs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = LiveRestrictedPolicy(
        args.checkpoint,
        args.encoder_checkpoint,
        112,
        1.5,
        0.25,
        device,
    )
    reference = Reference()
    perception_config = load_config(
        ROOT / "configs/restricted_perception.json"
    )
    episodes = []
    existing = {}
    if args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        existing = {
            episode["episode_key"]: episode
            for episode in payload.get("episodes", [])
        }
    for spec in specs[spec_start:spec_end]:
        if (
            spec["episode_key"] in existing
            and not (
                args.rerun_resource_failures
                and _resource_failure(existing[spec["episode_key"]])
            )
        ):
            episodes.append(existing[spec["episode_key"]])
            continue
        target = manifest["final_test_targets"][spec["target_index"]]
        try:
            episode = _run_episode(
                policy_kind="student",
                student_policy=policy,
                reference=reference,
                perception_config=perception_config,
                image_settings={
                    "gaussian_blur_radius": 1.5,
                    "center_occlusion_fraction": 0.25,
                },
                teacher_gain=0.15,
                target_robot_mm=np.asarray(
                    target["point_mm"],
                    dtype=float,
                ),
                target_id=spec["target_id"],
                target_index=spec["target_index"],
                seed=spec["seed"],
                max_steps=int(stage4["max_steps"]),
                threshold_mm=float(min(stage4["thresholds_um"])) / 1000.0,
                start_error_mm=float(stage4["start_error_um"]) / 1000.0,
                scenario="stage4",
                stagnation_steps=int(stage4["stagnation_steps"]),
            )
            episode["episode_key"] = spec["episode_key"]
            episode["replicate"] = spec["replicate"]
            for key in ("states", "actions", "errors_mm"):
                episode.pop(key, None)
        except Exception as error:
            episode = {
                "episode_key": spec["episode_key"],
                "target_id": spec["target_id"],
                "target_index": spec["target_index"],
                "replicate": spec["replicate"],
                "seed": spec["seed"],
                "success": False,
                "steps": 0,
                "final_error_mm": 30.0,
                "minimum_error_mm": 30.0,
                "maximum_error_mm": 30.0,
                "safety_limited_steps": 0,
                "early_stop_reason": f"EXCEPTION:{type(error).__name__}:{error}",
                "success_by_threshold_um": {
                    "20": False,
                    "10": False,
                    "5": False,
                },
            }
        episodes.append(episode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "result": "IN_PROGRESS",
                    "policy": stage4["primary_policy"],
                    "episodes": episodes,
                    "completed": len(episodes),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    result = {
        "result": "COMPLETE",
        "policy": stage4["primary_policy"],
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_trajectory_threshold_um": 5.0,
        "success_derivation": "three consecutive settled frames below each threshold",
        "episodes": episodes,
        "completed": len(episodes),
        "spec_start": spec_start,
        "spec_end": spec_end,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": "COMPLETE", "episodes": len(episodes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
