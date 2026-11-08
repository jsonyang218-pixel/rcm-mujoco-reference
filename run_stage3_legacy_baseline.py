"""Closed-loop Stage 3 comparisons for archived diffusion and regression models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT / "lerobot_training") not in sys.path:
    sys.path.insert(0, str(ROOT / "lerobot_training"))

from eval_policy import PolicyEvaluator  # noqa: E402
from restricted_perception.stage34 import load_stage34_protocol  # noqa: E402
from rcm_scene_v9 import RCMSceneV9  # noqa: E402
from rcm_strict_rgbd import StrictRGBDSensor  # noqa: E402
from rcm_visual_servo import visual_servo_settings  # noqa: E402


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=40)
    parser.add_argument("--num-action-samples", type=int, default=7)
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    stage3 = protocol["stage3"]
    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    targets = manifest["development_targets"]
    target_start = int(args.target_start)
    target_end = min(int(args.target_end), len(targets))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = PolicyEvaluator(
        str(args.checkpoint),
        device=str(device),
        num_action_samples=int(args.num_action_samples),
    )
    episodes_by_target = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        episodes_by_target = {
            episode["target_id"]: episode
            for episode in existing.get("episodes", [])
        }
    episodes = []
    for target_index in range(target_start, target_end):
        target = targets[target_index]
        target_id = target["id"]
        if target_id in episodes_by_target:
            episodes.append(episodes_by_target[target_id])
            continue
        scene = RCMSceneV9()
        sensor = None
        try:
            target_robot_mm = np.asarray(target["point_mm"], dtype=float)
            target_world_mm = scene.robot_to_world(target_robot_mm)
            scene.set_target_world(target_world_mm)
            if evaluator.use_oct:
                settings = visual_servo_settings(scene)
                settings["perception_backend"] = "strict_rgbd"
                sensor = StrictRGBDSensor(scene, settings)
            result = evaluator.evaluate_episode(
                scene,
                target_world_mm,
                max_steps=int(stage3["max_steps"]),
                error_threshold_mm=float(stage3["threshold_um"]) / 1000.0,
                control_period_s=0.05,
                early_stop_error_mm=30.0,
                max_consecutive_safety_limited=20,
                seed=int(protocol["seed"]) + target_index,
                sensor=sensor,
                start_error_mm=float(stage3["start_error_um"]) / 1000.0,
                stagnation_steps=int(stage3["stagnation_steps"]),
            )
            result.update(
                {
                    "target_id": target_id,
                    "target_index": target_index,
                    "target_robot_mm": target_robot_mm.tolist(),
                    "seed": int(protocol["seed"]) + target_index,
                    "variant": args.variant,
                    "num_action_samples": int(args.num_action_samples),
                }
            )
        except Exception as error:
            result = {
                "success": False,
                "steps": 0,
                "final_error_mm": 30.0,
                "max_error_mm": 30.0,
                "safety_limited_steps": 0,
                "early_stop_reason": f"EXCEPTION:{type(error).__name__}:{error}",
                "target_id": target_id,
                "target_index": target_index,
                "target_robot_mm": target["point_mm"],
                "seed": int(protocol["seed"]) + target_index,
                "variant": args.variant,
                "num_action_samples": int(args.num_action_samples),
                "trajectory": {
                    "states": [],
                    "actions": [],
                    "errors": [],
                },
            }
        finally:
            if sensor is not None:
                sensor.close()
            scene.close()
        episodes.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "result": "IN_PROGRESS",
                    "variant": args.variant,
                    "episodes": episodes,
                    "completed": len(episodes),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    successes = int(sum(episode["success"] for episode in episodes))
    report = {
        "result": "COMPLETE",
        "variant": args.variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "num_action_samples": int(args.num_action_samples),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / len(episodes),
        "target_start": target_start,
        "target_end": target_end,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": report["result"],
                "variant": args.variant,
                "success_rate": report["success_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
