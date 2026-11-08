"""Run the final-diffusion seven-sample median control on the frozen subset."""

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
from restricted_perception.stage34 import (  # noqa: E402
    load_stage34_protocol,
    median_control_specs,
)
from rcm_scene_v9 import RCMSceneV9  # noqa: E402
from rcm_strict_rgbd import StrictRGBDSensor  # noqa: E402
from rcm_visual_servo import visual_servo_settings  # noqa: E402


def _resource_failure(episode: dict) -> bool:
    reason = str(episode.get("early_stop_reason", ""))
    return reason.startswith("EXCEPTION:") and any(
        token in reason
        for token in ("MemoryError", "FatalError", "Could not allocate")
    )


def _success_by_threshold(episode: dict) -> dict:
    trajectory = episode.get("trajectory", {})
    errors = trajectory.get("errors", [])
    velocities = trajectory.get("velocities", [])
    result = {}
    for threshold_um in (20.0, 10.0, 5.0):
        streak = 0
        maximum = 0
        for error_mm, velocity in zip(errors, velocities):
            settled = np.all(
                np.abs(np.asarray(velocity, dtype=float))
                <= np.array([0.05, 0.05, 0.01])
            )
            if settled and error_mm < threshold_um / 1000.0:
                streak += 1
                maximum = max(maximum, streak)
            else:
                streak = 0
        result[f"{threshold_um:g}"] = maximum >= 3
    return result


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
        default=ROOT / "outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-start", type=int, default=0)
    parser.add_argument("--spec-end", type=int, default=60)
    parser.add_argument("--rerun-resource-failures", action="store_true")
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    stage4 = protocol["stage4"]
    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    specs = median_control_specs(manifest, protocol)
    spec_start = int(args.spec_start)
    spec_end = min(int(args.spec_end), len(specs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = PolicyEvaluator(
        str(args.checkpoint),
        device=str(device),
        num_action_samples=7,
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
        scene = RCMSceneV9()
        sensor = None
        try:
            target_robot_mm = np.asarray(
                spec["target_robot_mm"],
                dtype=float,
            )
            target_world_mm = scene.robot_to_world(target_robot_mm)
            scene.set_target_world(target_world_mm)
            settings = visual_servo_settings(scene)
            settings["perception_backend"] = "strict_rgbd"
            sensor = StrictRGBDSensor(scene, settings)
            episode = evaluator.evaluate_episode(
                scene,
                target_world_mm,
                max_steps=int(stage4["max_steps"]),
                error_threshold_mm=5.0 / 1000.0,
                control_period_s=0.05,
                early_stop_error_mm=30.0,
                max_consecutive_safety_limited=20,
                seed=spec["seed"],
                sensor=sensor,
                start_error_mm=float(stage4["start_error_um"]) / 1000.0,
                stagnation_steps=int(stage4["stagnation_steps"]),
            )
            episode["success_by_threshold_um"] = _success_by_threshold(episode)
            episode["episode_key"] = spec["episode_key"]
            episode["target_id"] = spec["target_id"]
            episode["target_index"] = spec["target_index"]
            episode["replicate"] = spec["replicate"]
            episode["seed"] = spec["seed"]
            episode["target_robot_mm"] = target_robot_mm.tolist()
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
                "safety_limited_steps": 0,
                "early_stop_reason": f"EXCEPTION:{type(error).__name__}:{error}",
                "target_robot_mm": spec["target_robot_mm"],
                "success_by_threshold_um": {
                    "20": False,
                    "10": False,
                    "5": False,
                },
                "trajectory": {
                    "states": [],
                    "actions": [],
                    "errors": [],
                    "velocities": [],
                },
            }
        finally:
            if sensor is not None:
                sensor.close()
            scene.close()
        episodes.append(episode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "result": "IN_PROGRESS",
                    "policy": stage4["median_control_policy"],
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
        "policy": stage4["median_control_policy"],
        "checkpoint": str(args.checkpoint.resolve()),
        "num_action_samples": 7,
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
