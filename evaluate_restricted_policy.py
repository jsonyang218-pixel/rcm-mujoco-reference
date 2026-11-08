"""Closed-loop evaluation for a restricted action policy in the v9 scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from restricted_perception.config import load_config
from restricted_perception.image_ops import degrade_image
from restricted_perception.online import RestrictedPerceptionAdapter
from restricted_perception.policy import RestrictedActionMLP
from restricted_perception.sidecar import _kinematic_tips
from restricted_perception.stage2_policy import Stage2ActionPolicy
from restricted_perception.stage3_policy import Stage3ResidualActionPolicy
from rcm_reference import Reference
from rcm_scene_v9 import RCMSceneV9
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import visual_servo_settings

ROOT = Path(__file__).resolve().parent
if str(ROOT / "lerobot_training") not in sys.path:
    sys.path.insert(0, str(ROOT / "lerobot_training"))


TASK_TO_DEG_DEG_MM = np.array(
    [180.0 / np.pi, 180.0 / np.pi, 1000.0],
    dtype=np.float64,
)
DEFAULT_ACTION_LIMIT_DEG_DEG_MM = np.array(
    [0.2897999882698059, 0.14434358477592468, 0.04271206632256508],
    dtype=np.float64,
)


def task_to_state_abc_mm(task):
    return np.asarray(task, dtype=np.float64) * TASK_TO_DEG_DEG_MM


def state_abc_mm_to_task(state):
    return np.asarray(state, dtype=np.float64) / TASK_TO_DEG_DEG_MM


class LiveRestrictedPolicy:
    def __init__(
        self,
        checkpoint_path: Path,
        encoder_checkpoint: Path,
        image_resolution: int,
        gaussian_blur_radius: float,
        center_occlusion_fraction: float,
        device: torch.device,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.stage3_residual = (
            checkpoint.get("policy_type")
            == "restricted_stage3_residual_distilled"
        )
        self.stage2 = checkpoint.get("policy_type") in (
            "restricted_stage2",
            "restricted_stage3_distilled",
        )
        if self.stage3_residual:
            self.model = Stage3ResidualActionPolicy(
                vision_dim=int(self.config["vision_dim"]),
                state_dim=int(self.config["state_dim"]),
                restricted_dim=int(self.config["restricted_dim"]),
                use_velocity=bool(self.config["use_velocity"]),
                residual_limit=self.config.get(
                    "residual_limit",
                    [0.005, 0.003, 0.001],
                ),
            ).to(device)
            self.model.base.load_state_dict(
                checkpoint["base_model_state_dict"]
            )
            self.model.base.set_normalization(
                **checkpoint["normalization"]
            )
            self.model.residual.load_state_dict(
                checkpoint["residual_state_dict"]
            )
        elif self.stage2:
            self.model = Stage2ActionPolicy(
                vision_dim=int(self.config["vision_dim"]),
                state_dim=int(self.config["state_dim"]),
                restricted_dim=int(self.config["restricted_dim"]),
                use_velocity=bool(self.config["use_velocity"]),
                use_stop_head=bool(self.config["use_stop_head"]),
                residual_distance_scale=self.config["residual_distance_scale"],
            ).to(device)
        else:
            self.model = RestrictedActionMLP(
                vision_dim=int(self.config["image_dim"]),
                state_dim=int(self.config["state_dim"]),
                restricted_dim=int(self.config["restricted_dim"]),
                n_action_steps=int(self.config["n_action_steps"]),
                use_image=bool(self.config["use_image"]),
            ).to(device)
        if not self.stage3_residual:
            self.model.set_normalization(**checkpoint["normalization"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device
        self.image_resolution = int(image_resolution)
        self.gaussian_blur_radius = float(gaussian_blur_radius)
        self.center_occlusion_fraction = float(center_occlusion_fraction)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        from lerobot_training.prepare_vision_feature_cache import load_feature_encoder

        self.encoder, _ = load_feature_encoder(encoder_checkpoint, device)

    def action(
        self,
        rgb: np.ndarray,
        state_abc: np.ndarray,
        restricted_vector: np.ndarray,
        velocity_abc: np.ndarray | None = None,
    ) -> np.ndarray:
        features = self.encode_features(
            rgb,
            state_abc,
            restricted_vector,
            velocity_abc,
        )
        return self.action_from_features(features)

    def encode_features(
        self,
        rgb: np.ndarray,
        state_abc: np.ndarray,
        restricted_vector: np.ndarray,
        velocity_abc: np.ndarray | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
            image = degrade_image(
                image,
                resolution=self.image_resolution,
                gaussian_blur_radius=self.gaussian_blur_radius,
                center_occlusion_fraction=self.center_occlusion_fraction,
            )
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            vision = self.encoder(image_tensor)
            parts = [
                vision,
                torch.from_numpy(state_abc.astype(np.float32)).unsqueeze(0).to(self.device),
                torch.from_numpy(restricted_vector).unsqueeze(0).to(self.device),
            ]
            if (
                (self.stage2 or self.stage3_residual)
                and self.config["use_velocity"]
            ):
                if velocity_abc is None:
                    raise ValueError("this policy requires velocity input")
                parts.append(
                    torch.from_numpy(velocity_abc.astype(np.float32))
                    .unsqueeze(0)
                    .to(self.device)
                )
            features = torch.cat(parts, dim=1)
            return features

    def action_from_features(self, features: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            output = self.model(features)
            if self.stage2 or self.stage3_residual:
                action, stop_logit = output
                if (
                    stop_logit is not None
                    and float(torch.sigmoid(stop_logit[0, 0])) >= 0.5
                ):
                    action = torch.zeros_like(action)
                return action[0].cpu().numpy()
            return output[0, 0].cpu().numpy()


class MismatchVectorBank:
    def __init__(
        self,
        data_dir: Path,
        seed: int,
        sidecar_dir: Path | None = None,
    ):
        with np.load(data_dir / "data" / "dataset.npz") as dataset:
            self.episodes = dataset["episode_index"].astype(np.int64)
            fine = dataset["oct_navigation"].astype(np.float32)
            fine_valid = dataset["oct_navigation_valid"].astype(bool)
        sidecar_path = (
            data_dir / "restricted_sidecar" / "restricted_sidecar.npz"
            if sidecar_dir is None
            else sidecar_dir / "restricted_sidecar.npz"
        )
        with np.load(sidecar_path) as sidecar:
            from restricted_perception.sidecar import restricted_feature_matrix

            features = restricted_feature_matrix(sidecar)
        fused = features[:, :6].copy()
        coarse_valid = features[:, 6] > 0.5
        fine_distance = np.linalg.norm(fine, axis=1)
        fused_distance = np.linalg.norm(fused[:, :3], axis=1)
        self.distance = np.where(fine_valid, fine_distance, fused_distance)
        keep = fine_valid | coarse_valid
        self.features = features[keep]
        self.episodes = self.episodes[keep]
        self.distance = self.distance[keep]
        self.bins = np.floor(self.distance / 0.1).astype(np.int64)
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def query(self, distance_mm: float) -> np.ndarray:
        bin_index = int(np.floor(max(0.0, float(distance_mm)) / 0.1))
        if bin_index not in self.cache:
            candidates = np.flatnonzero(self.bins == bin_index)
            if not len(candidates):
                nearest = np.argsort(np.abs(self.distance - float(distance_mm)))[:64]
                candidates = nearest
            selected = int(self.rng.choice(candidates))
            self.cache[bin_index] = self.features[selected, :6].copy()
        return self.cache[bin_index]


def evaluate_episode(
    policy: LiveRestrictedPolicy,
    reference: Reference,
    perception_config,
    target_robot_mm: np.ndarray,
    *,
    seed: int,
    max_steps: int,
    threshold_mm: float,
    vector_ablation: str,
    start_error_mm: float,
    mismatch_bank: MismatchVectorBank | None,
    stagnation_steps: int = 0,
) -> dict:
    scene = RCMSceneV9()
    sensor = None
    try:
        target_world = scene.robot_to_world(target_robot_mm)
        scene.set_target_world(target_world)
        target_task_units = scene.target_task(validate=True) * TASK_TO_DEG_DEG_MM
        start_task_units = target_task_units.copy()
        start_task_units[2] -= float(start_error_mm)
        scene.set_task(state_abc_mm_to_task(start_task_units))
        settings = visual_servo_settings(scene)
        settings["perception_backend"] = "strict_rgbd"
        sensor = StrictRGBDSensor(scene, settings)
        scene.enable_motor_execution()
        scene.reset_motor_time(0.0)
        motor = scene.motor_execution
        adapter = RestrictedPerceptionAdapter(
            perception_config,
            target_robot_mm=target_robot_mm,
            seed=seed,
            episode_id=f"closed_loop_{seed}",
        )
        success_streak = 0
        actions = []
        errors = []
        states = []
        success = False
        safety_limited_steps = 0
        safety_limited_streak = 0
        early_stop_reason = None
        best_error_mm = float("inf")
        last_improvement_step = 0
        for step_index in range(max_steps):
            sensor.observation_time_s = motor.time_s
            observation = sensor.observe()
            navigation = observation.get("oct_navigation_depth")
            delta_xyz_m = None if navigation is None else navigation.get("delta_xyz_m")
            fine_valid = (
                delta_xyz_m is not None
                and len(delta_xyz_m) == 3
                and np.all(np.isfinite(delta_xyz_m))
            )
            fine_camera = (
                np.asarray(delta_xyz_m, dtype=float) * 1000.0
                if fine_valid
                else None
            )
            state_abc = task_to_state_abc_mm(scene.current_task())
            tip_robot = _kinematic_tips(state_abc[None, :], reference)[0]
            restricted = adapter.observe(
                state=state_abc,
                kinematic_tip_robot_mm=tip_robot,
                fine_goal_vector_camera_mm=fine_camera,
                fine_valid=fine_valid,
                timestamp_s=motor.time_s,
            )
            restricted_vector = restricted.policy_vector().copy()
            if vector_ablation == "zero":
                restricted_vector[:6] = 0.0
            elif vector_ablation == "drop_valid":
                restricted_vector[6:8] = 0.0
            elif vector_ablation == "mismatch":
                if mismatch_bank is None:
                    raise ValueError("mismatch ablation requires a mismatch vector bank")
                measured_vector = (
                    restricted.fine_goal_vector_robot_mm
                    if restricted.fine_goal_vector_robot_mm is not None
                    else restricted.coarse_goal_vector_robot_mm
                )
                distance_mm = (
                    0.0
                    if measured_vector is None
                    else float(np.linalg.norm(measured_vector))
                )
                restricted_vector[:6] = mismatch_bank.query(distance_mm)
            velocity_abc = (
                np.asarray(scene.motor_execution.actual_velocity, dtype=np.float64)
                * TASK_TO_DEG_DEG_MM
            )
            action = policy.action(
                observation["rgb"],
                state_abc,
                restricted_vector,
                velocity_abc,
            )
            if restricted.fallback_state.value == "PRECISION_HOLD":
                action = np.zeros(3, dtype=np.float32)
            action = np.clip(
                np.asarray(action, dtype=np.float64),
                -DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
                DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
            )
            state = scene.current_task()
            current_tip = scene.site_mm("needle_tip")
            error_mm = float(np.linalg.norm(current_tip - target_world))
            errors.append(error_mm)
            states.append(state_abc.tolist())
            actions.append(action.tolist())
            if error_mm < best_error_mm - 0.0005:
                best_error_mm = error_mm
                last_improvement_step = step_index
            if (
                int(stagnation_steps) > 0
                and step_index - last_improvement_step >= int(stagnation_steps)
            ):
                early_stop_reason = (
                    f"STAGNATION_{int(stagnation_steps)}_STEPS"
                )
                break
            if error_mm > 30.0:
                early_stop_reason = "ERROR_EXCEEDED_30MM"
                break
            velocity = np.asarray(motor.actual_velocity) * TASK_TO_DEG_DEG_MM
            if (
                error_mm < threshold_mm
                and np.all(np.abs(velocity) <= np.array([0.05, 0.05, 0.01]))
            ):
                success_streak += 1
            else:
                success_streak = 0
            if success_streak >= 3:
                success = True
                break
            new_task = state_abc_mm_to_task(state_abc + action)
            try:
                scene.set_task(new_task, motor_dt=0.05)
                safety_limited_streak = 0
            except ValueError:
                safety_limited_steps += 1
                safety_limited_streak += 1
                if safety_limited_streak >= 20:
                    early_stop_reason = "SAFETY_LIMITED_20_STEPS"
                    break
            if safety_limited_streak >= 20:
                break
        final_tip = scene.site_mm("needle_tip")
        return {
            "success": success,
            "steps": len(errors),
            "final_error_mm": float(np.linalg.norm(final_tip - target_world)),
            "minimum_error_mm": float(np.min(errors)),
            "safety_limited_steps": safety_limited_steps,
            "early_stop_reason": early_stop_reason,
            "seed": int(seed),
            "target_robot_mm": target_robot_mm.tolist(),
            "target_world_mm": target_world.tolist(),
            "vector_ablation": vector_ablation,
            "states": states,
            "actions": actions,
            "errors_mm": errors,
        }
    finally:
        if sensor is not None:
            sensor.close()
        scene.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "final_test"), default="development")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--threshold-um", type=float, default=20.0)
    parser.add_argument("--image-resolution", type=int, default=112)
    parser.add_argument("--gaussian-blur-radius", type=float, default=0.0)
    parser.add_argument("--center-occlusion-fraction", type=float, default=0.0)
    parser.add_argument(
        "--vector-ablation",
        choices=("none", "zero", "drop_valid", "mismatch"),
        default="none",
    )
    parser.add_argument("--start-error-um", type=float, default=100.0)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=None)
    parser.add_argument(
        "--mismatch-data-dir",
        type=Path,
        default=Path("data/lerobot_rcm_val_recovery_full_224"),
    )
    parser.add_argument("--mismatch-sidecar-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261207)
    parser.add_argument("--stagnation-steps", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    key = "development_targets" if args.split == "development" else "final_test_targets"
    target_start = int(args.target_start)
    target_end = (
        len(manifest[key])
        if args.target_end is None
        else min(int(args.target_end), len(manifest[key]))
    )
    targets = manifest[key][target_start:target_end]
    targets = targets[: args.num_episodes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = LiveRestrictedPolicy(
        args.checkpoint,
        args.encoder_checkpoint,
        args.image_resolution,
        args.gaussian_blur_radius,
        args.center_occlusion_fraction,
        device,
    )
    reference = Reference()
    perception_config = load_config(
        Path(__file__).resolve().parent / "configs" / "restricted_perception.json"
    )
    mismatch_bank = (
        MismatchVectorBank(
            args.mismatch_data_dir,
            args.seed,
            args.mismatch_sidecar_dir,
        )
        if args.vector_ablation == "mismatch"
        else None
    )
    episodes = []
    existing_by_id = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing_by_id = {
            episode.get("target_id"): episode
            for episode in existing.get("episodes", [])
            if episode.get("target_id") is not None
        }
    for index, target in enumerate(targets):
        target_index = target_start + index
        target_id = target.get("id", f"target_{target_index:03d}")
        if target_id in existing_by_id:
            episodes.append(existing_by_id[target_id])
            continue
        try:
            episode = evaluate_episode(
                policy,
                reference,
                perception_config,
                np.asarray(target["point_mm"], dtype=float),
                seed=args.seed + target_index,
                max_steps=args.max_steps,
                threshold_mm=args.threshold_um / 1000.0,
                vector_ablation=args.vector_ablation,
                start_error_mm=args.start_error_um / 1000.0,
                mismatch_bank=mismatch_bank,
                stagnation_steps=args.stagnation_steps,
            )
        except Exception as error:
            episode = {
                "success": False,
                "steps": 0,
                "final_error_mm": 30.0,
                "minimum_error_mm": 30.0,
                "safety_limited_steps": 0,
                "early_stop_reason": f"EXCEPTION:{type(error).__name__}:{error}",
                "seed": args.seed + target_index,
                "target_robot_mm": target["point_mm"],
                "vector_ablation": args.vector_ablation,
                "states": [],
                "actions": [],
                "errors_mm": [],
            }
        episode["target_id"] = target_id
        episode["target_index"] = target_index
        episodes.append(episode)
        partial = {
            "result": "IN_PROGRESS",
            "episodes": episodes,
            "completed": len(episodes),
            "target_start": target_start,
            "target_end": target_end,
            "vector_ablation": args.vector_ablation,
            "threshold_um": args.threshold_um,
            "start_error_um": args.start_error_um,
            "stagnation_steps": args.stagnation_steps,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    success_count = sum(int(episode["success"]) for episode in episodes)
    result = {
        "result": "PASS" if success_count == len(episodes) else "FAIL",
        "episodes": episodes,
        "successes": success_count,
        "success_rate": success_count / len(episodes),
        "vector_ablation": args.vector_ablation,
        "threshold_um": args.threshold_um,
        "start_error_um": args.start_error_um,
        "stagnation_steps": args.stagnation_steps,
        "target_start": target_start,
        "target_end": target_end,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": result["result"], "success_rate": result["success_rate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
