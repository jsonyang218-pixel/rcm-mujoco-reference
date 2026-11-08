"""
在 MuJoCo 仿真环境中评估训练好的策略

用法:
    python eval_policy.py --model outputs/diffusion_policy/best_model.pth --episodes 10
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcm_scene_v9 import RCMSceneV9
from rcm_fundus import make_random_fundus_targets
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import visual_servo_settings
from train_diffusion_policy import SimpleDiffusionPolicy
from train_oct_guided_regression import (
    GuidedOCTResidualPolicy,
    proportional_guided_action,
)
from rcm_reference import Reference


TASK_TO_DEG_DEG_MM = np.array(
    [180.0 / math.pi, 180.0 / math.pi, 1000.0],
    dtype=np.float64,
)

# Per-axis maximum of the final recovery training set actions.  The previous
# 2 deg / 2 deg / 0.2 mm allowance let one bad diffusion sample move far beyond
# anything demonstrated by the teacher.
DEFAULT_ACTION_LIMIT_DEG_DEG_MM = (
    0.2897999882698059,
    0.14434358477592468,
    0.04271206632256508,
)


def task_to_state_abc_mm(task):
    """Convert scene task [A_rad, B_rad, C_m] to dataset [A_deg, B_deg, C_mm]."""
    task = np.asarray(task, dtype=np.float64)
    if task.shape != (3,):
        raise ValueError(f"scene task must have shape (3,), got {task.shape}")
    return task * TASK_TO_DEG_DEG_MM


def state_abc_mm_to_task(state):
    """Convert dataset [A_deg, B_deg, C_mm] to scene task [A_rad, B_rad, C_m]."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (3,):
        raise ValueError(f"policy state must have shape (3,), got {state.shape}")
    return state / TASK_TO_DEG_DEG_MM


class PolicyEvaluator:
    """策略评估器"""

    def __init__(self, model_path: str, device='cuda', num_action_samples=1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_action_samples = int(num_action_samples)
        if self.num_action_samples < 1:
            raise ValueError("num_action_samples must be at least 1")

        # 加载模型
        print(f"加载模型: {model_path}")
        self.checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        config = self.checkpoint.get('config', {})
        self.policy_type = self.checkpoint.get(
            'policy_type',
            'oct_diffusion_policy',
        )
        self.is_regression = self.policy_type == 'state_oct_mse_regression'
        self.is_guided_regression = (
            self.policy_type == 'oct_guided_residual_regression'
        )
        self.n_obs_steps = int(config.get('n_obs_steps', 2))
        self.n_action_steps = int(config.get('n_action_steps', 8))
        self.num_diffusion_iters = int(config.get('num_diffusion_iters', 10))
        self.diffusion_schedule = config.get('diffusion_schedule', 'linear')
        self.use_oct = bool(config.get('use_oct', False))
        self.guided_oct = bool(config.get('guided_oct_mode', False))
        self.history_mode = config.get('history_mode', 'none')
        oct_dim = 4 if self.use_oct else 0

        if self.is_regression:
            from train_oct_regression import OCTStateRegressionPolicy
            self.n_obs_steps = 1
            self.use_oct = True
            self.model = OCTStateRegressionPolicy(
                history_mode=self.history_mode,
                state_mean=config.get('state_mean'),
                state_std=config.get('state_std'),
                oct_mean=config.get('oct_mean'),
                oct_std=config.get('oct_std'),
                velocity_mean=config.get('velocity_mean'),
                velocity_std=config.get('velocity_std'),
                previous_action_mean=config.get('previous_action_mean'),
                previous_action_std=config.get('previous_action_std'),
                action_mean=config.get('action_mean'),
                action_std=config.get('action_std'),
            ).to(self.device)
        elif self.is_guided_regression:
            self.n_obs_steps = 1
            self.use_oct = True
            self.model = GuidedOCTResidualPolicy(
                state_mean=config['state_mean'],
                state_std=config['state_std'],
                log_distance_mean=config['log_distance_mean'],
                log_distance_std=config['log_distance_std'],
                base_mean=config['base_mean'],
                base_std=config['base_std'],
                far_residual_limit=config['far_residual_limit'],
                full_effect_distance_mm=config.get(
                    'full_effect_distance_mm',
                    1.0,
                ),
                near_fraction=config.get('near_fraction', 0.1),
                hidden_dim=config.get('hidden_dim', 128),
            ).to(self.device)
        else:
            self.model = SimpleDiffusionPolicy(
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                num_diffusion_iters=self.num_diffusion_iters,
                diffusion_schedule=self.diffusion_schedule,
                pred_x0_clip=config.get('pred_x0_clip'),
                oct_dim=oct_dim,
                oct_scale=config.get('oct_scale', [1.0, 1.0, 1.0, 1.0]),
                state_mean=config.get('state_mean', [0.0, 0.0, 0.0]),
                state_std=config.get('state_std', [1.0, 1.0, 1.0]),
                oct_mean=config.get('oct_mean', [0.0, 0.0, 0.0, 0.0]),
                oct_std=config.get('oct_std', [1.0, 1.0, 1.0, 1.0]),
                action_mean=config.get('action_mean', [0.0, 0.0, 0.0]),
                action_std=config.get('action_std', [1.0, 1.0, 1.0]),
                vision_mode=config.get('vision_mode', 'global'),
                guided_oct_mode=self.guided_oct,
                guided_base_mean=config.get('guided_base_mean'),
                guided_base_std=config.get('guided_base_std'),
                log_distance_mean=config.get('log_distance_mean', 0.0),
                log_distance_std=config.get('log_distance_std', 1.0),
                far_residual_limit=config.get('far_residual_limit'),
                full_effect_distance_mm=config.get(
                    'full_effect_distance_mm',
                    1.0,
                ),
                near_fraction=config.get('near_fraction', 0.1),
            ).to(self.device)

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        train_loss = self.checkpoint.get('train_loss', self.checkpoint.get('loss'))
        val_loss = self.checkpoint.get('val_loss', float('nan'))
        train_loss_text = (
            f"{float(train_loss):.6f}" if train_loss is not None else "n/a"
        )
        val_loss_text = (
            f"{float(val_loss):.6f}" if val_loss is not None else "n/a"
        )
        print(
            "模型加载完成 "
            f"(type={self.policy_type}, epoch={self.checkpoint['epoch']}, "
            f"train_loss={train_loss_text}, "
            f"val_loss={val_loss_text})"
        )

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 观测历史缓冲
        self.obs_buffer = []
        self.action_generator = torch.Generator(device=self.device)
        self.regression_previous_state = None
        self.regression_previous_action = np.zeros(3, dtype=np.float32)
        self.reference = None
        self.camera_rotation = None
        if self.guided_oct or self.is_guided_regression:
            self.reference = Reference()
            _, self.camera_rotation = self.reference.camera_pose()

    def reset_buffer(self, seed=None):
        """重置观测缓冲"""
        self.obs_buffer = []
        self.regression_previous_state = None
        self.regression_previous_action = np.zeros(3, dtype=np.float32)
        if seed is not None:
            self.action_generator.manual_seed(int(seed))

    def record_applied_action(self, action):
        """Record the command actually executed for the next regression step."""
        self.regression_previous_action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

    def preprocess_image(self, rgb_array):
        """预处理图像"""
        # rgb_array: [H, W, 3] numpy array
        img = Image.fromarray(rgb_array.astype(np.uint8))
        # Training images were first converted from 1920x1440 to 640x480 and
        # then resized to ResNet input. Preserve that exact preprocessing.
        img = img.resize((640, 480), Image.Resampling.BILINEAR)
        return self.transform(img)  # [3, 224, 224]

    def get_action(self, rgb, state, oct_feature):
        """
        从观测获取动作

        Args:
            rgb: [H, W, 3] numpy array
            state: [3] numpy array [A_deg, B_deg, C_mm]

        Returns:
            action: [3] numpy array [dA_deg, dB_deg, dC_mm]
        """
        # 预处理
        img_tensor = (
            None
            if self.is_regression or self.is_guided_regression
            else self.preprocess_image(rgb)
        )
        state_tensor = torch.from_numpy(state).float()

        # 添加到缓冲
        self.obs_buffer.append({
            'image': img_tensor,
            'state': state_tensor,
            'oct': torch.from_numpy(oct_feature).float(),
        })

        # 如果缓冲不足，返回零动作
        if len(self.obs_buffer) < self.n_obs_steps:
            return np.zeros(3, dtype=np.float32)

        # 保持固定长度
        if len(self.obs_buffer) > self.n_obs_steps:
            self.obs_buffer.pop(0)

        # 准备输入
        obs_states = torch.stack([obs['state'] for obs in self.obs_buffer])  # [T, 3]
        obs_oct = torch.stack([obs['oct'] for obs in self.obs_buffer])
        obs_states = obs_states.unsqueeze(0).to(self.device)  # [1, T, 3]
        obs_oct = obs_oct.unsqueeze(0).to(self.device)
        if self.is_regression:
            state_array = np.asarray(state, dtype=np.float64)
            if self.regression_previous_state is None:
                velocity = np.zeros(3, dtype=np.float32)
            else:
                velocity = (
                    (state_array - self.regression_previous_state) / 0.05
                ).astype(np.float32)
            with torch.no_grad():
                action = self.model(
                    obs_states[:, -1],
                    obs_oct[:, -1],
                    velocity=torch.from_numpy(velocity).unsqueeze(0).to(self.device),
                    previous_action=torch.from_numpy(
                        self.regression_previous_action.copy()
                    ).unsqueeze(0).to(self.device),
                )[0].cpu().numpy()
            self.regression_previous_state = state_array.copy()
            return action
        if self.is_guided_regression:
            base = proportional_guided_action(
                self.reference,
                np.asarray(state, dtype=np.float64),
                np.asarray(oct_feature[:3], dtype=np.float64),
                self.camera_rotation,
                valid=bool(oct_feature[3] > 0.5),
                gain=float(self.checkpoint['config']['gain']),
                saturation=np.asarray(
                    self.checkpoint['config']['base_saturation'],
                    dtype=np.float64,
                ),
                relative_damping=float(
                    self.checkpoint['config'].get(
                        'relative_damping',
                        1e-3,
                    )
                ),
            )
            with torch.no_grad():
                action = self.model(
                    obs_states[:, -1],
                    obs_oct[:, -1],
                    torch.from_numpy(base.astype(np.float32))
                    .unsqueeze(0)
                    .to(self.device),
                )[0][0].cpu().numpy()
            return action

        obs_images = torch.stack([obs['image'] for obs in self.obs_buffer])
        obs_images = obs_images.unsqueeze(0).to(self.device)  # [1, T, 3, 224, 224]
        guided_base = None
        if self.guided_oct:
            guided_base_array = proportional_guided_action(
                self.reference,
                np.asarray(state, dtype=np.float64),
                np.asarray(oct_feature[:3], dtype=np.float64),
                self.camera_rotation,
                valid=bool(oct_feature[3] > 0.5),
                gain=float(self.checkpoint['config']['guided_gain']),
                saturation=np.asarray(
                    self.checkpoint['config']['guided_saturation'],
                    dtype=np.float64,
                ),
                relative_damping=float(
                    self.checkpoint['config'].get(
                        'guided_relative_damping',
                        1e-3,
                    )
                ),
            )
            guided_base = torch.from_numpy(
                guided_base_array.astype(np.float32)
            ).unsqueeze(0).to(self.device)

        # 推理。逐次采样后取中位数，避免单个低质量扩散采样直接推动闭环。
        # 顺序采样而非常量 batch repeat，避免多路并行评估时增加峰值显存。
        sample_count = self.num_action_samples
        with torch.no_grad():
            sampled_first_actions = []
            for _ in range(sample_count):
                actions = self.model(
                    obs_images,
                    obs_states,
                    obs_oct=obs_oct if self.use_oct else None,
                    generator=self.action_generator,
                    guided_base=guided_base,
                )  # [1, n_action_steps, 3]
                sampled_first_actions.append(actions[0, 0])
        first_actions = torch.stack(sampled_first_actions, dim=0)
        if sample_count > 1:
            action = first_actions.median(dim=0).values.cpu().numpy()
        else:
            action = first_actions[0].cpu().numpy()

        return action

    def evaluate_episode(
        self,
        scene: RCMSceneV9,
        target_world_mm,
        max_steps=1600,
        error_threshold_mm=0.020,
        control_period_s=0.05,
        action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
        settled_velocity_deg_deg_mm_per_s=(0.05, 0.05, 0.01),
        early_stop_error_mm=30.0,
        max_consecutive_safety_limited=20,
        seed=0,
        sensor=None,
        start_error_mm=None,
        stagnation_steps=0,
    ):
        """
        评估单个 episode

        Args:
            scene: RCMSceneV9 实例
            target_world_mm: [3] world target position [x, y, z] in mm
            max_steps: 最大步数
            error_threshold_mm: 误差阈值 (mm)

        Returns:
            result: dict with success, steps, final_error, trajectory
        """
        self.reset_buffer(seed=seed)
        if start_error_mm is not None:
            target_task_units = scene.target_task(validate=True) * TASK_TO_DEG_DEG_MM
            start_task_units = target_task_units.copy()
            start_task_units[2] -= float(start_error_mm)
            scene.set_task(start_task_units / TASK_TO_DEG_DEG_MM)
        if scene.motor_execution is None:
            scene.enable_motor_execution()
            scene.reset_motor_time(0.0)
        motor = scene.motor_execution
        if self.use_oct and sensor is None:
            raise ValueError("OCT-enabled policy requires a strict RGB-D sensor")
        if not np.isclose(motor.control_period_s, control_period_s):
            raise ValueError(
                "Evaluation control period and MotorBackend period differ: "
                f"{control_period_s} != {motor.control_period_s}"
            )
        action_limit = np.asarray(action_limit_deg_deg_mm, dtype=np.float64)
        if action_limit.shape != (3,) or not np.all(np.isfinite(action_limit)):
            raise ValueError("action_limit_deg_deg_mm must contain three finite values")
        if np.any(action_limit <= 0):
            raise ValueError("action limits must be positive")
        if not np.isfinite(float(early_stop_error_mm)) or early_stop_error_mm <= 0:
            raise ValueError("early_stop_error_mm must be finite and positive")
        if int(max_consecutive_safety_limited) < 1:
            raise ValueError("max_consecutive_safety_limited must be positive")
        settled_velocity_limit = np.asarray(
            settled_velocity_deg_deg_mm_per_s,
            dtype=np.float64,
        )

        trajectory = {
            'states': [],
            'actions': [],
            'errors': [],
            'velocities': [],
            'oct_navigation_mm': [],
            'oct_navigation_norm_mm': [],
            'oct_navigation_valid': [],
            'simulation_time_s': [],
        }

        success = False
        consecutive_success = 0
        required_consecutive = 3  # 需要连续3帧达标
        target_world_mm = np.asarray(target_world_mm, dtype=np.float64)
        safety_limited_steps = 0
        safety_limited_streak = 0
        last_safety_error = None
        early_stop_reason = None
        best_error_mm = float('inf')
        last_improvement_step = 0

        for step in range(max_steps):
            # 获取与训练数据同源的控制观测。OCT navigation 由当前 RGB-D
            # 感知生成；真值只用于下方独立误差评价。
            oct_feature = np.zeros(4, dtype=np.float32)
            if sensor is None:
                rgb = scene.render_camera()
            else:
                sensor.observation_time_s = motor.time_s
                observation = sensor.observe()
                rgb = observation['rgb']
                navigation = observation.get('oct_navigation_depth')
                delta_xyz_m = None if navigation is None else navigation.get('delta_xyz_m')
                if (delta_xyz_m is not None
                        and len(delta_xyz_m) == 3
                        and np.all(np.isfinite(delta_xyz_m))):
                    oct_feature[:3] = np.asarray(delta_xyz_m, dtype=np.float64) * 1000.0
                    oct_feature[3] = 1.0
            state = scene.current_task()  # [A_rad, B_rad, C_m]
            state_abc = task_to_state_abc_mm(state)  # [A_deg, B_deg, C_mm]
            velocity_abc = (
                np.asarray(motor.actual_velocity, dtype=np.float64)
                * TASK_TO_DEG_DEG_MM
            )

            # Ground truth is only used for evaluation, never for selecting the action.
            current_tip = scene.site_mm('needle_tip')
            error = np.linalg.norm(current_tip - target_world_mm)

            # 记录轨迹
            trajectory['states'].append(state_abc.tolist())
            trajectory['errors'].append(float(error))
            trajectory['velocities'].append(velocity_abc.tolist())
            trajectory['oct_navigation_mm'].append(oct_feature[:3].tolist())
            trajectory['oct_navigation_norm_mm'].append(
                float(np.linalg.norm(oct_feature[:3]))
            )
            trajectory['oct_navigation_valid'].append(bool(oct_feature[3] > 0.5))
            trajectory['simulation_time_s'].append(float(motor.time_s))

            if error < best_error_mm - 0.0005:
                best_error_mm = error
                last_improvement_step = step
            if (
                int(stagnation_steps) > 0
                and step - last_improvement_step >= int(stagnation_steps)
            ):
                early_stop_reason = (
                    f"STAGNATION_{int(stagnation_steps)}_STEPS"
                )
                break

            if error > float(early_stop_error_mm):
                early_stop_reason = (
                    f"ERROR_EXCEEDED_{float(early_stop_error_mm):g}MM"
                )
                print(f"  发散早停: {error:.3f} mm")
                break

            # 检查成功
            motor_settled = bool(
                np.all(np.abs(velocity_abc) <= settled_velocity_limit)
            )
            if error < error_threshold_mm and motor_settled:
                consecutive_success += 1
                if consecutive_success >= required_consecutive:
                    success = True
                    print(f"  成功! 步数: {step}, 最终误差: {error:.3f} mm")
                    break
            else:
                consecutive_success = 0

            # 获取策略动作
            action = self.get_action(rgb, state_abc, oct_feature)  # [dA, dB, dC]

            # 限制动作增量
            action = np.clip(action, -action_limit, action_limit)
            trajectory['actions'].append(action.tolist())

            # 执行动作 (增量控制)
            new_state_abc = state_abc + action

            # 限制状态范围
            new_state_abc[0] = np.clip(new_state_abc[0], -30, 30)  # A: ±30度
            new_state_abc[1] = np.clip(new_state_abc[1], -30, 30)  # B: ±30度
            new_state_abc[2] = np.clip(new_state_abc[2], 0, 50)    # C: 0-50mm

            target_task = state_abc_mm_to_task(new_state_abc)
            applied_action = action
            applied = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0):
                trial_state = state_abc + scale * action
                trial_task = state_abc_mm_to_task(trial_state)
                try:
                    scene.set_task(trial_task, motor_dt=control_period_s)
                    applied = True
                    applied_action = scale * action
                    if scale < 1.0:
                        safety_limited_steps += 1
                        safety_limited_streak += 1
                    else:
                        safety_limited_streak = 0
                    break
                except ValueError as error:
                    last_safety_error = str(error)
            if not applied:
                trajectory['actions'].pop()
                print(f"  安全停止: {last_safety_error}")
                break
            trajectory['actions'][-1] = applied_action.tolist()
            self.record_applied_action(applied_action)
            if safety_limited_streak >= int(max_consecutive_safety_limited):
                early_stop_reason = (
                    f"SAFETY_LIMITED_{int(max_consecutive_safety_limited)}_STEPS"
                )
                print(
                    "  安全限位早停: "
                    f"{safety_limited_streak} consecutive steps"
                )
                break

        # 最终误差
        final_tip = scene.site_mm('needle_tip')
        final_error = np.linalg.norm(final_tip - target_world_mm)
        final_velocity = (
            np.asarray(motor.actual_velocity, dtype=np.float64)
            * TASK_TO_DEG_DEG_MM
        )

        result = {
            'success': success,
            'steps': len(trajectory['states']),
            'final_error_mm': float(final_error),
            'max_error_mm': max(trajectory['errors']),
            'final_state_A_deg_B_deg_C_mm': task_to_state_abc_mm(
                scene.current_task()
            ).tolist(),
            'final_velocity_deg_deg_mm_per_s': final_velocity.tolist(),
            'final_tip_world_mm': final_tip.tolist(),
            'target_world_mm': target_world_mm.tolist(),
            'motor_summary': motor.summary(),
            'ground_truth_used_for_control': False,
            'ground_truth_used_for_success': True,
            'execution_backend': 'MotorBackend',
            'state_action_units': 'A_deg,B_deg,C_mm',
            'oct_navigation_used': self.use_oct,
            'safety_limited_steps': safety_limited_steps,
            'last_safety_error': last_safety_error,
            'early_stop_reason': early_stop_reason,
            'trajectory': trajectory,
        }

        return result

    def evaluate(
        self,
        num_episodes=10,
        output_dir=None,
        random_targets=False,
        seed=42,
        max_steps=1600,
        error_threshold_mm=0.020,
        control_period_s=0.05,
        action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
        settled_velocity_deg_deg_mm_per_s=(0.05, 0.05, 0.01),
        early_stop_error_mm=30.0,
        max_consecutive_safety_limited=20,
        target_episodes=None,
        target_start=0,
        target_end=None,
        start_error_mm=None,
    ):
        """
        评估多个 episodes

        Args:
            num_episodes: episode 数量
            output_dir: 输出目录 (保存结果)
            random_targets: 是否使用随机目标
            seed: 随机种子
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        results = []
        rng = np.random.default_rng(seed)

        if target_episodes is None:
            selected_targets = [None] * int(num_episodes)
        else:
            end = len(target_episodes) if target_end is None else int(target_end)
            selected_targets = list(target_episodes[int(target_start):end])[: int(num_episodes)]
            num_episodes = len(selected_targets)
        for local_ep_idx, target_entry in enumerate(tqdm(selected_targets, desc="评估 episodes")):
            ep_idx = int(target_start) + local_ep_idx
            print(f"\nEpisode {ep_idx + 1}/{num_episodes}")

            # 创建场景
            scene = RCMSceneV9()
            sensor = None

            try:
                # 选择目标
                if target_entry is not None:
                    target_robot_mm = np.asarray(
                        target_entry["point_mm"],
                        dtype=np.float64,
                    )
                elif random_targets:
                    target_spec = make_random_fundus_targets(scene, rng, 1)
                    target_robot_mm = np.asarray(
                        target_spec['targets'][0]['point_mm'],
                        dtype=np.float64,
                    )
                else:
                    # Default P3 is stored in robot coordinates.
                    target_robot_mm = np.array([1.92, 8.478, 22.043])

                target_world_mm = scene.robot_to_world(target_robot_mm)
                scene.set_target_world(target_world_mm)
                if self.use_oct:
                    settings = visual_servo_settings(scene)
                    settings['perception_backend'] = 'strict_rgbd'
                    sensor = StrictRGBDSensor(scene, settings)
                print(
                    "  目标世界坐标: "
                    f"[{target_world_mm[0]:.3f}, {target_world_mm[1]:.3f}, "
                    f"{target_world_mm[2]:.3f}] mm"
                )

                # 评估
                result = self.evaluate_episode(
                    scene,
                    target_world_mm,
                    max_steps=max_steps,
                    error_threshold_mm=error_threshold_mm,
                    control_period_s=control_period_s,
                    action_limit_deg_deg_mm=action_limit_deg_deg_mm,
                    settled_velocity_deg_deg_mm_per_s=settled_velocity_deg_deg_mm_per_s,
                    early_stop_error_mm=early_stop_error_mm,
                    max_consecutive_safety_limited=max_consecutive_safety_limited,
                    seed=seed + ep_idx,
                    sensor=sensor,
                    start_error_mm=start_error_mm,
                )
                result['episode_index'] = ep_idx
                result['target_id'] = (
                    target_entry.get('id') if target_entry is not None else None
                )
                result['seed'] = seed + ep_idx
                result['target_robot_mm'] = target_robot_mm.tolist()

                results.append(result)
                if output_dir:
                    partial = {
                        'result': 'IN_PROGRESS',
                        'episodes': results,
                        'completed': len(results),
                    }
                    with open(output_dir / 'evaluation_results.json', 'w') as stream:
                        json.dump(partial, stream, indent=2)

                # 打印结果
                if result['success']:
                    print(f"  ✓ 成功: {result['steps']} 步, 误差 {result['final_error_mm']:.3f} mm")
                else:
                    print(f"  ✗ 失败: {result['steps']} 步, 误差 {result['final_error_mm']:.3f} mm")

            finally:
                if sensor is not None:
                    sensor.close()
                scene.close()

        # 汇总统计
        success_rate = float(sum(r['success'] for r in results) / len(results))
        avg_steps = float(np.mean([r['steps'] for r in results]))
        avg_error = float(np.mean([r['final_error_mm'] for r in results]))

        summary = {
            'num_episodes': num_episodes,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_final_error_mm': avg_error,
            'median_final_error_mm': float(np.median([
                r['final_error_mm'] for r in results
            ])),
            'evaluation_contract': {
                'execution_backend': 'MotorBackend',
                'control_period_s': control_period_s,
                'state_action_units': 'A_deg,B_deg,C_mm',
                'rgb_preprocessing': '1920x1440 -> 640x480 -> 224x224',
                'ground_truth_used_for_control': False,
                'ground_truth_used_for_success': True,
                'require_motor_settled_for_success': True,
                'early_stop_error_mm': early_stop_error_mm,
                'max_consecutive_safety_limited': max_consecutive_safety_limited,
                'oct_navigation_used': self.use_oct,
                'action_limit_deg_deg_mm': list(action_limit_deg_deg_mm),
                'num_action_samples': self.num_action_samples,
                'action_aggregation': 'median' if self.num_action_samples > 1 else 'single_sample',
                'model_epoch': int(self.checkpoint['epoch']),
                'policy_type': self.policy_type,
                'model_train_loss': (
                    None if self.checkpoint.get(
                        'train_loss',
                        self.checkpoint.get('loss'),
                    ) is None else float(self.checkpoint.get(
                        'train_loss',
                        self.checkpoint.get('loss'),
                    ))
                ),
                'model_val_loss': (
                    None if self.checkpoint.get('val_loss') is None
                    else float(self.checkpoint.get('val_loss'))
                ),
                'model_best_normalized_mae': self.checkpoint.get(
                    'best_normalized_mae'
                ),
            },
            'episodes': results,
        }

        print("\n" + "="*60)
        print("评估结果汇总:")
        print(f"  成功率: {success_rate*100:.1f}% ({sum(r['success'] for r in results)}/{len(results)})")
        print(f"  平均步数: {avg_steps:.1f}")
        print(f"  平均最终误差: {avg_error:.3f} mm")
        print("="*60)

        # 保存结果
        if output_dir:
            with open(output_dir / "evaluation_results.json", 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n结果已保存到: {output_dir / 'evaluation_results.json'}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="评估训练好的策略")
    parser.add_argument("--model", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数量")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="输出目录")
    parser.add_argument("--random-targets", action="store_true", help="使用随机目标")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--error-threshold-mm", type=float, default=0.020)
    parser.add_argument("--control-period-s", type=float, default=0.05)
    parser.add_argument(
        "--action-limit-deg-deg-mm",
        type=float,
        nargs=3,
        default=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
    )
    parser.add_argument("--num-action-samples", type=int, default=7)
    parser.add_argument(
        "--settled-velocity-deg-deg-mm-per-s",
        type=float,
        nargs=3,
        default=(0.05, 0.05, 0.01),
    )
    parser.add_argument("--early-stop-error-mm", type=float, default=30.0)
    parser.add_argument(
        "--max-consecutive-safety-limited",
        type=int,
        default=20,
    )
    parser.add_argument("--target-manifest", type=Path, default=None)
    parser.add_argument(
        "--target-split",
        choices=("development", "final_test"),
        default="development",
    )
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=None)
    parser.add_argument("--start-error-um", type=float, default=None)

    args = parser.parse_args()
    if args.num_action_samples < 1:
        parser.error("--num-action-samples must be at least 1")
    if args.early_stop_error_mm <= 0:
        parser.error("--early-stop-error-mm must be positive")
    if args.max_consecutive_safety_limited < 1:
        parser.error("--max-consecutive-safety-limited must be positive")

    # 创建评估器
    evaluator = PolicyEvaluator(
        args.model,
        device=args.device,
        num_action_samples=args.num_action_samples,
    )

    # 评估
    target_episodes = None
    if args.target_manifest is not None:
        manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
        key = (
            "development_targets"
            if args.target_split == "development"
            else "final_test_targets"
        )
        target_episodes = manifest[key]
    evaluator.evaluate(
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        random_targets=args.random_targets,
        seed=args.seed,
        max_steps=args.max_steps,
        error_threshold_mm=args.error_threshold_mm,
        control_period_s=args.control_period_s,
        action_limit_deg_deg_mm=args.action_limit_deg_deg_mm,
        settled_velocity_deg_deg_mm_per_s=args.settled_velocity_deg_deg_mm_per_s,
        early_stop_error_mm=args.early_stop_error_mm,
        max_consecutive_safety_limited=args.max_consecutive_safety_limited,
        target_episodes=target_episodes,
        target_start=args.target_start,
        target_end=args.target_end,
        start_error_mm=(
            None if args.start_error_um is None else args.start_error_um / 1000.0
        ),
    )


if __name__ == "__main__":
    main()
