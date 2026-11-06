"""
将 v9_motor_dataset 转换为 LeRobot 格式

用法:
    python convert_to_lerobot.py --input-dir runs/v9_motor_dataset --output-dir data/lerobot_rcm
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from tqdm import tqdm


class V9ToLeRobotConverter:
    """转换 v9 数据集到 LeRobot 格式"""

    TASK_RAD_RAD_M_TO_DEG_DEG_MM = np.array(
        [180.0 / math.pi, 180.0 / math.pi, 1000.0],
        dtype=np.float64,
    )

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        reuse_existing_images: bool = False,
        image_workers: int = 1,
        action_source: str = "executed",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reuse_existing_images = bool(reuse_existing_images)
        self.image_workers = int(image_workers)
        if self.image_workers < 1:
            raise ValueError("image_workers 必须至少为 1")
        if action_source not in ("executed", "teacher"):
            raise ValueError("action_source must be 'executed' or 'teacher'")
        self.action_source = action_source
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LeRobot 数据结构
        self.episodes = []
        self.frames = []

    def load_episode(self, episode_dir: Path) -> Dict:
        """加载单个 episode 的数据"""

        dagger_file = episode_dir / "dagger_samples.jsonl"
        dagger_samples = []
        if dagger_file.is_file():
            with open(dagger_file, "r", encoding="utf-8") as f:
                dagger_samples = [json.loads(line) for line in f if line.strip()]

        # 读取 motor_steps.jsonl (动作序列)
        motor_steps = []
        motor_steps_file = episode_dir / "motor_steps.jsonl"
        if not dagger_samples and not motor_steps_file.is_file():
            raise FileNotFoundError(f"缺少电机轨迹文件: {motor_steps_file}")
        if motor_steps_file.is_file():
            with open(motor_steps_file, 'r', encoding='utf-8') as f:
                for line in f:
                    motor_steps.append(json.loads(line))

        # 读取 frame_audit.jsonl (观测序列)
        frame_audits = []
        frame_audit_file = episode_dir / "frame_audit.jsonl"
        if not dagger_samples and not frame_audit_file.is_file():
            raise FileNotFoundError(f"缺少观测审计文件: {frame_audit_file}")
        if frame_audit_file.is_file():
            with open(frame_audit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    frame_audits.append(json.loads(line))

        oct_observations = []
        oct_file = episode_dir / "oct_observations.jsonl"
        if oct_file.is_file():
            with open(oct_file, 'r', encoding='utf-8') as f:
                for line in f:
                    oct_observations.append(json.loads(line))

        # 读取控制报告
        report_candidates = (
            episode_dir / "visual_servo_report.json",
            episode_dir / "control_report.json",
        )
        report_file = next(
            (path for path in report_candidates if path.is_file()),
            None,
        )
        report = {}
        if report_file is None:
            raise FileNotFoundError(
                f"缺少控制报告: {episode_dir}/visual_servo_report.json"
            )
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # 图像目录
        images_dir = episode_dir / "images"
        if not images_dir.exists():
            # 尝试其他可能的位置
            alt_images_dir = self.input_dir / "dataset" / "images" / episode_dir.name
            if alt_images_dir.exists():
                images_dir = alt_images_dir
        if not images_dir.is_dir():
            raise FileNotFoundError(f"缺少图像目录: {episode_dir}/images")

        return {
            'motor_steps': motor_steps,
            'frame_audits': frame_audits,
            'dagger_samples': dagger_samples,
            'oct_observations': oct_observations,
            'report': report,
            'images_dir': images_dir,
            'episode_dir': episode_dir
        }

    def extract_frames(self, episode_data: Dict, episode_idx: int) -> List[Dict]:
        """从 episode 提取帧数据"""

        frames = []
        dagger_samples = episode_data.get('dagger_samples', [])
        if dagger_samples:
            if self.action_source != 'teacher':
                raise ValueError(
                    "DAgger datasets require --action-source teacher"
                )
            images_dir = episode_data['images_dir']
            for source_index, row in enumerate(dagger_samples):
                if not row.get('teacher_valid'):
                    continue
                teacher_delta = row.get('teacher_action_deg_deg_mm')
                if teacher_delta is None:
                    continue
                state = self._finite_task_vector(
                    row['state_A_deg_B_deg_C_mm'],
                    'state_A_deg_B_deg_C_mm',
                )
                delta_action = self._finite_task_vector(
                    teacher_delta,
                    'teacher_action_deg_deg_mm',
                )
                action = state + delta_action
                image_path = images_dir / Path(row['image_path']).name
                if not image_path.is_file():
                    raise FileNotFoundError(f"图像不存在: {image_path}")
                oct_delta = self._finite_task_vector(
                    row['oct_navigation_mm'],
                    'oct_navigation_mm',
                )
                frames.append({
                    'episode_index': episode_idx,
                    'frame_index': len(frames),
                    'timestamp': float(row['timestamp_s']),
                    'source_frame_index': int(source_index),
                    'image_path': str(image_path),
                    'state': state.tolist(),
                    'action': action.tolist(),
                    'delta_action': delta_action.tolist(),
                    'oct_navigation_mm': oct_delta.tolist(),
                    'oct_navigation_valid': bool(
                        row.get('oct_navigation_valid')
                    ),
                })
            return frames
        motor_steps = episode_data['motor_steps']
        frame_audits = episode_data['frame_audits']
        oct_observations = episode_data['oct_observations']
        images_dir = episode_data['images_dir']

        if len(motor_steps) != max(0, len(frame_audits) - 1):
            raise ValueError(
                "motor_steps 与 frame_audits 数量不匹配: "
                f"{len(motor_steps)} != {len(frame_audits) - 1}"
            )
        oct_by_iteration = {
            int(row['iteration']): row
            for row in oct_observations
            if 'iteration' in row
        }

        # 遍历观测帧
        for frame_idx, audit in enumerate(frame_audits):
            # 图像路径
            frame_id = f"frame_{frame_idx:06d}"
            image_path = images_dir / f"{frame_id}.png"

            if not image_path.exists():
                raise FileNotFoundError(f"图像不存在: {image_path}")

            # 当前状态 (从 frame_audit 提取)
            if 'state_A_deg_B_deg_C_mm' in audit:
                state = self._finite_task_vector(
                    audit['state_A_deg_B_deg_C_mm'],
                    'state_A_deg_B_deg_C_mm',
                )
            else:
                raise KeyError(
                    "frame_audit 缺少权威状态字段 state_A_deg_B_deg_C_mm"
                )

            # 对应的动作 (从 motor_steps 提取)
            if frame_idx < len(motor_steps):
                motor_step = motor_steps[frame_idx]
                action = self._action_from_motor_step(motor_step)
            else:
                # 最后一帧：保持当前位置
                action = state

            # 计算增量动作 (更适合学习)
            delta_action = action - state
            if not np.all(np.isfinite(delta_action)):
                raise ValueError(f"frame {frame_idx} 的增量动作不是有限数")

            oct_row = oct_by_iteration.get(frame_idx)
            navigation = (
                None if oct_row is None
                else oct_row.get('navigation_depth')
            )
            oct_delta_m = None if navigation is None else navigation.get('delta_xyz_m')
            oct_valid = bool(
                oct_delta_m is not None
                and len(oct_delta_m) == 3
                and all(value is not None for value in oct_delta_m)
            )
            oct_navigation_mm = (
                (np.asarray(oct_delta_m, dtype=np.float64) * 1000.0).tolist()
                if oct_valid else [0.0, 0.0, 0.0]
            )

            frames.append({
                'episode_index': episode_idx,
                'frame_index': frame_idx,
                'timestamp': audit.get('timestamp_s', frame_idx * 0.05),
                'image_path': str(image_path),
                'state': state.tolist(),
                'action': action.tolist(),
                'delta_action': delta_action.tolist(),
                'oct_navigation_mm': oct_navigation_mm,
                'oct_navigation_valid': oct_valid,
            })

        return frames

    @staticmethod
    def _finite_task_vector(values, field_name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError(f"{field_name} 必须是三个数，实际为 {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field_name} 包含非有限数: {array.tolist()}")
        return array

    @classmethod
    def _action_from_motor_step(cls, motor_step: Dict) -> np.ndarray:
        """读取执行器实际跟踪的任务目标，并统一为 deg/deg/mm。"""
        if 'commanded_task_rad_rad_m' in motor_step:
            values = cls._finite_task_vector(
                motor_step['commanded_task_rad_rad_m'],
                'commanded_task_rad_rad_m',
            )
            return values * cls.TASK_RAD_RAD_M_TO_DEG_DEG_MM
        if 'requested_task_rad_rad_m' in motor_step:
            values = cls._finite_task_vector(
                motor_step['requested_task_rad_rad_m'],
                'requested_task_rad_rad_m',
            )
            return values * cls.TASK_RAD_RAD_M_TO_DEG_DEG_MM
        if 'requested_task' in motor_step:
            return cls._finite_task_vector(
                motor_step['requested_task'],
                'requested_task',
            )
        raise KeyError(
            "motor_step 缺少动作字段 commanded_task_rad_rad_m、"
            "requested_task_rad_rad_m 或 requested_task"
        )

    def convert(self):
        """执行完整转换"""

        print(f"开始转换: {self.input_dir} -> {self.output_dir}")
        self.episodes = []
        self.frames = []

        # 查找所有 episode
        episode_dirs = sorted(self.input_dir.glob("episode_*"))

        if len(episode_dirs) == 0:
            # 尝试 dataset 子目录
            alt_dir = self.input_dir / "dataset"
            if alt_dir.exists():
                episode_dirs = sorted(alt_dir.glob("episode_*"))

        print(f"发现 {len(episode_dirs)} 个 episodes")

        all_frames = []

        for ep_idx, ep_dir in enumerate(tqdm(episode_dirs, desc="处理 episodes")):
            try:
                # 加载 episode 数据
                episode_data = self.load_episode(ep_dir)

                # 提取帧
                frames = self.extract_frames(episode_data, ep_idx)

                print(f"  Episode {ep_idx}: {len(frames)} 帧")

                all_frames.extend(frames)

                # 保存 episode 元数据
                self.episodes.append({
                    'episode_index': ep_idx,
                    'num_frames': len(frames),
                    'success': episode_data['report'].get('result') == 'PASS',
                    'final_error_um': episode_data['report'].get(
                        'final_tip_target_error_um'
                    ),
                })

            except Exception as e:
                raise RuntimeError(f"Episode {ep_idx} 转换失败: {e}") from e

        print(f"\n总共 {len(all_frames)} 帧")

        # 保存为 LeRobot 格式
        self.save_lerobot_format(all_frames)

        print(f"转换完成! 输出: {self.output_dir}")

    def save_lerobot_format(self, frames: List[Dict]):
        """保存为 LeRobot 兼容格式"""

        # 创建目录结构
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        if self.image_workers == 1:
            image_paths = [
                self._prepare_image(frame)
                for frame in tqdm(frames, desc="保存数据")
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.image_workers) as executor:
                image_paths = list(tqdm(
                    executor.map(self._prepare_image, frames),
                    total=len(frames),
                    desc="保存数据",
                ))

        observations = [
            {'image': image_path, 'state': frame['state']}
            for frame, image_path in zip(frames, image_paths)
        ]
        actions = [frame['delta_action'] for frame in frames]
        oct_navigation = [frame['oct_navigation_mm'] for frame in frames]
        oct_navigation_valid = [
            frame['oct_navigation_valid'] for frame in frames
        ]
        episode_indices = [frame['episode_index'] for frame in frames]
        frame_indices = [frame['frame_index'] for frame in frames]

        states = np.array([obs['state'] for obs in observations], dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        oct_navigation = np.asarray(oct_navigation, dtype=np.float32)
        oct_navigation_valid = np.asarray(oct_navigation_valid, dtype=np.bool_)
        episode_indices_array = np.array(episode_indices, dtype=np.int32)
        frame_indices_array = np.array(frame_indices, dtype=np.int32)
        self._validate_dataset_arrays(
            states,
            actions,
            oct_navigation,
            oct_navigation_valid,
            episode_indices_array,
            frame_indices_array,
        )

        # 保存为 numpy 格式
        np.savez(
            self.output_dir / "data" / "dataset.npz",
            episode_index=episode_indices_array,
            frame_index=frame_indices_array,
            action=actions,
            state=states,
            oct_navigation=oct_navigation,
            oct_navigation_valid=oct_navigation_valid,
        )

        # 保存图像路径列表
        with open(self.output_dir / "data" / "image_paths.json", 'w') as f:
            json.dump([obs['image'] for obs in observations], f, indent=2)

        # 保存元数据
        metadata = {
            'num_episodes': len(self.episodes),
            'num_frames': len(frames),
            'fps': 20,  # 50ms 控制周期 = 20 Hz
            'observation_space': {
                'image': {'shape': [480, 640, 3], 'dtype': 'uint8'},
                'state': {'shape': [3], 'dtype': 'float32', 'names': ['A_deg', 'B_deg', 'C_mm']},
                'oct_navigation': {
                    'shape': [3],
                    'dtype': 'float32',
                    'units': 'mm',
                    'frame': 'camera_optical',
                    'definition': 'target_minus_tip',
                },
            },
            'action_space': {
                'shape': [3],
                'dtype': 'float32',
                'names': ['dA_deg', 'dB_deg', 'dC_mm'],
                'type': 'delta',  # 增量动作
            },
            'action_source': self.action_source,
            'episodes': self.episodes,
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n数据统计:")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  总帧数: {len(frames)}")
        print(f"  平均每集: {len(frames) / max(len(self.episodes), 1):.1f} 帧")
        print(f"  State min/max: {states.min(axis=0)} / {states.max(axis=0)}")
        print(f"  Action min/max: {actions.min(axis=0)} / {actions.max(axis=0)}")
        print(f"  OCT navigation valid: {int(oct_navigation_valid.sum())}/{len(frames)}")

    def _prepare_image(self, frame: Dict) -> str:
        """缩放单帧，返回相对图像路径。"""
        src_image = Path(frame['image_path'])
        dst_image = (
            self.output_dir
            / "images"
            / f"ep{frame['episode_index']:06d}_frame{frame['frame_index']:06d}.png"
        )
        if not src_image.is_file():
            raise FileNotFoundError(f"图像不存在: {src_image}")
        if not self._image_is_reusable(dst_image):
            with Image.open(src_image) as img:
                img_resized = img.convert('RGB').resize(
                    (640, 480),
                    Image.Resampling.BILINEAR,
                )
                img_resized.save(dst_image)
        return str(dst_image.relative_to(self.output_dir))

    def _image_is_reusable(self, destination: Path) -> bool:
        if not self.reuse_existing_images:
            return False
        if not destination.is_file() or destination.stat().st_size == 0:
            return False
        try:
            with Image.open(destination) as image:
                return image.size == (640, 480)
        except (OSError, ValueError):
            return False

    @staticmethod
    def _validate_dataset_arrays(
        states: np.ndarray,
        actions: np.ndarray,
        oct_navigation: np.ndarray,
        oct_navigation_valid: np.ndarray,
        episode_indices: np.ndarray,
        frame_indices: np.ndarray,
    ) -> None:
        if len(states) == 0:
            raise ValueError("转换结果没有样本")
        for name, array in (
            ('state', states),
            ('action', actions),
            ('episode_index', episode_indices),
            ('frame_index', frame_indices),
        ):
            if array.shape[0] != len(states):
                raise ValueError(f"{name} 样本数与 state 不一致")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError("state 或 action 包含非有限数")
        if np.all(np.ptp(states, axis=0) == 0):
            raise ValueError("state 所有维度均为常量，拒绝生成无效数据集")
        if np.max(np.abs(actions)) == 0:
            raise ValueError("action 全为 0，拒绝生成无效数据集")
        if oct_navigation.shape != (len(states), 3):
            raise ValueError("oct_navigation 形状必须为 (N, 3)")
        if oct_navigation_valid.shape != (len(states),):
            raise ValueError("oct_navigation_valid 形状必须为 (N,)")
        if not np.any(oct_navigation_valid):
            raise ValueError("没有可用的 OCT navigation 样本")
        for episode_index in np.unique(episode_indices):
            frame_index = frame_indices[episode_indices == episode_index]
            expected = np.arange(len(frame_index), dtype=frame_index.dtype)
            if not np.array_equal(frame_index, expected):
                raise ValueError(
                    f"episode {int(episode_index)} 的 frame_index 不连续"
                )


def main():
    parser = argparse.ArgumentParser(description="转换 v9 数据集到 LeRobot 格式")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入目录 (e.g., runs/v9_motor_dataset)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录 (e.g., data/lerobot_rcm)")
    parser.add_argument(
        "--reuse-existing-images",
        action="store_true",
        help="仅重建数值数据；复用输出目录中已有的非空图像",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="并行图像转换线程数（默认最多 8）",
    )
    parser.add_argument(
        "--action-source",
        choices=("executed", "teacher"),
        default="executed",
        help="Use executed protocol targets or DAgger teacher labels",
    )

    args = parser.parse_args()

    converter = V9ToLeRobotConverter(
        args.input_dir,
        args.output_dir,
        reuse_existing_images=args.reuse_existing_images,
        image_workers=args.image_workers,
        action_source=args.action_source,
    )
    converter.convert()


if __name__ == "__main__":
    main()
