"""
使用 Diffusion Policy 训练 RCM 视觉伺服策略

用法:
    python train_diffusion_policy.py --data-dir data/lerobot_rcm_train \
        --val-data-dir data/lerobot_rcm_val --output-dir outputs/diffusion_policy
"""

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import random
import sys
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be local only")
from typing import Dict, Tuple


DEFAULT_EPOCHS = 1000


def effective_epoch_count(requested_epochs):
    """Validate the requested epoch count."""
    requested_epochs = int(requested_epochs)
    if requested_epochs < 1:
        raise ValueError("requested_epochs must be at least 1")
    return requested_epochs


def diffusion_beta_schedule(timesteps, schedule="cosine", cosine_s=0.008):
    """Return an increasing beta schedule and its cumulative alpha product."""
    timesteps = int(timesteps)
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    if schedule == "linear":
        betas = torch.linspace(0.0001, 0.02, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alpha_bar = torch.cos(
            ((x / timesteps + cosine_s) / (1 + cosine_s)) * (torch.pi / 2)
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, 1e-8, 0.999).float()
    else:
        raise ValueError(f"Unknown diffusion schedule: {schedule}")
    alphas = 1 - betas
    return betas, torch.cumprod(alphas, dim=0)


def image_transform_for_dataset(data_dir, augment=False, geometric_augment=False):
    """Build the ResNet transform matching the stored image resolution."""
    metadata = json.loads(
        (Path(data_dir) / "metadata.json").read_text(encoding="utf-8")
    )
    image_shape = metadata.get("observation_space", {}).get(
        "image", {}
    ).get("shape", [480, 640, 3])
    operations = []
    if augment and geometric_augment:
        operations.append(
            transforms.RandomResizedCrop(
                224,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    elif tuple(image_shape[:2]) != (224, 224):
        operations.append(transforms.Resize((224, 224)))
    if augment:
        operations.append(
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
            )
        )
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(operations)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ThreadBatchLoader:
    """Small in-process loader for environments where process workers are blocked."""

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        num_threads=4,
        seed=0,
        pin_memory=False,
        prefetch_batches=2,
        sample_weights=None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_threads = int(num_threads)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1")
        self.prefetch_batches = int(prefetch_batches)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)
            if sample_weights.shape != (len(dataset),):
                raise ValueError(
                    "sample_weights must contain one value per dataset sample"
                )
            if not np.all(np.isfinite(sample_weights)):
                raise ValueError("sample_weights must be finite")
            if np.any(sample_weights < 0) or not np.any(sample_weights > 0):
                raise ValueError(
                    "sample_weights must be nonnegative with a positive total"
                )
        self.sample_weights = sample_weights
        self.epoch = 0

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.sample_weights is not None:
            probability = self.sample_weights / self.sample_weights.sum()
            indices = rng.choice(
                len(self.dataset),
                size=len(self.dataset),
                replace=True,
                p=probability,
            )
        else:
            indices = np.arange(len(self.dataset), dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
        self.epoch += 1
        batch_queue = queue.Queue(maxsize=self.prefetch_batches)
        stop = threading.Event()

        def produce():
            try:
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    for start in range(0, len(indices), self.batch_size):
                        if stop.is_set():
                            break
                        batch_indices = indices[start:start + self.batch_size]
                        samples = list(executor.map(
                            self.dataset.__getitem__,
                            batch_indices.tolist(),
                        ))
                        batch = default_collate(samples)
                        if self.pin_memory:
                            batch = {
                                key: value.pin_memory() if torch.is_tensor(value) else value
                                for key, value in batch.items()
                            }
                        while not stop.is_set():
                            try:
                                batch_queue.put(('batch', batch), timeout=0.1)
                                break
                            except queue.Full:
                                continue
            except BaseException as error:
                batch_queue.put(('error', error))
            finally:
                batch_queue.put(('done', None))

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            while True:
                kind, payload = batch_queue.get()
                if kind == 'done':
                    break
                if kind == 'error':
                    raise payload
                yield payload
        finally:
            stop.set()
            producer.join(timeout=5)


class TeeStream:
    """Write one stream to the console and a UTF-8 log file."""

    def __init__(self, console, log_stream):
        self.console = console
        self.log_stream = log_stream

    def write(self, text):
        written = self.console.write(text) if self.console is not None else None
        self.log_stream.write(text)
        if "\n" in text or "\r" in text:
            self.flush()
        return written

    def flush(self):
        if self.console is not None:
            self.console.flush()
        self.log_stream.flush()

    def isatty(self):
        return bool(self.console is not None and self.console.isatty())

    def fileno(self):
        if self.console is None:
            raise OSError("No console stream available")
        return self.console.fileno()

    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")


def configure_stream_logging(output_dir: Path, log_file=None):
    log_path = Path(log_file) if log_file else output_dir / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = TeeStream(original_stdout, log_stream)
    stderr_tee = TeeStream(original_stderr, log_stream)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    def close_logging():
        # Restore normal streams before closing so interpreter shutdown never
        # flushes through a TeeStream whose file is already closed.
        if sys.stdout is stdout_tee:
            sys.stdout = original_stdout
        if sys.stderr is stderr_tee:
            sys.stderr = original_stderr
        log_stream.flush()
        log_stream.close()

    atexit.register(close_logging)
    print(f"训练日志: {log_path}", flush=True)
    return log_path


class RCMDataset(Dataset):
    """RCM 数据集加载器"""

    def __init__(
        self,
        data_dir: Path,
        transform=None,
        n_obs_steps=2,
        n_action_steps=8,
        use_vision_feature_cache=False,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.n_obs_steps = n_obs_steps  # 观测历史帧数
        self.n_action_steps = n_action_steps  # 预测未来动作步数
        self.use_vision_feature_cache = bool(use_vision_feature_cache)

        # 加载元数据
        with open(data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # 加载数据
        with np.load(data_dir / "data" / "dataset.npz") as data:
            self.has_oct_navigation = 'oct_navigation' in data.files
            self.episode_indices = data['episode_index'].copy()
            self.frame_indices = data['frame_index'].copy()
            self.actions = data['action'].copy()  # [N, 3] delta actions
            self.states = data['state'].copy()    # [N, 3] current states
            if self.has_oct_navigation:
                self.oct_navigation = data['oct_navigation'].copy()
                self.oct_navigation_valid = data['oct_navigation_valid'].copy()
            else:
                self.oct_navigation = np.zeros((len(self.states), 3), dtype=np.float32)
                self.oct_navigation_valid = np.zeros(len(self.states), dtype=np.bool_)
        guided_base_path = data_dir / "data" / "guided_base_action.npy"
        guided_limit_path = data_dir / "data" / "guided_residual_limit.npy"
        self.has_guided_oct_action = (
            guided_base_path.is_file() and guided_limit_path.is_file()
        )
        if self.has_guided_oct_action:
            self.guided_base_action = np.load(
                guided_base_path,
                allow_pickle=False,
            ).astype(np.float32)
            self.guided_residual_limit = np.load(
                guided_limit_path,
                allow_pickle=False,
            ).astype(np.float32)
            if (
                self.guided_base_action.shape != self.actions.shape
                or self.guided_residual_limit.shape != self.actions.shape
            ):
                raise ValueError(
                    "guided OCT sidecars must match the action array shape"
                )
        else:
            self.guided_base_action = np.zeros_like(
                self.actions,
                dtype=np.float32,
            )
            self.guided_residual_limit = np.zeros_like(
                self.actions,
                dtype=np.float32,
            )
        feature_path = data_dir / "data" / "vision_features.npy"
        feature_metadata_path = data_dir / "data" / "vision_features.json"
        self.has_vision_feature_cache = feature_path.is_file()
        if self.use_vision_feature_cache and not self.has_vision_feature_cache:
            raise FileNotFoundError(feature_path)
        if self.has_vision_feature_cache:
            self.vision_features = np.load(
                feature_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if self.vision_features.shape != (len(self.actions), 512):
                raise ValueError(
                    "vision feature cache must have shape "
                    f"({len(self.actions)}, 512), got "
                    f"{self.vision_features.shape}"
                )
            self.vision_feature_metadata = (
                json.loads(feature_metadata_path.read_text(encoding="utf-8"))
                if feature_metadata_path.is_file()
                else {}
            )
        else:
            self.vision_features = None
            self.vision_feature_metadata = {}

        # 加载图像路径
        with open(data_dir / "data" / "image_paths.json", 'r') as f:
            self.image_paths = json.load(f)

        # 构建有效样本索引 (确保有足够的历史和未来帧)
        self.valid_indices = self._build_valid_indices()

        print(f"加载数据集: {len(self.valid_indices)} 个有效样本 (from {len(self.actions)} total frames)")

    def _build_valid_indices(self):
        """构建有效样本索引"""
        valid_indices = []

        for i in range(len(self.actions)):
            ep_idx = self.episode_indices[i]
            frame_idx = self.frame_indices[i]

            # 检查是否有足够的历史帧
            if frame_idx < self.n_obs_steps - 1:
                continue

            # 检查历史帧是否在同一 episode
            history_valid = True
            for j in range(1, self.n_obs_steps):
                if i - j < 0 or self.episode_indices[i - j] != ep_idx:
                    history_valid = False
                    break

            if not history_valid:
                continue

            # 检查是否有足够的未来动作
            if i + self.n_action_steps > len(self.actions):
                continue

            # 检查未来动作是否在同一 episode
            future_valid = True
            for j in range(1, self.n_action_steps):
                if i + j >= len(self.actions) or self.episode_indices[i + j] != ep_idx:
                    future_valid = False
                    break

            if not future_valid:
                continue

            valid_indices.append(i)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        i = self.valid_indices[idx]

        # 加载观测历史
        obs_images = []
        obs_vision_features = []
        obs_states = []
        obs_oct = []

        for j in range(self.n_obs_steps):
            hist_idx = i - (self.n_obs_steps - 1 - j)

            if self.use_vision_feature_cache:
                obs_vision_features.append(
                    torch.from_numpy(
                        np.asarray(
                            self.vision_features[hist_idx],
                            dtype=np.float16,
                        ).copy()
                    )
                )
            else:
                # 加载图像
                img_path = self.data_dir / self.image_paths[hist_idx]
                with Image.open(img_path) as source_image:
                    img = source_image.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                obs_images.append(img)

            # 加载状态
            obs_states.append(torch.from_numpy(self.states[hist_idx]))
            oct_feature = np.r_[
                self.oct_navigation[hist_idx],
                float(self.oct_navigation_valid[hist_idx]),
            ].astype(np.float32)
            obs_oct.append(torch.from_numpy(oct_feature))

        # 堆叠观测
        obs_images = (
            None
            if self.use_vision_feature_cache
            else torch.stack(obs_images)
        )
        obs_vision_features = (
            torch.stack(obs_vision_features)
            if self.use_vision_feature_cache
            else None
        )
        obs_states = torch.stack(obs_states)  # [n_obs_steps, 3]
        obs_oct = torch.stack(obs_oct)  # [n_obs_steps, 4]

        # 加载未来动作序列
        actions = []
        for j in range(self.n_action_steps):
            actions.append(torch.from_numpy(self.actions[i + j]))
        actions = torch.stack(actions)  # [n_action_steps, 3]

        sample = {
            'observation.state': obs_states,
            'observation.oct': obs_oct,
            'observation.guided_base': torch.from_numpy(
                self.guided_base_action[i].copy()
            ),
            'observation.guided_limit': torch.from_numpy(
                self.guided_residual_limit[i].copy()
            ),
            'action': actions,
        }
        if self.use_vision_feature_cache:
            sample['observation.vision_feature'] = obs_vision_features
        else:
            sample['observation.image'] = obs_images
        return sample


class SpatialSoftmax(nn.Module):
    """Map spatial feature maps to expected x/y coordinates per channel."""

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        weights = torch.softmax(
            feature.reshape(batch, channels, height * width),
            dim=-1,
        )
        y_axis = torch.linspace(
            -1.0,
            1.0,
            height,
            device=feature.device,
            dtype=feature.dtype,
        )
        x_axis = torch.linspace(
            -1.0,
            1.0,
            width,
            device=feature.device,
            dtype=feature.dtype,
        )
        y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        expected_x = (weights * x.reshape(1, 1, -1)).sum(dim=-1)
        expected_y = (weights * y.reshape(1, 1, -1)).sum(dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)


class ResNetEncoder(nn.Module):
    """ResNet18 视觉编码器"""

    def __init__(self, output_dim=512, vision_mode="global"):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if vision_mode not in ("global", "spatial_softmax"):
            raise ValueError(f"Unknown vision_mode: {vision_mode}")
        self.vision_mode = vision_mode
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        children = list(resnet.children())
        if vision_mode == "spatial_softmax":
            # Keep the 7x7 feature map; remove average pooling and classifier.
            self.encoder = nn.Sequential(*children[:-2])
            self.spatial_softmax = SpatialSoftmax()
            self.fc = nn.Linear(1024, output_dim)
        else:
            self.encoder = nn.Sequential(*children[:-1])
            self.fc = nn.Linear(512, output_dim)
        self.freeze_features = False

    def forward(self, x):
        if self.freeze_features:
            self.encoder.eval()
        # x: [B, T, C, H, W] 或 [B, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            feat = self.encoder(x)
            if self.vision_mode == "spatial_softmax":
                feat = self.spatial_softmax(feat)
            else:
                feat = feat.squeeze(-1).squeeze(-1)
            feat = self.fc(feat)  # [B*T, output_dim]
            feat = feat.reshape(B, T, -1)  # [B, T, output_dim]
        else:
            feat = self.encoder(x)
            if self.vision_mode == "spatial_softmax":
                feat = self.spatial_softmax(feat)
            else:
                feat = feat.squeeze(-1).squeeze(-1)
            feat = self.fc(feat)
        return feat


class SimpleDiffusionPolicy(nn.Module):
    """简化的 Diffusion Policy (不依赖 LeRobot)"""

    def __init__(
        self,
        vision_dim=512,
        state_dim=3,
        action_dim=3,
        n_obs_steps=2,
        n_action_steps=8,
        num_diffusion_iters=10,
        diffusion_schedule="linear",
        pred_x0_clip=None,
        oct_dim=0,
        oct_scale=None,
        state_mean=None,
        state_std=None,
        oct_mean=None,
        oct_std=None,
        action_mean=None,
        action_std=None,
        loss_weights=None,
        vision_mode="global",
        oct_noise_std_mm=0.0,
        oct_feature_dropout=0.0,
        oct_invalid_prob=0.0,
        guided_oct_mode=False,
        guided_base_mean=None,
        guided_base_std=None,
        log_distance_mean=0.0,
        log_distance_std=1.0,
        far_residual_limit=None,
        full_effect_distance_mm=1.0,
        near_fraction=0.1,
        use_cached_vision_features=False,
    ):
        super().__init__()

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_diffusion_iters = num_diffusion_iters
        self.diffusion_schedule = diffusion_schedule
        self.pred_x0_clip = pred_x0_clip
        self.oct_dim = int(oct_dim)
        self.oct_noise_std_mm = float(oct_noise_std_mm)
        self.oct_feature_dropout = float(oct_feature_dropout)
        self.oct_invalid_prob = float(oct_invalid_prob)
        self.guided_oct_mode = bool(guided_oct_mode)
        self.full_effect_distance_mm = float(full_effect_distance_mm)
        self.near_fraction = float(near_fraction)
        self.use_cached_vision_features = bool(
            use_cached_vision_features
        )
        if self.guided_oct_mode and self.oct_dim != 4:
            raise ValueError("guided OCT mode requires four OCT inputs")
        if self.full_effect_distance_mm <= 0:
            raise ValueError("full_effect_distance_mm must be positive")
        if not 0 <= self.near_fraction <= 1:
            raise ValueError("near_fraction must be in [0, 1]")
        if self.oct_noise_std_mm < 0:
            raise ValueError("oct_noise_std_mm must be nonnegative")
        if not 0 <= self.oct_feature_dropout <= 1:
            raise ValueError("oct_feature_dropout must be in [0, 1]")
        if not 0 <= self.oct_invalid_prob <= 1:
            raise ValueError("oct_invalid_prob must be in [0, 1]")
        if oct_scale is None:
            oct_scale = torch.ones(max(self.oct_dim, 1), dtype=torch.float32)
        oct_scale = torch.as_tensor(oct_scale, dtype=torch.float32)
        if self.oct_dim > 0 and oct_scale.shape != (self.oct_dim,):
            raise ValueError("oct_scale must have one value per OCT feature")
        self.register_buffer('oct_scale', oct_scale, persistent=False)
        self.register_buffer(
            'state_mean',
            torch.as_tensor(
                [0.0] * state_dim if state_mean is None else state_mean,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            'state_std',
            torch.as_tensor(
                [1.0] * state_dim if state_std is None else state_std,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            'oct_mean',
            torch.as_tensor(
                [0.0] * self.oct_dim if oct_mean is None else oct_mean,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            'oct_std',
            torch.as_tensor(
                [1.0] * self.oct_dim if oct_std is None else oct_std,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        guided_base_mean = (
            [0.0] * action_dim
            if guided_base_mean is None else guided_base_mean
        )
        guided_base_std = (
            [1.0] * action_dim
            if guided_base_std is None else guided_base_std
        )
        far_residual_limit = (
            [1.0] * action_dim
            if far_residual_limit is None else far_residual_limit
        )
        self.register_buffer(
            'guided_base_mean',
            torch.as_tensor(guided_base_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'guided_base_std',
            torch.as_tensor(guided_base_std, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'log_distance_mean',
            torch.as_tensor(log_distance_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'log_distance_std',
            torch.as_tensor(log_distance_std, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'far_residual_limit',
            torch.as_tensor(far_residual_limit, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'action_mean',
            torch.as_tensor(
                [0.0] * action_dim if action_mean is None else action_mean,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            'action_std',
            torch.as_tensor(
                [1.0] * action_dim if action_std is None else action_std,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        if loss_weights is None:
            loss_weights = torch.ones(action_dim, dtype=torch.float32)
        loss_weights = torch.as_tensor(loss_weights, dtype=torch.float32)
        if loss_weights.shape != (action_dim,) or torch.any(loss_weights <= 0):
            raise ValueError("loss_weights must contain one positive value per action axis")
        self.register_buffer('loss_weights', loss_weights, persistent=False)

        # 编码器
        self.vision_encoder = ResNetEncoder(vision_dim, vision_mode)
        state_input_dim = (
            state_dim + 8
            if self.guided_oct_mode
            else state_dim + self.oct_dim
        )
        self.state_encoder = nn.Linear(state_input_dim, 64)

        # 条件编码
        obs_dim = vision_dim + 64  # vision + state
        self.obs_fusion = nn.Sequential(
            nn.Linear(obs_dim * n_obs_steps, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        # Diffusion 去噪网络
        self.noise_pred_net = nn.Sequential(
            nn.Linear(action_dim * n_action_steps + 256 + 1, 512),  # noisy_action + condition + timestep
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * n_action_steps),
        )

        # Noise schedule (简化版 DDPM)
        betas, alphas_cumprod = diffusion_beta_schedule(
            num_diffusion_iters,
            diffusion_schedule,
        )
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def _guided_oct_components(self, oct_input):
        xyz = oct_input[..., :3]
        distance = torch.linalg.norm(xyz, dim=-1)
        direction = torch.where(
            (distance > 1e-9)[..., None],
            xyz / torch.clamp(distance[..., None], min=1e-9),
            torch.zeros_like(xyz),
        )
        log_distance = torch.log(torch.clamp(distance, min=1e-4))
        valid = oct_input[..., 3:4]
        normalized_log_distance = (
            log_distance - self.log_distance_mean
        ) / self.log_distance_std
        scale = torch.clamp(
            distance / self.full_effect_distance_mm,
            min=self.near_fraction,
            max=1.0,
        )
        scale = torch.where(
            distance > 1e-9,
            scale,
            torch.zeros_like(scale),
        )
        scale = torch.where(
            valid[..., 0] > 0.5,
            scale,
            torch.zeros_like(scale),
        )
        limit_shape = (1,) * scale.ndim + (3,)
        limits = (
            scale[..., None]
            * self.far_residual_limit.reshape(limit_shape)
        )
        return direction, normalized_log_distance, valid, limits

    def encode_observation(
        self,
        obs_image,
        obs_state,
        obs_oct=None,
        guided_base=None,
        obs_vision_feature=None,
    ):
        """编码观测"""
        # obs_image: [B, T, C, H, W]
        # obs_state: [B, T, 3]

        B = obs_state.shape[0]

        # 视觉特征
        if self.use_cached_vision_features:
            if obs_vision_feature is None:
                raise ValueError(
                    "observation.vision_feature is required when using "
                    "cached vision features"
                )
            vision_feat = obs_vision_feature.to(obs_state.dtype)
        else:
            if obs_image is None:
                raise ValueError("observation.image is required")
            vision_feat = self.vision_encoder(obs_image)  # [B, T, vision_dim]

        # 状态特征
        if self.oct_dim > 0:
            if obs_oct is None:
                raise ValueError("OCT navigation features are required")
            oct_input = obs_oct
            if self.training:
                oct_input = oct_input.clone()
                if self.oct_invalid_prob:
                    invalid = (
                        torch.rand(
                            obs_oct.shape[:-1],
                            device=obs_oct.device,
                        )
                        < self.oct_invalid_prob
                    ).unsqueeze(-1)
                    oct_input[..., :3] = oct_input[..., :3].masked_fill(
                        invalid,
                        0.0,
                    )
                    oct_input[..., 3:4] = oct_input[..., 3:4].masked_fill(
                        invalid,
                        0.0,
                    )
                if self.oct_feature_dropout:
                    dropped = (
                        torch.rand(
                            obs_oct.shape[:-1],
                            device=obs_oct.device,
                        )
                        < self.oct_feature_dropout
                    ).unsqueeze(-1)
                    oct_input[..., :3] = oct_input[..., :3].masked_fill(
                        dropped,
                        0.0,
                    )
                    oct_input[..., 3:4] = oct_input[..., 3:4].masked_fill(
                        dropped,
                        0.0,
                    )
                if self.oct_noise_std_mm:
                    valid = oct_input[..., 3:4] > 0.5
                    oct_input[..., :3] = oct_input[..., :3] + (
                        torch.randn_like(oct_input[..., :3])
                        * self.oct_noise_std_mm
                        * valid.to(oct_input.dtype)
                    )
            normalized_state = (
                obs_state - self.state_mean.reshape(1, 1, -1)
            ) / self.state_std.reshape(1, 1, -1)
            if self.guided_oct_mode:
                if guided_base is None:
                    raise ValueError("guided_base is required in guided OCT mode")
                direction, log_distance, valid, _ = (
                    self._guided_oct_components(oct_input)
                )
                normalized_base = (
                    guided_base - self.guided_base_mean.reshape(1, 3)
                ) / self.guided_base_std.reshape(1, 3)
                normalized_base = normalized_base[:, None, :].expand(
                    -1,
                    obs_state.shape[1],
                    -1,
                )
                state_input = torch.cat(
                    [
                        normalized_state,
                        direction,
                        log_distance[..., None],
                        valid,
                        normalized_base,
                    ],
                    dim=-1,
                )
            else:
                oct_xyz = (
                    oct_input[..., :3]
                    - self.oct_mean[:3].reshape(1, 1, 3)
                ) / self.oct_std[:3].reshape(1, 1, 3)
                state_input = torch.cat(
                    [
                        normalized_state,
                        torch.cat(
                            [oct_xyz, oct_input[..., 3:4]],
                            dim=-1,
                        ) * self.oct_scale.reshape(1, 1, -1),
                    ],
                    dim=-1,
                )
        else:
            state_input = (
                obs_state - self.state_mean.reshape(1, 1, -1)
            ) / self.state_std.reshape(1, 1, -1)
        state_feat = self.state_encoder(state_input)  # [B, T, 64]

        # 融合
        obs_feat = torch.cat([vision_feat, state_feat], dim=-1)  # [B, T, vision_dim+64]
        obs_feat = obs_feat.reshape(B, -1)  # [B, T*(vision_dim+64)]

        condition = self.obs_fusion(obs_feat)  # [B, 256]

        return condition

    def forward(
        self,
        obs_image,
        obs_state,
        actions=None,
        obs_oct=None,
        generator=None,
        guided_base=None,
        obs_vision_feature=None,
    ):
        """
        训练时: 返回去噪损失
        推理时: 返回预测动作
        """
        B = obs_state.shape[0]

        # 编码观测
        condition = self.encode_observation(
            obs_image,
            obs_state,
            obs_oct,
            guided_base,
            obs_vision_feature,
        )
        residual_limits = None
        if self.guided_oct_mode:
            if guided_base is None:
                raise ValueError("guided_base is required in guided OCT mode")
            _, _, _, limits = self._guided_oct_components(obs_oct[:, -1])
            residual_limits = limits
            if actions is not None:
                actions = actions - guided_base[:, None, :]
                actions = torch.maximum(
                    torch.minimum(
                        actions,
                        residual_limits[:, None, :],
                    ),
                    -residual_limits[:, None, :],
                )

        if actions is not None:
            # 训练模式: 计算去噪损失
            # actions: [B, n_action_steps, 3]
            actions_normalized = (
                actions - self.action_mean.reshape(1, 1, -1)
            ) / self.action_std.reshape(1, 1, -1)
            actions_flat = actions_normalized.reshape(B, -1)

            # 随机时间步
            timesteps = torch.randint(
                0,
                self.num_diffusion_iters,
                (B,),
                device=actions.device,
                generator=generator,
            )

            # 加噪
            noise = torch.randn(
                actions_flat.shape,
                dtype=actions_flat.dtype,
                device=actions_flat.device,
                generator=generator,
            )
            alpha_t = self.alphas_cumprod[timesteps].reshape(-1, 1)
            noisy_actions = torch.sqrt(alpha_t) * actions_flat + torch.sqrt(1 - alpha_t) * noise

            # 预测噪声
            timestep_embed = timesteps.float().reshape(-1, 1) / self.num_diffusion_iters
            noise_pred = self.noise_pred_net(
                torch.cat([noisy_actions, condition, timestep_embed], dim=-1)
            )

            # 损失
            squared = (noise_pred - noise).pow(2).reshape(
                B,
                self.n_action_steps,
                -1,
            )
            loss = (
                squared
                * self.loss_weights.reshape(1, 1, -1)
            ).mean()
            return loss

        else:
            # 推理模式: DDIM 采样
            return self.sample(
                condition,
                generator=generator,
                guided_base=guided_base,
                residual_limits=residual_limits,
            )

    @torch.no_grad()
    def sample(
        self,
        condition,
        generator=None,
        guided_base=None,
        residual_limits=None,
    ):
        """DDIM 采样"""
        B = condition.shape[0]
        device = condition.device

        # 从纯噪声开始
        actions = torch.randn(
            B,
            self.n_action_steps * 3,
            device=device,
            generator=generator,
        )

        # 反向去噪
        for t in reversed(range(self.num_diffusion_iters)):
            timestep = torch.full((B,), t, device=device, dtype=torch.long)
            timestep_embed = timestep.float().reshape(-1, 1) / self.num_diffusion_iters

            # 预测噪声
            noise_pred = self.noise_pred_net(
                torch.cat([actions, condition, timestep_embed], dim=-1)
            )

            # DDIM 更新
            alpha_t = self.alphas_cumprod[t]
            if t > 0:
                alpha_prev = self.alphas_cumprod[t - 1]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            pred_x0 = (actions - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            if self.pred_x0_clip is not None:
                pred_x0 = torch.clamp(
                    pred_x0,
                    -float(self.pred_x0_clip),
                    float(self.pred_x0_clip),
                )
            actions = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred

        # 重塑为动作序列
        actions = actions.reshape(B, self.n_action_steps, 3)
        actions = (
            actions
            * self.action_std.reshape(1, 1, -1)
            + self.action_mean.reshape(1, 1, -1)
        )
        if self.guided_oct_mode:
            if guided_base is None or residual_limits is None:
                raise ValueError(
                    "guided sampling requires base actions and residual limits"
                )
            actions = torch.maximum(
                torch.minimum(
                    actions,
                    residual_limits[:, None, :],
                ),
                -residual_limits[:, None, :],
            )
            actions = guided_base[:, None, :] + actions
        return actions


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    amp=False,
    amp_dtype=torch.float16,
    scaler=None,
):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    sample_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        obs_image = (
            batch['observation.image'].to(device, non_blocking=True)
            if 'observation.image' in batch
            else None
        )
        obs_vision_feature = (
            batch['observation.vision_feature'].to(
                device,
                non_blocking=True,
            )
            if 'observation.vision_feature' in batch
            else None
        )
        obs_state = batch['observation.state'].to(device, non_blocking=True)
        actions = batch['action'].to(device, non_blocking=True)
        obs_oct = (
            batch.get('observation.oct').to(device, non_blocking=True)
            if model.oct_dim > 0 else None
        )
        guided_base = (
            batch['observation.guided_base'].to(
                device,
                non_blocking=True,
            )
            if model.guided_oct_mode else None
        )

        # 前向传播
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp,
        ):
            loss = model(
                obs_image,
                obs_state,
                actions,
                obs_oct=obs_oct,
                guided_base=guided_base,
                obs_vision_feature=obs_vision_feature,
            )

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = actions.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        pbar.set_postfix({'loss': loss.item()})

    if sample_count == 0:
        raise RuntimeError("训练集没有可用样本")
    return total_loss / sample_count


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    device,
    seed,
    epoch=None,
    amp=False,
    amp_dtype=torch.float16,
):
    """计算确定性的验证扩散损失，不更新参数。"""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    description = "Validation" if epoch is None else f"Validation {epoch}"
    pbar = tqdm(dataloader, desc=description)
    for batch in pbar:
        obs_image = (
            batch['observation.image'].to(device, non_blocking=True)
            if 'observation.image' in batch
            else None
        )
        obs_vision_feature = (
            batch['observation.vision_feature'].to(
                device,
                non_blocking=True,
            )
            if 'observation.vision_feature' in batch
            else None
        )
        obs_state = batch['observation.state'].to(device, non_blocking=True)
        actions = batch['action'].to(device, non_blocking=True)
        obs_oct = (
            batch.get('observation.oct').to(device, non_blocking=True)
            if model.oct_dim > 0 else None
        )
        guided_base = (
            batch['observation.guided_base'].to(
                device,
                non_blocking=True,
            )
            if model.guided_oct_mode else None
        )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp,
        ):
            loss = model(
                obs_image,
                obs_state,
                actions,
                obs_oct=obs_oct,
                generator=generator,
                guided_base=guided_base,
                obs_vision_feature=obs_vision_feature,
            )
        batch_size = actions.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        pbar.set_postfix({'loss': loss.item()})

    if sample_count == 0:
        raise RuntimeError("验证集没有可用样本")
    return total_loss / sample_count


def _atomic_torch_save(payload, destination: Path):
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_loss,
    val_loss,
    args,
):
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        # Backward-compatible field retained for older tooling.
        'loss': train_loss,
        'config': vars(args),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--val-data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--resume-history", type=str, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="请求训练轮数",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--thread-workers",
        type=int,
        default=4,
        help="num_workers=0 时用于并行图像加载的线程数",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--num-diffusion-iters", type=int, default=100)
    parser.add_argument(
        "--diffusion-schedule",
        choices=("linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--pred-x0-clip", type=float, default=None)
    parser.add_argument(
        "--use-oct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 OCT 导航相对向量加入策略输入",
    )
    parser.add_argument("--val-seed", type=int, default=20261103)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 自动混合精度（默认开启）",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="默认写入 <output-dir>/training.log，同时输出到控制台",
    )
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()
    requested_epochs = args.epochs
    try:
        args.epochs = effective_epoch_count(args.epochs)
    except ValueError as error:
        parser.error(str(error))

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_stream_logging(output_dir, args.log_file)
    if requested_epochs != args.epochs:
        print(f"训练轮数: {args.epochs}", flush=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("警告: AMP 仅在 CUDA 设备启用，当前使用 FP32")
    if use_amp and args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 bfloat16")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    print(f"混合精度: {args.amp_dtype if use_amp else 'disabled'}")

    # 数据增强
    transform = image_transform_for_dataset(args.data_dir)

    # 数据集
    dataset = RCMDataset(
        Path(args.data_dir),
        transform=transform,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    validation_dataset = RCMDataset(
        Path(args.val_data_dir),
        transform=transform,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    if args.use_oct and not (
        dataset.has_oct_navigation and validation_dataset.has_oct_navigation
    ):
        raise ValueError(
            "Dataset lacks OCT navigation fields; rerun convert_to_lerobot.py"
        )
    print(
        f"OCT navigation: {'enabled' if args.use_oct else 'disabled'} "
        f"(train={int(dataset.oct_navigation_valid.sum())}/"
        f"{len(dataset.states)}, "
        f"val={int(validation_dataset.oct_navigation_valid.sum())}/"
        f"{len(validation_dataset.states)})"
    )

    if args.num_workers > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )
        loader_mode = f"process workers={args.num_workers}"
    else:
        dataloader = ThreadBatchLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_threads=args.thread_workers,
            seed=args.seed,
            pin_memory=device.type == "cuda",
            prefetch_batches=args.prefetch_factor,
        )
        validation_loader = ThreadBatchLoader(
            validation_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_threads=max(1, min(args.thread_workers, 4)),
            seed=args.val_seed,
            pin_memory=device.type == "cuda",
            prefetch_batches=args.prefetch_factor,
        )
        loader_mode = f"thread workers={args.thread_workers}"
    print(f"数据加载: {loader_mode}")

    state_mean = dataset.states.mean(axis=0)
    state_std = np.maximum(dataset.states.std(axis=0), 1e-6)
    action_mean = dataset.actions.mean(axis=0)
    action_std = np.maximum(dataset.actions.std(axis=0), 1e-6)
    oct_mean = np.zeros(4, dtype=np.float32)
    oct_std = np.ones(4, dtype=np.float32)
    if args.use_oct:
        valid_oct = dataset.oct_navigation[dataset.oct_navigation_valid]
        oct_mean[:3] = valid_oct.mean(axis=0)
        oct_std[:3] = np.maximum(valid_oct.std(axis=0), 1e-6)
    print(
        f"归一化 state_std={state_std.tolist()} "
        f"action_std={action_std.tolist()} "
        f"oct_std={oct_std[:3].tolist()}",
        flush=True,
    )

    # 模型
    model = SimpleDiffusionPolicy(
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_diffusion_iters=args.num_diffusion_iters,
        diffusion_schedule=args.diffusion_schedule,
        pred_x0_clip=args.pred_x0_clip,
        oct_dim=4 if args.use_oct else 0,
        state_mean=state_mean,
        state_std=state_std,
        oct_mean=oct_mean,
        oct_std=oct_std,
        action_mean=action_mean,
        action_std=action_std,
    ).to(device)

    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype == torch.float16,
    )

    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    if args.resume_from:
        resume_path = Path(args.resume_from)
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        checkpoint_config = checkpoint.get('config', {})
        for name in ('n_obs_steps', 'n_action_steps', 'num_diffusion_iters'):
            saved_value = checkpoint_config.get(name)
            if saved_value is not None and int(saved_value) != int(getattr(args, name)):
                raise ValueError(
                    f"Resume config mismatch for {name}: "
                    f"{saved_value} != {getattr(args, name)}"
                )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_val_loss = float(checkpoint['val_loss'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        if 'scaler_state_dict' in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        history_source = (
            Path(args.resume_history)
            if args.resume_history
            else resume_path.parent / "training_history.json"
        )
        if history_source.is_file():
            history = json.loads(history_source.read_text(encoding="utf-8"))
        history_path = output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)

        if not (output_dir / "best_model.pth").exists():
            _atomic_torch_save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    int(checkpoint['epoch']),
                    float(checkpoint.get('train_loss', checkpoint['loss'])),
                    best_val_loss,
                    args,
                ),
                output_dir / "best_model.pth",
            )
        print(
            f"从 checkpoint 恢复: {resume_path}；"
            f"已完成到 epoch {start_epoch - 1}，"
            f"下一轮为 epoch {start_epoch}（第 {start_epoch + 1} 个 epoch）",
            flush=True,
        )

    # WandB
    if args.use_wandb:
        if WANDB_AVAILABLE:
            wandb.init(project="rcm-diffusion-policy", config=vars(args))

    # 训练
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            device,
            epoch,
            amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_loss = evaluate_epoch(
            model,
            validation_loader,
            device,
            seed=args.val_seed,
            epoch=epoch,
            amp=use_amp,
            amp_dtype=amp_dtype,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'best_val_loss': best_val_loss,
            'improved': improved,
        })
        history_path = output_dir / "training_history.json"
        history_tmp = history_path.with_name(history_path.name + ".tmp")
        with open(history_tmp, "w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        history_tmp.replace(history_path)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}"
        )

        if args.use_wandb:
            if WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "lr": learning_rate,
                })

        # 只按验证集损失选择 best；训练损失仅用于监控。
        if improved:
            _atomic_torch_save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    train_loss,
                    val_loss,
                    args,
                ),
                output_dir / "best_model.pth",
            )

        if (epoch + 1) % 100 == 0:
            _atomic_torch_save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    train_loss,
                    val_loss,
                    args,
                ),
                output_dir / f"checkpoint_epoch_{epoch+1}.pth",
            )

    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
