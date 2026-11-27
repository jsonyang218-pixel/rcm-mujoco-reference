"""Spatial, geometric, and temporal diffusion policy candidate for v13."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def diffusion_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    cosine_s: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return beta and cumulative alpha schedules."""
    timesteps = int(timesteps)
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    if schedule == "linear":
        betas = torch.linspace(0.0001, 0.02, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alpha_bar = torch.cos(
            ((x / timesteps + cosine_s) / (1 + cosine_s))
            * (torch.pi / 2)
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(
            1 - alpha_bar[1:] / alpha_bar[:-1],
            1e-8,
            0.999,
        ).float()
    else:
        raise ValueError(f"Unknown diffusion schedule: {schedule}")
    alphas = 1.0 - betas
    return betas, torch.cumprod(alphas, dim=0)


def _replace_batch_norm(module: nn.Module, groups: int = 32) -> None:
    """Replace BatchNorm2d recursively while preserving convolution weights."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(
                num_groups=min(groups, child.num_features),
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            if child.affine:
                with torch.no_grad():
                    replacement.weight.fill_(1.0)
                    replacement.bias.zero_()
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child, groups=groups)


class GroupNormResNet18Encoder(nn.Module):
    """ResNet18 feature map encoder with GroupNorm and no global pooling."""

    def __init__(self, pretrained: bool = False, groups: int = 32):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        _replace_batch_norm(backbone, groups=groups)
        # Remove avgpool and fc, retaining the 7x7 layer4 feature map.
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.out_channels = 512

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            batch, obs_steps, channels, height, width = image.shape
            feature = self.features(
                image.reshape(
                    batch * obs_steps,
                    channels,
                    height,
                    width,
                )
            )
            return feature.reshape(
                batch,
                obs_steps,
                self.out_channels,
                feature.shape[-2],
                feature.shape[-1],
            )
        if image.ndim != 4:
            raise ValueError(
                "image must have shape [B,T,C,H,W] or [B,C,H,W]"
            )
        return self.features(image)


class SpatialSoftmax(nn.Module):
    """Map channel feature maps to expected 2-D coordinates."""

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
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
        expected_x = (
            weights * x.reshape(1, 1, -1)
        ).sum(dim=-1)
        expected_y = (
            weights * y.reshape(1, 1, -1)
        ).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(
            batch,
            channels * 2,
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half,
                device=timestep.device,
                dtype=torch.float32,
            )
            / max(half - 1, 1)
        )
        phase = timestep.float()[:, None] * frequency[None, :]
        embedding = torch.cat(
            [torch.sin(phase), torch.cos(phase)],
            dim=-1,
        )
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(
                embedding,
                (0, 1),
            )
        return embedding


class ConditionalResidualBlock1D(nn.Module):
    """Residual 1-D convolution block with zero-initialized FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        kernel_size: int = 3,
        groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(
            min(groups, in_channels),
            in_channels,
        )
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        self.film = nn.Linear(condition_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(
            min(groups, out_channels),
            out_channels,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual(hidden)
        hidden = self.conv1(
            torch.nn.functional.silu(self.norm1(hidden))
        )
        scale, shift = self.film(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale[..., None]) + shift[..., None]
        hidden = self.conv2(
            torch.nn.functional.silu(self.norm2(hidden))
        )
        return hidden + residual


class ConditionalUNet1D(nn.Module):
    """Two-scale temporal U-Net using FiLM in every residual block."""

    def __init__(
        self,
        action_dim: int = 3,
        action_steps: int = 16,
        condition_dim: int = 256,
        base_channels: int = 128,
        middle_channels: int = 256,
        top_channels: int = 512,
    ):
        super().__init__()
        if action_steps % 4:
            raise ValueError(
                "action_steps must be divisible by four for two downsampling levels"
            )
        self.action_dim = int(action_dim)
        self.action_steps = int(action_steps)
        self.condition_dim = int(condition_dim)
        self.input_conv = nn.Conv1d(
            action_dim,
            base_channels,
            kernel_size=3,
            padding=1,
        )
        self.block_in = ConditionalResidualBlock1D(
            base_channels,
            base_channels,
            condition_dim,
        )
        self.down1 = nn.Conv1d(
            base_channels,
            middle_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.block1 = ConditionalResidualBlock1D(
            middle_channels,
            middle_channels,
            condition_dim,
        )
        self.down2 = nn.Conv1d(
            middle_channels,
            top_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.block2 = ConditionalResidualBlock1D(
            top_channels,
            top_channels,
            condition_dim,
        )
        self.up1 = nn.ConvTranspose1d(
            top_channels,
            middle_channels,
            kernel_size=2,
            stride=2,
        )
        self.block3 = ConditionalResidualBlock1D(
            middle_channels * 2,
            middle_channels,
            condition_dim,
        )
        self.up2 = nn.ConvTranspose1d(
            middle_channels,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.block4 = ConditionalResidualBlock1D(
            base_channels * 2,
            base_channels,
            condition_dim,
        )
        self.output_conv = nn.Conv1d(
            base_channels,
            action_dim,
            kernel_size=1,
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.shape[1:] != (
            self.action_dim,
            self.action_steps,
        ):
            raise ValueError(
                "noisy_action must have shape "
                f"[B,{self.action_dim},{self.action_steps}], "
                f"got {tuple(noisy_action.shape)}"
            )
        hidden0 = self.block_in(self.input_conv(noisy_action), condition)
        hidden1 = self.block1(self.down1(hidden0), condition)
        hidden2 = self.block2(self.down2(hidden1), condition)
        recovered1 = self.up1(hidden2)
        recovered1 = self.block3(
            torch.cat([recovered1, hidden1], dim=1),
            condition,
        )
        recovered0 = self.up2(recovered1)
        recovered0 = self.block4(
            torch.cat([recovered0, hidden0], dim=1),
            condition,
        )
        return self.output_conv(recovered0)


class SpatialFiLMUNetDiffusionPolicy(nn.Module):
    """v13 candidate policy with spatial keypoints and temporal U-Net."""

    def __init__(
        self,
        n_obs_steps: int = 4,
        n_action_steps: int = 16,
        num_diffusion_iters: int = 100,
        diffusion_schedule: str = "cosine",
        pred_x0_clip: float | None = 4.0,
        oct_dim: int = 4,
        state_mean=None,
        state_std=None,
        oct_mean=None,
        oct_std=None,
        action_mean=None,
        action_std=None,
        include_velocity: bool = True,
        include_previous_action: bool = True,
        keypoint_channels: int = 32,
        token_dim: int = 256,
        condition_dim: int = 256,
        base_channels: int = 128,
        middle_channels: int = 256,
        top_channels: int = 512,
        attention_heads: int = 4,
        pretrained_vision: bool = False,
    ):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.diffusion_schedule = diffusion_schedule
        self.pred_x0_clip = pred_x0_clip
        self.oct_dim = int(oct_dim)
        self.include_velocity = bool(include_velocity)
        self.include_previous_action = bool(include_previous_action)
        self.guided_oct_mode = False
        self.keypoint_channels = int(keypoint_channels)
        self.token_dim = int(token_dim)
        self.condition_dim = int(condition_dim)
        if self.oct_dim not in (0, 4):
            raise ValueError("oct_dim must be zero or four")
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        if self.n_action_steps % 4:
            raise ValueError("n_action_steps must be divisible by four")

        self.register_buffer(
            "state_mean",
            torch.as_tensor(
                [0.0, 0.0, 0.0] if state_mean is None else state_mean,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "state_std",
            torch.as_tensor(
                [1.0, 1.0, 1.0] if state_std is None else state_std,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "oct_mean",
            torch.as_tensor(
                [0.0] * max(self.oct_dim, 1)
                if oct_mean is None
                else oct_mean,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "oct_std",
            torch.as_tensor(
                [1.0] * max(self.oct_dim, 1)
                if oct_std is None
                else oct_std,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_mean",
            torch.as_tensor(
                [0.0, 0.0, 0.0]
                if action_mean is None
                else action_mean,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_std",
            torch.as_tensor(
                [1.0, 1.0, 1.0]
                if action_std is None
                else action_std,
                dtype=torch.float32,
            ),
        )

        self.vision_encoder = GroupNormResNet18Encoder(
            pretrained=pretrained_vision,
        )
        self.keypoint_proj = nn.Conv2d(
            self.vision_encoder.out_channels,
            self.keypoint_channels,
            kernel_size=1,
        )
        self.spatial_softmax = SpatialSoftmax()
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(self.keypoint_channels * 2, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.frame_embedding = nn.Parameter(
            torch.zeros(self.n_obs_steps, token_dim)
        )
        nn.init.normal_(self.frame_embedding, std=0.02)

        geometry_input_dim = (
            3
            + (3 if self.include_velocity else 0)
            + (3 if self.include_previous_action else 0)
            + self.oct_dim
            + (2 if self.oct_dim else 0)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_input_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            token_dim,
            attention_heads,
            batch_first=True,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.fusion_norm = nn.LayerNorm(token_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(token_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.time_embedding = SinusoidalPosEmb(128)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim + 128, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.denoiser = ConditionalUNet1D(
            action_dim=3,
            action_steps=self.n_action_steps,
            condition_dim=condition_dim,
            base_channels=base_channels,
            middle_channels=middle_channels,
            top_channels=top_channels,
        )
        betas, alphas_cumprod = diffusion_beta_schedule(
            self.num_diffusion_iters,
            self.diffusion_schedule,
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def encode_observation(
        self,
        obs_image: torch.Tensor,
        obs_state: torch.Tensor,
        obs_oct: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ):
        """Return fused condition, keypoints, and attention weights."""
        batch, obs_steps = obs_state.shape[:2]
        feature_map = self.vision_encoder(obs_image)
        if feature_map.shape[1:2] != (obs_steps,):
            raise ValueError("visual and state observation histories differ")
        flat_feature = feature_map.reshape(
            batch * obs_steps,
            feature_map.shape[2],
            feature_map.shape[3],
            feature_map.shape[4],
        )
        keypoints = self.spatial_softmax(
            self.keypoint_proj(flat_feature)
        ).reshape(batch, obs_steps, self.keypoint_channels * 2)
        visual_tokens = (
            self.keypoint_encoder(keypoints)
            + self.frame_embedding[:obs_steps][None, :, :]
        )

        current_state = obs_state[:, -1]
        normalized_state = (
            current_state - self.state_mean.reshape(1, -1)
        ) / self.state_std.reshape(1, -1)
        geometry_parts = [normalized_state]
        if self.include_velocity:
            if velocity is None:
                raise ValueError(
                    "velocity is required by the configured history"
                )
            geometry_parts.append(velocity)
        if self.include_previous_action:
            if previous_action is None:
                raise ValueError(
                    "previous_action is required by the configured history"
                )
            geometry_parts.append(previous_action)

        if self.oct_dim:
            if obs_oct is None:
                raise ValueError(
                    "OCT navigation is required by the configured policy"
                )
            current_oct = obs_oct[:, -1]
            normalized_oct_xyz = (
                current_oct[:, :3] - self.oct_mean[:3].reshape(1, -1)
            ) / self.oct_std[:3].reshape(1, -1)
            depth = current_oct[:, 2:3].abs().clamp_min(1e-3)
            projection = current_oct[:, :2] / depth
            geometry_parts.extend(
                [normalized_oct_xyz, current_oct[:, 3:4], projection]
            )
        geometry_input = torch.cat(geometry_parts, dim=-1)
        geometry_token = self.geometry_encoder(geometry_input)
        query = geometry_token[:, None, :]
        key_value = visual_tokens.reshape(batch, -1, self.token_dim)
        attended, attention_weights = self.cross_attention(
            query,
            key_value,
            key_value,
            need_weights=True,
        )
        gate = torch.sigmoid(self.fusion_gate(geometry_token))
        fused = self.fusion_norm(
            geometry_token + gate * attended[:, 0]
        )
        condition = self.condition_projection(fused)
        return (
            condition,
            keypoints,
            attention_weights.squeeze(1),
        )

    def _condition_with_time(
        self,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        time_feature = self.time_embedding(timestep)
        return self.condition_encoder(
            torch.cat([condition, time_feature], dim=-1)
        )

    def forward(
        self,
        obs_image,
        obs_state,
        actions=None,
        obs_oct=None,
        generator=None,
        guided_base=None,
        obs_vision_feature=None,
        velocity=None,
        previous_action=None,
    ):
        if guided_base is not None:
            raise ValueError("v13 policy does not use guided residual actions")
        if obs_vision_feature is not None:
            raise ValueError("v13 policy does not use cached vision features")
        batch = obs_state.shape[0]
        condition, _, _ = self.encode_observation(
            obs_image,
            obs_state,
            obs_oct=obs_oct,
            velocity=velocity,
            previous_action=previous_action,
        )
        if actions is None:
            return self.sample(condition, generator=generator)

        if actions.shape != (batch, self.n_action_steps, 3):
            raise ValueError(
                "actions must have shape "
                f"[B,{self.n_action_steps},3], got {tuple(actions.shape)}"
            )
        normalized = (
            actions - self.action_mean.reshape(1, 1, -1)
        ) / self.action_std.reshape(1, 1, -1)
        clean = normalized.transpose(1, 2)
        timestep = torch.randint(
            0,
            self.num_diffusion_iters,
            (batch,),
            device=clean.device,
            generator=generator,
        )
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        alpha = self.alphas_cumprod[timestep].reshape(-1, 1, 1)
        noisy = torch.sqrt(alpha) * clean + torch.sqrt(1 - alpha) * noise
        global_condition = self._condition_with_time(condition, timestep)
        predicted_noise = self.denoiser(noisy, global_condition)
        return torch.mean((predicted_noise - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        action = torch.randn(
            batch,
            3,
            self.n_action_steps,
            device=condition.device,
            generator=generator,
        )
        for timestep_index in reversed(range(self.num_diffusion_iters)):
            timestep = torch.full(
                (batch,),
                timestep_index,
                dtype=torch.long,
                device=condition.device,
            )
            global_condition = self._condition_with_time(
                condition,
                timestep,
            )
            predicted_noise = self.denoiser(action, global_condition)
            alpha = self.alphas_cumprod[timestep_index]
            alpha_previous = (
                self.alphas_cumprod[timestep_index - 1]
                if timestep_index > 0
                else torch.ones_like(alpha)
            )
            predicted_clean = (
                action - torch.sqrt(1 - alpha) * predicted_noise
            ) / torch.sqrt(alpha)
            if self.pred_x0_clip is not None:
                predicted_clean = torch.clamp(
                    predicted_clean,
                    -float(self.pred_x0_clip),
                    float(self.pred_x0_clip),
                )
            action = (
                torch.sqrt(alpha_previous) * predicted_clean
                + torch.sqrt(1 - alpha_previous) * predicted_noise
            )
        action = action.transpose(1, 2)
        return (
            action * self.action_std.reshape(1, 1, -1)
            + self.action_mean.reshape(1, 1, -1)
        )
