"""1D temporal U-Net with FiLM conditioning and flow-matching actions."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _group_norm(channels):
    return nn.GroupNorm(math.gcd(32, int(channels)), int(channels))


def sinusoidal_embedding(t, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    phase = t.reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class SpatialSoftmax(nn.Module):
    def __init__(self, input_channels, keypoints=32, hidden_dim=128):
        super().__init__()
        self.keypoints = int(keypoints)
        self.project = nn.Conv2d(input_channels, self.keypoints, 1)
        self.out = nn.Linear(self.keypoints * 2, hidden_dim)

    def forward(self, feature):
        weights = self.project(feature)
        batch, channels, height, width = weights.shape
        weights = torch.softmax(weights.reshape(batch, channels, -1), dim=-1)
        y_axis = torch.linspace(
            -1.0, 1.0, height, device=feature.device, dtype=feature.dtype
        )
        x_axis = torch.linspace(
            -1.0, 1.0, width, device=feature.device, dtype=feature.dtype
        )
        y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        expected_x = (weights * x.reshape(1, 1, -1)).sum(dim=-1)
        expected_y = (weights * y.reshape(1, 1, -1)).sum(dim=-1)
        keypoints = torch.stack(
            [expected_x, expected_y], dim=-1
        ).reshape(batch, self.keypoints * 2)
        return self.out(keypoints)


class SpatialResNetEncoder(nn.Module):
    def __init__(self, hidden_dim=128, keypoints=32):
        super().__init__()
        pretrained = resnet18(weights="IMAGENET1K_V1")
        network = resnet18(weights=None, norm_layer=_group_norm)
        pretrained_state = pretrained.state_dict()
        compatible = {
            name: value
            for name, value in network.state_dict().items()
            if name in pretrained_state
            and pretrained_state[name].shape == value.shape
        }
        network.load_state_dict(compatible, strict=False)
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.spatial_softmax = SpatialSoftmax(512, keypoints, hidden_dim)

    def forward(self, image):
        batch, obs_steps, channels, height, width = image.shape
        feature = self.features(
            image.reshape(batch * obs_steps, channels, height, width)
        )
        encoded = self.spatial_softmax(feature)
        return encoded.reshape(batch, obs_steps, -1)


class ObservationConditioner(nn.Module):
    def __init__(
        self,
        n_obs_steps,
        hidden_dim=128,
        cond_dim=256,
        n_heads=4,
    ):
        super().__init__()
        self.vision = SpatialResNetEncoder(hidden_dim)
        self.geometry = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * n_obs_steps, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, image, state, oct_feature, velocity, previous_action):
        batch, obs_steps, _ = state.shape
        vision = self.vision(image)
        geometry = torch.cat(
            [state, oct_feature, velocity, previous_action],
            dim=-1,
        )
        query = self.geometry(geometry)
        query_flat = query.reshape(batch * obs_steps, 1, -1)
        vision_tokens = vision.reshape(batch * obs_steps, -1, query.shape[-1])
        attended, _ = self.cross_attention(
            query_flat,
            vision_tokens,
            vision_tokens,
            need_weights=False,
        )
        attended = attended.reshape(batch, obs_steps, -1)
        gate = self.gate(torch.cat([query, attended], dim=-1))
        fused = query + gate * attended
        return self.output(fused.reshape(batch, obs_steps, -1).flatten(1))


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, condition):
        scale, bias = self.cond(condition).reshape(
            condition.shape[0], 2, -1, 1
        ).unbind(dim=1)
        h = self.norm1(self.conv1(x))
        h = F.silu(h * (1 + scale) + bias)
        h = self.norm2(self.conv2(h))
        h = F.silu(h * (1 + scale) + bias)
        return h + self.residual(x)


class TemporalUNet1D(nn.Module):
    def __init__(
        self,
        action_dim=3,
        horizon=16,
        cond_dim=256,
        base_channels=256,
    ):
        super().__init__()
        if horizon % 4:
            raise ValueError("flow action horizon must be divisible by 4")
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
        )
        film_dim = cond_dim + 64
        self.time_mlp = nn.Sequential(
            nn.Linear(film_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.input_proj = nn.Conv1d(action_dim, channels[0], 1)
        self.down_blocks = nn.ModuleList([
            FiLMResidualBlock1D(channels[0], channels[0], cond_dim),
            FiLMResidualBlock1D(channels[0], channels[1], cond_dim),
            FiLMResidualBlock1D(channels[1], channels[2], cond_dim),
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv1d(channels[0], channels[0], 4, 2, 1),
            nn.Conv1d(channels[1], channels[1], 4, 2, 1),
        ])
        self.mid = FiLMResidualBlock1D(
            channels[2], channels[2], cond_dim
        )
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose1d(
                channels[2], channels[1], 4, 2, 1
            ),
            nn.ConvTranspose1d(
                channels[1], channels[0], 4, 2, 1
            ),
        ])
        self.up_blocks = nn.ModuleList([
            FiLMResidualBlock1D(
                channels[1] * 2,
                channels[1],
                cond_dim,
            ),
            FiLMResidualBlock1D(
                channels[0] * 2,
                channels[0],
                cond_dim,
            ),
        ])
        self.output_norm = _group_norm(channels[0])
        self.output = nn.Conv1d(channels[0], action_dim, 1)

    def forward(self, noisy_actions, condition, timestep):
        time = sinusoidal_embedding(timestep, 64)
        cond = self.time_mlp(torch.cat([condition, time], dim=-1))
        h = self.input_proj(noisy_actions)
        skips = []
        h = self.down_blocks[0](h, cond)
        skips.append(h)
        h = self.downsamples[0](h)
        h = self.down_blocks[1](h, cond)
        skips.append(h)
        h = self.downsamples[1](h)
        h = self.down_blocks[2](h, cond)
        h = self.mid(h, cond)
        h = self.upsamples[0](h)
        h = self.up_blocks[0](torch.cat([h, skips[-1]], dim=1), cond)
        h = self.upsamples[1](h)
        h = self.up_blocks[1](torch.cat([h, skips[0]], dim=1), cond)
        return self.output(F.silu(self.output_norm(h)))


class FlowTemporalUNetPolicy(nn.Module):
    policy_type = "flow_temporal_unet"

    def __init__(
        self,
        n_obs_steps=2,
        n_action_steps=16,
        action_dim=3,
        oct_dim=4,
        state_mean=None,
        state_std=None,
        oct_mean=None,
        oct_std=None,
        action_mean=None,
        action_std=None,
        velocity_mean=None,
        velocity_std=None,
        previous_action_mean=None,
        previous_action_std=None,
        cond_dim=256,
        base_channels=256,
        flow_integration_steps=4,
    ):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.action_dim = int(action_dim)
        self.oct_dim = int(oct_dim)
        self.flow_integration_steps = int(flow_integration_steps)
        for name, value, size in (
            ("state_mean", state_mean, 3),
            ("state_std", state_std, 3),
            ("oct_mean", oct_mean, oct_dim),
            ("oct_std", oct_std, oct_dim),
            ("action_mean", action_mean, action_dim),
            ("action_std", action_std, action_dim),
            ("velocity_mean", velocity_mean, 3),
            ("velocity_std", velocity_std, 3),
            ("previous_action_mean", previous_action_mean, action_dim),
            ("previous_action_std", previous_action_std, action_dim),
        ):
            if value is None or len(value) != size:
                raise ValueError(f"{name} must contain {size} values")
            self.register_buffer(
                name,
                torch.as_tensor(value, dtype=torch.float32),
                persistent=False,
            )
        self.observation = ObservationConditioner(
            self.n_obs_steps,
            hidden_dim=128,
            cond_dim=cond_dim,
        )
        self.unet = TemporalUNet1D(
            action_dim=action_dim,
            horizon=self.n_action_steps,
            cond_dim=cond_dim,
            base_channels=base_channels,
        )

    def _normalize_observation(
        self, state, oct_feature, velocity, previous_action
    ):
        if velocity.dim() == 2:
            velocity = velocity[:, None, :].expand(
                -1, state.shape[1], -1
            )
        if previous_action.dim() == 2:
            previous_action = previous_action[:, None, :].expand(
                -1, state.shape[1], -1
            )
        return (
            (state - self.state_mean) / self.state_std,
            (oct_feature - self.oct_mean) / self.oct_std,
            (velocity - self.velocity_mean) / self.velocity_std,
            (
                previous_action - self.previous_action_mean
            ) / self.previous_action_std,
        )

    def _predict_velocity_from_condition(
        self,
        noisy_actions,
        timestep,
        condition,
    ):
        return self.unet(
            noisy_actions.transpose(1, 2),
            condition,
            timestep,
        ).transpose(1, 2)

    def predict_velocity(
        self,
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
        noisy_actions,
        timestep,
    ):
        state, oct_feature, velocity, previous_action = (
            self._normalize_observation(
                state, oct_feature, velocity, previous_action
            )
        )
        condition = self.observation(
            image,
            state,
            oct_feature,
            velocity,
            previous_action,
        )
        return self._predict_velocity_from_condition(
            noisy_actions,
            timestep,
            condition,
        )

    def flow_loss(
        self,
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
        actions,
    ):
        normalized = (
            actions - self.action_mean.reshape(1, 1, -1)
        ) / self.action_std.reshape(1, 1, -1)
        noise = torch.randn_like(normalized)
        tau = torch.rand(
            normalized.shape[0],
            device=normalized.device,
            dtype=normalized.dtype,
        )
        tau_reshape = tau.reshape(-1, 1, 1)
        interpolated = (1 - tau_reshape) * noise + tau_reshape * normalized
        target_velocity = normalized - noise
        predicted = self.predict_velocity(
            image,
            state,
            oct_feature,
            velocity,
            previous_action,
            interpolated,
            tau,
        )
        return F.mse_loss(predicted, target_velocity)

    @torch.no_grad()
    def sample(
        self,
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
        steps=None,
        generator=None,
    ):
        steps = int(
            self.flow_integration_steps if steps is None else steps
        )
        if steps < 1:
            raise ValueError("flow integration steps must be positive")
        actions = torch.randn(
            state.shape[0],
            self.n_action_steps,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        state, oct_feature, velocity, previous_action = (
            self._normalize_observation(
                state,
                oct_feature,
                velocity,
                previous_action,
            )
        )
        condition = self.observation(
            image,
            state,
            oct_feature,
            velocity,
            previous_action,
        )
        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full(
                (state.shape[0],),
                (index + 0.5) * dt,
                device=state.device,
                dtype=state.dtype,
            )
            actions = actions + dt * self._predict_velocity_from_condition(
                actions,
                tau,
                condition,
            )
        return (
            actions * self.action_std.reshape(1, 1, -1)
            + self.action_mean.reshape(1, 1, -1)
        )
