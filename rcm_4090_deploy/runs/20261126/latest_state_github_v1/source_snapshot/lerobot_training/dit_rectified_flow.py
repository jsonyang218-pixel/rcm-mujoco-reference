"""DiT policy trained with rectified-flow matching on action trajectories."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _group_norm(channels):
    return nn.GroupNorm(math.gcd(32, int(channels)), int(channels))


def _sinusoidal(timestep, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=timestep.dtype)
        / max(half - 1, 1)
    )
    phase = timestep.reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads, context_dim=None):
        super().__init__()
        if dim % heads:
            raise ValueError("DiT dimension must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.q = nn.Linear(self.dim, self.dim)
        context_dim = self.dim if context_dim is None else int(context_dim)
        self.kv = (
            nn.Linear(self.dim, self.dim * 2)
            if context_dim == self.dim
            else nn.Linear(context_dim, self.dim * 2)
        )
        self.output = nn.Linear(self.dim, self.dim)

    def forward(self, value, context=None):
        context = value if context is None else context
        batch, tokens, _ = value.shape
        context_tokens = context.shape[1]
        query = self.q(value).reshape(
            batch,
            tokens,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)
        key, value_context = self.kv(context).reshape(
            batch,
            context_tokens,
            self.heads,
            self.head_dim * 2,
        ).chunk(2, dim=-1)
        key = key.transpose(1, 2)
        value_context = value_context.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value_context,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch,
            tokens,
            self.dim,
        )
        return self.output(attended)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero modulation and visual cross-attn."""

    def __init__(self, dim, heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attention = MultiheadAttention(dim, heads)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attention = MultiheadAttention(
            dim,
            heads,
            context_dim=dim,
        )
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 9),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(value, shift, scale):
        return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, tokens, condition_tokens, global_condition):
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(global_condition).chunk(9, dim=-1)
        value = tokens + gate_self.unsqueeze(1) * self.self_attention(
            self._modulate(
                self.norm_self(tokens),
                shift_self,
                scale_self,
            )
        )
        value = value + gate_cross.unsqueeze(1) * self.cross_attention(
            self._modulate(
                self.norm_cross(value),
                shift_cross,
                scale_cross,
            ),
            condition_tokens,
        )
        value = value + gate_mlp.unsqueeze(1) * self.mlp(
            self._modulate(
                self.norm_mlp(value),
                shift_mlp,
                scale_mlp,
            )
        )
        return value


class SpatialResNetTokens(nn.Module):
    def __init__(self, output_dim):
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
        self.project = nn.Conv2d(512, output_dim, 1)

    def forward(self, image):
        batch, obs_steps, channels, height, width = image.shape
        feature = self.features(
            image.reshape(batch * obs_steps, channels, height, width)
        )
        feature = self.project(feature)
        tokens = feature.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, obs_steps, tokens.shape[1], -1)


class DiTRectifiedFlowPolicy(nn.Module):
    policy_type = "dit_rectified_flow"

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
        dit_dim=384,
        dit_depth=6,
        dit_heads=6,
        flow_integration_steps=4,
    ):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.action_dim = int(action_dim)
        self.oct_dim = int(oct_dim)
        self.dit_dim = int(dit_dim)
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
        self.vision = SpatialResNetTokens(self.dit_dim)
        self.observation_temporal_embedding = nn.Parameter(
            torch.zeros(1, self.n_obs_steps, 1, self.dit_dim)
        )
        geometry_dim = 3 + self.oct_dim + 3 + self.action_dim
        self.geometry_token = nn.Sequential(
            nn.Linear(geometry_dim, self.dit_dim),
            nn.SiLU(),
            nn.Linear(self.dit_dim, self.dit_dim),
        )
        self.global_condition = nn.Sequential(
            nn.Linear(geometry_dim * self.n_obs_steps, self.dit_dim),
            nn.SiLU(),
            nn.Linear(self.dit_dim, self.dit_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.dit_dim, self.dit_dim),
            nn.SiLU(),
            nn.Linear(self.dit_dim, self.dit_dim),
        )
        self.action_input = nn.Linear(action_dim, self.dit_dim)
        self.action_position = nn.Parameter(
            torch.randn(1, self.n_action_steps, self.dit_dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.dit_dim,
                int(dit_heads),
                self.dit_dim,
            )
            for _ in range(int(dit_depth))
        ])
        self.final_norm = nn.LayerNorm(
            self.dit_dim,
            elementwise_affine=False,
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dit_dim, self.dit_dim * 2),
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        self.output = nn.Linear(self.dit_dim, action_dim)

    def _normalize_observation(
        self,
        state,
        oct_feature,
        velocity,
        previous_action,
    ):
        if velocity.dim() == 2:
            velocity = velocity[:, None, :].expand(
                -1,
                state.shape[1],
                -1,
            )
        if previous_action.dim() == 2:
            previous_action = previous_action[:, None, :].expand(
                -1,
                state.shape[1],
                -1,
            )
        return (
            (state - self.state_mean) / self.state_std,
            (oct_feature - self.oct_mean) / self.oct_std,
            (velocity - self.velocity_mean) / self.velocity_std,
            (
                previous_action - self.previous_action_mean
            ) / self.previous_action_std,
        )

    def _condition(
        self,
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
    ):
        state, oct_feature, velocity, previous_action = (
            self._normalize_observation(
                state,
                oct_feature,
                velocity,
                previous_action,
            )
        )
        vision_tokens = (
            self.vision(image)
            + self.observation_temporal_embedding
        ).flatten(1, 2)
        geometry = torch.cat(
            [state, oct_feature, velocity, previous_action],
            dim=-1,
        )
        geometry_tokens = self.geometry_token(geometry)
        condition_tokens = torch.cat(
            [vision_tokens, geometry_tokens],
            dim=1,
        )
        global_condition = self.global_condition(
            geometry.flatten(1)
        )
        return condition_tokens, global_condition

    def _predict_velocity_from_condition(
        self,
        interpolated_actions,
        timestep,
        condition_tokens,
        global_condition,
    ):
        timestep_embedding = self.time_mlp(
            _sinusoidal(timestep, self.dit_dim)
        )
        global_condition = global_condition + timestep_embedding
        tokens = (
            self.action_input(interpolated_actions)
            + self.action_position
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                condition_tokens,
                global_condition,
            )
        shift, scale = self.final_modulation(
            global_condition
        ).chunk(2, dim=-1)
        tokens = (
            self.final_norm(tokens)
            * (1 + scale.unsqueeze(1))
            + shift.unsqueeze(1)
        )
        return self.output(tokens)

    def predict_velocity(
        self,
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
        interpolated_actions,
        timestep,
    ):
        condition_tokens, global_condition = self._condition(
            image,
            state,
            oct_feature,
            velocity,
            previous_action,
        )
        return self._predict_velocity_from_condition(
            interpolated_actions,
            timestep,
            condition_tokens,
            global_condition,
        )

    def rectified_flow_loss(
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
            raise ValueError("rectified-flow steps must be positive")
        actions = torch.randn(
            state.shape[0],
            self.n_action_steps,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        condition_tokens, global_condition = self._condition(
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
                condition_tokens,
                global_condition,
            )
        return (
            actions * self.action_std.reshape(1, 1, -1)
            + self.action_mean.reshape(1, 1, -1)
        )
