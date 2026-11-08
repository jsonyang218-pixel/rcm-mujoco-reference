"""Deterministic residual single-step distillation policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from .stage2_policy import Stage2ActionPolicy


class Stage3ResidualActionPolicy(nn.Module):
    def __init__(
        self,
        *,
        vision_dim: int,
        state_dim: int,
        restricted_dim: int,
        use_velocity: bool,
        hidden_dim: int = 256,
        residual_limit=(0.005, 0.003, 0.001),
    ):
        super().__init__()
        self.base = Stage2ActionPolicy(
            vision_dim=vision_dim,
            state_dim=state_dim,
            restricted_dim=restricted_dim,
            use_velocity=use_velocity,
            residual_distance_scale=None,
            use_stop_head=False,
            action_limit=[0.2898, 0.14434, 0.04271],
        )
        input_dim = (
            int(vision_dim)
            + int(state_dim)
            + int(restricted_dim)
            + (3 if use_velocity else 0)
        )
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer(
            "residual_limit",
            torch.as_tensor(residual_limit, dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, None]:
        base_action, _ = self.base(features)
        normalized = (
            features - self.base.feature_mean
        ) / self.base.feature_std
        correction = torch.tanh(self.residual(normalized)) * self.residual_limit
        return base_action + correction, None
