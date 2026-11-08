"""Stage 2 ablations built on the Stage 1 restricted representation."""

from __future__ import annotations

import torch
import torch.nn as nn


class Stage2ActionPolicy(nn.Module):
    def __init__(
        self,
        *,
        vision_dim: int,
        state_dim: int,
        restricted_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 512,
        use_velocity: bool = False,
        use_stop_head: bool = False,
        residual_distance_scale: float | None = None,
        action_limit=None,
    ):
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.state_dim = int(state_dim)
        self.restricted_dim = int(restricted_dim)
        self.action_dim = int(action_dim)
        self.use_velocity = bool(use_velocity)
        self.use_stop_head = bool(use_stop_head)
        self.residual_distance_scale = residual_distance_scale
        input_dim = (
            self.vision_dim
            + self.state_dim
            + self.restricted_dim
            + (3 if self.use_velocity else 0)
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.stop_head = (
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
            if self.use_stop_head
            else None
        )
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer(
            "action_limit",
            torch.as_tensor(
                [0.3, 0.15, 0.05] if action_limit is None else action_limit,
                dtype=torch.float32,
            ),
        )

    def set_normalization(
        self,
        feature_mean,
        feature_std,
        action_mean,
        action_std,
    ) -> None:
        self.feature_mean.copy_(torch.as_tensor(feature_mean, dtype=torch.float32))
        self.feature_std.copy_(torch.as_tensor(feature_std, dtype=torch.float32))
        self.action_mean.copy_(torch.as_tensor(action_mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(action_std, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = (features - self.feature_mean) / self.feature_std
        action = self.network(normalized) * self.action_std + self.action_mean
        if self.residual_distance_scale is not None:
            offset = self.vision_dim + self.state_dim
            coarse = features[:, offset:offset + 3]
            fine = features[:, offset + 3:offset + 6]
            fine_valid = features[:, offset + 6:offset + 7] > 0.5
            goal = torch.where(fine_valid, fine, coarse)
            distance = torch.linalg.norm(goal, dim=1, keepdim=True)
            scale = torch.clamp(
                distance / float(self.residual_distance_scale),
                min=0.05,
                max=1.0,
            )
            action = torch.clamp(
                action,
                min=-self.action_limit * scale,
                max=self.action_limit * scale,
            )
        stop_logit = self.stop_head(normalized) if self.stop_head is not None else None
        return action, stop_logit
