"""Lightweight single-step restricted-observation imitation policies."""

from __future__ import annotations

import torch
import torch.nn as nn


class RestrictedActionMLP(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        state_dim: int,
        restricted_dim: int,
        action_dim: int = 3,
        n_action_steps: int = 8,
        hidden_dim: int = 512,
        use_image: bool = True,
    ):
        super().__init__()
        self.vision_dim = int(vision_dim) if use_image else 0
        self.state_dim = int(state_dim)
        self.restricted_dim = int(restricted_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.use_image = bool(use_image)
        input_dim = self.vision_dim + self.state_dim + self.restricted_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim * self.n_action_steps),
        )
        self.register_buffer(
            "feature_mean",
            torch.zeros(input_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_std",
            torch.ones(input_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_mean",
            torch.zeros(self.action_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_std",
            torch.ones(self.action_dim, dtype=torch.float32),
            persistent=False,
        )
        self.action_clip = None

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, feature_dim]")
        normalized = (features - self.feature_mean) / self.feature_std
        output = self.network(normalized)
        output = output.reshape(-1, self.n_action_steps, self.action_dim)
        output = output * self.action_std + self.action_mean
        if self.action_clip is not None:
            clip = torch.as_tensor(self.action_clip, device=output.device)
            output = torch.clamp(output, -clip, clip)
        return output
