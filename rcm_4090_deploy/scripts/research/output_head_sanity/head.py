"""Shared deterministic training/inference transforms, without privileged inputs."""
import torch
from torch import nn

SCALE = (.003, .002, .001)  # A degrees, B degrees, C millimeters.


def symlog(action):
    scale = action.new_tensor(SCALE)
    return action.sign() * torch.log1p(action.abs() / scale)


def symexp(value):
    scale = value.new_tensor(SCALE)
    return value.sign() * scale * torch.expm1(value.abs())


class RegressionHead(nn.Module):
    def __init__(self, dimension=384):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, 384), nn.LayerNorm(384), nn.SiLU(),
            nn.Linear(384, 384), nn.LayerNorm(384), nn.SiLU(),
            nn.Linear(384, 48),
        )

    def forward(self, features):
        return self.layers(features).reshape(-1, 16, 3)


def geom_condition(encoder, state, oct_feature, velocity, previous_action):
    # Exact original F_geom_cond. The baseline has no global visual fusion.
    if getattr(encoder, 'oct_symlog_enabled', False) or getattr(encoder, 'use_local_oct_branch', False):
        raise ValueError('Expected unmodified baseline geometry encoder')
    normalized = encoder._normalize_observation(state, oct_feature, velocity, previous_action)
    return encoder.global_condition(torch.cat(normalized, dim=-1).flatten(1))


def physical_output(head, encoder, features, arm):
    value = head(features).float()
    if arm == 'R1':
        return value * encoder.action_std + encoder.action_mean
    if arm == 'R2':
        return symexp(value)
    raise ValueError(arm)


def make_sample(head, encoder, arm):
    @torch.no_grad()
    def sample(image, state, oct_feature, velocity, previous_action, **kwargs):
        features = geom_condition(encoder, state, oct_feature, velocity, previous_action)
        action = physical_output(head, encoder, features, arm)
        if not torch.isfinite(action).all():
            raise FloatingPointError('Nonfinite deterministic action; no clamping/fallback')
        return action
    return sample
