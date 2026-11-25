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
    # Large cache GEMMs and single-observation GEMVs select different kernels.
    # Keep this frozen map in IEEE FP32 on both paths; leave head-training AMP
    # and the caller's original matmul setting unchanged.
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        with torch.autocast(device_type=state.device.type, enabled=False):
            normalized = encoder._normalize_observation(
                state.float(), oct_feature.float(), velocity.float(), previous_action.float())
            return encoder.global_condition(torch.cat(normalized, dim=-1).flatten(1))
    finally:
        torch.set_float32_matmul_precision(precision)


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
