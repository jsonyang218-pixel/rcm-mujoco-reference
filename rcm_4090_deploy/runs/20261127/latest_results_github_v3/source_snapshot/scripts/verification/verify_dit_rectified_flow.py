"""Smoke-test the DiT rectified-flow policy without environment rollout."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lerobot_training"))

from dit_rectified_flow import DiTRectifiedFlowPolicy


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    policy = DiTRectifiedFlowPolicy(
        n_obs_steps=2,
        n_action_steps=16,
        state_mean=[0, 0, 0],
        state_std=[1, 1, 1],
        oct_mean=[0, 0, 0, 0],
        oct_std=[1, 1, 1, 1],
        action_mean=[0, 0, 0],
        action_std=[1, 1, 1],
        velocity_mean=[0, 0, 0],
        velocity_std=[1, 1, 1],
        previous_action_mean=[0, 0, 0],
        previous_action_std=[1, 1, 1],
        dit_dim=192,
        dit_depth=2,
        dit_heads=6,
        flow_integration_steps=4,
    ).to(device)
    image = torch.rand(2, 2, 3, 224, 224, device=device)
    state = torch.randn(2, 2, 3, device=device)
    oct_feature = torch.randn(2, 2, 4, device=device)
    velocity = torch.randn(2, 3, device=device)
    previous_action = torch.randn(2, 3, device=device)
    actions = torch.randn(2, 16, 3, device=device)
    loss = policy.rectified_flow_loss(
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
        actions,
    )
    assert torch.isfinite(loss)
    loss.backward()
    sampled = policy.sample(
        image,
        state,
        oct_feature,
        velocity,
        previous_action,
        steps=2,
    )
    assert sampled.shape == (2, 16, 3)
    assert torch.isfinite(sampled).all()
    print(json.dumps({
        "result": "PASS",
        "policy_type": policy.policy_type,
        "parameters_m": (
            sum(parameter.numel() for parameter in policy.parameters())
            / 1e6
        ),
        "loss": float(loss.detach()),
        "sample_shape": list(sampled.shape),
        "flow": "rectified_flow_euler",
        "backbone": "diffusion_transformer",
    }, indent=2))


if __name__ == "__main__":
    main()
