"""Contract tests for guided-OCT image diffusion residual integration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from lerobot_training.train_diffusion_policy import (
    RCMDataset,
    SimpleDiffusionPolicy,
)


def _policy():
    return SimpleDiffusionPolicy(
        n_obs_steps=1,
        n_action_steps=2,
        num_diffusion_iters=4,
        oct_dim=4,
        guided_oct_mode=True,
        state_mean=np.zeros(3),
        state_std=np.ones(3),
        oct_mean=np.zeros(4),
        oct_std=np.ones(4),
        action_mean=np.zeros(3),
        action_std=np.ones(3),
        guided_base_mean=np.zeros(3),
        guided_base_std=np.ones(3),
        log_distance_mean=0.0,
        log_distance_std=1.0,
        far_residual_limit=np.array([0.29, 0.15, 0.05]),
        full_effect_distance_mm=1.0,
        near_fraction=0.1,
        pred_x0_clip=4.0,
    )


def test_guided_diffusion_training_and_sampling():
    torch.manual_seed(20261105)
    model = _policy()
    image = torch.zeros(3, 1, 3, 8, 8)
    state = torch.tensor([
        [0.0, 0.0, 18.0],
        [1.0, -1.0, 19.0],
        [2.0, 1.0, 20.0],
    ])[:, None, :]
    oct_navigation = torch.tensor([
        [0.0, 0.0, 0.01, 1.0],
        [0.0, 0.0, 0.1, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    base = torch.tensor([
        [0.001, 0.002, 0.003],
        [0.01, 0.02, 0.03],
        [0.0, 0.0, 0.0],
    ])
    actions = base[:, None, :] + torch.full((3, 2, 3), 0.001)

    loss = model(
        image,
        state,
        actions,
        obs_oct=oct_navigation[:, None, :],
        guided_base=base,
        generator=torch.Generator().manual_seed(20261105),
    )
    assert torch.isfinite(loss)
    loss.backward()

    with torch.no_grad():
        sampled = model(
            image,
            state,
            obs_oct=oct_navigation[:, None, :],
            guided_base=base,
            generator=torch.Generator().manual_seed(20261105),
        )
    assert sampled.shape == (3, 2, 3)
    limits = torch.tensor([
        [0.029, 0.015, 0.005],
        [0.29, 0.15, 0.05],
        [0.0, 0.0, 0.0],
    ])
    assert torch.all(
        torch.abs(sampled - base[:, None, :]) <= limits[:, None, :] + 1e-6
    )
    assert torch.allclose(sampled[2], torch.zeros_like(sampled[2]), atol=1e-7)


def test_guided_detail_state_encoder_width():
    model = _policy()
    assert model.state_encoder.in_features == 11
    assert model.guided_oct_mode


def test_dataset_guided_sidecars_align():
    dataset = RCMDataset(
        Path("data/lerobot_rcm_train_recovery_full_224").resolve(),
        transforms.ToTensor(),
        n_obs_steps=2,
        n_action_steps=8,
    )
    assert dataset.has_guided_oct_action
    sample = dataset[0]
    assert sample["observation.guided_base"].shape == (3,)
    assert sample["observation.guided_limit"].shape == (3,)
    assert torch.all(sample["observation.guided_limit"] >= 0)
    assert torch.any(sample["observation.guided_base"] != 0)


def test_cached_vision_feature_forward():
    torch.manual_seed(20261106)
    model = SimpleDiffusionPolicy(
        n_obs_steps=1,
        n_action_steps=2,
        num_diffusion_iters=4,
        oct_dim=4,
        guided_oct_mode=True,
        use_cached_vision_features=True,
        state_mean=np.zeros(3),
        state_std=np.ones(3),
        oct_mean=np.zeros(4),
        oct_std=np.ones(4),
        action_mean=np.zeros(3),
        action_std=np.ones(3),
        guided_base_mean=np.zeros(3),
        guided_base_std=np.ones(3),
        log_distance_mean=0.0,
        log_distance_std=1.0,
        far_residual_limit=np.array([0.29, 0.15, 0.05]),
        full_effect_distance_mm=1.0,
        near_fraction=0.1,
        pred_x0_clip=4.0,
    )
    state = torch.zeros(2, 1, 3)
    features = torch.randn(2, 1, 512)
    oct_navigation = torch.tensor([
        [[0.0, 0.0, 0.01, 1.0]],
        [[0.0, 0.0, 1.0, 1.0]],
    ])
    base = torch.zeros(2, 3)
    actions = torch.zeros(2, 2, 3)
    loss = model(
        None,
        state,
        actions,
        obs_oct=oct_navigation,
        guided_base=base,
        obs_vision_feature=features,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.isfinite(loss)
    with torch.no_grad():
        sampled = model(
            None,
            state,
            obs_oct=oct_navigation,
            guided_base=base,
            obs_vision_feature=features,
            generator=torch.Generator().manual_seed(2),
        )
    assert sampled.shape == (2, 2, 3)
    assert torch.isfinite(sampled).all()


def test_dataset_uses_cached_vision_features():
    dataset = RCMDataset(
        Path("runs/checks/feature_cache_smoke_20261106").resolve(),
        transform=None,
        n_obs_steps=1,
        n_action_steps=1,
        use_vision_feature_cache=True,
    )
    sample = dataset[0]
    assert "observation.image" not in sample
    assert sample["observation.vision_feature"].shape == (1, 512)
    assert sample["observation.vision_feature"].dtype == torch.float16


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
