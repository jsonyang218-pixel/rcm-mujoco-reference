"""Tests for the direct-OCT guided residual regression structure."""
from __future__ import annotations

import numpy as np

from rcm_reference import Reference
from lerobot_training.train_oct_guided_regression import (
    TASK_TO_SI,
    camera_error_to_world,
    distance_aware_oct_features,
    proportional_guided_action,
    residual_limit_for_distance,
    task_jacobian,
)


def _direction(vector):
    norm = float(np.linalg.norm(vector))
    assert norm > 0
    return np.asarray(vector, dtype=np.float64) / norm


def test_camera_error_round_trip():
    rng = np.random.default_rng(20261105)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    camera_error = np.array([0.003, -0.004, 0.005])
    world_error = camera_error_to_world(camera_error, rotation)
    recovered = rotation.T @ world_error
    recovered *= np.array([1.0, -1.0, -1.0])
    np.testing.assert_allclose(recovered, camera_error, atol=1e-12, rtol=0)


def test_numeric_jacobian_predicts_tip_delta():
    reference = Reference()
    q = np.array([0.01, -0.005, 0.018])
    jacobian = task_jacobian(reference, q)
    delta = np.array([0.0002, -0.0001, 0.00003])
    predicted = jacobian @ delta
    tip = reference.robot_to_world(reference.forward_kinematics(q))
    shifted = reference.robot_to_world(
        reference.forward_kinematics(q + delta)
    )
    actual = shifted - tip
    relative_error = np.linalg.norm(predicted - actual) / np.linalg.norm(actual)
    assert relative_error < 2e-3, (predicted, actual, relative_error)


def test_proportional_guided_action_direction_saturation_and_invalid():
    reference = Reference()
    _, rotation = reference.camera_pose()
    state = np.array([2.0, -1.0, 18.0])
    camera_error = np.array([0.004, -0.003, 0.002])
    saturation = np.array([0.29, 0.15, 0.05])
    gain = 0.15
    action = proportional_guided_action(
        reference,
        state,
        camera_error,
        rotation,
        gain=gain,
        saturation=saturation,
    )
    assert np.all(np.abs(action) <= gain * saturation + 1e-12)
    jacobian = task_jacobian(reference, state * TASK_TO_SI)
    world_step = jacobian @ (action / np.array([180.0 / np.pi, 180.0 / np.pi, 1000.0]))
    expected_direction = camera_error_to_world(camera_error, rotation)
    assert _direction(world_step) @ _direction(expected_direction) > 0.999

    huge = proportional_guided_action(
        reference,
        state,
        camera_error * 1e6,
        rotation,
        gain=gain,
        saturation=saturation,
    )
    np.testing.assert_allclose(
        np.abs(huge),
        gain * saturation,
        atol=1e-12,
    )
    invalid = proportional_guided_action(
        reference,
        state,
        camera_error,
        rotation,
        valid=False,
        gain=gain,
        saturation=saturation,
    )
    np.testing.assert_allclose(invalid, 0.0, atol=0)


def test_distance_features_and_residual_limit():
    oct_navigation = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.02, 1.0],
        [0.0, 0.0, 0.1, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    direction, log_distance, distance = distance_aware_oct_features(
        oct_navigation
    )
    np.testing.assert_allclose(direction[1], [0.0, 0.0, 1.0], atol=0)
    assert np.all(np.diff(log_distance[1:4]) > 0)
    np.testing.assert_allclose(distance, [0.0, 0.02, 0.1, 1.0, 0.0])

    limits = residual_limit_for_distance(
        distance,
        oct_navigation[:, 3] > 0.5,
        far_limit=np.array([0.08, 0.04, 0.02]),
        full_effect_distance_mm=1.0,
        near_fraction=0.1,
    )
    assert np.all(limits[1, 2] <= limits[2, 2] <= limits[3, 2])
    assert np.all(limits[0] == 0)
    assert np.all(limits[4] == 0)


def test_residual_policy_is_bounded():
    import torch

    from lerobot_training.train_oct_guided_regression import (
        GuidedOCTResidualPolicy,
    )

    model = GuidedOCTResidualPolicy(
        state_mean=np.zeros(3),
        state_std=np.ones(3),
        log_distance_mean=0.0,
        log_distance_std=1.0,
        base_mean=np.zeros(3),
        base_std=np.ones(3),
        far_residual_limit=np.array([0.08, 0.04, 0.02]),
        full_effect_distance_mm=1.0,
        near_fraction=0.1,
        hidden_dim=32,
    )
    with torch.no_grad():
        for parameter in model.network[-1].parameters():
            parameter.fill_(100.0)
    state = torch.zeros(4, 3)
    oct_navigation = torch.tensor([
        [0.0, 0.0, 0.01, 1.0],
        [0.0, 0.0, 0.1, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    base = torch.zeros(4, 3)
    action, residual, limits, _ = model(
        state,
        oct_navigation,
        base,
    )
    assert action.shape == (4, 3)
    assert torch.all(torch.abs(residual) <= limits + 1e-6)
    assert torch.all(torch.abs(action - base) <= limits + 1e-6)


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
