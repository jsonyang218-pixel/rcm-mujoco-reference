from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from restricted_perception.policy import RestrictedActionMLP
from restricted_perception.sidecar import (
    RESTRICTED_FEATURE_NAMES,
    restricted_feature_matrix,
)
from restricted_perception.stage2_policy import Stage2ActionPolicy
from train_restricted_policy import _source_allowed


class RestrictedPolicyTests(unittest.TestCase):
    def test_feature_matrix_shape(self):
        count = 2
        data = {
            "episode_index": np.zeros(count, dtype=np.int64),
            "coarse_goal_vector_robot_mm": np.zeros((count, 3), dtype=np.float32),
            "fine_goal_vector_robot_mm": np.zeros((count, 3), dtype=np.float32),
            "coarse_valid": np.ones(count, dtype=bool),
            "fine_valid": np.zeros(count, dtype=bool),
            "coarse_age_s": np.zeros(count, dtype=np.float32),
            "fine_age_s": np.full(count, np.nan, dtype=np.float32),
            "coarse_sigma_mm": np.ones(count, dtype=np.float32),
            "fine_sigma_mm": np.ones(count, dtype=np.float32),
            "switch_weight": np.zeros(count, dtype=np.float32),
            "fallback_state": np.zeros(count, dtype=np.int8),
            "precision_sufficient": np.zeros(count, dtype=bool),
        }
        features = restricted_feature_matrix(data)
        self.assertEqual(features.shape, (count, len(RESTRICTED_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())

    def test_policy_output_shape_and_action_clip(self):
        model = RestrictedActionMLP(
            vision_dim=8,
            state_dim=3,
            restricted_dim=len(RESTRICTED_FEATURE_NAMES),
            n_action_steps=4,
        )
        model.action_clip = [0.1, 0.1, 0.01]
        output = model(torch.zeros(2, 8 + 3 + len(RESTRICTED_FEATURE_NAMES)))
        self.assertEqual(tuple(output.shape), (2, 4, 3))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool((output.abs() <= torch.tensor([0.1, 0.1, 0.01])).all()))

    def test_source_filter_returns_boolean_mask(self):
        episodes = np.array([0, 1, 2, 3], dtype=np.int64)
        metadata = {
            "source_datasets": [
                {"directory": "recovery", "episode_offset": 0, "episodes": 2},
                {"directory": "dagger", "episode_offset": 2, "episodes": 2},
            ]
        }
        include = _source_allowed(episodes, metadata, True)
        exclude = _source_allowed(episodes, metadata, False)
        self.assertEqual(include.dtype, np.bool_)
        self.assertTrue(include.all())
        self.assertEqual(exclude.tolist(), [True, True, False, False])

    def test_residual_cap_uses_ten_micrometre_scale(self):
        model = Stage2ActionPolicy(
            vision_dim=512,
            state_dim=3,
            restricted_dim=len(RESTRICTED_FEATURE_NAMES),
            use_velocity=True,
            residual_distance_scale=0.010,
            action_limit=[0.3, 0.15, 0.05],
        )
        features = torch.zeros(2, 512 + 3 + len(RESTRICTED_FEATURE_NAMES) + 3)
        offset = 512 + 3
        features[:, offset + 3] = torch.tensor([0.005, 0.020])
        features[:, offset + 7] = 1.0
        actions, _ = model(features)
        self.assertTrue(
            bool(
                (actions[0].abs() <= torch.tensor([0.15, 0.075, 0.025]) + 1e-6)
                .all()
            )
        )
        self.assertTrue(
            bool(
                (actions[1].abs() <= torch.tensor([0.3, 0.15, 0.05]) + 1e-6)
                .all()
            )
        )


if __name__ == "__main__":
    unittest.main()
