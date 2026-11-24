"""Stage 4.1 OCT symlog train/eval parity contracts."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OCTSymlogParityTests(unittest.TestCase):
    def test_train_and_eval_share_exact_normalization(self):
        import torch
        from lerobot_training.dit_state_attention import (
            StateConditionedDiTRectifiedFlowPolicy,
        )

        model = StateConditionedDiTRectifiedFlowPolicy(
            n_obs_steps=2,
            n_action_steps=16,
            state_mean=[0., 0., 0.],
            state_std=[1., 1., 1.],
            oct_mean=[0., 0., 0., 0.],
            oct_std=[1.2167125, .9585331, 5.9810953, .170527],
            action_mean=[0., 0., 0.],
            action_std=[1., 1., 1.],
            velocity_mean=[0., 0., 0.],
            velocity_std=[1., 1., 1.],
            previous_action_mean=[0., 0., 0.],
            previous_action_std=[1., 1., 1.],
            oct_symlog_enabled=True,
        )
        oct_feature = torch.tensor(
            [[.02, .02, .02, 1.], [1., 1., 1., 1.]],
            dtype=torch.float32,
        )
        model.train()
        train_value, _ = model.normalize_oct_symlog(oct_feature)
        model.eval()
        eval_value, _ = model.normalize_oct_symlog(oct_feature)
        torch.testing.assert_close(
            train_value, eval_value, rtol=0, atol=0,
        )
        self.assertTrue(np.all(
            (train_value[0].numpy() >= .30)
            & (train_value[0].numpy() <= .45)
        ))
        expected = (
            np.log1p(.02 / .001)
            / np.log1p(np.array([1.2167125, .9585331, 5.9810953]) / .001)
        )
        np.testing.assert_allclose(
            eval_value[0].numpy(), expected, rtol=1e-6, atol=1e-7,
        )

    def test_invalid_oct_is_zero_and_masked(self):
        import torch
        from lerobot_training.dit_state_attention import (
            StateConditionedDiTRectifiedFlowPolicy,
        )

        model = StateConditionedDiTRectifiedFlowPolicy(
            n_obs_steps=2,
            n_action_steps=16,
            state_mean=[0., 0., 0.],
            state_std=[1., 1., 1.],
            oct_mean=[0., 0., 0., 0.],
            oct_std=[1., 1., 1., 1.],
            action_mean=[0., 0., 0.],
            action_std=[1., 1., 1.],
            velocity_mean=[0., 0., 0.],
            velocity_std=[1., 1., 1.],
            previous_action_mean=[0., 0., 0.],
            previous_action_std=[1., 1., 1.],
            oct_symlog_enabled=True,
        )
        value, valid = model.normalize_oct_symlog(
            torch.tensor([[1., 2., 3., 0.]])
        )
        self.assertFalse(bool(valid.any()))
        self.assertTrue(bool((value == 0.).all()))


if __name__ == "__main__":
    unittest.main()
