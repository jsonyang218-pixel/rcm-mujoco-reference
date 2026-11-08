from __future__ import annotations

import unittest

import torch

from restricted_perception.stage3_policy import Stage3ResidualActionPolicy


class Stage3PolicyTests(unittest.TestCase):
    def test_zero_initialized_residual_preserves_base_action(self):
        policy = Stage3ResidualActionPolicy(
            vision_dim=16,
            state_dim=3,
            restricted_dim=18,
            use_velocity=False,
        )
        policy.eval()
        features = torch.randn(4, 16 + 3 + 18)
        with torch.no_grad():
            combined, stop_logit = policy(features)
            base, _ = policy.base(features)
        self.assertIsNone(stop_logit)
        self.assertEqual(tuple(combined.shape), (4, 3))
        self.assertTrue(torch.allclose(combined, base, atol=1e-7))

    def test_residual_is_bounded(self):
        policy = Stage3ResidualActionPolicy(
            vision_dim=16,
            state_dim=3,
            restricted_dim=18,
            use_velocity=True,
        )
        torch.nn.init.constant_(policy.residual[-1].bias, 100.0)
        features = torch.randn(2, 16 + 3 + 18 + 3)
        with torch.no_grad():
            combined, _ = policy(features)
            base, _ = policy.base(features)
        correction = combined - base
        limit = policy.residual_limit.reshape(1, 3)
        self.assertTrue(bool((correction.abs() <= limit + 1e-7).all()))


if __name__ == "__main__":
    unittest.main()
