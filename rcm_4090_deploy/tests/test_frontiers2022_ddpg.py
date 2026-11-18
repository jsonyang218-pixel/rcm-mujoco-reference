"""Numerical/autograd checks for the paper-to-RCM DDPG adaptation."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_training"))
import dit_rectified_flow as flow
import dit_replay_actor_critic as replay
import frontiers2022_ddpg as ddpg
from experience_replay import ACTION_LIMIT


def small_policy(seed=17):
    statistics = {}
    for name, width in (("state", 3), ("oct", 4), ("action", 3),
                        ("velocity", 3), ("previous_action", 3)):
        statistics[name + "_mean"] = [0.0] * width
        statistics[name + "_std"] = [1.0] * width
    statistics["action_std"] = [0.01, 0.01, 0.001]
    # Exercise real ResNet/DiT/Flow, without an ImageNet download in tests.
    def local_resnet(**kwargs):
        return resnet18(**{**kwargs, "weights": None})
    with patch.object(flow, "resnet18", local_resnet):
        model = ddpg.DeterministicFlowPolicy(
            **statistics, n_action_steps=4, dit_dim=24, dit_depth=1,
            dit_heads=3, flow_integration_steps=2, fixed_noise_seed=seed,
        )
    torch.nn.init.normal_(model.output.weight, std=0.01)
    return model


def batch():
    return {"observation." + name: torch.zeros(shape) for name, shape in (
        ("image", (2, 2, 3, 32, 32)), ("state", (2, 2, 3)),
        ("oct", (2, 2, 4)), ("velocity", (2, 3)), ("previous_action", (2, 3)),
    )}


class PaperDDPGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reward_units_and_optional_obstacles(self):
        self.assertAlmostEqual(float(ddpg.paper_reward(1.0)), -250.0)
        self.assertAlmostEqual(float(ddpg.paper_reward(100e-6)), -2.5e-6)
        self.assertAlmostEqual(float(ddpg.paper_reward(0.0, 0.0)), -15.0)
        np.testing.assert_allclose(ddpg.paper_reward([0.0, 1.0], 1.0),
                                   np.array([0.0, -250.0]) - 15.0 / 2 ** 35)

    def test_replay_reward_is_next_oct_metres_and_preserves_gaps(self):
        count = 8
        data = types.SimpleNamespace(
            states=np.zeros((count, 3), np.float32),
            executed_actions=np.zeros((count, 3), np.float32),
            episode_indices=np.zeros(count, np.int64),
            source_frame_indices=np.arange(count), timestamps_s=np.arange(count) * .05,
            oct_navigation=np.column_stack([np.arange(count) * .1, np.zeros((count, 2))]),
            oct_navigation_valid=np.ones(count, bool), has_history_contract=True,
        )
        data.source_frame_indices[4:] += 2
        records = ddpg.transition_records(data, 3)
        np.testing.assert_array_equal(records["row"], [1, 2, 5, 6])
        np.testing.assert_allclose(records["reward"], [-1e-5, -2.25e-5, -9e-5, -1.225e-4])
        self.assertFalse(records["terminated"].any())

    def test_actual_flow_is_deterministic_and_reload_recreates_noise(self):
        torch.manual_seed(3)
        model, observation = small_policy(), batch()
        arguments = replay.observation_arguments(observation)
        first = model.sample(*arguments, generator=torch.Generator().manual_seed(1))
        torch.randn(100)
        model.eval()
        second = model.sample(*arguments, generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        # State dict remains compatible with the original v1 warm-start keys.
        self.assertNotIn("fixed_flow_noise", model.state_dict())
        restored = small_policy(seed=model.fixed_noise_seed).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(first, restored.sample(*arguments), rtol=0, atol=0)

    def test_q_updates_real_dit_output_without_updating_critic(self):
        torch.manual_seed(3)
        model, observation = small_policy(), batch()
        critic = ddpg.SingleQ(26, hidden_dim=8)
        for parameter in critic.parameters():
            parameter.data.fill_(0.1)
        original_critic = copy.deepcopy(critic.state_dict())
        original_output = model.output.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metric, objective, q = ddpg.update_actor(
            model, critic, optimizer, None, observation, torch.tensor(ACTION_LIMIT),
            fm_weight=0.0, q_weight=1.0,
        )
        self.assertFalse(torch.equal(original_output, model.output.weight))
        self.assertAlmostEqual(objective, -q)
        self.assertEqual(metric, objective)
        for name, value in critic.state_dict().items():
            torch.testing.assert_close(value, original_critic[name], rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in critic.parameters()))

    def test_single_q_target_masks_true_terminals_and_soft_updates(self):
        torch.manual_seed(3)
        model, observation = small_policy(), batch()
        critic = ddpg.SingleQ(26, hidden_dim=8)
        target = copy.deepcopy(critic).requires_grad_(False)
        for parameter in target.parameters():
            parameter.data.zero_()
        target.network[-1].bias.data.fill_(2.0)
        before = copy.deepcopy(target.state_dict())
        transition = (observation, observation, torch.zeros(2, 3), torch.full((2,), 3.0),
                      torch.tensor([True, False]))
        _, _, mean = ddpg.update_critic(
            model, model, critic, target, torch.optim.SGD(critic.parameters(), lr=.01),
            transition, torch.tensor(ACTION_LIMIT), gamma=.9, tau=.1,
        )
        self.assertAlmostEqual(mean, 3.9, places=5)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, .9 * before[name] + .1 * critic.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
