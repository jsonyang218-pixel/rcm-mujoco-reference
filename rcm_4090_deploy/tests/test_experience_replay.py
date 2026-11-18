"""Check transition alignment and that Q can really update the Flow actor."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_training"))
from experience_replay import ACTION_LIMIT, TransitionReplayBuffer, transition_records


def trajectory(count=8):
    return types.SimpleNamespace(
        states=np.zeros((count, 3), dtype=np.float32),
        actions=np.full((count, 3), 99.0, dtype=np.float32),
        executed_actions=np.zeros((count, 3), dtype=np.float32),
        episode_indices=np.zeros(count, dtype=np.int64),
        source_frame_indices=np.arange(count),
        timestamps_s=np.arange(count) * 0.05,
        oct_navigation=np.column_stack([np.linspace(1.0, 0.3, count), np.zeros((count, 2))]),
        oct_navigation_valid=np.ones(count, dtype=bool), has_history_contract=True,
    )


class ReplayDataTests(unittest.TestCase):
    def test_last_observation_is_available_without_false_terminal(self):
        data = trajectory()
        records = transition_records(data, 0)
        np.testing.assert_array_equal(records["row"], np.arange(1, 7))
        np.testing.assert_array_equal(records["next_row"], np.arange(2, 8))
        self.assertFalse(records["terminated"].any())
        explicit = np.zeros(8, dtype=bool)
        explicit[6] = True
        labeled = transition_records(data, 0, terminated=explicit)
        np.testing.assert_array_equal(labeled["terminated"], [False] * 5 + [True])

    def test_source_gaps_and_episode_boundaries_break_transitions(self):
        data = trajectory()
        data.source_frame_indices[4:] += 3
        records = transition_records(data, 0)
        np.testing.assert_array_equal(records["row"], [1, 2, 5, 6])
        data = trajectory()
        data.episode_indices[4:] = 1
        records = transition_records(data, 0)
        np.testing.assert_array_equal(records["row"], [1, 2, 5, 6])
        data = trajectory()
        data.timestamps_s[4:] += 0.05
        records = transition_records(data, 0)
        np.testing.assert_array_equal(records["row"], [1, 2, 5, 6])

    def test_reward_uses_executed_action_and_does_not_change_validity(self):
        data = trajectory()
        data.executed_actions[2] = ACTION_LIMIT
        data.oct_navigation[5, 0] = np.nan
        valid_before = data.oct_navigation_valid.copy()
        records = transition_records(data, 0)
        reward = records["reward"][records["row"] == 2][0]
        self.assertAlmostEqual(float(reward), 0.1 - 0.1 * 0.7 - 0.001, places=6)
        self.assertNotIn(4, records["row"])
        self.assertNotIn(5, records["row"])
        self.assertNotIn(6, records["row"])
        np.testing.assert_array_equal(data.oct_navigation_valid, valid_before)
        data.actions[:] = -1234
        changed = transition_records(data, 0)
        np.testing.assert_array_equal(changed, records)

    def test_reservoir_is_bounded_and_reproducible_across_sources(self):
        left = TransitionReplayBuffer(capacity=5, seed=7)
        right = TransitionReplayBuffer(capacity=5, seed=7)
        for source in range(8):
            records = transition_records(trajectory(), source)
            left.extend(records)
            right.extend(records)
        self.assertEqual(left.seen, 48)
        self.assertEqual(len(left), 5)
        np.testing.assert_array_equal(left.records, right.records)
        np.testing.assert_array_equal(left.sample(12), right.sample(12))


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))


@unittest.skipUnless(HAS_TORCH, "PyTorch/torchvision unavailable; gradient checks require the training environment")
class ReplayGradientTests(unittest.TestCase):
    def setUp(self):
        import torch
        from torch import nn
        import dit_replay_actor_critic as module
        self.torch, self.module = torch, module

        class TinyFlow(nn.Module):
            n_action_steps, action_dim, flow_integration_steps = 4, 3, 2

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.2))
                self.register_buffer("action_mean", torch.zeros(3))
                self.register_buffer("action_std", torch.tensor([0.01, 0.01, 0.001]))

            def _condition(self, image, state, oct_feature, velocity, previous_action):
                return state, state

            def _predict_velocity_from_condition(self, actions, tau, tokens, condition):
                return self.weight.expand_as(actions)

            def _normalize_observation(self, state, oct_feature, velocity, previous_action):
                return (state, oct_feature, velocity[:, None].expand(-1, 2, -1),
                        previous_action[:, None].expand(-1, 2, -1))

            def rectified_flow_loss(self, *arguments):
                return self.weight.square()

        self.actor = TinyFlow()
        self.batch = {"observation." + name: torch.zeros(shape) for name, shape in (
            ("image", (8, 2, 3, 8, 8)), ("state", (8, 2, 3)),
            ("oct", (8, 2, 4)), ("velocity", (8, 3)), ("previous_action", (8, 3)),
        )}
        self.batch["action"] = torch.zeros(8, 4, 3)
        self.limits = torch.tensor(ACTION_LIMIT)

    def test_q_gradient_alone_updates_actor_and_keeps_critic_fixed(self):
        torch, module = self.torch, self.module
        torch.manual_seed(3)
        critic = module.TwinQ(26, hidden_dim=8)
        for parameter in critic.parameters():
            parameter.data.fill_(0.1)
        before_actor = self.actor.weight.detach().clone()
        before_critic = [parameter.detach().clone() for parameter in critic.parameters()]
        optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.1)
        module.update_actor(self.actor, critic, optimizer, self.batch, self.batch,
                            self.limits, fm_weight=0.0, q_weight=1.0)
        self.assertNotEqual(float(before_actor), float(self.actor.weight.detach()))
        for before, after in zip(before_critic, critic.parameters()):
            torch.testing.assert_close(before, after)
            self.assertTrue(after.requires_grad)

    def test_true_terminal_masks_bootstrap(self):
        torch, module = self.torch, self.module
        class ConstantTarget(torch.nn.Module):
            def forward(self, geometry, action):
                values = torch.full((len(geometry),), 2.0)
                return values, values
        critic = module.TwinQ(26, hidden_dim=8)
        optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
        transition = (self.batch, self.batch, torch.zeros(8, 3), torch.full((8,), 3.0),
                      torch.tensor([True] * 4 + [False] * 4))
        _, _, target_mean = module.update_critic(
            self.actor, self.actor, critic, ConstantTarget(), optimizer, transition,
            self.limits, gamma=0.9, tau=0.005,
        )
        self.assertAlmostEqual(target_mean, 3.9, places=5)


if __name__ == "__main__":
    unittest.main()
