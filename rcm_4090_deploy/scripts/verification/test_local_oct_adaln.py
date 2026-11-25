"""Contract checks for the local OCT AdaLN conditioning path."""
import unittest
from unittest import mock

import torch
from torch import nn

from lerobot_training import dit_state_attention as attention


class LocalOctAdaLNTests(unittest.TestCase):
    def make_policy(self, enabled):
        policy = attention.StateConditionedDiTRectifiedFlowPolicy.__new__(
            attention.StateConditionedDiTRectifiedFlowPolicy
        )
        nn.Module.__init__(policy)
        policy.oct_symlog_enabled = False
        policy.use_local_oct_branch = enabled
        policy.local_oct_tanh_scale_mm = 0.05
        policy.local_oct_log_scale_mm = 0.005
        policy.local_oct_log_reference_mm = 0.5
        if enabled:
            policy.local_oct_mlp = nn.Sequential(nn.Linear(7, 4))
        return policy

    def test_disabled_preserves_baseline_condition_exactly(self):
        policy = self.make_policy(False)
        tokens, global_condition = torch.randn(2, 3, 4), torch.randn(2, 4)
        with mock.patch.object(attention._OriginalPolicy, "_condition",
                               return_value=(tokens, global_condition)):
            actual = policy._condition(None, None, torch.zeros(2, 1, 4), None, None)
        self.assertIs(actual[0], tokens)
        self.assertIs(actual[1], global_condition)
        self.assertTrue(torch.equal(actual[0], tokens))
        self.assertTrue(torch.equal(actual[1], global_condition))

    def test_local_transform_is_identical_in_training_and_eval(self):
        policy = self.make_policy(True)
        oct_data = torch.tensor([[[0.02, -0.01, 0.0, 1.0],
                                  [0.1, 0.2, 0.3, 0.0]]])
        policy.train()
        train_features, valid = policy.local_oct_features(oct_data)
        policy.eval()
        eval_features, _ = policy.local_oct_features(oct_data)
        self.assertTrue(torch.equal(train_features, eval_features))
        self.assertTrue(torch.equal(train_features[0, 1], torch.zeros(7)))
        self.assertEqual(valid[0, 0, 0].item(), 1)

    def test_local_branch_leaves_tokens_and_motion_condition_untouched(self):
        policy = self.make_policy(True)
        tokens, base = torch.randn(2, 3, 4), torch.randn(2, 4)
        oct_data = torch.tensor([[[0.02, -0.01, 0., 1.]]] * 2)
        with mock.patch.object(attention._OriginalPolicy, "_condition",
                               return_value=(tokens, base)):
            actual_tokens, (motion, residual) = policy._condition(
                None, None, oct_data, None, None
            )
        self.assertIs(actual_tokens, tokens)
        self.assertIs(motion, base)
        self.assertEqual(tuple(residual.shape), tuple(base.shape))
        residual.sum().backward()
        self.assertGreater(policy.local_oct_mlp[0].weight.grad.abs().sum().item(), 0)


class WorkflowGateTests(unittest.TestCase):
    def test_gate_requires_all_five_and_exact_card_thresholds(self):
        from scripts.research.local_oct_adaln_v1 import dev20_gate
        self.assertTrue(dev20_gate(dict(completed_targets=5, final_mean_um=57.9,
                                       min_mean_um=24.9, entered_20_um_count=0)))
        self.assertFalse(dev20_gate(dict(completed_targets=5, final_mean_um=58.,
                                        min_mean_um=24.9, entered_20_um_count=0)))
        self.assertTrue(dev20_gate(dict(completed_targets=5, final_mean_um=90.,
                                       min_mean_um=50., entered_20_um_count=2)))
        self.assertFalse(dev20_gate(dict(completed_targets=2, final_mean_um=10.,
                                        min_mean_um=5., entered_20_um_count=2)))

    def test_k4_samples_reach_evaluation_worker(self):
        from types import SimpleNamespace
        from scripts.research.local_oct_adaln_v1 import _evaluate_worker
        with mock.patch('scripts.research.homog_replay_evaluate.evaluate') as evaluate, \
                mock.patch('scripts.research.homog_replay_common.identity', return_value={}):
            _evaluate_worker(SimpleNamespace(checkpoint='selected.pth', samples=4,
                indices='6,10', destination='test_output', workers=2, action_aggregation='mean'))
        args, kwargs = evaluate.call_args
        self.assertEqual(args[1], 4)
        self.assertEqual(args[3], (6, 10))
        self.assertEqual(kwargs['action_aggregation'], 'mean')

    def test_report_handles_failed_preflight(self):
        import tempfile
        from pathlib import Path
        from scripts.research import local_oct_adaln_v1 as workflow
        with tempfile.TemporaryDirectory(prefix='rcm_oct_report_') as directory:
            with mock.patch.object(workflow, 'OUTPUT', Path(directory)):
                workflow.write_report({'status': 'failed', 'reason': 'missing asset'})
            self.assertIn('missing asset', (Path(directory) / 'report.md').read_text())


if __name__ == "__main__":
    unittest.main()
