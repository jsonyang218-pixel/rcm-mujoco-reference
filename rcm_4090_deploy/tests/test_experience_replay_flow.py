"""Unit checks for pure Flow experience-replay sampling."""
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_training"))
import dit_experience_replay as replay


def source_dataset():
    return types.SimpleNamespace(
        metadata={
            "episodes": [
                {"episode_index": 0, "action_source": "executed"},
                {"episode_index": 1, "action_source": "teacher"},
                {"episode_index": 2, "action_source": "teacher"},
                {"episode_index": 3, "action_source": "executed"},
            ]
        },
        episode_indices=np.array([0, 0, 1, 1, 2, 3], dtype=np.int64),
        valid_indices=[0, 1, 2, 3, 4, 5],
    )


class ExperienceReplayFlowTests(unittest.TestCase):
    def test_split_uses_episode_supervision_source(self):
        original, correction = replay.split_experience_indices(source_dataset())
        np.testing.assert_array_equal(original, [0, 1, 5])
        np.testing.assert_array_equal(correction, [2, 3, 4])

    def test_balanced_sampler_has_exact_half_split(self):
        sampler = replay.BalancedReplacementBatchSampler(
            original_count=5,
            correction_count=7,
            batches=4,
            batch_size=8,
            seed=11,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertEqual(len(batch), 8)
            self.assertTrue(all(0 <= index < 5 for index in batch[:4]))
            self.assertTrue(all(5 <= index < 12 for index in batch[4:]))

    def test_balanced_sampler_rejects_odd_batch_size(self):
        with self.assertRaises(ValueError):
            replay.BalancedReplacementBatchSampler(2, 2, 1, 7, 3)

    def test_sampling_is_reproducible_for_same_seed(self):
        first = list(replay.BalancedReplacementBatchSampler(3, 4, 5, 16, 17))
        second = list(replay.BalancedReplacementBatchSampler(3, 4, 5, 16, 17))
        self.assertEqual(first, second)

    def test_index_subset_rejects_out_of_range_indices(self):
        with self.assertRaises(IndexError):
            replay.IndexSubset(list(range(3)), [0, 3])

    def test_valid_window_has_no_extra_boundary_frame(self):
        count = 20
        indices, diagnostics = replay.valid_window_indices(
            np.zeros(count, dtype=np.int64),
            np.arange(count, dtype=np.int64),
            np.arange(count, dtype=np.float64) * 0.05,
            n_obs_steps=2,
            n_action_steps=16,
            control_period_s=0.05,
            oct_navigation=np.zeros((count, 3), dtype=np.float32),
            oct_navigation_valid=np.ones(count, dtype=np.bool_),
            return_diagnostics=True,
        )
        np.testing.assert_array_equal(indices, [1, 2, 3, 4])
        self.assertEqual(diagnostics["valid_windows"], 4)

    def test_ratio_sampler_reaches_requested_long_run_ratio(self):
        sampler = replay.RatioReplacementBatchSampler(
            original_count=5,
            correction_count=5,
            correction_ratio=0.1,
            batches=1000,
            batch_size=16,
            seed=17,
        )
        correction = sum(
            index >= 5
            for batch in sampler
            for index in batch
        )
        self.assertAlmostEqual(correction / 16000, 0.1, places=2)

    def test_unknown_label_source_fails_loudly(self):
        dataset = source_dataset()
        dataset.metadata["episodes"].pop()
        with self.assertRaisesRegex(ValueError, "no declared action source"):
            replay.split_experience_indices(dataset)


if __name__ == "__main__":
    unittest.main()
