from __future__ import annotations

import unittest

from restricted_perception.statistics import paired_comparison


def _episode(target, seed, success, error):
    return {
        "target_robot_mm": list(target),
        "seed": seed,
        "success": success,
        "final_error_mm": error,
    }


class PairedStatisticsTests(unittest.TestCase):
    def test_pairs_by_target_and_seed(self):
        baseline = [
            _episode([0, 0, 2], 1, False, 0.5),
            _episode([1, 0, 2], 1, True, 0.01),
        ]
        candidate = [
            _episode([1, 0, 2], 1, True, 0.005),
            _episode([0, 0, 2], 1, True, 0.01),
            _episode([2, 0, 2], 1, True, 0.01),
        ]
        report = paired_comparison(
            candidate,
            baseline,
            bootstrap_samples=100,
        )
        self.assertEqual(report["paired_episodes"], 2)
        self.assertAlmostEqual(report["success_rate_difference"], 0.5)
        self.assertLess(report["paired_error_difference_mean_um"], 0.0)


if __name__ == "__main__":
    unittest.main()
