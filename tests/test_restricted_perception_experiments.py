from __future__ import annotations

from pathlib import Path
import unittest

from restricted_perception.config import load_config
from restricted_perception.experiments import (
    run_closed_loop,
    run_precision_scan,
    run_switch_scan,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "restricted_perception.json"


class RestrictedPerceptionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_switch_scan_has_required_distances_and_gate(self):
        report = run_switch_scan(self.config, episode_count=20)
        self.assertEqual(
            [row["switch_distance_mm"] for row in report["scan"][0]["distances"]],
            [1.0, 2.0, 3.0],
        )
        self.assertIn("selected_transition_half_width_mm", report)
        self.assertIn("pass", report)

    def test_closed_loop_is_deterministic(self):
        first = run_closed_loop(
            self.config,
            threshold_um=20.0,
            seeds_per_target=1,
            max_frames=180,
        )
        second = run_closed_loop(
            self.config,
            threshold_um=20.0,
            seeds_per_target=1,
            max_frames=180,
        )
        for key in (
            "episodes",
            "successes",
            "final_error_p50_um",
            "final_error_p95_um",
            "fallback_state_frames",
        ):
            self.assertEqual(first[key], second[key])

    def test_closed_loop_does_not_stop_on_early_precision_hold(self):
        report = run_closed_loop(
            self.config,
            threshold_um=20.0,
            seeds_per_target=1,
            max_frames=1600,
        )
        self.assertGreaterEqual(report["success_rate"], 0.8)
        self.assertLess(report["final_error_p95_um"], 100.0)

    def test_precision_scan_exposes_all_levels(self):
        report = run_precision_scan(self.config, seeds_per_target=1)
        self.assertEqual(
            [row["threshold_um"] for row in report["levels"]],
            [5.0, 10.0, 20.0],
        )
        self.assertIn("selected_threshold_um", report)


if __name__ == "__main__":
    unittest.main()
