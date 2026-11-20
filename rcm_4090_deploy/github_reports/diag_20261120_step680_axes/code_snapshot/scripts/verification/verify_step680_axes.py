"""Regression checks for time alignment, branch gates and teacher units."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.diagnose_step680_axes import (
    UNIT, choose_branch, crossings, metrics, zero_velocity_teacher,
)


class Step680AxisTests(unittest.TestCase):
    def test_zero_crossings_ignore_exact_zero_plateaus(self):
        self.assertEqual(crossings(np.array([1, 0, 0, -1, 0, 1])), 2)
        self.assertEqual(crossings(np.zeros(8)), 0)

    def test_last_command_uses_separate_post_budget_state(self):
        states = np.zeros((1600, 3))
        final = np.array([0.002, 0.003, 0.004])
        commands = np.ones_like(states)
        rows, _, movement = metrics(states, final, commands, np.zeros(3),
                                    np.ones_like(states), commands, commands)
        np.testing.assert_array_equal(movement[-1], final)
        self.assertEqual(rows["A"]["active_command_frames"], 900)
        self.assertEqual(rows["A"]["deadzone_frames"], 899)
        self.assertAlmostEqual(rows["A"]["minimum_nonzero_step"], 0.002)

    def test_branch_uses_deadzone_without_claiming_physical_floor(self):
        row = {"deadzone_fraction_active_commands": 0.44, "minimum_nonzero_step_tip_um": 0.2}
        selected = choose_branch({"E00": {"axes": {"A": row}}})
        self.assertEqual(selected["case"], 1)
        self.assertIn("not proof", selected["causal_limit"])
        row["deadzone_fraction_active_commands"] = 0.01
        self.assertIsNone(choose_branch({"E00": {"axes": {"A": row}}})["case"])

    def test_zero_velocity_law_matches_existing_teacher(self):
        from rcm.reference import Reference
        from rcm.visual_servo import VisualServoController, visual_servo_settings
        reference = Reference()
        q = np.array([0.01, 0.02, 0.019])
        jacobian = np.diag([20.0, 25.0, 1000.0])
        settings = visual_servo_settings(reference)
        settings["sample_period_s"] = 0.05
        context = SimpleNamespace(reference=reference, settings=settings,
                                  control_axes=np.ones(3, dtype=bool), completion_axes=np.ones(3, dtype=bool),
                                  measurement_goal=None, target_transform=None,
                                  _model_jacobian=lambda _: jacobian)
        context._visual_goal = lambda target, tip, obs: np.asarray(target)
        for error in (np.array([0.02, -0.03, 0.01]), np.array([0.0001, 0, 0])):
            # A synthetic observation tests algebra only, never used as experiment data.
            observation = {"target_pixel": [0, 0], "target_depth_m": 0.01, "tip_depth_m": 0.01,
                           "geometry": {"tip_world_mm": [0, 0, 0], "target_world_mm": error,
                                        "tip_quality": {"fit_quality_mm": 0},
                                        "target_quality": {"fit_quality_mm": 0}}}
            expected = VisualServoController.geometry_action_from_observation(
                context, q, observation, previous_velocity=np.zeros(3), command_q=q)
            self.assertTrue(expected["valid"])
            actual = zero_velocity_teacher(jacobian, error, reference.config["navigation"],
                                           settings["position_tolerance_um"])
            np.testing.assert_allclose(actual, expected["action_deg_deg_mm"], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
