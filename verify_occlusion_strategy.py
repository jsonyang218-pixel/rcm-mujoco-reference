#!/usr/bin/env python3
"""Fast sensor-injection tests for S0a-S3 occlusion policies."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import rcm_visual_servo as servo
from rcm_occlusion_policy import (
    S0A_STOP_INSTANT, S0B_STOP_BRAKE, S1_OPEN_LOOP, S2_TARGET_MEMORY,
    S3_RETRACT_REOBSERVE,
)
from verify_visual_motion import COUPLED, OFFSET, LinearPlant, MeasuredSensor


OUTPUT = Path("runs/checks/occlusion_strategy")


class IntervenedSensor(MeasuredSensor):
    """Hide the target from selected observations without editing image data."""

    def __init__(self, reference, settings, hidden, start=6):
        super().__init__(reference, settings)
        self.hidden = hidden
        self.start = int(start)
        self.loss_c_m = None
        self.was_hidden = False

    def observe(self):
        index = self.calls
        if index == self.start:
            self.loss_c_m = float(self.reference.current_task()[2])
        observation = super().observe()
        hide = index >= self.start and self.hidden(
            index, float(self.reference.current_task()[2]), self.loss_c_m)
        if hide:
            self.was_hidden = True
            observation["geometry"]["target_world_mm"] = None
            observation["geometry"]["target_error"] = "TARGET_NOT_VISIBLE"
            observation.update(target_pixel=None, target_depth_m=None, target_pixels=0)
        return observation


def make(mode, hidden, *, goal=None, generous_acceleration=True,
         max_iterations=600):
    goal = np.asarray(
        [.005, -.004, .0001] if goal is None else goal, dtype=float)
    target = OFFSET+COUPLED@goal
    plant = LinearPlant(COUPLED, target,
                        generous_acceleration=generous_acceleration)
    settings = {
        **servo.DEFAULT_SERVO,
        "position_feedback": "geometry",
        "perception_backend": "simulation_ids",
        "occlusion_handling": mode,
        "position_tolerance_um": 8.,
        "endpoint_tolerance_um": 20.,
        "require_settled": True,
        "settle_frames": 3,
        "save_diagnostics": False,
        "max_iterations": max_iterations,
        "sample_period_s": .05,
        "geometry_gain_per_s": 3.,
        "occlusion_reacquire_frames": 3,
        "occlusion_max_recovery_attempts": 3,
        "occlusion_memory_max_age_s": 2.2,
        "occlusion_retract_mm": 1.,
        "occlusion_retract_speed_mm_s": .3,
        "occlusion_recovery_timeout_s": 5.,
    }
    factory = lambda reference, options: IntervenedSensor(
        reference, options, hidden)
    with patch.object(servo, "VisualServoSensor", factory):
        controller = servo.VisualServoController(plant, settings=settings)
    controller._model_jacobian = lambda q: COUPLED.copy()
    return controller, plant


class OcclusionStrategyChecks(unittest.TestCase):
    def test_s0a_immediate_stop_records_abrupt_hold(self):
        controller, _ = make(
            S0A_STOP_INSTANT, lambda index, c, origin: True,
            goal=[.03, -.02, .002], generous_acceleration=False)
        report = controller.run()
        self.assertEqual(report["result"], "FAIL")
        self.assertEqual(report["controller_reason"], "TARGET_NOT_VISIBLE")
        self.assertTrue(report["simulation_abort_hold"])
        self.assertFalse(report["trajectory_rate_limits_pass"])

    def test_s0b_brakes_with_limits_then_fails_closed(self):
        controller, _ = make(
            S0B_STOP_BRAKE, lambda index, c, origin: True,
            generous_acceleration=False)
        report = controller.run()
        self.assertEqual(report["result"], "FAIL")
        self.assertEqual(
            report["controller_reason"],
            "STOP_BRAKE_COMPLETE_WITHOUT_REACQUISITION")
        self.assertFalse(report["simulation_abort_hold"])
        self.assertTrue(report["trajectory_rate_limits_pass"], report)
        np.testing.assert_array_equal(
            report["final_command_velocity_A_deg_B_deg_C_mm_per_s"], np.zeros(3))

    def test_s1_reacquires_after_timed_target_loss(self):
        controller, _ = make(
            S1_OPEN_LOOP, lambda index, c, origin: 6 <= index < 12)
        report = controller.run()
        self.assertEqual(report["result"], "PASS", report)
        self.assertGreaterEqual(report["recovery_attempt_count"], 1)
        self.assertGreaterEqual(report["recovery_success_count"], 1)
        self.assertTrue(controller.sensor.was_hidden)

    def test_s2_uses_bounded_target_memory_and_reacquires(self):
        controller, _ = make(
            S2_TARGET_MEMORY, lambda index, c, origin: 6 <= index < 12)
        report = controller.run()
        self.assertEqual(report["result"], "PASS", report)
        self.assertGreater(report["memory_frame_count"], 0)
        self.assertGreaterEqual(report["memory_success_count"], 1)
        self.assertTrue(controller.sensor.was_hidden)

    def test_s3_retraction_ends_tool_coupled_occlusion(self):
        released = {"value": False}

        def hidden(index, current_c_m, loss_c_m):
            if index < 6 or loss_c_m is None:
                return False
            if current_c_m <= loss_c_m-.0003:
                released["value"] = True
            return not released["value"]
        controller, _ = make(S3_RETRACT_REOBSERVE, hidden)
        report = controller.run()
        self.assertEqual(report["result"], "PASS", report)
        self.assertGreaterEqual(report["recovery_attempt_count"], 1)
        self.assertGreaterEqual(report["recovery_success_count"], 1)
        self.assertTrue(controller.sensor.was_hidden)
        self.assertLess(min(row["C_mm"] for row in controller.rows), 0.)

    def test_s3_timeout_covers_braking_and_retracting_states(self):
        controller, _ = make(
            S3_RETRACT_REOBSERVE, lambda index, c, origin: True,
            max_iterations=400)
        report = controller.run()
        self.assertEqual(report["result"], "FAIL")
        self.assertEqual(
            report["controller_reason"], "OCCLUSION_RECOVERY_TIMEOUT")
        self.assertIn(report["strategy_state"], {
            "braking", "retracting", "reobserving"})

    def test_policy_can_reach_invalid_input_error(self):
        with self.assertRaisesRegex(ValueError, "occlusion_handling"):
            servo.visual_servo_settings(type("Reference", (), {
                "config": {"navigation": {"visual_servo": {
                    "occlusion_handling": "not_a_strategy"}}},
                "visual_perception_backend": "strict_rgbd",
            })())


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        OcclusionStrategyChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "result": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "scope": ("Synthetic measured-geometry injection for strategy state "
                  "transitions; no renderer or hardware claim."),
    }
    (OUTPUT/"strategy_verification.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
