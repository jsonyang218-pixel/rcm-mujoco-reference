"""Short numerical regressions for measured-geometry visual motion.

These tests exercise the real controller loop against an independent coupled
linear plant and measured-position sensor. No renderer, privileged goal pose,
or application dependency beyond NumPy is required.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import rcm_visual_servo as servo


COUPLED = np.array([[12., 3., 250.], [2., 10., -100.], [1., -2., 1000.]])
OFFSET = np.array([0., 0., 40.])


class LinearPlant:
    def __init__(self, matrix, measured_target, truth_offset=None, generous_acceleration=False):
        self.matrix = np.asarray(matrix, dtype=float)
        self.q = np.zeros(3)
        self.measured_target = np.asarray(measured_target, dtype=float)
        self.truth_target = self.measured_target + (
            np.zeros(3) if truth_offset is None else np.asarray(truth_offset, dtype=float))
        self.config = {"navigation": {
            "max_velocity_A_deg_B_deg_C_mm_per_s": [4., 2., .6],
            "max_acceleration_A_deg_B_deg_C_mm_per_s2": (
                [1000., 1000., 100.] if generous_acceleration else [10., 5., 1.5]),
        }}
        self.applied = []

    def current_task(self):
        return self.q.copy()

    def validate_task(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("invalid plant state")
        return q

    def set_task(self, q):
        self.q = self.validate_task(q).copy()
        self.applied.append(self.q.copy())

    def tip(self):
        return OFFSET+self.matrix@self.q

    def metrics(self):
        tip = self.tip()
        return {"tip_world_mm": tip.tolist(),
                "target_world_mm": self.truth_target.tolist(),
                "tip_target_error_um": float(np.linalg.norm(tip-self.truth_target)*1000.)}

    def target_task(self, *args, **kwargs):
        raise AssertionError("The measured-geometry controller read a privileged target pose")


class Projector:
    @staticmethod
    def project(world_mm):
        p = np.asarray(world_mm)
        return p[:2].tolist(), float(p[2]*.001)


class MeasuredSensor:
    camera_name = "eye_internal"
    width = 640
    height = 480

    def __init__(self, reference, settings, fail_at=None):
        self.reference = reference
        self.geometry = Projector()
        self.calls = 0
        self.fail_at = fail_at
        self.closed = False

    def observe(self):
        index = self.calls
        self.calls += 1
        target, tip = self.reference.measured_target, self.reference.tip()
        quality = {"source": "independent_linear_sensor", "support": 100,
                   "fit_quality_mm": .001, "residual_mm": 0.}
        geometry = {"target_world_mm": target.tolist(), "tip_world_mm": tip.tolist(),
                    "target_quality": dict(quality), "tip_quality": dict(quality)}
        if self.fail_at is not None and index >= self.fail_at:
            geometry["tip_world_mm"] = None
            geometry["tip_error"] = "TIP_NOT_OBSERVABLE"
        return {"geometry": geometry, "target_pixel": target[:2].tolist(),
                "tip_pixel": tip[:2].tolist(), "target_depth_m": float(target[2]*.001),
                "tip_depth_m": float(tip[2]*.001), "target_pixels": 10, "tip_pixels": 10,
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

    def close(self):
        self.closed = True


def make_controller(goal_q, *, matrix=COUPLED, truth_offset=None,
                    control_axes=None, completion_axes=None, initial_velocity=None,
                    fail_at=None, generous_acceleration=False, overrides=None):
    target = OFFSET+np.asarray(matrix)@np.asarray(goal_q)
    plant = LinearPlant(matrix, target, truth_offset, generous_acceleration)
    settings = {**servo.DEFAULT_SERVO, "position_feedback": "geometry",
                "position_tolerance_um": 8., "endpoint_tolerance_um": 20.,
                "require_settled": True, "settle_frames": 3,
                "save_diagnostics": False, "max_iterations": 300,
                "sample_period_s": .05, "geometry_gain_per_s": 3.}
    settings.update(overrides or {})
    factory = lambda reference, options: MeasuredSensor(reference, options, fail_at)
    with patch.object(servo, "VisualServoSensor", factory):
        controller = servo.VisualServoController(
            plant, settings=settings, initial_velocity=initial_velocity,
            control_axes=control_axes, completion_axes=completion_axes)
    # Supply only the calibrated actuation derivative, never a target solution.
    controller._model_jacobian = lambda q: np.asarray(matrix).copy()
    return controller, plant


class GeometryMotionRegression(unittest.TestCase):
    def test_coupled_motion_reaches_measured_target_and_settles(self):
        goal = np.array([.005, -.004, .0001])
        controller, plant = make_controller(goal, generous_acceleration=True)
        observed = []
        controller.iteration_callback = lambda iteration, observation: observed.append(plant.q.copy()) or True
        report = controller.run()
        self.assertEqual(report["result"], "PASS", report)
        self.assertEqual(report["controller_reason"], "POSITION_ERROR_AND_SETTLED")
        self.assertLess(report["final_tip_target_error_um"], 20.)
        self.assertTrue(report["terminal_stop_validated"])
        self.assertGreaterEqual(report["settled_frames"], 3)
        self.assertTrue(report["trajectory_rate_limits_pass"])
        self.assertTrue(controller.sensor.closed)
        # A fully coupled linear plant must move toward its known reachable
        # joint solution. Independent A/B-image and C-depth blocks do not do so.
        np.testing.assert_allclose(observed[1], .15*goal, rtol=2e-5, atol=1e-10)
        np.testing.assert_allclose(observed[-3:], np.tile(observed[-1], (3, 1)), atol=1e-14)
        np.testing.assert_array_equal(report["final_command_velocity_A_deg_B_deg_C_mm_per_s"], np.zeros(3))
        self.assertFalse(report["known_target_pose_used_for_feature_calibration"])

    def test_truth_is_only_an_independent_post_run_acceptance(self):
        goal = [.005, -.004, .0001]
        good, plant_good = make_controller(goal)
        poisoned, plant_poisoned = make_controller(goal, truth_offset=[1., 0., 0.])
        good_result, poisoned_result = good.run(), poisoned.run()
        self.assertEqual(good_result["result"], "PASS", good_result)
        self.assertEqual(poisoned_result["controller_result"], "PASS", poisoned_result)
        self.assertEqual(poisoned_result["result"], "FAIL")
        self.assertEqual(poisoned_result["reason"], "INDEPENDENT_ACCURACY_REJECTED")
        self.assertTrue(poisoned_result["terminal_stop_validated"])
        self.assertEqual(len(good.rows), len(poisoned.rows))
        np.testing.assert_allclose(plant_good.q, plant_poisoned.q, atol=1e-14)
        for a, b in zip(good.rows, poisoned.rows):
            for key in ("A_deg", "B_deg", "C_mm", "estimated_position_error_um"):
                self.assertEqual(a[key], b[key], key)

    def test_partial_insertion_holds_orientation_and_rejects_lateral_miss(self):
        controller, plant = make_controller([.01, 0., .0002],
            control_axes=[False, False, True], completion_axes=[False, False, True])
        report = controller.run()
        self.assertEqual(report["controller_result"], "PASS", report)
        self.assertTrue(report["terminal_stop_validated"])
        self.assertEqual(report["result"], "FAIL")
        self.assertEqual(report["reason"], "INDEPENDENT_ACCURACY_REJECTED")
        self.assertGreater(report["final_tip_target_error_um"], 20.)
        self.assertLess(controller.rows[-1]["completion_error_um"], 8.)
        for row in controller.rows:
            self.assertEqual(row["A_deg"], 0.)
            self.assertEqual(row["B_deg"], 0.)

    def test_partial_insertion_reaches_a_collinear_target(self):
        controller, _ = make_controller([0., 0., .0002],
            control_axes=[False, False, True], completion_axes=[False, False, True])
        report = controller.run()
        self.assertEqual(report["result"], "PASS", report)
        self.assertTrue(report["terminal_stop_validated"])
        self.assertLess(report["final_tip_target_error_um"], 20.)

    def test_nonzero_velocity_on_forbidden_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "(inactive|disabled|control|forbidden|masked|fixed)"):
            controller, _ = make_controller([0., 0., .0002],
                control_axes=[False, False, True], completion_axes=[False, False, True],
                initial_velocity=[.001, 0., 0.])
            controller.run()

    def assert_abort_hold_is_reported(self, report):
        self.assertEqual(report["result"], "FAIL", report)
        self.assertTrue(report["simulation_abort_hold"], report)
        self.assertFalse(report["physical_braking_validated"])
        self.assertFalse(report["terminal_stop_validated"])
        before = np.asarray(report["velocity_before_abort_hold_A_deg_B_deg_C_mm_per_s"])
        self.assertGreater(np.linalg.norm(before), 0.)
        np.testing.assert_array_equal(report["final_command_velocity_A_deg_B_deg_C_mm_per_s"], np.zeros(3))
        acceleration = np.asarray(report["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
        self.assertTrue(np.all(acceleration+1e-8 >= np.abs(before)/.05), report)
        self.assertFalse(report["trajectory_rate_limits_pass"], report)

    def test_lost_observation_records_abrupt_hold_acceleration(self):
        controller, _ = make_controller([.03, -.02, .002], fail_at=7)
        report = controller.run()
        self.assertEqual(report["controller_reason"], "TIP_NOT_OBSERVABLE")
        self.assert_abort_hold_is_reported(report)
        self.assertTrue(controller.sensor.closed)

    def test_user_callback_can_abort_and_is_not_called_success(self):
        controller, _ = make_controller([.03, -.02, .002])
        controller.iteration_callback = lambda iteration, observation: iteration < 7
        report = controller.run()
        self.assertEqual(report["controller_reason"], "USER_ABORTED")
        self.assertEqual(report["iterations"], 8)
        self.assert_abort_hold_is_reported(report)

    def test_iteration_limit_does_not_apply_an_unobserved_last_motion(self):
        controller, plant = make_controller([.03, -.02, .002],
                                            overrides={"max_iterations": 4})
        observed = []
        controller.iteration_callback = lambda iteration, observation: observed.append(plant.q.copy()) or True
        report = controller.run()
        self.assertEqual(report["controller_reason"], "MAX_ITERATIONS")
        self.assertEqual(report["iterations"], 4)
        self.assertEqual(controller.sensor.calls, 4)
        np.testing.assert_array_equal(plant.q, observed[-1])
        self.assert_abort_hold_is_reported(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
