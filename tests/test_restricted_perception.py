from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from restricted_perception.config import load_config
from restricted_perception.fusion import CoarseFineFusion
from restricted_perception.reporting import precision_floor, switch_step_reduction
from restricted_perception.schema import FallbackState, POLICY_FEATURE_KEYS
from restricted_perception.simulator import CoarseFinePerceptionSimulator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "restricted_perception.json"


class RestrictedPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_config_round_trip(self):
        self.assertEqual(self.config.schema_version, 1)
        self.assertEqual(self.config.fusion.precision_levels_um, (5.0, 10.0, 20.0))

    def test_extrinsic_matches_rcm_scene_v9_optical_camera_snapshot(self):
        from rcm_scene_v9 import RCMSceneV9

        scene = RCMSceneV9()
        _, rotation = scene.camera_pose()
        scene.close()
        expected = rotation @ np.diag([1.0, -1.0, -1.0])
        self.assertTrue(
            np.allclose(
                self.config.extrinsic.camera_to_robot_rotation,
                expected,
                atol=1e-10,
            )
        )

    def test_fine_optical_vector_rotates_back_to_target_minus_tip(self):
        from rcm_scene_v9 import RCMSceneV9

        scene = RCMSceneV9()
        _, rotation = scene.camera_pose()
        scene.close()
        truth_robot = np.array([0.012, -0.031, 0.094], dtype=float)
        optical_camera = (
            np.diag([1.0, -1.0, -1.0]) @ rotation.T @ truth_robot
        )
        recovered = (
            self.config.extrinsic.camera_to_robot_rotation
            @ optical_camera
        )
        self.assertTrue(np.allclose(recovered, truth_robot, atol=1e-12))

    def test_truth_is_not_in_policy_features(self):
        simulator = CoarseFinePerceptionSimulator(self.config, seed=11)
        frame = simulator.update(
            np.array([0.0, 0.0, 10.0]),
            np.array([0.0, 0.0, 8.5]),
            timestamp_s=0.0,
            fine_enabled=False,
        )
        observation = CoarseFineFusion(self.config).step(frame, state=np.zeros(3))
        features = observation.policy_features()
        self.assertEqual(set(features), set(POLICY_FEATURE_KEYS))
        forbidden = (
            "target_robot_mm",
            "target_world_mm",
            "tip_world_mm",
            "teacher",
            "ground_truth",
        )
        self.assertFalse(any(any(token in key for token in forbidden) for key in features))
        vector = observation.policy_vector()
        self.assertEqual(vector.shape, (18,))
        self.assertTrue(np.isfinite(vector).all())

    def test_coarse_bias_is_fixed_per_episode_and_varies_by_seed(self):
        simulator_a = CoarseFinePerceptionSimulator(self.config, seed=22)
        simulator_b = CoarseFinePerceptionSimulator(self.config, seed=22)
        simulator_c = CoarseFinePerceptionSimulator(self.config, seed=23)
        self.assertTrue(np.array_equal(simulator_a.coarse_bias, simulator_b.coarse_bias))
        self.assertFalse(np.array_equal(simulator_a.coarse_bias, simulator_c.coarse_bias))

    def test_coarse_sigma_increases_with_distance(self):
        near = CoarseFinePerceptionSimulator(self.config, seed=30)._coarse_sigma(1.0)
        far = CoarseFinePerceptionSimulator(self.config, seed=31)._coarse_sigma(20.0)
        self.assertGreater(far, near)

    def test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip(self):
        simulator = CoarseFinePerceptionSimulator(self.config, seed=32)
        target = np.array([0.0, 0.0, 8.0])
        first_tip = np.array([0.0, 0.0, 6.5])
        second_tip = first_tip + np.array([0.0, 0.0, 0.08])
        simulator.update(target, first_tip, timestamp_s=0.0)
        frame = simulator.update(target, second_tip, timestamp_s=0.05)
        self.assertIsNotNone(frame.fine_goal_vector_camera_mm)
        observation = CoarseFineFusion(self.config).step(frame, state=np.zeros(3))
        aligned_error = np.linalg.norm(
            observation.fine_goal_vector_robot_mm - (target - second_tip)
        )
        stale_error = np.linalg.norm(
            self.config.extrinsic.camera_to_robot_rotation
            @ frame.fine_goal_vector_camera_mm
            - (target - second_tip)
        )
        self.assertLess(aligned_error * 1000.0, 5.0)
        self.assertGreater(stale_error * 1000.0, 70.0)

    def test_fine_loss_coasts_two_frames_then_falls_back(self):
        fusion = CoarseFineFusion(self.config)
        simulator = CoarseFinePerceptionSimulator(self.config, seed=40)
        target = np.array([0.0, 0.0, 8.001])
        tip = np.array([0.0, 0.0, 6.5])
        acquisition_frames = self.config.fine.delay_frames + 1
        states = []
        for index in range(acquisition_frames + 3):
            frame = simulator.update(
                target,
                tip,
                timestamp_s=index * 0.05,
                fine_enabled=index < acquisition_frames,
            )
            observation = fusion.step(frame, state=np.array([0.0, 0.0, 6.5]))
            states.append(observation.fallback_state)
        tracking_index = acquisition_frames - 1
        self.assertEqual(
            states[:tracking_index],
            [FallbackState.PRECISION_HOLD] * tracking_index,
        )
        self.assertEqual(states[tracking_index], FallbackState.FINE_TRACKING)
        self.assertEqual(
            states[tracking_index + 1:tracking_index + 3],
            [FallbackState.FINE_COAST, FallbackState.FINE_COAST],
        )
        self.assertIn(
            states[tracking_index + 3],
            (FallbackState.COARSE_FALLBACK, FallbackState.PRECISION_HOLD),
        )

    def test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction(self):
        fusion = CoarseFineFusion(self.config)
        simulator = CoarseFinePerceptionSimulator(self.config, seed=41)
        target = np.array([0.0, 0.0, 8.0])
        tip = np.array([0.0, 0.0, 7.0])
        observations = []
        for index in range(14):
            frame = simulator.update(
                target,
                tip,
                timestamp_s=index * 0.05,
                fine_enabled=False,
            )
            observation = fusion.step(frame, state=np.array([0.0, 0.0, 7.0]))
            observations.append(observation)
        hold = observations[-1]
        self.assertEqual(hold.fallback_state, FallbackState.PRECISION_HOLD)
        self.assertIsNone(hold.selected_goal_vector_robot_mm)
        self.assertTrue(np.array_equal(hold.action_mask_xyz, np.zeros(3)))
        self.assertTrue(hold.retraction_active)
        self.assertLessEqual(float(np.linalg.norm(hold.retraction_command_robot_mm)), 0.2 * 0.05 + 1e-12)
        expected_axis = self.config.extrinsic.camera_to_robot_rotation @ np.array(
            [0.0, 0.0, 1.0]
        )
        actual_axis = -hold.retraction_command_robot_mm
        actual_axis = actual_axis / np.linalg.norm(actual_axis)
        self.assertTrue(np.allclose(actual_axis, expected_axis))

    def test_precision_floor_requires_both_accuracy_and_closed_loop(self):
        report = precision_floor(self.config, p95_perception_error_um=2.4, deterministic_closed_loop_success=0.97)
        self.assertTrue(report["pass"])
        self.assertEqual(report["selected_threshold_um"], 5.0)
        failed = precision_floor(self.config, p95_perception_error_um=2.4, deterministic_closed_loop_success=0.9)
        self.assertFalse(failed["pass"])

    def test_switch_gate_rejects_insufficient_reduction(self):
        report = switch_step_reduction([0.3, 0.3, 0.3] * 20, [0.2, 0.2, 0.2] * 20)
        self.assertFalse(report["pass"])
        passed = switch_step_reduction([0.3, 0.3, 0.3] * 20, [0.02, 0.02, 0.02] * 20)
        self.assertTrue(passed["pass"])


if __name__ == "__main__":
    unittest.main()
