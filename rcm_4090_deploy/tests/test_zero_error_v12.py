"""Focused regressions for the v12 zero-error data construction."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rcm.reference import HERE
from rcm.scene_v9 import RCMSceneV9
from rcm.waypoint_dataset import write_manifest
from scripts.datasets.collect_zero_error_dataset import (
    SI_UNITS,
    limited_joint_path,
    load_source_episodes,
    collect_episode,
    stabilize_terminal_pose,
)
from scripts.datasets.collect_zero_error_parallel import _split_ranges


class StrictTargetTests(unittest.TestCase):
    def test_target_site_and_marker_use_exact_needle_tip_meters(self):
        scene = RCMSceneV9()
        try:
            scene.set_task(np.array([0.013, -0.007, 0.012]))
            tip_before = scene.data.site_xpos[scene.sites["needle_tip"]].copy()
            target = scene.set_target_from_needle_tip()
            np.testing.assert_array_equal(
                target, scene.data.site_xpos[scene.sites["lesion"]]
            )
            np.testing.assert_array_equal(
                target,
                scene.model.geom_pos[
                    __import__("mujoco").mj_name2id(
                        scene.model,
                        __import__("mujoco").mjtObj.mjOBJ_GEOM,
                        "lesion_marker",
                    )
                ],
            )
            np.testing.assert_array_equal(
                target,
                scene.model.geom_pos[
                    __import__("mujoco").mj_name2id(
                        scene.model,
                        __import__("mujoco").mjtObj.mjOBJ_GEOM,
                        "lesion_display_marker",
                    )
                ],
            )
            # The tip is unchanged by target placement.
            np.testing.assert_array_equal(
                tip_before,
                scene.data.site_xpos[scene.sites["needle_tip"]],
            )
            self.assertEqual(scene.metrics()["tip_target_error_um"], 0.0)
        finally:
            scene.close()

    def test_integer_terminal_reproduction_and_target_are_exactly_equal(self):
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        config["motor_execution"] = {"enabled": False}
        source = load_source_episodes(
            HERE / "data" / "lerobot_rcm_train_guided_dagger1_full_224"
        )[0]
        result = stabilize_terminal_pose(
            config,
            source.terminal_state_A_deg_B_deg_C_mm,
        )
        self.assertEqual(
            result["terminal_tip_world_m"],
            result["terminal_target_world_m"],
        )
        self.assertEqual(result["final_error_um"], 0.0)
        self.assertEqual(
            result["target_minus_source_terminal_tip_m"],
            [0.0, 0.0, 0.0],
        )
        self.assertGreaterEqual(len(result["endpoint_robot_ABCE"]), 4)
        self.assertEqual(result["endpoint_robot_ABCE"][-1], 0)


class FallbackTests(unittest.TestCase):
    def test_joint_path_is_terminal_exact_and_respects_limits(self):
        start = np.array([0.0, 0.0, 0.0])
        terminal = np.array([0.03, -0.02, 0.004])
        vmax = np.array([4.0, 2.0, 0.6]) * SI_UNITS
        amax = np.array([10.0, 5.0, 1.5]) * SI_UNITS
        dt = 0.05
        commands = limited_joint_path(start, terminal, vmax, amax, dt)
        self.assertGreater(len(commands), 3)
        np.testing.assert_array_equal(commands[-1], terminal)
        points = np.vstack([start, np.asarray(commands)])
        velocity = np.diff(points, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        self.assertTrue(np.all(np.abs(velocity) <= vmax + 1e-10))
        self.assertTrue(np.all(np.abs(acceleration) <= amax + 1e-8))

    def test_zero_distance_path_returns_exact_terminal(self):
        terminal = np.array([0.01, -0.02, 0.003])
        commands = limited_joint_path(
            terminal,
            terminal,
            np.ones(3),
            np.ones(3),
            0.05,
        )
        self.assertEqual(len(commands), 1)
        np.testing.assert_array_equal(commands[0], terminal)


class SourceTests(unittest.TestCase):
    def test_source_indexes_and_per_source_traceback_are_preserved(self):
        source = load_source_episodes(
            HERE / "data" / "lerobot_rcm_train_guided_dagger1_full_224"
        )
        self.assertEqual(len(source), 608)
        self.assertEqual(source[0].source_episode_index, 0)
        self.assertEqual(source[489].source_episode_index, 489)
        self.assertEqual(source[490].source_episode_index, 0)
        self.assertEqual(
            source[490].source_dataset,
            "data/dagger_guided_v6_round1_all_224",
        )
        self.assertTrue(source[0].start_state_A_deg_B_deg_C_mm.shape == (3,))
        self.assertTrue(
            source[0].terminal_state_A_deg_B_deg_C_mm.shape == (3,)
        )

    def test_manifest_write_is_atomic(self):
        with tempfile.TemporaryDirectory(
                prefix="rcm_v12_manifest_test_") as temporary:
            root = Path(temporary)
            write_manifest(
                root,
                [{"episode_index": 0, "frames": 3, "result": "PASS"}],
                [224, 224, 3],
            )
            self.assertTrue((root / "manifest.json").is_file())
            self.assertFalse((root / "manifest.json.tmp").exists())
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["total_frames"], 3)

    def test_parallel_ranges_cover_missing_indices_once(self):
        missing = [21, 23, 24, 100]
        ranges = _split_ranges(missing, 3)
        self.assertLessEqual(len(ranges), 3)
        covered = [
            index
            for start, end in ranges
            for index in range(start, end + 1)
            if index in missing
        ]
        self.assertEqual(covered, missing)


class CollectorIntegrationTests(unittest.TestCase):
    def test_visual_pass_then_exact_terminal_lock(self):
        source = load_source_episodes(
            HERE / "data" / "lerobot_rcm_train_guided_dagger1_full_224"
        )[206]
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        config["motor_execution"] = {
            "enabled": True,
            "control_period_s": float(
                config["navigation"]["visual_servo"]["sample_period_s"]
            ),
        }
        config["navigation"]["visual_servo"]["perception_backend"] = (
            "strict_rgbd"
        )
        config["reference_scene"]["eye_camera"]["resolution_px"] = [1920, 1440]
        with tempfile.TemporaryDirectory(
                prefix="rcm_v12_visual_pass_test_") as temporary:
            episode = collect_episode(
                config,
                Path(temporary),
                source,
                seed=20261114,
                max_visual_iterations=100,
            )
        self.assertEqual(episode["result"], "PASS")
        self.assertEqual(episode["visual_servo_result"], "PASS")
        self.assertFalse(episode["fallback_used"])
        self.assertEqual(episode["final_error_um"], 0.0)
        self.assertGreater(
            episode["execution_source_counts"]["strict_rgbd_teacher"], 0
        )
        self.assertGreater(
            episode["execution_source_counts"]["exact_terminal_lock"], 0
        )
        self.assertEqual(
            episode["execution_source_counts"]["privileged_joint_fallback"],
            0,
        )

    def test_visual_failure_uses_motor_fallback_then_exact_terminal_lock(self):
        source = load_source_episodes(
            HERE / "data" / "lerobot_rcm_train_guided_dagger1_full_224"
        )[206]
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        config["motor_execution"] = {
            "enabled": True,
            "control_period_s": float(
                config["navigation"]["visual_servo"]["sample_period_s"]
            ),
        }
        config["navigation"]["visual_servo"]["perception_backend"] = (
            "strict_rgbd"
        )
        config["reference_scene"]["eye_camera"]["resolution_px"] = [1920, 1440]
        with tempfile.TemporaryDirectory(
                prefix="rcm_v12_visual_failure_test_") as temporary:
            episode = collect_episode(
                config,
                Path(temporary),
                source,
                seed=20261114,
                max_visual_iterations=1,
            )
        self.assertEqual(episode["result"], "PASS")
        self.assertEqual(episode["visual_servo_result"], "FAIL")
        self.assertTrue(episode["fallback_used"])
        self.assertEqual(
            episode["fallback_mode"],
            "privileged_joint_space_velocity_acceleration_limited",
        )
        self.assertEqual(episode["final_error_um"], 0.0)
        self.assertGreater(
            episode["execution_source_counts"]["privileged_joint_fallback"],
            0,
        )
        self.assertGreater(
            episode["execution_source_counts"]["exact_terminal_lock"],
            0,
        )
        self.assertEqual(
            episode["terminal_robot_ABCE"],
            episode["endpoint_robot_ABCE"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
