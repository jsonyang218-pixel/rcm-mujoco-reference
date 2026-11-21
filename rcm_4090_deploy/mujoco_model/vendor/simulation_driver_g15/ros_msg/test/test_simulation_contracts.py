#!/usr/bin/env python3
"""Simulation defaults, effective-topic isolation and G15 protocol contracts."""

import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from robot_control.experiment_io import software_fingerprints
from robot_control.kinematics import (
    C_ENCODER_COUNTS_PER_HANDLE, CONTROL_GEAR, cartesian_handle_delta_float,
    cartesian_handle_target, limit_cartesian_packet_delta,
    model_tip_from_robot_encoders, predicted_robot_encoders,
)
from robot_control.simulation_topics import resolve_experiment_topics


PACKAGE = Path(__file__).resolve().parents[1]


class SimulationContractsTest(unittest.TestCase):
    def test_shared_sources_and_messages_match_hardware_package_in_repository(self):
        hardware = PACKAGE.parents[1] / "src/ros_msg"
        if not hardware.is_dir() or hardware.resolve() == PACKAGE:
            self.skipTest("standalone package; hardware checkout unavailable")
        paths = [path.relative_to(PACKAGE) for path in (PACKAGE / "src/robot_control").glob("*.py")
                 if path.name not in ("experiment_io.py", "simulation_topics.py")]
        paths.extend(path.relative_to(PACKAGE) for path in (PACKAGE / "msg").glob("*.msg"))
        for relative in paths:
            with self.subTest(path=str(relative)):
                actual, expected = (PACKAGE / relative).read_bytes(), (hardware / relative).read_bytes()
                if relative.suffix == ".msg":
                    # ROS ignores trailing whitespace; keep diff --check clean.
                    actual, expected = actual.rstrip(), expected.rstrip()
                self.assertEqual(actual, expected)

    def test_both_clis_default_to_simulation_and_keep_explicit_execution_gate(self):
        for name in ("base_position_response", "base_sine_response"):
            spec = importlib.util.spec_from_file_location(name, PACKAGE / "scripts" / (name + ".py"))
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            with patch.object(cli.rospy, "myargv", return_value=[name, "--output", "/tmp/unused"]):
                args = cli.arguments()
            self.assertEqual(args.feedback_topic, "/simulation/RobotState")
            self.assertEqual(args.control_topic, "/simulation/controlSignal")
            self.assertEqual(args.data_source, "simulation")
            self.assertFalse(args.execute)
            with patch.object(cli.rospy, "myargv", return_value=[name, "--output", "/tmp/unused", "--execute"]):
                with self.assertRaises(SystemExit):
                    cli.arguments()

    def test_isolation_checks_both_effective_names_after_remapping(self):
        feedback, control = "/simulation/RobotState", "/simulation/controlSignal"
        for effective in ((feedback, control), ("/robot/RobotState", control),
                          (feedback, "/robot/controlSignal"), ("/simulation/", control)):
            with patch("rospy.resolve_name", side_effect=effective):
                if effective == (feedback, control):
                    self.assertEqual(resolve_experiment_topics(feedback, control, True), effective)
                else:
                    with self.assertRaises(ValueError):
                        resolve_experiment_topics(feedback, control, True)

    def test_g15_forward_and_inverse_agree_for_both_needle_offsets(self):
        self.assertEqual(CONTROL_GEAR, 15)
        self.assertEqual(C_ENCODER_COUNTS_PER_HANDLE, 112.5)
        robot = (0, 1466021, 925000, 0)
        # Known G15 A and C increments, independent of the inverse helper.
        moved = predicted_robot_encoders(robot, (4, 0, 2, 0))
        self.assertAlmostEqual(moved[0] - robot[0], 50)
        self.assertAlmostEqual(moved[2] - robot[2], 225)
        for sign in (-1, 1):
            tip = model_tip_from_robot_encoders(robot, 4, sign)
            target = model_tip_from_robot_encoders(moved, 4, sign)
            delta = tuple(b - a for a, b in zip(tip, target))
            position = model_tip_from_robot_encoders(robot, 0)
            _, raw, _ = cartesian_handle_delta_float(robot, position, delta, tip_offset_sign=sign)
            _, handles, _ = cartesian_handle_target(robot, (9, 7, 1, 0), position, delta, tip_offset_sign=sign)
            self.assertEqual(handles, (13, 7, 3, 0))
            self.assertTrue(all(math.isfinite(value) for value in raw))

    def test_packet_budget_rejects_e_and_invalid_input(self):
        self.assertEqual(limit_cartesian_packet_delta((192, -16, 64, 0), (96, 16, 64, 0)), (96, -8, 32, 0))
        for delta in ((1, 1, 1, 1), (float("nan"), 0, 0, 0), (0.5, 0, 0, 0)):
            with self.assertRaises(ValueError):
                limit_cartesian_packet_delta(delta, (96, 16, 64, 0))

    def test_fingerprints_work_without_ui_modules(self):
        fingerprints = software_fingerprints()
        self.assertIn("kinematics.py", fingerprints)
        self.assertTrue(all(len(value) == 64 for value in fingerprints.values()))


if __name__ == "__main__":
    unittest.main()
