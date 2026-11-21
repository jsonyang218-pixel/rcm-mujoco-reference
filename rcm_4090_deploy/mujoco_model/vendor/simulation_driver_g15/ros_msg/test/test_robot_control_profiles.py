#!/usr/bin/env python3
"""Regression tests for the shared needle/profile layer."""

import math
import os
import sys
import unittest


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PACKAGE_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_SRC_DIR)

from robot_control.kinematics import (  # noqa: E402
    b_distance_from_alpha,
    new_needle_pose_from_robot_encoders,
    new_needle_solid_segment,
)
from robot_control.profiles import (  # noqa: E402
    LEGACY_NEEDLE_PROFILE,
    NEW_NEEDLE_PROFILE,
)


class NeedleProfileTest(unittest.TestCase):
    def test_variants_share_geometry_but_have_explicit_policies(self):
        self.assertEqual(LEGACY_NEEDLE_PROFILE.tip_offset_mm,
                         NEW_NEEDLE_PROFILE.tip_offset_mm)
        self.assertEqual(LEGACY_NEEDLE_PROFILE.offset_sign, 1.0)
        self.assertEqual(NEW_NEEDLE_PROFILE.offset_sign, -1.0)
        self.assertEqual(NEW_NEEDLE_PROFILE.solid_length_mm, 1.0)

    def test_new_needle_solid_segment_uses_profile_length(self):
        robot_b = round(4 * 1024 * 15.5817175 * b_distance_from_alpha(53.0))
        pose = new_needle_pose_from_robot_encoders((0, robot_b, 0, 0), (0.0, 0.0, 0.0))
        start, end = new_needle_solid_segment(
            (0.0, 0.0, 0.0), pose, NEW_NEEDLE_PROFILE.solid_length_mm,
        )
        self.assertEqual(start, pose["display_tip"])
        length = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
        self.assertAlmostEqual(length, NEW_NEEDLE_PROFILE.solid_length_mm)


if __name__ == "__main__":
    unittest.main()
