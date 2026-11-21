#!/usr/bin/env python3
"""Fixed validation catalog and guarded route-planning regressions."""

import math
import unittest

from robot_control.kinematics import needle_direction_from_robot_encoders
from robot_control.validation_experiments import (
    EXPERIMENT_STEP_MM,
    EXPERIMENT_TOLERANCE_UM,
    VALIDATION_EXPERIMENTS,
    experiment_targets,
    plan_validation_experiment,
)


class ValidationCatalogTest(unittest.TestCase):
    def test_catalog_has_twelve_unique_buttons_and_fixed_parameters(self):
        self.assertEqual(len(VALIDATION_EXPERIMENTS), 12)
        self.assertEqual(len({item.identifier for item in VALIDATION_EXPERIMENTS}), 12)
        self.assertEqual(EXPERIMENT_STEP_MM, 0.050)
        self.assertEqual(EXPERIMENT_TOLERANCE_UM, 2.0)
        self.assertEqual(sum(item.kind == "circle" for item in VALIDATION_EXPERIMENTS), 3)
        self.assertEqual(sum(item.kind == "rectangle" for item in VALIDATION_EXPERIMENTS), 3)
        self.assertEqual(sum(item.kind == "sine" for item in VALIDATION_EXPERIMENTS), 3)
        self.assertEqual({item.kind for item in VALIDATION_EXPERIMENTS[-3:]},
                         {"diagonal", "box", "curve"})

    def test_plane_sizes_and_space_amplitudes_are_expanded(self):
        start = (0.0, 0.0, 0.0)
        samples, _, params = experiment_targets("xy_circle", start)
        radius = max(math.dist(start, point) for point in samples)
        self.assertAlmostEqual(radius, 0.8)
        self.assertEqual(params["circle_points"], 128)
        samples, _, params = experiment_targets("xy_rectangle", start)
        self.assertEqual(params["width_um"], 800)
        self.assertEqual(params["height_um"], 400)
        self.assertGreater(max(point[0] for point in samples) - min(point[0] for point in samples), 0.79)
        samples, _, params = experiment_targets("space_curve", start)
        self.assertEqual(len(samples), 513)
        self.assertAlmostEqual(max(point[0] for point in samples), 0.8, places=6)
        self.assertAlmostEqual(max(point[1] for point in samples), 0.4, places=6)
        self.assertAlmostEqual(samples[256][2], 0.8, places=6)

    def test_each_plan_stays_with_step_and_returns_when_required(self):
        robot = (0, 1466021, 893518, 0)
        start = (0.0, 0.0, 0.0)
        for item in VALIDATION_EXPERIMENTS:
            with self.subTest(item=item.identifier):
                route, metadata = plan_validation_experiment(
                    item.identifier, start, robot, 4.0, -1.0,
                )
                self.assertGreater(len(route), 1)
                self.assertLessEqual(max(math.dist(a, b) for a, b in zip(route, route[1:])),
                                     EXPERIMENT_STEP_MM + 1e-9)
                if item.kind in ("circle", "rectangle", "sine", "diagonal", "curve"):
                    self.assertEqual(route[-1], start)
                self.assertEqual(metadata["experiment_id"], item.identifier)
                self.assertEqual(len(metadata["waypoint_phases"]), len(route))
                previous = needle_direction_from_robot_encoders(robot)
                # The fixed planner inserts axis-safe subdivisions before execution.
                self.assertIsNotNone(previous)

    def test_invalid_robot_feedback_is_rejected_before_route(self):
        with self.assertRaises(ValueError):
            plan_validation_experiment("xy_circle", (0, 0, 0), (0, 1466021, -1, 0), 4.0, -1.0)


if __name__ == "__main__":
    unittest.main()
