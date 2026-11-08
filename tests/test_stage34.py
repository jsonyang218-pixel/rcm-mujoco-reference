from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from restricted_perception.config import load_config
from restricted_perception.stage34 import (
    geometric_teacher_action,
    load_stage34_protocol,
    median_control_specs,
    scenario_configuration,
    stage4_run_specs,
)


ROOT = Path(__file__).resolve().parents[1]


class Stage34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_stage34_protocol(
            ROOT / "configs/stage34_protocol.json"
        )
        cls.manifest = json.loads(
            (ROOT / "configs/target_split_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cls.config = load_config(
            ROOT / "configs/restricted_perception.json"
        )

    def test_run_specs_are_frozen_and_pairing_safe(self):
        specs = stage4_run_specs(self.manifest, self.protocol)
        controls = median_control_specs(self.manifest, self.protocol)
        self.assertEqual(len(specs), 180)
        self.assertEqual(len(controls), 60)
        self.assertEqual(len({spec["episode_key"] for spec in specs}), 180)
        self.assertTrue(
            {spec["episode_key"] for spec in controls}.issubset(
                {spec["episode_key"] for spec in specs}
            )
        )

    def test_perturbations_change_one_sensor_factor(self):
        delay, delay_image = scenario_configuration(
            self.config,
            "delay",
            self.protocol,
        )
        self.assertGreater(delay.fine.delay_frames, self.config.fine.delay_frames)
        self.assertEqual(delay.coarse.noise_floor_mm, self.config.coarse.noise_floor_mm)
        self.assertEqual(delay_image["center_occlusion_fraction"], 0.25)
        occlusion, image = scenario_configuration(
            self.config,
            "occlusion",
            self.protocol,
        )
        self.assertEqual(
            occlusion.fine.dropout_probability,
            self.config.fine.dropout_probability,
        )
        self.assertEqual(image["center_occlusion_fraction"], 0.45)

    def test_geometric_action_moves_toward_target_minus_tip(self):
        from rcm_reference import Reference

        reference = Reference()
        state = np.zeros(3, dtype=float)
        goal = np.array([0.0, 0.0, 0.1])
        tip_before = np.asarray(reference.forward_kinematics(state * [np.pi / 180, np.pi / 180, 0.001]))
        action = geometric_teacher_action(
            reference,
            state,
            goal,
            gain=0.15,
        )
        tip_after = np.asarray(
            reference.forward_kinematics(
                (state + action) * [np.pi / 180, np.pi / 180, 0.001]
            )
        )
        before_error = np.linalg.norm((tip_before + goal) - tip_before)
        after_error = np.linalg.norm((tip_before + goal) - tip_after)
        self.assertLess(after_error, before_error)


if __name__ == "__main__":
    unittest.main()
