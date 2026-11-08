from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from restricted_perception.config import load_config
from restricted_perception.sidecar import (
    RESTRICTED_FEATURE_NAMES,
    _kinematic_tips,
    build_restricted_sidecar,
    restricted_feature_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "restricted_perception.json"


class RestrictedSidecarTests(unittest.TestCase):
    def test_batched_kinematics_matches_reference(self):
        from rcm_reference import Reference

        reference = Reference()
        states = np.array(
            [[0.0, 0.0, 19.0], [5.0, -2.0, 20.0], [-4.0, 3.0, 18.0]],
            dtype=float,
        )
        batched = _kinematic_tips(states, reference)
        expected = np.asarray(
            [
                reference.forward_kinematics(
                    np.array(
                        [
                            np.radians(state[0]),
                            np.radians(state[1]),
                            state[2] * 0.001,
                        ]
                    )
                )
                for state in states
            ],
            dtype=float,
        )
        self.assertTrue(np.allclose(batched, expected, atol=1e-10))

    def test_sidecar_schema_and_truth_omission(self):
        temporary = ROOT / "outputs" / "test_restricted_sidecar"
        temporary.mkdir(parents=True, exist_ok=True)
        dataset = temporary / "dataset.npz"
        state = np.array(
            [
                [0.0, 0.0, 19.0],
                [0.0, 0.0, 19.002],
                [0.0, 0.0, 19.004],
            ],
            dtype=np.float32,
        )
        oct_navigation = np.array(
            [[0.0, 0.0, 3.0], [0.0, 0.0, 2.98], [0.0, 0.0, 2.96]],
            dtype=np.float32,
        )
        np.savez(
            dataset,
            episode_index=np.zeros(3, dtype=np.int64),
            frame_index=np.arange(3, dtype=np.int64),
            state=state,
            action=np.zeros((3, 3), dtype=np.float32),
            oct_navigation=oct_navigation,
            oct_navigation_valid=np.ones(3, dtype=bool),
        )
        output = temporary / "sidecar"
        metadata = build_restricted_sidecar(
            dataset,
            load_config(CONFIG_PATH),
            output,
        )
        self.assertEqual(metadata["samples"], 3)
        self.assertFalse(metadata["truth_used_for_policy"])
        sidecar = np.load(output / "restricted_sidecar.npz")
        self.assertEqual(sidecar["state"].shape, (3, 3))
        self.assertIn("fine_goal_vector_robot_mm", sidecar.files)
        self.assertNotIn("target_robot_mm", sidecar.files)
        self.assertNotIn("target_world_mm", sidecar.files)
        features = restricted_feature_matrix(sidecar)
        self.assertEqual(features.shape, (3, len(RESTRICTED_FEATURE_NAMES)))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
