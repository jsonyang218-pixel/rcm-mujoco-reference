from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from restricted_perception.target_split import (
    canonical_point,
    nearest_neighbor_distances,
    target_hash,
)


ROOT = Path(__file__).resolve().parents[1]


class TargetSplitTests(unittest.TestCase):
    def test_hash_is_stable_and_seed_sensitive(self):
        point = [1.0, 2.0, 3.0]
        self.assertEqual(target_hash(point, 7), target_hash(point, 7))
        self.assertNotEqual(target_hash(point, 7), target_hash(point, 8))
        self.assertEqual(canonical_point([1.0 + 1e-9, 2.0, 3.0]), (1.0, 2.0, 3.0))

    def test_nearest_neighbor_and_self_exclusion(self):
        points = [[0, 0, 0], [0, 0, 1.2], [0, 0, 3.0]]
        self.assertEqual(
            nearest_neighbor_distances(points, points, exclude_self=True),
            [1.2, 1.2, 1.8],
        )

    def test_checked_in_manifest_has_frozen_separation(self):
        manifest_path = ROOT / "configs" / "target_split_manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertGreater(manifest_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
