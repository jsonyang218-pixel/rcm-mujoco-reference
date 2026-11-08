from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from restricted_perception.causal import image_leakage_gate, summarize_counterfactual
from restricted_perception.image_ops import degrade_image


class CausalGateTests(unittest.TestCase):
    def test_counterfactual_contribution_and_gate(self):
        normal_success = np.ones(40, dtype=bool)
        ablated_success = np.zeros(40, dtype=bool)
        summary = summarize_counterfactual(
            normal_success,
            ablated_success,
            np.full(40, 8.0),
            np.full(40, 30.0),
        )
        self.assertTrue(summary["representation_contribution"])
        gate = image_leakage_gate(summary, summary)
        self.assertEqual(gate["result"], "PASS")

    def test_no_effect_is_leakage(self):
        outcomes = np.array([True] * 20 + [False] * 20)
        summary = summarize_counterfactual(
            outcomes,
            outcomes,
            np.linspace(5.0, 10.0, 40),
            np.linspace(5.0, 10.0, 40),
        )
        self.assertFalse(summary["representation_contribution"])
        self.assertEqual(image_leakage_gate(summary, summary)["result"], "FAIL")

    def test_one_factor_degradation(self):
        source = Image.new("RGB", (64, 48), (200, 100, 50))
        resized = degrade_image(source, resolution=32)
        self.assertEqual(resized.size, (32, 32))
        blurred = degrade_image(source, gaussian_blur_radius=1.5)
        self.assertEqual(blurred.size, source.size)
        occluded = degrade_image(source, center_occlusion_fraction=0.4)
        self.assertEqual(occluded.getpixel((24, 16)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
