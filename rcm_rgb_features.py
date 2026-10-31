"""Image-only feature masks for the calibrated single-marker eye benchmark.

The input is an ordinary HWC RGB image. Color assumptions are explicit: an
orange target, a blue-gray proximal tube, a pale distal needle, and an optional
green physical fiducial. No model, geom IDs, depth, poses, or target coordinates
are available to this detector. These fixed colors are a benchmark detector,
not a trained clinical segmentation model.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class RGBFeatureDetector:
    """Conservative RGB masks with no hidden-object or motion-state fallback."""

    def __init__(self, settings=None):
        settings = dict(settings or {})
        self.target_red_green_ratio = float(settings.get('rgb_target_red_green_ratio', 1.45))
        self.target_green_blue_ratio = float(settings.get('rgb_target_green_blue_ratio', 1.35))
        self.proximal_blue_red_ratio = float(settings.get('rgb_proximal_blue_red_ratio', 1.18))
        self.needle_min_value = float(settings.get('rgb_needle_min_value', 18.))
        self.target_min_pixels = int(settings.get('rgb_target_min_pixels', 6))
        self.apex_min_pixels = int(settings.get('rgb_apex_min_pixels', 2))
        self.needle_min_component_pixels = int(settings.get('rgb_needle_min_component_pixels', 3))
        for value in (self.target_red_green_ratio, self.target_green_blue_ratio,
                      self.proximal_blue_red_ratio, self.needle_min_value):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('RGB thresholds must be finite and positive')
        if min(self.target_min_pixels, self.apex_min_pixels,
               self.needle_min_component_pixels) < 1:
            raise ValueError('RGB component sizes must be positive')

    @staticmethod
    def _components(mask, min_pixels, largest_only=False):
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if not count:
            return np.zeros(mask.shape, dtype=bool)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if largest_only:
            index = int(np.argmax(sizes))
            return labels == index if sizes[index] >= min_pixels else np.zeros(mask.shape, dtype=bool)
        keep = sizes >= min_pixels
        keep[0] = False
        return keep[labels]

    def detect(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('RGB detector expects uint8 HWC with three channels')
        color = rgb.astype(np.float32)
        red, green, blue = color[..., 0], color[..., 1], color[..., 2]

        # Orange marker vs the red/pink retinal background. Ratios tolerate
        # multiplicative shading; minimum brightness rejects dark boundaries.
        target = ((red > 65.) & (green > 25.)
                  & (red > self.target_red_green_ratio * green)
                  & (green > self.target_green_blue_ratio * blue))
        target = self._components(target, self.target_min_pixels, largest_only=True)

        # This is an optional visible artificial fiducial, not the real fine
        # needle apex. An absent/occluded fiducial produces an empty mask.
        apex = ((green > 35.) & (green > 1.5 * red) & (green > 1.25 * blue))
        apex = self._components(apex, self.apex_min_pixels, largest_only=True)

        # Neutral/blue-gray material excludes both the retina and colored
        # markers. It deliberately does not infer hidden pixels behind them.
        value = np.max(color, axis=2)
        saturation = (value - np.min(color, axis=2)) / np.maximum(value, 1.)
        needle = ((value >= self.needle_min_value) & (saturation < .38)
                  & (blue >= red + 1.) & (green >= red - 2.)
                  & (blue >= green - 2.) & ~target & ~apex)
        needle = self._components(needle, self.needle_min_component_pixels)

        # The proximal tube's calibrated blue/gray finish has a higher B/R
        # ratio than the pale distal finish. Recover only immediately adjacent
        # neutral highlights; dilation never fills retinal/occluded regions.
        proximal_seed = needle & (blue > self.proximal_blue_red_ratio * np.maximum(red, 1.))
        proximal = needle & ndimage.binary_dilation(proximal_seed, iterations=1)
        proximal = self._components(proximal, self.needle_min_component_pixels)
        distal = self._components(needle & ~proximal, self.needle_min_component_pixels)
        return {'target': target, 'proximal': proximal, 'distal': distal, 'apex': apex}
