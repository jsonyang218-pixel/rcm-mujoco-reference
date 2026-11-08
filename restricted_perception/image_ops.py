"""One-factor image degradation ladder for the Stage 1 causal gate."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def degrade_image(
    image: Image.Image | np.ndarray,
    *,
    resolution: int | None = None,
    gaussian_blur_radius: float = 0.0,
    center_occlusion_fraction: float = 0.0,
) -> Image.Image:
    result = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    )
    if resolution is not None:
        if int(resolution) < 1:
            raise ValueError("resolution must be positive")
        result = result.resize(
            (int(resolution), int(resolution)),
            Image.Resampling.BILINEAR,
        )
    if gaussian_blur_radius < 0.0:
        raise ValueError("gaussian_blur_radius must be nonnegative")
    if gaussian_blur_radius > 0.0:
        result = result.filter(ImageFilter.GaussianBlur(gaussian_blur_radius))
    if not 0.0 <= center_occlusion_fraction <= 1.0:
        raise ValueError("center_occlusion_fraction must be in [0, 1]")
    if center_occlusion_fraction > 0.0:
        width, height = result.size
        side = int(round(min(width, height) * center_occlusion_fraction))
        if side > 0:
            left = (width - side) // 2
            top = (height - side) // 2
            draw = ImageDraw.Draw(result)
            draw.rectangle((left, top, left + side - 1, top + side - 1), fill=(0, 0, 0))
    return result
