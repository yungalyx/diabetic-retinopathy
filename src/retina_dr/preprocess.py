"""Image preprocessing: crop to the fundus, pad to square, resize, and fingerprint for dedupe."""

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

DEFAULT_SIZE: Final = 512
JPEG_QUALITY: Final = 95
BLACK_THRESHOLD: Final = 10
"""Mean channel value at or below which a pixel counts as background."""
WHITE_THRESHOLD: Final = 235
"""Minimum channel value at or above which a pixel counts as white background (D-022)."""

FINGERPRINT_SIZE: Final = 32
"""Side of the grayscale thumbnail used to find duplicate images."""

Pixels = NDArray[np.uint8]


def crop_fundus(pixels: Pixels, threshold: int = BLACK_THRESHOLD) -> Pixels:
    """Crop to the bounding box of the fundus, then pad to a centred square.

    Black and white pixels count as background. After the crop, white pixels outside the
    inscribed ellipse become black, so every image has a black background (D-022).
    """
    black = pixels.mean(axis=2) <= threshold
    white = pixels.min(axis=2) >= WHITE_THRESHOLD
    mask = ~(black | white)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return pixels
    crop = pixels[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()
    height, width = crop.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    outside = ((yy - (height - 1) / 2) / (height / 2)) ** 2 + (
        (xx - (width - 1) / 2) / (width / 2)
    ) ** 2 > 1.0
    crop[outside & (crop.min(axis=2) >= WHITE_THRESHOLD)] = 0
    side = max(height, width)
    square = np.zeros((side, side, pixels.shape[2]), dtype=pixels.dtype)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = crop
    return square


def fingerprint(image: Image.Image) -> Pixels:
    """Flat 32x32 grayscale thumbnail (area average). Duplicates have near-equal fingerprints."""
    small = image.convert("L").resize((FINGERPRINT_SIZE, FINGERPRINT_SIZE), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.uint8).reshape(-1)


def fingerprint_existing(path: Path) -> Pixels:
    """Fingerprint of a processed image on disk."""
    with Image.open(path) as image:
        return fingerprint(image)


def process_image(src: Path, dst: Path, size: int = DEFAULT_SIZE) -> Pixels:
    """Crop, resize, and save src as a JPEG at dst. Return the saved image's fingerprint."""
    with Image.open(src) as raw:
        pixels: Pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    out = Image.fromarray(crop_fundus(pixels)).resize((size, size), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="JPEG", quality=JPEG_QUALITY)
    return fingerprint_existing(dst)
