"""Image preprocessing: crop to the fundus, pad to square, resize, and hash for deduplication."""

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

DEFAULT_SIZE: Final = 512
JPEG_QUALITY: Final = 95
BLACK_THRESHOLD: Final = 10
"""Mean channel value at or below which a pixel counts as background."""

Pixels = NDArray[np.uint8]


def crop_fundus(pixels: Pixels, threshold: int = BLACK_THRESHOLD) -> Pixels:
    """Crop to the bounding box of non-black pixels, then pad to a centred square."""
    mask = pixels.mean(axis=2) > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return pixels
    crop = pixels[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    height, width = crop.shape[:2]
    side = max(height, width)
    square = np.zeros((side, side, pixels.shape[2]), dtype=pixels.dtype)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = crop
    return square


def dhash(image: Image.Image) -> int:
    """64-bit difference hash. Equal hashes mean visually identical images."""
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def process_image(src: Path, dst: Path, size: int = DEFAULT_SIZE) -> int:
    """Crop, resize, and save src as a JPEG at dst. Return the output image's dhash."""
    with Image.open(src) as raw:
        pixels: Pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    out = Image.fromarray(crop_fundus(pixels)).resize((size, size), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="JPEG", quality=JPEG_QUALITY)
    return dhash(out)


def hash_existing(path: Path) -> int:
    """Return the dhash of an already-processed image, for resumed runs."""
    with Image.open(path) as image:
        return dhash(image)
