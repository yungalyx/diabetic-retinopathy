from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from retina_dr.preprocess import (
    FINGERPRINT_SIZE,
    crop_fundus,
    fingerprint,
    fingerprint_existing,
    process_image,
)
from tests.conftest import MakeImage


def test_crop_removes_black_border_and_returns_a_square() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:80, 120:180] = 200
    cropped = crop_fundus(image)
    assert cropped.shape == (60, 60, 3)
    assert cropped.min() == 200


def test_crop_pads_a_wide_fundus_to_square_with_black() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[40:60, 50:150] = 200
    cropped = crop_fundus(image)
    assert cropped.shape == (100, 100, 3)
    assert cropped[0].max() == 0
    assert cropped[50].min() == 200


def _fundus_on(background: int) -> NDArray[np.uint8]:
    image = np.full((100, 200, 3), background, dtype=np.uint8)
    yy, xx = np.mgrid[:100, :200]
    disc = (yy - 50) ** 2 + (xx - 100) ** 2 <= 30**2
    image[disc] = (150, 80, 40)
    return image


def test_crop_removes_a_white_background() -> None:
    cropped = crop_fundus(_fundus_on(255))
    assert cropped.shape == (61, 61, 3)
    for corner in (cropped[0, 0], cropped[0, -1], cropped[-1, 0], cropped[-1, -1]):
        assert corner.max() == 0
    assert tuple(cropped[30, 30]) == (150, 80, 40)


def test_crop_keeps_a_white_optic_disc_inside_the_fundus() -> None:
    image = _fundus_on(0)
    image[45:55, 115:125] = 255
    cropped = crop_fundus(image)
    assert cropped.shape == (61, 61, 3)
    assert cropped[30, 45].min() == 255


def test_crop_returns_input_unchanged_when_image_is_all_black() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert crop_fundus(image) is image


def test_process_writes_a_square_jpeg_at_target_size(tmp_path: Path, make_image: MakeImage) -> None:
    src = make_image(tmp_path / "in.png", 1)
    dst = tmp_path / "out" / "x.jpg"
    process_image(src, dst, size=64)
    with Image.open(dst) as out:
        assert out.size == (64, 64)
        assert out.format == "JPEG"


def test_fingerprint_matches_for_the_same_image_and_differs_for_another(
    tmp_path: Path, make_image: MakeImage
) -> None:
    a = process_image(make_image(tmp_path / "a.png", 1), tmp_path / "a.jpg", size=64)
    a_again = process_image(make_image(tmp_path / "b.png", 1), tmp_path / "b.jpg", size=64)
    other = process_image(make_image(tmp_path / "c.png", 2), tmp_path / "c.jpg", size=64)
    assert np.array_equal(a, a_again)
    assert np.abs(a.astype(int) - other.astype(int)).mean() > 4


def test_fingerprint_of_an_existing_file_matches_the_one_from_processing(
    tmp_path: Path, make_image: MakeImage
) -> None:
    from_processing = process_image(make_image(tmp_path / "a.png", 1), tmp_path / "a.jpg", size=64)
    assert np.array_equal(fingerprint_existing(tmp_path / "a.jpg"), from_processing)


def test_fingerprint_is_a_flat_grayscale_vector() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (50, 50, 3), dtype=np.uint8)
    fp = fingerprint(Image.fromarray(noise))
    assert fp.shape == (FINGERPRINT_SIZE * FINGERPRINT_SIZE,)
    assert fp.dtype == np.uint8
