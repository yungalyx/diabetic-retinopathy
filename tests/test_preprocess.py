from pathlib import Path

import numpy as np
from PIL import Image

from retina_dr.preprocess import crop_fundus, dhash, process_image
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


def test_dhash_is_equal_for_the_same_image_and_differs_for_another(
    tmp_path: Path, make_image: MakeImage
) -> None:
    a = process_image(make_image(tmp_path / "a.png", 1), tmp_path / "a.jpg", size=64)
    a_again = process_image(make_image(tmp_path / "b.png", 1), tmp_path / "b.jpg", size=64)
    other = process_image(make_image(tmp_path / "c.png", 2), tmp_path / "c.jpg", size=64)
    assert a == a_again
    assert a != other


def test_dhash_fits_in_64_bits() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (50, 50, 3), dtype=np.uint8)
    assert 0 <= dhash(Image.fromarray(noise)) < 2**64
