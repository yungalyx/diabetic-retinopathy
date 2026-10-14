from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

MakeImage = Callable[[Path, int], Path]


def write_fundus(path: Path, seed: int, size: tuple[int, int] = (160, 120)) -> Path:
    """Write a fake fundus: a bright noisy disc on a black background."""
    rng = np.random.default_rng(seed)
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    radius = min(width, height) * 0.4
    disc = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= radius**2
    pixels = rng.integers(60, 255, size=(height, width, 3), dtype=np.uint8)
    pixels[~disc] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture
def make_image() -> MakeImage:
    return write_fundus


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Build a tiny copy of each source's Kaggle layout."""
    root = tmp_path / "raw"

    eyepacs = root / "eyepacs"
    (eyepacs).mkdir(parents=True)
    (eyepacs / "trainLabels.csv").write_text("image,level\n10_left,0\n10_right,3\n13_left,1\n")
    for i, name in enumerate(["10_left", "10_right", "13_left"]):
        write_fundus(eyepacs / "train" / f"{name}.jpeg", seed=i)

    aptos = root / "aptos"
    aptos.mkdir(parents=True)
    (aptos / "train.csv").write_text("id_code,diagnosis\nabc123,2\ndef456,4\n")
    for i, name in enumerate(["abc123", "def456"]):
        write_fundus(aptos / "train_images" / f"{name}.png", seed=10 + i)

    ddr = root / "ddr" / "DR_grading"
    ddr.mkdir(parents=True)
    (ddr / "train.txt").write_text("007-0001-000.jpg 0\n007-0002-000.jpg 5\n")
    (ddr / "test.txt").write_text("007-0003-000.jpg 2\n")
    write_fundus(ddr / "train" / "007-0001-000.jpg", seed=20)
    write_fundus(ddr / "train" / "007-0002-000.jpg", seed=21)
    write_fundus(ddr / "test" / "007-0003-000.jpg", seed=22)

    odir = root / "odir"
    odir.mkdir(parents=True)
    (odir / "full_df.csv").write_text(
        "ID,Left-Fundus,Right-Fundus,Left-Diagnostic Keywords,Right-Diagnostic Keywords\n"
        "1,1_left.jpg,1_right.jpg,normal fundus,moderate non proliferative retinopathy\n"
        "1,1_left.jpg,1_right.jpg,normal fundus,moderate non proliferative retinopathy\n"
        "2,2_left.jpg,2_right.jpg,low image quality,cataract\n"
    )
    for i, name in enumerate(["1_left", "1_right", "2_left", "2_right"]):
        write_fundus(odir / "ODIR-5K" / "Training Images" / f"{name}.jpg", seed=30 + i)
    return root
