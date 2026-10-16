"""Dataset over manifest rows, and the train and eval image transforms (D-010)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2

from retina_dr.quality import REJECT
from retina_dr.train.backbone import Normalization
from retina_dr.train.sampling import NO_GRADE

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)
IMAGENET: Final[Normalization] = (IMAGENET_MEAN, IMAGENET_STD)

Transform = Callable[[Image.Image], Tensor]
"""Takes a PIL image, returns a normalised float tensor. PIL ops run in C and are fast."""
Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
"""images, binary label, grade (0 when missing), has_grade, reject, has_quality. All float."""
NO_QUALITY: Final = -1


@dataclass(frozen=True, slots=True)
class Labels:
    binary: NDArray[np.int64]
    grade: NDArray[np.int64]
    """Grade 0-4, or NO_GRADE for rows without a clinician grade."""
    quality: NDArray[np.int64]
    """EyeQ code 0-2 (2 is reject), or NO_QUALITY when the row has no quality label."""


def labels_of(frame: pl.DataFrame) -> Labels:
    binary = frame["binary_label"].cast(pl.Int64).to_numpy()
    grade = frame["grade"].cast(pl.Int64).fill_null(NO_GRADE).to_numpy()
    if "quality" in frame.columns:
        quality = frame["quality"].cast(pl.Int64).fill_null(NO_QUALITY).to_numpy()
    else:
        quality = np.full(frame.height, NO_QUALITY)
    return Labels(binary.astype(np.int64), grade.astype(np.int64), quality.astype(np.int64))


def train_transform(size: int, norm: Normalization = IMAGENET) -> Transform:
    transform: Transform = v2.Compose(
        [
            v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(norm[0], norm[1]),
        ]
    )
    return transform


def eval_transform(size: int, norm: Normalization = IMAGENET) -> Transform:
    transform: Transform = v2.Compose(
        [
            v2.Resize((size, size), antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(norm[0], norm[1]),
        ]
    )
    return transform


class FundusDataset(Dataset[Batch]):
    """Reads processed JPEGs listed in a manifest frame. Paths are relative to `root`."""

    def __init__(self, frame: pl.DataFrame, root: Path, transform: Transform) -> None:
        self.paths = [str(root / p) for p in frame["path"].to_list()]
        labels = labels_of(frame)
        self.binary = torch.from_numpy(labels.binary.astype(np.float32))
        self.has_grade = torch.from_numpy((labels.grade >= 0).astype(np.float32))
        self.grade = torch.from_numpy(np.clip(labels.grade, 0, None).astype(np.float32))
        self.has_quality = torch.from_numpy((labels.quality >= 0).astype(np.float32))
        self.reject = torch.from_numpy((labels.quality == REJECT).astype(np.float32))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Batch:
        with Image.open(self.paths[index]) as raw:
            pixels = self.transform(raw.convert("RGB"))
        return (
            pixels,
            self.binary[index],
            self.grade[index],
            self.has_grade[index],
            self.reject[index],
            self.has_quality[index],
        )
