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
from torchvision.transforms.v2.functional import pil_to_tensor

from retina_dr.train.sampling import NO_GRADE

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)

Transform = Callable[[Tensor], Tensor]
Batch = tuple[Tensor, Tensor, Tensor, Tensor]
"""images, binary label (float), grade (float, 0 when missing), has_grade (float 0/1)."""


@dataclass(frozen=True, slots=True)
class Labels:
    binary: NDArray[np.int64]
    grade: NDArray[np.int64]
    """Grade 0-4, or NO_GRADE for rows without a clinician grade."""


def labels_of(frame: pl.DataFrame) -> Labels:
    binary = frame["binary_label"].cast(pl.Int64).to_numpy()
    grade = frame["grade"].cast(pl.Int64).fill_null(NO_GRADE).to_numpy()
    return Labels(binary.astype(np.int64), grade.astype(np.int64))


def train_transform(size: int) -> Transform:
    transform: Transform = v2.Compose(
        [
            v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform


def eval_transform(size: int) -> Transform:
    transform: Transform = v2.Compose(
        [
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Batch:
        with Image.open(self.paths[index]) as raw:
            image: Tensor = pil_to_tensor(raw.convert("RGB"))
        pixels: Tensor = self.transform(image)
        return pixels, self.binary[index], self.grade[index], self.has_grade[index]
