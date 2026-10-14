"""Domain types shared by every pipeline stage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

Source = Literal["eyepacs", "aptos", "ddr", "odir"]
SOURCES: Final[tuple[Source, ...]] = ("eyepacs", "aptos", "ddr", "odir")

Split = Literal["train", "val", "test", "holdout"]


@dataclass(frozen=True, slots=True)
class Record:
    """One labelled fundus image from one source.

    `label_grade` is the grade the binary label derives from.
    `grade` is the grade the severity head trains on. It is None when the
    source has no clinician grade (ODIR keyword grades are too coarse).
    """

    source: Source
    image_id: str
    raw_path: Path
    patient_id: str
    label_grade: int
    grade: int | None
