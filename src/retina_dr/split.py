"""Patient-level split assignment by hashing, so the split is a column you can regenerate."""

import hashlib
from dataclasses import dataclass

from retina_dr.records import Source, Split


@dataclass(frozen=True, slots=True)
class SplitConfig:
    holdout_source: Source | None = "ddr"
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0


def _unit_interval(key: str) -> float:
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def assign_split(source: Source, patient_id: str, config: SplitConfig) -> Split:
    """Return the split for a patient. All images of one patient get the same split."""
    if source == config.holdout_source:
        return "holdout"
    u = _unit_interval(f"{config.seed}:{source}:{patient_id}")
    if u < config.test_fraction:
        return "test"
    if u < config.test_fraction + config.val_fraction:
        return "val"
    return "train"
