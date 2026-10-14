"""Label mapping onto the ICDR diabetic retinopathy scale (0 = no DR ... 4 = proliferative DR)."""

import re
from typing import Final

MAX_GRADE: Final = 4
UNGRADABLE: Final = 5
"""Sentinel grade for images that cannot be graded. Rows with this grade are dropped."""

DEFAULT_BINARY_CUTOFF: Final = 2
"""Grade 2 and above is "referable DR", the standard clinical screening cutoff."""

# ODIR terms after normalization (lower case, no spaces, no hyphens).
# Check order matters: "nonproliferative" contains "proliferative".
_ODIR_GRADED_TERMS: Final = (
    ("mildnonproliferative", 1),
    ("moderatenonproliferative", 2),
    ("severenonproliferative", 3),
)
_ODIR_PROLIFERATIVE: Final = "proliferativediabeticretinopathy"
_ODIR_UNGRADABLE_TERMS: Final = ("lowimagequality", "nofundusimage")
_ODIR_UNSPECIFIED_DR: Final = "diabeticretinopathy"
_ODIR_SEPARATORS: Final = re.compile("[,;\uff0c\uff1b]")  # ODIR uses fullwidth comma (U+FF0C)


def binary_from_grade(grade: int, cutoff: int) -> int:
    """Return 1 when the grade is at or above the cutoff, else 0."""
    if not 0 <= grade <= MAX_GRADE:
        raise ValueError(f"grade {grade} is outside the ICDR scale 0-{MAX_GRADE}")
    return int(grade >= cutoff)


def _normalize(term: str) -> str:
    return term.lower().replace(" ", "").replace("-", "").strip()


def _term_grade(term: str) -> int | None:
    """Return the DR grade one ODIR keyword states, or None when it states no grade."""
    for pattern, grade in _ODIR_GRADED_TERMS:
        if pattern in term:
            return grade
    if _ODIR_PROLIFERATIVE in term and "nonproliferative" not in term:
        return MAX_GRADE
    return None


def odir_eye_grade(keywords: str) -> int:
    """Map one eye's ODIR diagnostic keywords to an ICDR grade.

    Keywords with no DR term give grade 0. Unreadable images and DR of
    unspecified severity give UNGRADABLE.
    """
    terms = [_normalize(t) for t in _ODIR_SEPARATORS.split(keywords) if t.strip()]
    grades = [g for g in map(_term_grade, terms) if g is not None]
    if grades:
        return max(grades)
    unreadable = any(u in t for t in terms for u in _ODIR_UNGRADABLE_TERMS)
    unspecified_dr = any(_ODIR_UNSPECIFIED_DR in t for t in terms)
    if unreadable or unspecified_dr:
        return UNGRADABLE
    return 0
