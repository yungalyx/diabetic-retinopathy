import pytest

from retina_dr.labels import UNGRADABLE, binary_from_grade, odir_eye_grade


@pytest.mark.parametrize(
    ("grade", "cutoff", "expected"),
    [(0, 2, 0), (1, 2, 0), (2, 2, 1), (4, 2, 1), (1, 1, 1), (0, 1, 0)],
)
def test_binary_label_is_one_at_or_above_cutoff(grade: int, cutoff: int, expected: int) -> None:
    assert binary_from_grade(grade, cutoff) == expected


def test_binary_label_rejects_grade_outside_icdr_scale() -> None:
    with pytest.raises(ValueError, match="grade"):
        binary_from_grade(5, 2)


@pytest.mark.parametrize(
    ("keywords", "expected"),
    [
        ("normal fundus", 0),
        ("cataract", 0),
        ("mild nonproliferative retinopathy", 1),
        ("moderate non proliferative retinopathy\uff0chypertensive retinopathy", 2),
        ("severe nonproliferative retinopathy", 3),
        ("proliferative diabetic retinopathy", 4),
        ("severe proliferative diabetic retinopathy", 4),
        ("low image quality", UNGRADABLE),
        ("no fundus image", UNGRADABLE),
        ("diabetic retinopathy", UNGRADABLE),
    ],
)
def test_odir_keywords_map_to_icdr_grade(keywords: str, expected: int) -> None:
    assert odir_eye_grade(keywords) == expected


def test_odir_takes_the_most_severe_grade_when_several_match() -> None:
    assert (
        odir_eye_grade("mild nonproliferative retinopathy\uff0csevere nonproliferative retinopathy")
        == 3
    )
