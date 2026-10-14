from collections import Counter

from retina_dr.split import SplitConfig, assign_split


def test_holdout_source_always_goes_to_holdout() -> None:
    config = SplitConfig(holdout_source="ddr")
    assert {assign_split("ddr", str(i), config) for i in range(100)} == {"holdout"}


def test_split_is_deterministic_for_the_same_patient() -> None:
    config = SplitConfig()
    assert assign_split("eyepacs", "10", config) == assign_split("eyepacs", "10", config)


def test_split_changes_with_the_seed_for_some_patients() -> None:
    a = [assign_split("eyepacs", str(i), SplitConfig(seed=1)) for i in range(200)]
    b = [assign_split("eyepacs", str(i), SplitConfig(seed=2)) for i in range(200)]
    assert a != b


def test_split_fractions_are_close_to_config() -> None:
    config = SplitConfig(val_fraction=0.1, test_fraction=0.1)
    counts = Counter(assign_split("aptos", str(i), config) for i in range(20_000))
    assert abs(counts["train"] / 20_000 - 0.8) < 0.02
    assert abs(counts["val"] / 20_000 - 0.1) < 0.02
    assert abs(counts["test"] / 20_000 - 0.1) < 0.02
