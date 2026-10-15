import numpy as np
from numpy.typing import NDArray

from retina_dr.dedupe import duplicate_groups, nearest_distances

DIM = 1024


def _fp(value: int) -> NDArray[np.uint8]:
    return np.full(DIM, value, dtype=np.uint8)


def test_identical_and_near_identical_fingerprints_share_a_group() -> None:
    fps = np.stack([_fp(100), _fp(100), _fp(101), _fp(180)])
    groups = duplicate_groups(fps, max_rms=4.0)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_groups_are_transitive() -> None:
    fps = np.stack([_fp(100), _fp(103), _fp(106)])
    groups = duplicate_groups(fps, max_rms=4.0)
    assert len(set(groups.tolist())) == 1


def test_group_id_is_the_smallest_member_index() -> None:
    fps = np.stack([_fp(10), _fp(200), _fp(10)])
    assert duplicate_groups(fps, max_rms=4.0).tolist() == [0, 1, 0]


def test_chunking_gives_the_same_groups_as_one_block() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (50, DIM), dtype=np.uint8)
    fps = np.concatenate([base, base[:10]])
    one_block = duplicate_groups(fps, max_rms=4.0, chunk=1000)
    small_chunks = duplicate_groups(fps, max_rms=4.0, chunk=7)
    assert one_block.tolist() == small_chunks.tolist()
    assert len(set(one_block.tolist())) == 50


def test_nearest_distance_is_rms_to_the_closest_other_image() -> None:
    fps = np.stack([_fp(100), _fp(102), _fp(200)])
    assert np.allclose(nearest_distances(fps), [2.0, 2.0, 98.0], atol=0.05)
