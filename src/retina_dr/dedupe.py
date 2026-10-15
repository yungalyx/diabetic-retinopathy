"""Find duplicate images by the RMS distance between grayscale fingerprints.

Distances come from one matrix product per chunk: |a-b|^2 = |a|^2 + |b|^2 - 2ab.
This is exact up to float32 rounding and fast enough for about 100,000 images.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

DUPLICATE_RMS: Final = 2.0
"""Fingerprint RMS distance (0-255 scale) at or below which two images are duplicates.

On the real data (57,952 images) the nearest-neighbour distances were 0-1: 516, 1-2: 28,
2-4: 684, 4-8: 23,643. Copies sit below 1; the gap is at 1-2; distinct images start above 2.
"""

DEFAULT_CHUNK: Final = 2048

Fingerprints = NDArray[np.uint8]
Groups = NDArray[np.intp]


def _squared_distances(
    x: NDArray[np.float32], sq: NDArray[np.float32], start: int, stop: int
) -> NDArray[np.float32]:
    block = x[start:stop]
    d2 = sq[start:stop, None] + sq[None, :] - 2.0 * (block @ x.T)
    np.maximum(d2, 0.0, out=d2)
    d2[np.arange(stop - start), np.arange(start, stop)] = np.inf
    return d2


def _find(parent: NDArray[np.intp], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = int(parent[i])
    return i


def duplicate_groups(
    fingerprints: Fingerprints, max_rms: float = DUPLICATE_RMS, chunk: int = DEFAULT_CHUNK
) -> Groups:
    """Return a group id per row. Rows in one group are duplicates (transitively).

    The group id is the smallest row index in the group.
    """
    n, dim = fingerprints.shape
    x = fingerprints.astype(np.float32)
    sq = (x * x).sum(axis=1)
    limit = np.float32(max_rms * max_rms * dim)
    parent = np.arange(n, dtype=np.intp)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows, cols = np.nonzero(_squared_distances(x, sq, start, stop) <= limit)
        for a, b in zip((rows + start).tolist(), cols.tolist(), strict=True):
            ra, rb = _find(parent, a), _find(parent, b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return np.array([_find(parent, i) for i in range(n)], dtype=np.intp)


def nearest_distances(
    fingerprints: Fingerprints, chunk: int = DEFAULT_CHUNK
) -> NDArray[np.float32]:
    """RMS distance from each row to its closest other row."""
    n, dim = fingerprints.shape
    x = fingerprints.astype(np.float32)
    sq = (x * x).sum(axis=1)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = _squared_distances(x, sq, start, stop).min(axis=1)
    result: NDArray[np.float32] = np.sqrt(out / np.float32(dim)).astype(np.float32)
    return result
