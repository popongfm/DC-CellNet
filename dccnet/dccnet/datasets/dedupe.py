"""Perceptual-hash (pHash) near-duplicate detection."""
from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


def phash_file(path: Path) -> int:
    """64-bit pHash (imagehash.phash, 8x8) of the image converted to RGB, as a Python int."""
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"))
    return int("".join("1" if b else "0" for b in h.hash.flatten()), 2)


def hash_paths(paths: list[Path]) -> np.ndarray:
    return np.array([phash_file(p) for p in paths], dtype=np.uint64)


def hamming_matrix(ha: np.ndarray, hb: np.ndarray) -> np.ndarray:
    """(len(ha), len(hb)) Hamming distances between uint64 hash arrays (vectorized popcount)."""
    if len(ha) == 0 or len(hb) == 0:
        return np.zeros((len(ha), len(hb)), dtype=np.int64)
    x = np.bitwise_xor(ha[:, None], hb[None, :])
    bits = np.unpackbits(np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8), axis=-1)
    return bits.sum(axis=-1, dtype=np.int64)


def find_duplicates(a: list[Path], b: list[Path], max_dist: int = 6) -> list[tuple[Path, Path, int]]:
    """All (a_i, b_j, hamming) with hamming <= max_dist. Pairs are returned once each."""
    a, b = list(a), list(b)
    d = hamming_matrix(hash_paths(a), hash_paths(b))
    ii, jj = np.nonzero(d <= max_dist)
    return [(a[i], b[j], int(d[i, j])) for i, j in zip(ii.tolist(), jj.tolist())]
