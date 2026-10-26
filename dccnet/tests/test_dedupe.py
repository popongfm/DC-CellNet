import numpy as np
from PIL import Image

from dccnet.datasets.dedupe import find_duplicates, phash_file


def _blobs(seed, size=(200, 200)):
    rng = np.random.default_rng(seed)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    for _ in range(12):
        cx, cy = rng.integers(10, size[0] - 10), rng.integers(10, size[1] - 10)
        r = rng.integers(8, 30)
        col = rng.integers(40, 255, 3)
        arr[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = col
    return Image.fromarray(arr)


def test_recompressed_resized_is_near(tmp_path):
    im = _blobs(0)
    a, b = tmp_path / "a.png", tmp_path / "b.jpg"
    im.save(a)
    im.resize((180, 180)).save(b, quality=70)
    d = bin(phash_file(a) ^ phash_file(b)).count("1")
    assert d <= 6


def test_different_images_far(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _blobs(1).save(a)
    _blobs(2).save(b)
    assert bin(phash_file(a) ^ phash_file(b)).count("1") > 6
    assert isinstance(phash_file(a), int)


def test_find_duplicates_returns_pair_once(tmp_path):
    im = _blobs(3)
    a1, b1, b2 = tmp_path / "a1.png", tmp_path / "b1.jpg", tmp_path / "b2.png"
    im.save(a1)
    im.resize((180, 180)).save(b1, quality=70)
    _blobs(4).save(b2)
    res = find_duplicates([a1], [b1, b2])
    assert len(res) == 1
    assert res[0][0] == a1 and res[0][1] == b1 and res[0][2] <= 6
