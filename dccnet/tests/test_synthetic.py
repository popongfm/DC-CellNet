from __future__ import annotations

import numpy as np

from dccnet.datasets.geometry import boxes_to_circles, overlap_levels
from dccnet.datasets.synthetic import make_synthetic


def test_deterministic(tmp_path):
    a = make_synthetic(tmp_path / "a", 3, seed=7)
    b = make_synthetic(tmp_path / "b", 3, seed=7)
    for sa, sb in zip(a, b):
        assert sa.image_path.name == sb.image_path.name
        assert sa.image_path.read_bytes() == sb.image_path.read_bytes()
        np.testing.assert_array_equal(sa.boxes, sb.boxes)
        np.testing.assert_array_equal(sa.labels, sb.labels)


def test_independent_of_n_images_and_start_index(tmp_path):
    a = make_synthetic(tmp_path / "a", 3, seed=1)
    b = make_synthetic(tmp_path / "b", 1, seed=1, start_index=2)
    assert b[0].image_path.name == "syn_00002.png"
    assert a[2].image_path.read_bytes() == b[0].image_path.read_bytes()
    np.testing.assert_array_equal(a[2].boxes, b[0].boxes)


def test_samples_well_formed(tmp_path):
    samples = make_synthetic(tmp_path, 5, seed=0, img_size=640)
    for s in samples:
        assert s.source == "SYN" and s.width == 640 and s.height == 640
        assert s.boxes.dtype == np.float32 and s.labels.dtype == np.int64
        assert s.boxes.shape == (len(s.labels), 4)
        assert s.image_path.exists()
        assert (s.labels == 0).sum() >= 1
        assert set(np.unique(s.labels)) <= {0, 1, 2}


def test_boxes_inside_image(tmp_path):
    for s in make_synthetic(tmp_path, 10, seed=3):
        assert (s.boxes[:, :2] >= 0).all()
        assert (s.boxes[:, 2] <= s.width).all() and (s.boxes[:, 3] <= s.height).all()
        w = s.boxes[:, 2] - s.boxes[:, 0]
        h = s.boxes[:, 3] - s.boxes[:, 1]
        np.testing.assert_allclose(w, h, atol=1e-3)


def test_overlap_bins_all_populated(tmp_path):
    samples = make_synthetic(tmp_path, 50, seed=0)
    lv = []
    for s in samples:
        c, _ = boxes_to_circles(s.boxes[s.labels == 0], s.width, s.height)
        lv.append(overlap_levels(c))
    lv = np.concatenate(lv)
    for k in range(4):
        assert (lv == k).sum() > 0, f"bin {k} empty"


def _disc_centroid(cls, cx, cy, r, size=128):
    """Weighted centroid (edge convention: pixel i covers [i, i+1), centre i + 0.5) of one drawn cell."""
    from dccnet.datasets import synthetic as syn

    bg = np.asarray(syn.BG_RGB, np.float32)
    img = np.empty((size, size, 3), np.float32)
    img[:] = bg
    syn._draw_cell(img, np.random.default_rng(0), cls, cx, cy, r)
    w = np.abs(img - bg).sum(axis=2).astype(np.float64)
    ys, xs = np.mgrid[0:size, 0:size] + 0.5
    return (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()


def test_drawn_disc_centroid_matches_gt_center():
    from dccnet.datasets.synthetic import PLT, RBC

    rng = np.random.default_rng(42)
    for cls, r in [(PLT, 7.3), (RBC, 21.6)]:  # PLT: uniform fill; RBC: fill + rim + pale centre (all concentric)
        for _ in range(5):
            cx, cy = rng.uniform(50, 78, 2)
            gx, gy = _disc_centroid(cls, float(cx), float(cy), r)
            assert abs(gx - cx) < 0.15 and abs(gy - cy) < 0.15, (cls, cx, cy, gx, gy)


def test_synthetic_version_constant():
    from dccnet.datasets.synthetic import SYNTHETIC_VERSION

    assert SYNTHETIC_VERSION == "2"


def test_make_synthetic_script_writes_version(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_synthetic as ms

    monkeypatch.setattr(ms, "SIZES", {"train": 1, "val": 1, "test": 1})
    out = tmp_path / "data" / "synthetic"
    ms.main(["--out", str(out), "--data-root", str(tmp_path / "data"), "--img-size", "128"])
    assert (out / "VERSION").read_text() == "2"  # no newline: read by cmd's `set /p`
    assert (tmp_path / "data" / "splits" / "synthetic.json").exists()
