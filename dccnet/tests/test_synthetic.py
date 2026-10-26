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
