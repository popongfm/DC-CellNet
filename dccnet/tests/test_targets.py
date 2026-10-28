import math

import numpy as np
import pytest
import torch

from dccnet.datasets.targets import build_targets


def _b(circles, labels, valid=None, **kw):
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    l = np.asarray(labels, dtype=np.int64).reshape(-1)
    v = np.ones(len(c), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return build_targets(c, l, v, **kw)


def test_single_circle_conventions():
    t = _b([[100.5, 200.25, 20.0]], [0])
    hm = t["heatmap"]
    assert hm.shape == (3, 160, 160) and hm.dtype == torch.float32
    assert hm[0].max().item() == 1.0
    iy, ix = np.unravel_index(int(hm[0].argmax()), (160, 160))
    assert (iy, ix) == (50, 25)
    assert hm[1:].abs().sum().item() == 0
    assert t["ind"].dtype == torch.int64 and t["ind"][0].item() == 50 * 160 + 25
    assert t["mask"].dtype == torch.bool and t["mask"].tolist()[:2] == [True, False]
    np.testing.assert_allclose(t["offset"][0].numpy(), [0.125, 0.0625], atol=1e-6)
    assert t["log_r"][0].item() == pytest.approx(math.log(20.0), abs=1e-6)
    np.testing.assert_allclose(t["circle"][0].numpy(), [100.5, 200.25, 20.0], atol=1e-4)
    assert t["cls"].dtype == torch.int64 and t["cls"][0].item() == 0
    assert t["r_valid"].dtype == torch.bool and t["r_valid"][0].item()
    for k in ("ind", "mask", "log_r", "offset", "r_valid", "cls", "circle"):
        assert t[k].shape[0] == 512
    assert t["density"].dtype == torch.float32 and t["density"].shape == (3, 160, 160)
    # unused slots zero
    assert t["mask"][1:].sum().item() == 0 and t["circle"][1:].abs().sum().item() == 0


def test_density_sums_to_count_including_border():
    t = _b([[4.0, 4.0, 10.0], [300.0, 300.0, 24.0], [638.0, 2.0, 6.0], [200.0, 100.0, 8.0]],
           [0, 0, 0, 1])
    d = t["density"]
    assert d[0].sum().item() == pytest.approx(3.0, abs=1e-3)
    assert d[1].sum().item() == pytest.approx(1.0, abs=1e-3)
    assert d[2].sum().item() == 0.0
    assert torch.isfinite(d).all()


def test_overlap_heatmap_is_max_with_two_peaks():
    t = _b([[200.0, 200.0, 20.0], [220.0, 200.0, 20.0]], [0, 0])
    hm = t["heatmap"][0]
    assert hm[50, 50].item() == 1.0 and hm[50, 55].item() == 1.0
    assert (hm <= 1.0).all()
    assert hm[50, 52].item() < 1.0


def test_zero_circles():
    t = _b(np.zeros((0, 3)), [])
    assert t["heatmap"].abs().sum().item() == 0 and t["density"].abs().sum().item() == 0
    assert not t["mask"].any() and not t["r_valid"].any()
    assert torch.isfinite(t["log_r"]).all()


def test_too_many_objects():
    n = 513
    c = np.tile([[100.0, 100.0, 5.0]], (n, 1))
    with pytest.raises(ValueError, match="max_objs.*513|513.*max_objs"):
        _b(c, np.zeros(n))


def test_nonpositive_radius_and_r_valid_passthrough():
    with pytest.raises(ValueError):
        _b([[10.0, 10.0, 0.0]], [0])
    t = _b([[10.0, 10.0, 5.0]], [1], valid=[False])
    assert not t["r_valid"][0].item() and t["mask"][0].item() and t["cls"][0].item() == 1


def test_small_radius_sigma_floor():
    t = _b([[10.0, 10.0, 1.0]], [2])
    assert t["heatmap"][2, 2, 2].item() == 1.0
    assert t["heatmap"][2, 2, 3].item() == pytest.approx(math.exp(-1 / (2 * 0.25)), rel=1e-5)


# ---- M2d: adaptive heatmap sigma ----

def _old_radius_heatmap(circles, labels, img_size=640, stride=4, num_classes=3):
    """Reference: the pre-M2d heatmap formula (sigma_h = max(r / (3 stride), 0.5) cells)."""
    H = W = img_size // stride
    heat = np.zeros((num_classes, H, W), dtype=np.float32)
    for (cx, cy, r), c in zip(np.asarray(circles, np.float64).reshape(-1, 3), labels):
        ix = int(min(max(math.floor(cx / stride), 0), W - 1))
        iy = int(min(max(math.floor(cy / stride), 0), H - 1))
        sigma = max(r / (3.0 * stride), 0.5)
        rad = math.ceil(3.0 * sigma)
        y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
        x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
        ys = np.arange(y0, y1, dtype=np.float64)[:, None]
        xs = np.arange(x0, x1, dtype=np.float64)[None, :]
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma**2))
        patch = heat[c, y0:y1, x0:x1]
        np.maximum(patch, g.astype(np.float32), out=patch)
    return heat


def _sigma_at_peak(hm, iy, ix):
    """Recover sigma (cells) from the value one cell right of an isolated peak."""
    return math.sqrt(-1.0 / (2.0 * math.log(hm[iy, ix + 1].item())))


def test_radius_sigma_default_is_bit_identical_to_old_formula():
    rng = np.random.default_rng(7)
    n = 60
    circles = np.stack([rng.uniform(0, 640, n), rng.uniform(0, 640, n), rng.uniform(1, 40, n)], 1)
    labels = rng.integers(0, 3, n)
    ref = _old_radius_heatmap(circles, labels)
    t_def = _b(circles, labels)
    t_rad = _b(circles, labels, heatmap_sigma="radius")
    assert np.array_equal(t_def["heatmap"].numpy(), ref)
    for k in t_def:
        assert torch.equal(t_def[k], t_rad[k]), k


def test_adaptive_sigma_two_same_class_cells_20px():
    # centers 20 px apart, r = 24, stride 4: sigma = max(min(24/12, 20/16), 0.5) = 1.25 cells
    t = _b([[200.0, 100.0, 24.0], [220.0, 100.0, 24.0]], [0, 0], heatmap_sigma="adaptive", stride=4)
    hm = t["heatmap"][0]
    assert hm[25, 50].item() == 1.0 and hm[25, 55].item() == 1.0
    # left peak's left neighbour is unaffected by the other cell
    assert hm[25, 49].item() == pytest.approx(math.exp(-1 / (2 * 1.25**2)), rel=1e-6)
    assert hm[26, 55].item() == pytest.approx(math.exp(-1 / (2 * 1.25**2)), rel=1e-6)


def test_adaptive_sigma_lone_cell_keeps_radius_sigma():
    t = _b([[200.0, 100.0, 24.0]], [0], heatmap_sigma="adaptive")
    r = _b([[200.0, 100.0, 24.0]], [0])
    assert torch.equal(t["heatmap"], r["heatmap"])
    assert _sigma_at_peak(t["heatmap"][0], 25, 50) == pytest.approx(24 / 12, rel=1e-5)


def test_adaptive_sigma_different_classes_keep_radius_sigma():
    circles = [[200.0, 100.0, 24.0], [212.0, 100.0, 24.0]]
    t = _b(circles, [0, 1], heatmap_sigma="adaptive")
    r = _b(circles, [0, 1])
    assert torch.equal(t["heatmap"], r["heatmap"])
    assert _sigma_at_peak(t["heatmap"][1], 25, 53) == pytest.approx(2.0, rel=1e-5)


def test_adaptive_sigma_floor_and_density_unchanged():
    circles = [[200.0, 100.0, 24.0], [201.0, 100.0, 24.0], [400.0, 400.0, 10.0]]
    t = _b(circles, [0, 0, 2], heatmap_sigma="adaptive")
    r = _b(circles, [0, 0, 2])
    # d_nn = 1 px -> 1/16 cell, floored at 0.5
    assert t["heatmap"][0, 25, 49].item() == pytest.approx(math.exp(-1 / (2 * 0.25)), rel=1e-6)
    for k in t:
        if k != "heatmap":
            assert torch.equal(t[k], r[k]), k


def test_adaptive_sigma_stride_2():
    # stride 2, centers 20 px apart, r = 24: sigma = max(min(24/6, 20/8), 0.5) = 2.5 cells
    t = _b([[200.0, 100.0, 24.0], [220.0, 100.0, 24.0]], [0, 0], heatmap_sigma="adaptive", stride=2)
    hm = t["heatmap"][0]
    assert hm.shape == (320, 320)
    assert hm[50, 99].item() == pytest.approx(math.exp(-1 / (2 * 2.5**2)), rel=1e-6)


def test_invalid_heatmap_sigma():
    with pytest.raises(ValueError, match="heatmap_sigma"):
        _b([[10.0, 10.0, 5.0]], [0], heatmap_sigma="bogus")


def _n_peaks_smoothed(hm, thr=0.3):
    """Local-maximum regions above thr after a sigma = 1 cell Gaussian blur: a proxy for the smooth
    map a conv head regresses (the max-combined GT itself always keeps both center cells at 1.0).
    Adjacent local-max cells (a plateau) count as one peak."""
    import cv2
    import torch.nn.functional as F

    x = torch.arange(-3, 4, dtype=torch.float64)
    k = torch.exp(-x**2 / 2)
    k = (k[:, None] * k[None, :]) / (k.sum() ** 2)
    b = F.conv2d(hm.double()[None, None], k[None, None], padding=3)
    lm = (b == F.max_pool2d(b, 3, stride=1, padding=1)) & (b > thr)
    n_labels, _ = cv2.connectedComponents(lm[0, 0].numpy().astype(np.uint8), connectivity=8)
    return n_labels - 1


def test_adaptive_sigma_keeps_close_same_class_peaks_separate():
    # 12 px apart (3 cells) with r = 24: radius sigma 2.0 merges, adaptive sigma 0.75 does not
    circles = [[200.0, 100.0, 24.0], [212.0, 100.0, 24.0]]
    rad = _b(circles, [0, 0])["heatmap"][0]
    ada = _b(circles, [0, 0], heatmap_sigma="adaptive")["heatmap"][0]
    assert _n_peaks_smoothed(rad) == 1
    assert _n_peaks_smoothed(ada) == 2
    # the valley between the two center cells is much deeper
    assert ada[25, 51].item() < 0.5 < rad[25, 51].item()
