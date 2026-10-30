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


# ---------------------------------------------------------------- P3a: sigma rules, divisor, repel shape
from dccnet.datasets.targets import REPEL_ALPHA, heatmap_sigmas  # noqa: E402

# two same-class cells 20 px apart, r = 24, stride 2 -> radius sigma r/(3s) = 4 cells
PAIR = [[200.0, 100.0, 24.0], [220.0, 100.0, 24.0]]
# a third same-class cell: A-B 20 px, A-C 30 px, B-C sqrt(20^2 + 30^2) px
TRIO = PAIR + [[200.0, 130.0, 24.0]]
BC = math.hypot(20.0, 30.0)


def _sig(circles, labels, rule, divisor=4.0, stride=2):
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    return heatmap_sigmas(c, np.asarray(labels, dtype=np.int64), stride, rule, divisor)


@pytest.mark.parametrize("rule, expected", [
    ("radius", [4.0, 4.0]),
    ("adaptive", [2.5, 2.5]),             # min(4, 20 / (4 * 2))
    ("adaptive_agnostic", [2.5, 2.5]),
    ("knn3", [2.5, 2.5]),                 # one neighbour -> mean = 20
    ("mcnn", [3.0, 3.0]),                 # 0.3 * 20 / 2
])
def test_sigma_rules_pair(rule, expected):
    np.testing.assert_allclose(_sig(PAIR, [0, 0], rule), expected, rtol=1e-12)


@pytest.mark.parametrize("rule, expected", [
    ("adaptive", [2.5, 2.5, 3.75]),                                    # d_nn = 20, 20, 30
    ("knn3", [25 / 8, (20 + BC) / 2 / 8, min(4.0, (30 + BC) / 2 / 8)]),  # mean over the 2 available neighbours
    ("mcnn", [0.3 * 25 / 2, 0.3 * (20 + BC) / 2 / 2, 0.3 * (30 + BC) / 2 / 2]),
])
def test_sigma_rules_trio_means(rule, expected):
    np.testing.assert_allclose(_sig(TRIO, [0, 0, 0], rule), expected, rtol=1e-12)


def test_knn3_uses_at_most_three_neighbours():
    # center cell plus 4 same-class neighbours at 10, 20, 30, 1000 px: mean of the 3 nearest = 20
    pts = [[300.0, 300.0, 24.0], [310.0, 300.0, 24.0], [300.0, 320.0, 24.0], [270.0, 300.0, 24.0],
           [300.0, 1300.0, 24.0]]
    s = _sig(pts, [0] * 5, "knn3")
    assert s[0] == pytest.approx(min(4.0, 20 / 8), rel=1e-12)
    m = _sig(pts, [0] * 5, "mcnn")
    assert m[0] == pytest.approx(0.3 * 20 / 2, rel=1e-12)


@pytest.mark.parametrize("rule", ["radius", "adaptive", "adaptive_agnostic", "knn3", "mcnn"])
def test_lone_cell_falls_back_to_radius_sigma(rule):
    np.testing.assert_allclose(_sig([[200.0, 100.0, 24.0]], [1], rule), [4.0], rtol=1e-12)
    # a same-class "pair" split over two classes also counts as lone, except for the agnostic rule
    s = _sig(PAIR, [0, 1], rule)
    np.testing.assert_allclose(s, [2.5, 2.5] if rule == "adaptive_agnostic" else [4.0, 4.0], rtol=1e-12)


@pytest.mark.parametrize("rule", ["radius", "adaptive", "adaptive_agnostic", "knn3", "mcnn"])
def test_sigma_floor_half_cell(rule):
    # 1 px apart, tiny radius: every rule floors at 0.5 cells
    s = _sig([[200.0, 100.0, 1.0], [201.0, 100.0, 1.0]], [0, 0], rule)
    np.testing.assert_array_equal(s, [0.5, 0.5])
    assert np.isfinite(_sig(np.zeros((0, 3)), [], rule)).all()


@pytest.mark.parametrize("divisor, expected", [(2.0, 4.0), (4.0, 2.5), (8.0, 1.25)])
@pytest.mark.parametrize("rule", ["adaptive", "adaptive_agnostic", "knn3"])
def test_divisor(rule, divisor, expected):
    # min(r / (3s) = 4, 20 / (D * 2))
    np.testing.assert_allclose(_sig(PAIR, [0, 0], rule, divisor), [expected] * 2, rtol=1e-12)


@pytest.mark.parametrize("rule", ["radius", "mcnn"])
def test_divisor_ignored_by_radius_and_mcnn(rule):
    np.testing.assert_array_equal(_sig(PAIR, [0, 0], rule, 2.0), _sig(PAIR, [0, 0], rule, 8.0))


@pytest.mark.parametrize("rule, sigma", [("adaptive_agnostic", 2.5), ("knn3", 2.5), ("mcnn", 3.0)])
def test_new_rules_shape_heatmap(rule, sigma):
    # stride 2: left peak at (iy, ix) = (50, 100); its left neighbour reflects its own sigma
    labels = [0, 1] if rule == "adaptive_agnostic" else [0, 0]
    t = _b(PAIR, labels, heatmap_sigma=rule, stride=2)
    hm = t["heatmap"][0]
    assert hm[50, 100].item() == 1.0
    assert hm[50, 99].item() == pytest.approx(math.exp(-1 / (2 * sigma**2)), rel=1e-6)
    r = _b(PAIR, labels, stride=2)
    for k in t:
        if k != "heatmap":
            assert torch.equal(t[k], r[k]), k


def test_divisor_shapes_adaptive_heatmap():
    for D, sigma in ((2.0, 4.0), (8.0, 1.25)):
        hm = _b(PAIR, [0, 0], heatmap_sigma="adaptive", heatmap_divisor=D, stride=2)["heatmap"][0]
        assert hm[50, 99].item() == pytest.approx(math.exp(-1 / (2 * sigma**2)), rel=1e-6)


def test_divisor_4_is_default_adaptive():
    a = _b(TRIO, [0, 0, 0], heatmap_sigma="adaptive", stride=2)
    b = _b(TRIO, [0, 0, 0], heatmap_sigma="adaptive", stride=2, heatmap_divisor=4.0, heatmap_shape="gaussian")
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("kw, match", [
    ({"heatmap_sigma": "knn5"}, "heatmap_sigma"),
    ({"heatmap_divisor": 0.0}, "heatmap_divisor"),
    ({"heatmap_divisor": -2.0}, "heatmap_divisor"),
    ({"heatmap_divisor": float("inf")}, "heatmap_divisor"),
    ({"heatmap_shape": "disk"}, "heatmap_shape"),
])
def test_invalid_p3a_values(kw, match):
    with pytest.raises(ValueError, match=match):
        _b([[10.0, 10.0, 5.0]], [0], **kw)


# ---- repel
def _repel(circles, labels, stride=2, **kw):
    return _b(circles, labels, heatmap_shape="repel", stride=stride, **kw)["heatmap"]


def test_repel_alpha():
    assert REPEL_ALPHA == 0.8


def test_repel_single_cell_form_and_support():
    # r = 24, s = 2 -> support radius 12 cells around (50, 100); d2 = inf -> value = 1 / (1 + a * d1)
    hm = _repel([[200.0, 100.0, 24.0]], [0])[0].numpy()
    assert hm[50, 100] == 1.0
    yy, xx = np.mgrid[0:320, 0:320]
    d1 = np.hypot(yy - 50, xx - 100)
    inside = d1 <= 12.0
    np.testing.assert_allclose(hm[inside], 1.0 / (1.0 + 0.8 * d1[inside]), rtol=1e-6)
    assert (hm[~inside] == 0).all()
    assert hm[50, 112] > 0 and hm[50, 113] == 0
    assert (hm[inside & (d1 > 0)] < 1.0).all()


def test_repel_peaks_and_valley_between_two_centers():
    # centers 6 cells apart (12 px at stride 2), r = 24: repel valley is far below the radius-gaussian valley
    circles = [[200.0, 100.0, 24.0], [212.0, 100.0, 24.0]]
    hm = _repel(circles, [0, 0])[0]
    gs = _b(circles, [0, 0], stride=2)["heatmap"][0]
    assert hm[50, 100].item() == 1.0 and hm[50, 106].item() == 1.0
    others = hm.clone()
    others[50, 100] = others[50, 106] = 0
    assert others.max().item() < 1.0
    # midpoint: d1 = d2 = 3 -> D' = 3 * 4 = 12
    assert hm[50, 103].item() == pytest.approx(1.0 / (1.0 + 0.8 * 12.0), rel=1e-6)
    assert hm[50, 103].item() < gs[50, 103].item()
    # monotone valley: decreasing from each peak to the midpoint
    row = hm[50, 100:107].tolist()
    assert row[0] > row[1] > row[2] > row[3] < row[4] < row[5] < row[6]
    # 1 cell right of the left peak: d1 = 1, d2 = 5 -> D' = 1 * (1.2)^2
    assert hm[50, 101].item() == pytest.approx(1.0 / (1.0 + 0.8 * 1.44), rel=1e-6)


def test_repel_is_per_class_and_other_targets_unchanged():
    circles = [[200.0, 100.0, 24.0], [212.0, 100.0, 24.0]]
    t = _b(circles, [0, 1], heatmap_shape="repel", stride=2)
    r = _b(circles, [0, 1], stride=2)
    hm = t["heatmap"]
    # different classes: each is a lone cell in its class -> 1/(1 + a*d1)
    assert hm[0, 50, 103].item() == pytest.approx(1.0 / (1.0 + 0.8 * 3.0), rel=1e-6)
    assert hm[1, 50, 103].item() == pytest.approx(1.0 / (1.0 + 0.8 * 3.0), rel=1e-6)
    assert hm[2].abs().sum().item() == 0
    for k in t:
        if k != "heatmap":
            assert torch.equal(t[k], r[k]), k


def test_repel_ignores_sigma_rule():
    a = _repel(TRIO, [0, 0, 0])
    for rule in ("adaptive", "knn3", "mcnn"):
        assert torch.equal(a, _repel(TRIO, [0, 0, 0], heatmap_sigma=rule, heatmap_divisor=8.0))


def test_repel_same_center_cell_and_finite():
    # two same-class centers in the same cell: d1 = d2 = 0 at that cell -> 1.0, never NaN
    hm = _repel([[200.2, 100.2, 24.0], [200.8, 100.9, 24.0], [5.0, 5.0, 3.0]], [0, 0, 2])
    assert torch.isfinite(hm).all()
    assert hm[0, 50, 100].item() == 1.0 and hm[2, 2, 2].item() == 1.0
    assert (hm <= 1.0).all() and (hm >= 0).all()


@pytest.mark.parametrize("shape", ["gaussian", "repel"])
@pytest.mark.parametrize("rule", ["radius", "adaptive", "adaptive_agnostic", "knn3", "mcnn"])
def test_zero_one_two_cells_per_class_finite(rule, shape):
    for circles, labels in (([], []), ([[50.0, 60.0, 9.0]], [2]), (PAIR, [1, 1]), (TRIO, [0, 1, 1])):
        t = _b(circles, labels, heatmap_sigma=rule, heatmap_shape=shape, stride=2, img_size=320)
        hm = t["heatmap"]
        assert torch.isfinite(hm).all() and hm.max().item() <= 1.0
        for (cx, cy, _), c in zip(circles, labels):
            assert hm[c, int(cy // 2), int(cx // 2)].item() == 1.0
