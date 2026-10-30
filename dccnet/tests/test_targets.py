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


# ---- P3a: heatmap target variants (sigma rules, divisor, repel shape) ----

import hashlib
import sys

from dccnet.datasets.targets import REPEL_ALPHA, heatmap_sigmas


def _frozen_build_targets_42b0203(circles, labels, radius_valid, img_size=640, stride=4, num_classes=3,
                                  max_objs=512, heatmap_sigma="radius"):
    """Verbatim copy of ``build_targets`` at commit 42b0203 (before P3a); do not edit."""
    assert heatmap_sigma in ("radius", "adaptive")
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    radius_valid = np.asarray(radius_valid).reshape(-1).astype(bool)
    n = len(circles)
    H = W = img_size // stride
    heat = np.zeros((num_classes, H, W), dtype=np.float32)
    dens = np.zeros((num_classes, H, W), dtype=np.float32)
    ind = np.zeros(max_objs, dtype=np.int64)
    mask = np.zeros(max_objs, dtype=bool)
    log_r = np.zeros(max_objs, dtype=np.float32)
    offset = np.zeros((max_objs, 2), dtype=np.float32)
    r_valid = np.zeros(max_objs, dtype=bool)
    cls = np.zeros(max_objs, dtype=np.int64)
    circle = np.zeros((max_objs, 3), dtype=np.float32)
    d_nn = None
    if heatmap_sigma == "adaptive":
        d_nn = np.full(len(circles), np.inf)
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            if len(idx) < 2:
                continue
            p = circles[idx, :2]
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            d_nn[idx] = d.min(axis=1)
    for k in range(n):
        cx, cy, r = circles[k]
        c = int(labels[k])
        u, v = cx / stride, cy / stride
        ix = int(min(max(math.floor(u), 0), W - 1))
        iy = int(min(max(math.floor(v), 0), H - 1))
        sigma = max(r / (3.0 * stride), 0.5)
        sigma_h = sigma if d_nn is None else max(min(r / (3.0 * stride), d_nn[k] / (4.0 * stride)), 0.5)
        rad = math.ceil(3.0 * sigma_h)
        y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
        x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
        ys = np.arange(y0, y1, dtype=np.float64)[:, None]
        xs = np.arange(x0, x1, dtype=np.float64)[None, :]
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma_h**2))
        patch = heat[c, y0:y1, x0:x1]
        np.maximum(patch, g.astype(np.float32), out=patch)
        rad = math.ceil(3.0 * sigma) + 1
        y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
        x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
        ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        g = np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2.0 * sigma**2))
        s = g.sum()
        if s > 0:
            dens[c, y0:y1, x0:x1] += (g / s).astype(np.float32)
        ind[k] = iy * W + ix
        mask[k] = True
        log_r[k] = math.log(r)
        offset[k] = (u - ix, v - iy)
        r_valid[k] = radius_valid[k]
        cls[k] = c
        circle[k] = (cx, cy, r)
    out = {"heatmap": heat, "density": dens, "ind": ind, "mask": mask, "log_r": log_r, "offset": offset,
           "r_valid": r_valid, "cls": cls, "circle": circle}
    return {k: torch.from_numpy(v) for k, v in out.items()}


def _regression_fixture(seed):
    rng = np.random.default_rng(seed)
    n = 80
    # half uniform, half clustered (close same-class neighbours exercise the adaptive rule)
    a = np.stack([rng.uniform(0, 640, n // 2), rng.uniform(0, 640, n // 2), rng.uniform(1, 40, n // 2)], 1)
    base = rng.uniform(100, 540, (n // 2, 2))
    b = np.concatenate([base + rng.normal(0, 8, base.shape), rng.uniform(4, 30, (n // 2, 1))], 1)
    return np.concatenate([a, b]), rng.integers(0, 3, n), rng.random(n) < 0.8


def _digest(t):
    h = hashlib.sha256()
    for k in sorted(t):
        h.update(k.encode())
        h.update(t[k].numpy().tobytes())
    return h.hexdigest()


# SHA-256 of every output tensor of build_targets as computed by the code at commit 42b0203 (Linux, numpy).
_REF_DIGESTS_42B0203 = {
    (11, "radius", 2): "eb43b45ca59e4f2cb7adde03e5b390a196b79f65dfe64981cd3fb3791fd337db",
    (11, "radius", 4): "099c97d027a2f52f2f46fbfffb5b17b59d096fa6cb9036678cc45b34e1278654",
    (11, "adaptive", 2): "d02655acf0ce6977897329f09776be93826ce59f4f1200b69766df2a7bf9b85c",
    (11, "adaptive", 4): "4982d57f56c0d50bef4e9bc9b6d75b9d2c2cdc406684d17d402f61500a3cad8a",
    (12, "radius", 2): "e5f21d7a25de8d6f1451eb198d5cc22bd860f64a482684c7df87f91f62d58d2d",
    (12, "radius", 4): "a7a2111c2c5855cccdcf61597a9cc0ade5ec945c37952c5c03ad7c40fd4cbe46",
    (12, "adaptive", 2): "88faa5641a6af329c973b84a20e65cb3f117cbaa8e00f21824dac6fa71923795",
    (12, "adaptive", 4): "7c6f60f78d7aba9406a5dba9aa59002ac64a4a6ba79d22bbe1c7b591b05eabe5",
}


@pytest.mark.parametrize("seed, rule, stride", sorted(_REF_DIGESTS_42B0203))
def test_default_path_bit_identical_to_42b0203(seed, rule, stride):
    c, l, v = _regression_fixture(seed)
    frozen = _frozen_build_targets_42b0203(c, l, v, stride=stride, heatmap_sigma=rule)
    implicit = build_targets(c, l, v, stride=stride, heatmap_sigma=rule)
    explicit = build_targets(c, l, v, stride=stride, heatmap_sigma=rule, heatmap_divisor=4.0,
                             heatmap_shape="gaussian")
    for k in frozen:
        assert torch.equal(implicit[k], frozen[k]), k
        assert torch.equal(explicit[k], frozen[k]), k
    if _digest(frozen) != _REF_DIGESTS_42B0203[(seed, rule, stride)]:
        if sys.platform != "linux":
            pytest.skip("stored digests were taken on Linux; exp() may differ in the last ulp elsewhere")
        pytest.fail("frozen reference no longer reproduces the stored 42b0203 digest")


R, S = 24.0, 2  # radius sigma r / (3 s) = 4.0 cells
TWO = [[200.0, 100.0, R], [220.0, 100.0, R]]  # same class, 20 px apart
THREE = TWO + [[250.0, 100.0, R]]  # distances: A-B 20, B-C 30, A-C 50


@pytest.mark.parametrize("rule, expected", [
    ("radius", [4.0, 4.0]),
    ("adaptive", [2.5, 2.5]),           # min(4, 20 / (4*2))
    ("adaptive_agnostic", [2.5, 2.5]),
    ("knn3", [2.5, 2.5]),               # mean of the one available neighbour = 20
    ("mcnn", [3.0, 3.0]),               # 0.3 * 20 / 2
])
def test_sigma_rules_two_same_class_cells(rule, expected):
    sig = heatmap_sigmas(np.asarray(TWO), np.array([0, 0]), stride=S, heatmap_sigma=rule)
    np.testing.assert_allclose(sig, expected, rtol=1e-12)
    hm = _b(TWO, [0, 0], stride=S, heatmap_sigma=rule)["heatmap"][0]
    assert hm[50, 100].item() == 1.0 and hm[50, 110].item() == 1.0
    # vertical neighbour of each center carries that center's own Gaussian
    for ix, sg in ((100, expected[0]), (110, expected[1])):
        assert hm[51, ix].item() == pytest.approx(math.exp(-1 / (2 * sg**2)), rel=1e-6)


@pytest.mark.parametrize("rule, expected", [
    ("adaptive", [2.5, 2.5, 3.75]),     # d_nn = 20, 20, 30 -> /8, capped at 4
    ("knn3", [4.0, 3.125, 4.0]),        # means 35, 25, 40 -> /8 = 4.375, 3.125, 5.0 -> capped at 4
    ("mcnn", [5.25, 3.75, 6.0]),        # 0.3 * (35, 25, 40) / 2, no cap
])
def test_sigma_rules_three_cells_use_mean_of_up_to_3_neighbours(rule, expected):
    sig = heatmap_sigmas(np.asarray(THREE), np.zeros(3, np.int64), stride=S, heatmap_sigma=rule)
    np.testing.assert_allclose(sig, expected, rtol=1e-12)
    hm = _b(THREE, [0, 0, 0], stride=S, heatmap_sigma=rule)["heatmap"][0]
    for ix, sg in ((100, expected[0]), (110, expected[1]), (125, expected[2])):
        assert hm[50, ix].item() == 1.0
        assert hm[51, ix].item() == pytest.approx(math.exp(-1 / (2 * sg**2)), rel=1e-6)


def test_knn3_mean_uses_only_3_nearest():
    # five cells on a line: the leftmost has neighbours at 10, 20, 30, 400 -> mean of 3 nearest = 20
    pts = [[100.0, 100.0, 60.0], [110.0, 100.0, 60.0], [120.0, 100.0, 60.0], [130.0, 100.0, 60.0],
           [500.0, 100.0, 60.0]]
    sig = heatmap_sigmas(np.asarray(pts), np.zeros(5, np.int64), stride=S, heatmap_sigma="knn3")
    assert sig[0] == pytest.approx(20.0 / 8.0, rel=1e-12)
    mc = heatmap_sigmas(np.asarray(pts), np.zeros(5, np.int64), stride=S, heatmap_sigma="mcnn")
    assert mc[0] == pytest.approx(0.3 * 20.0 / 2.0, rel=1e-12)


@pytest.mark.parametrize("rule", ["radius", "adaptive", "adaptive_agnostic", "knn3", "mcnn"])
def test_lone_cell_falls_back_to_radius_sigma(rule):
    sig = heatmap_sigmas(np.asarray([[200.0, 100.0, R]]), np.array([1]), stride=S, heatmap_sigma=rule)
    assert sig.tolist() == [4.0]
    t = _b([[200.0, 100.0, R]], [1], stride=S, heatmap_sigma=rule)
    assert torch.equal(t["heatmap"], _b([[200.0, 100.0, R]], [1], stride=S)["heatmap"])
    # tiny radius -> floor
    tiny = heatmap_sigmas(np.asarray([[200.0, 100.0, 1.0]]), np.array([1]), stride=S, heatmap_sigma=rule)
    assert tiny.tolist() == [0.5]


@pytest.mark.parametrize("rule", ["adaptive", "adaptive_agnostic", "knn3", "mcnn"])
def test_neighbour_rules_floor_at_half_cell(rule):
    pts = np.asarray([[200.0, 100.0, R], [201.0, 100.0, R], [200.0, 100.0, R]])  # incl. a coincident center
    sig = heatmap_sigmas(pts, np.zeros(3, np.int64), stride=S, heatmap_sigma=rule)
    assert sig.tolist() == [0.5, 0.5, 0.5]
    t = _b(pts, [0, 0, 0], stride=S, heatmap_sigma=rule)
    assert torch.isfinite(t["heatmap"]).all() and t["heatmap"].max().item() == 1.0


def test_agnostic_uses_different_class_neighbour():
    labels = np.array([0, 1])
    same = heatmap_sigmas(np.asarray(TWO), labels, stride=S, heatmap_sigma="adaptive")
    agn = heatmap_sigmas(np.asarray(TWO), labels, stride=S, heatmap_sigma="adaptive_agnostic")
    assert same.tolist() == [4.0, 4.0] and agn.tolist() == [2.5, 2.5]
    for rule in ("knn3", "mcnn"):  # same-class neighbour sets: a different-class cell is ignored
        assert heatmap_sigmas(np.asarray(TWO), labels, stride=S, heatmap_sigma=rule).tolist() == [4.0, 4.0]
    hm = _b(TWO, labels, stride=S, heatmap_sigma="adaptive_agnostic")["heatmap"]
    assert hm[0, 51, 100].item() == pytest.approx(math.exp(-1 / (2 * 2.5**2)), rel=1e-6)
    assert hm[1, 51, 110].item() == pytest.approx(math.exp(-1 / (2 * 2.5**2)), rel=1e-6)


@pytest.mark.parametrize("rule, divisor, expected", [
    ("adaptive", 2.0, 4.0),    # min(4, 20/4 = 5) -> capped by the radius rule
    ("adaptive", 8.0, 1.25),   # 20 / 16
    ("adaptive_agnostic", 8.0, 1.25),
    ("knn3", 2.0, 4.0),
    ("knn3", 8.0, 1.25),
    ("mcnn", 2.0, 3.0),        # mcnn uses its own 0.3 factor, not the divisor
    ("mcnn", 8.0, 3.0),
    ("radius", 8.0, 4.0),
])
def test_heatmap_divisor(rule, divisor, expected):
    sig = heatmap_sigmas(np.asarray(TWO), np.array([0, 0]), stride=S, heatmap_sigma=rule, heatmap_divisor=divisor)
    np.testing.assert_allclose(sig, [expected, expected], rtol=1e-12)
    hm = _b(TWO, [0, 0], stride=S, heatmap_sigma=rule, heatmap_divisor=divisor)["heatmap"][0]
    assert hm[51, 100].item() == pytest.approx(math.exp(-1 / (2 * expected**2)), rel=1e-6)


def test_heatmap_divisor_stride_4():
    # stride 4, divisor 8: min(24/12 = 2, 20/32 = 0.625) = 0.625
    sig = heatmap_sigmas(np.asarray(TWO), np.array([0, 0]), stride=4, heatmap_sigma="adaptive", heatmap_divisor=8)
    np.testing.assert_allclose(sig, [0.625, 0.625], rtol=1e-12)


@pytest.mark.parametrize("kw, match", [
    ({"heatmap_sigma": "bogus"}, "heatmap_sigma"),
    ({"heatmap_shape": "bogus"}, "heatmap_shape"),
    ({"heatmap_divisor": 0.0}, "heatmap_divisor"),
    ({"heatmap_divisor": -4.0}, "heatmap_divisor"),
    ({"heatmap_divisor": float("nan")}, "heatmap_divisor"),
    ({"heatmap_divisor": float("inf")}, "heatmap_divisor"),
    ({"heatmap_divisor": "4"}, "heatmap_divisor"),
])
def test_invalid_variant_values_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _b([[10.0, 10.0, 5.0]], [0], **kw)


def _repel_expected(centers_cells, radii_cells, shape, alpha=0.8):
    """Brute-force reference for one class: max over objects of the repel code on each object's support."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ctr = np.asarray(centers_cells, np.float64)
    d = np.sqrt((xx[..., None] - ctr[:, 0]) ** 2 + (yy[..., None] - ctr[:, 1]) ** 2)
    ds = np.sort(d, axis=-1)
    d1 = ds[..., 0]
    d2 = ds[..., 1] if len(ctr) > 1 else np.full_like(d1, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = np.where(d1 > 0, d1 * (1 + d1 / d2) ** 2, 0.0)
    val = 1.0 / (1.0 + alpha * dp)
    support = np.zeros((H, W), bool)
    for (x, y), rc in zip(ctr, radii_cells):
        support |= np.sqrt((xx - x) ** 2 + (yy - y) ** 2) <= rc
    return np.where(support, val, 0.0)


def test_repel_alpha_constant():
    assert REPEL_ALPHA == 0.8


def test_repel_single_cell_map_form():
    t = _b([[201.0, 101.0, R]], [2], stride=S, heatmap_shape="repel")
    hm = t["heatmap"][2].numpy().astype(np.float64)
    assert t["heatmap"][:2].abs().sum().item() == 0
    assert hm[50, 100] == 1.0 and (hm[np.arange(320) != 50] < 1).all()
    yy, xx = np.mgrid[0:320, 0:320]
    d1 = np.sqrt((xx - 100.0) ** 2 + (yy - 50.0) ** 2)
    expected = np.where(d1 <= R / S, 1.0 / (1.0 + 0.8 * d1), 0.0)
    np.testing.assert_allclose(hm, expected, rtol=1e-6, atol=0)
    assert hm[50, 112] == pytest.approx(1 / (1 + 0.8 * 12), rel=1e-6)  # edge of support (r/s = 12)
    assert hm[50, 113] == 0.0


def test_repel_two_centers_6_cells_apart_have_deep_valley():
    circles = [[200.0, 100.0, R], [212.0, 100.0, R]]  # stride 2 -> center cells (50,100), (50,106)
    rep = _b(circles, [0, 0], stride=S, heatmap_shape="repel")["heatmap"][0]
    gau = _b(circles, [0, 0], stride=S)["heatmap"][0]
    assert rep[50, 100].item() == 1.0 and rep[50, 106].item() == 1.0
    others = torch.ones_like(rep, dtype=torch.bool)
    others[50, 100] = others[50, 106] = False
    assert (rep[others] < 1.0).all()
    # midpoint: d1 = d2 = 3 -> D' = 3 * 4 = 12
    assert rep[50, 103].item() == pytest.approx(1 / (1 + 0.8 * 12), rel=1e-6)
    assert rep[50, 103].item() < gau[50, 103].item()
    assert rep[50, 103].item() < 0.5 < gau[50, 103].item()
    np.testing.assert_allclose(rep.numpy(), _repel_expected([(100, 50), (106, 50)], [12, 12], (320, 320)),
                               rtol=1e-6, atol=0)


def test_repel_matches_brute_force_per_class_and_ignores_other_classes():
    rng = np.random.default_rng(3)
    n = 40
    circles = np.stack([rng.uniform(0, 640, n), rng.uniform(0, 640, n), rng.uniform(2, 30, n)], 1)
    labels = rng.integers(0, 3, n)
    hm = _b(circles, labels, stride=4, heatmap_shape="repel")["heatmap"].numpy()
    for c in range(3):
        sel = labels == c
        ctr = np.stack([np.clip(np.floor(circles[sel, 0] / 4), 0, 159), np.clip(np.floor(circles[sel, 1] / 4), 0, 159)], 1)
        ref = _repel_expected(ctr, circles[sel, 2] / 4, (160, 160))
        np.testing.assert_allclose(hm[c], ref, rtol=1e-6, atol=0)


def test_repel_ignores_sigma_rule_and_divisor():
    circles, labels = np.asarray(THREE), [0, 0, 0]
    a = _b(circles, labels, stride=S, heatmap_shape="repel")
    b = _b(circles, labels, stride=S, heatmap_shape="repel", heatmap_sigma="mcnn", heatmap_divisor=8)
    assert torch.equal(a["heatmap"], b["heatmap"])


VARIANTS = [
    {"heatmap_sigma": "adaptive"},
    {"heatmap_sigma": "adaptive_agnostic"},
    {"heatmap_sigma": "knn3"},
    {"heatmap_sigma": "mcnn"},
    {"heatmap_sigma": "adaptive", "heatmap_divisor": 2},
    {"heatmap_sigma": "adaptive", "heatmap_divisor": 8},
    {"heatmap_shape": "repel"},
]


@pytest.mark.parametrize("kw", VARIANTS)
@pytest.mark.parametrize("stride", [2, 4])
def test_variants_change_only_heatmap(kw, stride):
    c, l, v = _regression_fixture(11)
    t = _b(c, l, v, stride=stride, **kw)
    ref = _b(c, l, v, stride=stride)
    for k in t:
        if k != "heatmap":
            assert torch.equal(t[k], ref[k]), k


@pytest.mark.parametrize("kw", VARIANTS)
@pytest.mark.parametrize("counts", [(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, 2, 1), (0, 2, 0)])
def test_variants_finite_with_0_1_2_cells_per_class(kw, counts):
    rng = np.random.default_rng(sum(counts))
    circles, labels = [], []
    for c, n in enumerate(counts):
        for _ in range(n):
            circles.append([rng.uniform(0, 640), rng.uniform(0, 640), rng.uniform(1, 40)])
            labels.append(c)
    t = _b(np.asarray(circles).reshape(-1, 3), labels, stride=2, **kw)
    hm = t["heatmap"]
    assert torch.isfinite(hm).all() and (hm >= 0).all() and (hm <= 1).all()
    for c, n in enumerate(counts):
        assert (hm[c] == 1.0).sum().item() == n  # random centers here never share a cell
