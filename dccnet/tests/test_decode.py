import numpy as np
import pytest
import torch

from dccnet.datasets.targets import build_targets
from dccnet.decode import decode


def dense_maps(t, img_size=640, stride=4):
    """Scatter build_targets sparse log_r/offset at `ind` into dense (1,1,H,W)/(1,2,H,W)."""
    H = W = img_size // stride
    log_r = torch.zeros(1, 1, H, W)
    offset = torch.zeros(1, 2, H, W)
    m = t["mask"]
    ind = t["ind"][m]
    iy, ix = ind // W, ind % W
    log_r[0, 0, iy, ix] = t["log_r"][m]
    offset[0, :, iy, ix] = t["offset"][m].T
    return log_r, offset


def sample_circles(n, rng, img_size=640, stride=4, min_sep_cells=3.0):
    H = img_size // stride
    pts, cells = [], []
    while len(pts) < n:
        cx, cy = rng.uniform(20, img_size - 20, 2)
        r = rng.uniform(8, 30)
        cell = np.array([cx / stride, cy / stride])
        if any(np.max(np.abs(cell - c)) < min_sep_cells for c in cells):
            continue
        cells.append(cell)
        pts.append((cx, cy, r))
    return np.array(pts, dtype=np.float32)


def test_round_trip_recovers_circles():
    rng = np.random.default_rng(0)
    circles = sample_circles(30, rng)
    labels = rng.integers(0, 3, 30)
    t = build_targets(circles, labels, np.ones(30, bool))
    log_r, offset = dense_maps(t)
    dets = decode(t["heatmap"][None], log_r, offset, score_thr=0.3, nms="none")
    assert len(dets) == 1
    d = dets[0]
    assert d.dtype == np.float32 and d.shape == (30, 5)
    assert np.all(np.diff(d[:, 4]) <= 0)
    for (cx, cy, r), c in zip(circles, labels):
        sel = d[(d[:, 0] == c)]
        dist = np.hypot(sel[:, 1] - cx, sel[:, 2] - cy)
        j = int(np.argmin(dist))
        assert dist[j] < 1.0
        assert abs(sel[j, 3] - r) / r < 0.05


def _density_case():
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 1, 1] = 0.9
    hm[0, 0, 1, 5] = 0.8
    hm[0, 0, 5, 3] = 0.2
    log_r = torch.full((1, 1, 8, 8), float(np.log(40.0)))
    offset = torch.zeros(1, 2, 8, 8)
    density = torch.zeros(1, 1, 8, 8)
    density[0, 0, 0, 0] = 3.0
    return hm, log_r, offset, density


def test_density_fill_adds_third_peak():
    hm, log_r, offset, density = _density_case()
    no_fill = decode(hm, log_r, offset, density, score_thr=0.3, density_fill=False, window=8)
    fill = decode(hm, log_r, offset, density, score_thr=0.3, density_fill=True, window=8, low_thr=0.1)
    assert len(no_fill[0]) == 2
    assert len(fill[0]) == 3
    assert fill[0][-1, 4] == pytest.approx(0.2)
    assert fill[0][-1, 1] == pytest.approx(3 * 4) and fill[0][-1, 2] == pytest.approx(5 * 4)


def test_density_fill_never_removes_and_respects_low_thr():
    hm, log_r, offset, density = _density_case()
    density[0, 0, 0, 0] = 1.0  # target 1 < 2 peaks: keep both
    out = decode(hm, log_r, offset, density, density_fill=True, window=8)
    assert len(out[0]) == 2
    density[0, 0, 0, 0] = 3.0
    out = decode(hm, log_r, offset, density, density_fill=True, window=8, low_thr=0.25)
    assert len(out[0]) == 2


def test_density_fill_requires_density():
    hm, log_r, offset, _ = _density_case()
    with pytest.raises(ValueError):
        decode(hm, log_r, offset, None, density_fill=True)


def test_empty_heatmap():
    out = decode(torch.zeros(2, 3, 16, 16), torch.zeros(2, 1, 16, 16), torch.zeros(2, 2, 16, 16))
    assert len(out) == 2
    for d in out:
        assert d.shape == (0, 5) and d.dtype == np.float32


def test_nms_and_max_dets_and_half_precision():
    hm = torch.zeros(1, 1, 16, 16)
    hm[0, 0, 4, 4] = 0.9
    hm[0, 0, 4, 6] = 0.8  # 2 cells = 8 px apart, r=40 -> IoU ~0.78
    hm[0, 0, 12, 12] = 0.7
    log_r = torch.full((1, 1, 16, 16), float(np.log(40.0)))
    offset = torch.zeros(1, 2, 16, 16)
    base = decode(hm.half(), log_r.half(), offset.half(), nms="none")
    assert len(base[0]) == 3
    for mode in ("circle", "box"):
        out = decode(hm, log_r, offset, nms=mode)
        assert len(out[0]) == 2
        assert out[0][0, 4] == pytest.approx(0.9)
    assert len(decode(hm, log_r, offset, max_dets=2)[0]) == 2
    with pytest.raises(ValueError):
        decode(hm, log_r, offset, nms="bogus")


def test_nms_is_class_wise():
    hm = torch.zeros(1, 2, 16, 16)
    hm[0, 0, 4, 4] = 0.9
    hm[0, 1, 4, 5] = 0.8
    log_r = torch.full((1, 1, 16, 16), float(np.log(40.0)))
    offset = torch.zeros(1, 2, 16, 16)
    assert len(decode(hm, log_r, offset, nms="circle")[0]) == 2


def test_round_trip_stride_2():
    rng = np.random.default_rng(1)
    circles = sample_circles(30, rng, stride=2)
    labels = rng.integers(0, 3, 30)
    t = build_targets(circles, labels, np.ones(30, bool), stride=2)
    assert t["heatmap"].shape == (3, 320, 320)
    log_r, offset = dense_maps(t, stride=2)
    d = decode(t["heatmap"][None], log_r, offset, score_thr=0.3, stride=2)[0]
    assert d.shape == (30, 5)
    for (cx, cy, r), c in zip(circles, labels):
        sel = d[d[:, 0] == c]
        dist = np.hypot(sel[:, 1] - cx, sel[:, 2] - cy)
        j = int(np.argmin(dist))
        assert dist[j] < 0.5
        assert abs(sel[j, 3] - r) / r < 0.05
