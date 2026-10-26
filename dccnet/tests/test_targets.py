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
