import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dccnet.datasets.targets import build_targets
from dccnet.losses import (
    cc_weight,
    compute_losses,
    count_consistency_loss,
    density_loss,
    focal_loss,
    offset_loss,
    radius_ciou_loss,
)

EPS = 1e-4


def _batched_targets(circle_sets, img_size=64, stride=4, num_classes=3, max_objs=8):
    ts = []
    for circles, labels, rvalid in circle_sets:
        ts.append(
            build_targets(
                np.asarray(circles, dtype=np.float64).reshape(-1, 3),
                np.asarray(labels, dtype=np.int64),
                np.asarray(rvalid, dtype=bool),
                img_size=img_size,
                stride=stride,
                num_classes=num_classes,
                max_objs=max_objs,
            )
        )
    return {k: torch.stack([t[k] for t in ts]) for k in ts[0]}


def _dense_from_targets(t, H, W):
    """Perfect predictions written at the GT center cells."""
    B, K = t["ind"].shape
    log_r = torch.zeros(B, 1, H * W)
    off = torch.zeros(B, 2, H * W)
    for b in range(B):
        for k in range(K):
            if t["mask"][b, k]:
                i = t["ind"][b, k]
                log_r[b, 0, i] = t["log_r"][b, k]
                off[b, :, i] = t["offset"][b, k]
    return log_r.view(B, 1, H, W), off.view(B, 2, H, W)


SETS = [
    ([[20.3, 22.7, 6.0], [41.1, 30.9, 9.0]], [0, 1], [True, True]),
    ([[10.5, 50.2, 5.0], [33.3, 12.8, 7.0], [60.0, 60.0, 4.0]], [0, 2, 0], [True, True, False]),
]


# ---------------------------------------------------------------- focal
def test_focal_pred_equals_gt_near_zero():
    t = _batched_targets(SETS)
    gt = t["heatmap"]
    pred = gt.clamp(EPS, 1 - EPS)
    loss = focal_loss(pred, gt)
    bad = focal_loss(torch.full_like(gt, 0.5), gt)
    assert loss.dim() == 0 and loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() < 0.1
    assert loss.item() < 0.01 * bad.item()


def test_focal_all_zero_gt_finite():
    gt = torch.zeros(2, 3, 16, 16)
    pred = torch.rand(2, 3, 16, 16).clamp(EPS, 1 - EPS).requires_grad_()
    loss = focal_loss(pred, gt)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_focal_hand_value():
    gt = torch.tensor([[[[1.0, 0.5]]]])
    pred = torch.tensor([[[[0.8, 0.3]]]])
    pos = -math.log(0.8) * 0.2**2
    neg = -math.log(0.7) * 0.3**2 * 0.5**4
    assert focal_loss(pred, gt).item() == pytest.approx(pos + neg, rel=1e-5)


def test_focal_half_precision_input_returns_float32():
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 1, 1] = 1.0
    pred = torch.full((1, 1, 4, 4), 0.2).half()
    loss = focal_loss(pred, gt)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


# ---------------------------------------------------------------- radius / offset
def test_radius_perfect_prediction_zero():
    t = _batched_targets(SETS)
    log_r, off = _dense_from_targets(t, 16, 16)
    l1, ciou = radius_ciou_loss(log_r, off, t, stride=4)
    assert l1.item() == pytest.approx(0.0, abs=1e-6)
    assert ciou.item() == pytest.approx(0.0, abs=1e-4)
    assert offset_loss(off, t).item() == pytest.approx(0.0, abs=1e-7)


def test_radius_hand_value_and_valid_only():
    t = _batched_targets(SETS)
    log_r, off = _dense_from_targets(t, 16, 16)
    # perturb log_r by +0.1 on every predicted cell -> l1 = 0.1 on valid objects
    l1, ciou = radius_ciou_loss(log_r + 0.1, off, t, stride=4)
    assert l1.item() == pytest.approx(0.1, abs=1e-6)
    # concentric circles r and r*e^0.1 -> IoU = e^-0.2
    assert ciou.item() == pytest.approx(1 - math.exp(-0.2), abs=1e-4)
    # an absurd prediction at the r_valid=False object (b=1, k=2) must not matter
    i = t["ind"][1, 2]
    lr2 = log_r.clone().view(2, 1, -1)
    lr2[1, 0, i] = 50.0
    l1b, ciou_b = radius_ciou_loss(lr2.view_as(log_r), off, t, stride=4)
    assert l1b.item() == pytest.approx(0.0, abs=1e-6)
    assert ciou_b.item() == pytest.approx(0.0, abs=1e-4)


def test_offset_hand_value_includes_border_objects():
    t = _batched_targets(SETS)
    _, off = _dense_from_targets(t, 16, 16)
    # +0.2 everywhere: every mask object (incl. r_valid=False) has error 0.2
    assert offset_loss(off + 0.2, t).item() == pytest.approx(0.2, abs=1e-6)


def test_radius_no_valid_objects_zero_and_backprop():
    t = _batched_targets([([[60.0, 60.0, 4.0]], [0], [False]), ([], [], [])])
    log_r = torch.randn(2, 1, 16, 16, requires_grad=True)
    off = torch.rand(2, 2, 16, 16, requires_grad=True)
    l1, ciou = radius_ciou_loss(log_r, off, t, stride=4)
    assert l1.item() == 0.0 and ciou.item() == 0.0
    assert not torch.isnan(l1) and not torch.isnan(ciou)
    (l1 + ciou).backward()
    assert log_r.grad is not None and torch.all(log_r.grad == 0)


def test_offset_no_objects_zero_and_backprop():
    t = _batched_targets([([], [], []), ([], [], [])])
    off = torch.rand(2, 2, 16, 16, requires_grad=True)
    loss = offset_loss(off, t)
    assert loss.item() == 0.0
    loss.backward()
    assert off.grad is not None and torch.all(off.grad == 0)


# ---------------------------------------------------------------- density
def test_density_loss_hand_value():
    gt = torch.zeros(1, 2, 2, 2)
    gt[0, 0, 0, 0] = 1.0
    pred = torch.zeros(1, 2, 2, 2)
    pred[0, 0, 0, 0] = 0.5
    pred[0, 1, 1, 1] = 1.0
    mse = (0.25 + 1.0) / 8
    count = (0.5 / 1.0 + 1.0 / 1.0) / 2  # |pred-gt| / max(gt, 1); class 1 has gt 0 -> denominator 1
    assert density_loss(pred, gt).item() == pytest.approx(mse + count, rel=1e-6)


def test_density_loss_count_term_normalized_by_gt():
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, :2, :2] = 1.0  # sum 4
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, :2] = 1.0  # sum 2
    mse = 2.0 / 16
    count = abs(2.0 - 4.0) / 4.0
    assert density_loss(pred, gt).item() == pytest.approx(mse + count, rel=1e-6)


def test_density_loss_empty_maps_zero():
    z = torch.zeros(2, 3, 8, 8)
    assert density_loss(z, z).item() == 0.0


# ---------------------------------------------------------------- count consistency
def _three_peaks(H=64, W=64):
    hm = torch.zeros(1, 1, H, W)
    pts = [(3, 4), (10, 20), (25, 7)]  # all inside window 0 (32x32)
    for y, x in pts:
        hm[0, 0, y, x] = 1.0
    return hm, pts


def _density_sum(total, H=64, W=64):
    d = torch.zeros(1, 1, H, W)
    d[0, 0, 5:15, 5:15] = total / 100.0  # all inside window 0
    return d


def test_cc_matching_counts_zero():
    hm, _ = _three_peaks()
    d = _density_sum(3.0)
    loss = count_consistency_loss(hm, d, window=32)
    assert loss.dim() == 0 and loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_cc_mismatch_positive_grad_at_peaks_and_stopgrad():
    hm, pts = _three_peaks()
    hm.requires_grad_()
    d = _density_sum(5.0).requires_grad_()
    loss = count_consistency_loss(hm, d, window=32, stop_grad_density=True)
    assert loss.item() > 0
    loss.backward()
    for y, x in pts:
        assert hm.grad[0, 0, y, x] != 0
    # neighbours of a peak are not local maxima -> no gradient
    assert hm.grad[0, 0, 3, 5] == 0
    assert d.grad is None or torch.all(d.grad == 0)


def test_cc_two_way_gradient_reaches_density():
    hm, _ = _three_peaks()
    d = _density_sum(5.0).requires_grad_()
    count_consistency_loss(hm, d, window=32, stop_grad_density=False).backward()
    assert d.grad is not None and torch.any(d.grad != 0)


def test_cc_global_equals_local_full_window():
    torch.manual_seed(0)
    hm = torch.rand(2, 3, 160, 160).clamp(EPS, 1 - EPS)
    d = torch.rand(2, 3, 160, 160) * 0.01
    g = count_consistency_loss(hm, d, scope="global")
    loc = count_consistency_loss(hm, d, window=160, scope="local")
    assert g.item() == pytest.approx(loc.item(), rel=1e-6)
    # sanity: global differs from the default 32-cell local loss
    assert g.item() != pytest.approx(count_consistency_loss(hm, d).item(), rel=1e-3)


def test_cc_empty_maps_zero_finite():
    hm = torch.zeros(2, 3, 160, 160, requires_grad=True)
    d = torch.zeros(2, 3, 160, 160)
    loss = count_consistency_loss(hm, d)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(hm.grad).all()


def test_cc_non_divisible_map_hand_value():
    # 10x10 map, window 4 -> zero-padded to 12x12 -> 3x3 tiles (last row/col partial)
    hm = torch.zeros(1, 1, 10, 10)
    hm[0, 0, 1, 1] = 0.8  # tile (0,0)
    hm[0, 0, 5, 9] = 0.5  # tile (1,2) partial
    hm[0, 0, 9, 9] = 0.6  # tile (2,2) partial
    d = torch.zeros(1, 1, 10, 10)
    d[0, 0, 2, 2] = 1.0  # tile (0,0)
    d[0, 0, 8, 8] = 2.0  # tile (2,2)

    def sl1(x):
        x = abs(x)
        return 0.5 * x * x if x < 1 else x - 0.5

    expected = (sl1(0.8 - 1.0) + sl1(0.5) + sl1(0.6 - 2.0)) / 9
    loss = count_consistency_loss(hm, d, window=4, count_fn="linear")
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_cc_bad_scope_raises():
    z = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        count_consistency_loss(z, z, scope="nope")


# ---------------------------------------------------------------- cc_weight
def test_cc_weight_schedule():
    for e in range(10):
        assert cc_weight(e, 0.5) == 0.0
    assert cc_weight(10, 0.5) == 0.0
    assert cc_weight(12, 0.5) == pytest.approx(0.2)
    for e in (15, 16, 100):
        assert cc_weight(e, 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------- compute_losses
def _cfg(density=True, cc=True):
    return {
        "model": {"heads": {"density": density}},
        "loss": {
            "lambda_r": 0.1,
            "lambda_o": 1.0,
            "lambda_d": 1.0,
            "lambda_cc": 0.5,
            "cc": {
                "enabled": cc,
                "scope": "local",
                "window": 8,
                "stop_grad_density": True,
                "peak_thr": 0.01,
                "count_fn": "sigmoid",
                "count_center": 0.3,
                "count_temp": 0.05,
                "warmup_epochs": 10,
                "ramp_epochs": 5,
            },
        },
        "data": {"img_size": 64, "stride": 4},
    }


def _fake_outputs(with_density, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {
        "heatmap": torch.rand(2, 3, 16, 16, generator=g).clamp(EPS, 1 - EPS).requires_grad_(),
        "log_r": (torch.randn(2, 1, 16, 16, generator=g) + 1.5).requires_grad_(),
        "offset": torch.rand(2, 2, 16, 16, generator=g).requires_grad_(),
    }
    if with_density:
        out["density"] = (torch.rand(2, 3, 16, 16, generator=g) * 0.05).requires_grad_()
    return out


KEYS = {
    "loss_focal", "loss_r", "loss_ciou", "loss_o", "loss_d",
    "loss_cc", "loss_cc_bg", "lambda_cc", "loss_total",
}


def test_compute_losses_density_on():
    t = _batched_targets(SETS)
    out = _fake_outputs(True)
    cfg = _cfg(True, True)
    L = compute_losses(out, t, cfg, epoch=12)
    assert set(L) == KEYS
    for v in L.values():
        assert v.dim() == 0 and v.dtype == torch.float32 and torch.isfinite(v)
    assert L["lambda_cc"].item() == pytest.approx(0.2)
    assert L["loss_d"].item() > 0 and L["loss_cc"].item() > 0
    expected = (
        L["loss_focal"]
        + 0.1 * (L["loss_r"] + L["loss_ciou"])
        + 1.0 * L["loss_o"]
        + 1.0 * L["loss_d"]
        + 0.2 * L["loss_cc"]
    )
    assert L["loss_total"].item() == pytest.approx(expected.item(), rel=1e-6)
    L["loss_total"].backward()
    for v in out.values():
        assert v.grad is not None and torch.isfinite(v.grad).all()


def test_compute_losses_warmup_cc_still_logged():
    t = _batched_targets(SETS)
    L = compute_losses(_fake_outputs(True), t, _cfg(True, True), epoch=3)
    assert L["lambda_cc"].item() == 0.0
    assert L["loss_cc"].item() > 0  # logged even while its weight is 0


def test_compute_losses_density_off():
    t = _batched_targets(SETS)
    out = _fake_outputs(False)
    L = compute_losses(out, t, _cfg(False, False), epoch=20)
    assert set(L) == KEYS
    assert L["loss_d"].item() == 0.0 and L["loss_cc"].item() == 0.0
    assert L["loss_cc_bg"].item() == 0.0
    assert L["lambda_cc"].item() == 0.0
    expected = L["loss_focal"] + 0.1 * (L["loss_r"] + L["loss_ciou"]) + L["loss_o"]
    assert L["loss_total"].item() == pytest.approx(expected.item(), rel=1e-6)
    L["loss_total"].backward()


def test_compute_losses_density_on_cc_disabled():
    t = _batched_targets(SETS)
    L = compute_losses(_fake_outputs(True), t, _cfg(True, False), epoch=20)
    assert L["loss_d"].item() > 0
    assert L["loss_cc"].item() == 0.0 and L["lambda_cc"].item() == 0.0
    assert L["loss_cc_bg"].item() == 0.0


# ---------------------------------------------------------------- fix round 1
def test_focal_fp16_exact_zero_and_one_finite():
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 1, 1] = 1.0
    pred = torch.zeros(1, 1, 4, 4, dtype=torch.float16)
    pred[0, 0, 1, 1] = 1.0  # what fp16 clamp(max=1-1e-4) produces
    pred[0, 0, 2, 2] = 1.0  # false positive at exactly 1.0
    loss = focal_loss(pred, gt)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_radius_huge_log_r_finite_ciou():
    t = _batched_targets(SETS)
    log_r, off = _dense_from_targets(t, 16, 16)
    big = torch.full_like(log_r, 100.0).requires_grad_()
    l1, ciou = radius_ciou_loss(big, off, t, stride=4)
    assert torch.isfinite(l1) and torch.isfinite(ciou)
    (l1 + ciou).backward()
    assert torch.isfinite(big.grad).all()


def _floor_probe(H=64, W=64):
    hm = torch.full((1, 1, H, W), EPS)
    pts = [(3, 4), (10, 20), (25, 7)]
    for y, x in pts:
        hm[0, 0, y, x] = 1.0
    return hm, pts


def _exact_density(n, H=64, W=64):
    d = torch.zeros(1, 1, H, W)
    for i in range(n):
        d[0, 0, 2 + i, 2] = 1.0  # exact integer mass inside window 0
    return d


def _peak_free(grad, pts):
    g = grad.clone()
    for y, x in pts:
        g[0, 0, y, x] = 0
    return g


@pytest.mark.parametrize("count_fn", ["linear", "sigmoid"])
@pytest.mark.parametrize("scope", ["local", "global"])
def test_cc_floor_background_ignored(scope, count_fn):
    # peaks at H = 1 count exactly 1 (linear) or 1 - 8e-7 (sigmoid at (1 - 0.3) / 0.05 = 14)
    hm, pts = _floor_probe()
    hm.requires_grad_()
    d = _exact_density(3)
    loss = count_consistency_loss(hm, d, window=32, scope=scope, count_fn=count_fn)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    loss.backward()
    assert torch.all(_peak_free(hm.grad, pts) == 0)  # floor plateau: no gradient
    if count_fn == "linear":
        assert loss.item() == 0.0 and torch.all(hm.grad == 0)


@pytest.mark.parametrize("count_fn", ["linear", "sigmoid"])
def test_cc_subthreshold_background_max_no_effect(count_fn):
    hm, pts = _floor_probe()
    hm[0, 0, 50, 50] = 0.005  # isolated local max below peak_thr, tile (1,1)
    hm.requires_grad_()
    d = _exact_density(3)
    loss = count_consistency_loss(hm, d, window=32, count_fn=count_fn)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    loss.backward()
    assert hm.grad[0, 0, 50, 50] == 0


def test_cc_weak_candidate_in_undercounted_tile_pushed_up():
    hm, pts = _floor_probe()
    hm[0, 0, 15, 15] = 0.05  # split candidate, above peak_thr but below decode.low_thr
    hm.requires_grad_()
    d = _density_sum(5.0)
    loss = count_consistency_loss(hm, d, window=32)
    assert loss.item() > 0
    loss.backward()
    # gradient descent (H -= lr * grad) increases H at the candidate
    assert hm.grad[0, 0, 15, 15] < 0


def test_cc_background_mass_diagnostic():
    from dccnet.losses import cc_background_mass

    hm = torch.zeros(2, 1, 8, 8)
    hm[0, 0, 1, 1] = 0.005  # counted (<= thr, local max)
    hm[0, 0, 5, 5] = 0.9    # real peak, not counted
    hm[1, 0, 4, 4] = 0.01   # == thr -> counted
    assert cc_background_mass(hm, 0.01).item() == pytest.approx((0.005 + 0.01) / 2, rel=1e-6)


def test_compute_losses_cc_bg_logged_not_in_total():
    t = _batched_targets(SETS)
    out = _fake_outputs(True)
    with torch.no_grad():
        out["heatmap"].clamp_(max=0.009)  # every local max is background
    L = compute_losses(out, t, _cfg(True, True), epoch=20)
    assert L["loss_cc_bg"].item() > 0 and not L["loss_cc_bg"].requires_grad
    expected = (
        L["loss_focal"] + 0.1 * (L["loss_r"] + L["loss_ciou"]) + L["loss_o"]
        + L["loss_d"] + 0.5 * L["loss_cc"]
    )
    assert L["loss_total"].item() == pytest.approx(expected.item(), rel=1e-6)


# ---------------------------------------------------------------- R21: soft count = sigmoid((H - c) / T)
def _peaks_at(vals, H=64, W=64):
    """Isolated peaks (all inside tile 0 of a 32 window) with the given heights, on a zero map."""
    hm = torch.zeros(1, 1, H, W)
    pts = [(3, 4), (10, 20), (25, 7), (15, 15), (28, 28)][: len(vals)]
    for (y, x), v in zip(pts, vals):
        hm[0, 0, y, x] = v
    return hm, pts


def test_cc_sigmoid_confident_peaks_matching_density_near_zero():
    hm, _ = _peaks_at([0.95, 0.95, 0.95])
    d = _exact_density(3)
    sig = count_consistency_loss(hm, d, window=32)  # default count_fn="sigmoid"
    lin = count_consistency_loss(hm, d, window=32, count_fn="linear")
    assert sig.item() == pytest.approx(0.0, abs=1e-8)
    # linear counts 3 * 0.95 = 2.85 against 3 -> keeps pushing confident peaks toward 1
    assert lin.item() > 1e-4
    assert lin.item() == pytest.approx(0.5 * 0.15 ** 2 / 4, rel=1e-5)  # one tile of 2x2 non-zero


def test_cc_sigmoid_weak_candidate_gets_larger_upward_gradient_than_confident_peak():
    hm, pts = _peaks_at([0.95, 0.95, 0.25])  # third is a weak split candidate
    hm.requires_grad_()
    d = _exact_density(4)  # under-counted tile
    loss = count_consistency_loss(hm, d, window=32)
    assert loss.item() > 0
    loss.backward()
    g_conf = hm.grad[0, 0, pts[0][0], pts[0][1]].item()
    g_weak = hm.grad[0, 0, pts[2][0], pts[2][1]].item()
    assert g_weak < 0 and g_conf <= 0  # descent raises both
    assert abs(g_weak) > 10 * abs(g_conf)


def test_cc_linear_reproduces_old_soft_count():
    torch.manual_seed(0)
    hm = torch.rand(2, 3, 40, 40).clamp(EPS, 1 - EPS)
    d = torch.rand(2, 3, 40, 40) * 0.02
    peak = ((hm == F.max_pool2d(hm, 3, stride=1, padding=1)) & (hm > 0.01)).float()

    def tiles(x, w=8):
        return x.reshape(2, 3, 5, w, 5, w).sum(dim=(3, 5)).reshape(2, 3, 25)

    expected = F.smooth_l1_loss(tiles(hm * peak), tiles(d), beta=1.0)
    got = count_consistency_loss(hm, d, window=8, count_fn="linear")
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_cc_sigmoid_value_and_peak_gate():
    hm, _ = _peaks_at([0.3, 0.005])  # 0.005 is a local max below peak_thr -> excluded
    d = torch.zeros_like(hm)
    loss = count_consistency_loss(hm, d, window=32, count_center=0.3, count_temp=0.05)
    assert loss.item() == pytest.approx(0.5 * 0.5 ** 2 / 4, rel=1e-6)  # sigmoid(0) = 0.5 in one of 4 tiles


def test_cc_bad_count_fn_raises():
    z = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        count_consistency_loss(z, z, count_fn="nope")


def test_compute_losses_wires_count_fn():
    t = _batched_targets(SETS)
    cfg_s, cfg_l = _cfg(True, True), _cfg(True, True)
    cfg_l["loss"]["cc"]["count_fn"] = "linear"
    ls = compute_losses(_fake_outputs(True), t, cfg_s, epoch=20)["loss_cc"]
    ll = compute_losses(_fake_outputs(True), t, cfg_l, epoch=20)["loss_cc"]
    out = _fake_outputs(True)
    ref = count_consistency_loss(out["heatmap"], out["density"], window=8, count_fn="linear")
    assert ll.item() == pytest.approx(ref.item(), rel=1e-6)
    assert ls.item() != pytest.approx(ll.item(), rel=1e-3)


def test_base_yaml_cc_count_keys():
    from pathlib import Path

    from dccnet.config import load_config

    cc = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")["loss"]["cc"]
    assert (cc["count_fn"], cc["count_center"], cc["count_temp"]) == ("sigmoid", 0.3, 0.05)
