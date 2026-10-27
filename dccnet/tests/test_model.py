import copy
from pathlib import Path

import pytest
import torch

from dccnet.config import load_config
from dccnet.models import DCCellNet


def _cfg(**model_over):
    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    cfg["model"]["pretrained"] = False
    for k, v in model_over.items():
        cfg["model"][k] = v
    return cfg


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DCCellNet(_cfg())


def test_shapes_and_ranges_640(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 640, 640))
    assert out["heatmap"].shape == (1, 3, 160, 160)
    assert out["log_r"].shape == (1, 1, 160, 160)
    assert out["offset"].shape == (1, 2, 160, 160)
    assert out["density"].shape == (1, 3, 160, 160)
    assert out["heatmap"].min() >= 1e-4 and out["heatmap"].max() <= 1 - 1e-4
    assert out["density"].min() > 0  # softplus: strictly positive, never a dead ReLU


def test_density_absent_when_disabled():
    cfg = _cfg()
    cfg["model"]["heads"]["density"] = False
    m = DCCellNet(cfg).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 128, 128))
    assert "density" not in out
    assert set(out) == {"heatmap", "log_r", "offset"}


def test_backward_finite_grads():
    m = DCCellNet(_cfg()).train()
    out = m(torch.randn(2, 3, 128, 128))
    sum(v.sum() for v in out.values()).backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_outputs_float32_under_autocast(model):
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randn(1, 3, 256, 256))
    for k, v in out.items():
        assert v.dtype == torch.float32, k


def test_initial_heatmap_mean():
    # Train-mode BN (batch statistics) is what the first training step sees; eval-mode BN with
    # untrained running stats inflates activations and says nothing about the head init.
    # Input is a realistic RGB image in [0,1] (normalized inside forward). Expected mean stays
    # sigmoid(-2.19) ~= 0.10: the BN after the stem re-standardizes, so the ImageNet normalization
    # does not move it (measured 0.1013 both before and after normalization was added).
    torch.manual_seed(0)
    m = DCCellNet(_cfg()).train()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 256, 256))
    assert abs(out["heatmap"].mean().item() - 0.10) < 0.03


def test_head_init(model):
    assert model.heads["heatmap"].out.bias.allclose(torch.tensor(-2.19))
    assert model.heads["log_r"].out.bias.item() == pytest.approx(2.7725887, abs=1e-5)
    assert model.heads["offset"].out.bias.abs().max() == 0
    assert model.heads["density"].out.bias.allclose(torch.tensor(-7.0))
    assert model.heads["density"].out.weight.std().item() == pytest.approx(0.01, rel=0.1)


def test_density_softplus_keeps_gradient_for_negative_preactivations():
    # ReLU heads died on WBC/PLT channels (R17); softplus must pass gradient at bias -7.
    torch.manual_seed(0)
    m = DCCellNet(_cfg()).train()
    out = m(torch.randn(2, 3, 128, 128))
    out["density"].sum().backward()
    g = m.heads["density"].out.bias.grad
    assert g is not None and (g > 0).all()


def test_not_implemented():
    with pytest.raises(NotImplementedError, match="M3"):
        DCCellNet(_cfg(repr="box"))
    with pytest.raises(NotImplementedError, match="M3"):
        DCCellNet(_cfg(backbone="convnext_t"))


def test_bad_input_size(model):
    with pytest.raises(ValueError, match="100"):
        model(torch.randn(1, 3, 100, 128))


def test_imagenet_normalization_applied_in_forward():
    m = DCCellNet(_cfg()).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.equal(m.pixel_mean, mean) and torch.equal(m.pixel_std, std)
    # constants, not weights: old checkpoints without them still load strictly
    assert not any(k.startswith("pixel_") for k in m.state_dict())
    seen = {}
    orig = m.backbone.forward

    def spy(x):
        seen["x"] = x.clone()
        return orig(x)

    m.backbone.forward = spy
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        m(x)
    assert torch.allclose(seen["x"], (x - mean) / std, atol=1e-6)


# ---- M2b: detached density branch ----

def _detached_cfg(branch="detached"):
    cfg = _cfg()
    cfg["model"]["density_branch"] = branch
    return cfg


def _backbone_neck_params(m):
    return [p for n, mod in (("backbone", m.backbone), ("neck", m.neck)) for p in mod.parameters()]


def _density_only_backward(m):
    m.train()
    out = m(torch.randn(2, 3, 128, 128))
    out["density"].sum().backward()


def _is_zero_or_none(p):
    return p.grad is None or bool((p.grad == 0).all())


def test_detached_density_does_not_reach_backbone_or_neck():
    torch.manual_seed(0)
    m = DCCellNet(_detached_cfg("detached"))
    _density_only_backward(m)
    assert all(_is_zero_or_none(p) for p in _backbone_neck_params(m))
    dens = list(m.heads["density"].parameters())
    assert dens and all(p.grad is not None and p.grad.abs().sum() > 0 for p in dens)


def test_shared_density_reaches_neck():
    torch.manual_seed(0)
    m = DCCellNet(_detached_cfg("shared"))
    _density_only_backward(m)
    assert any(not _is_zero_or_none(p) for p in m.neck.parameters())


def test_default_density_branch_is_shared():
    torch.manual_seed(0)
    m = DCCellNet(_cfg())
    _density_only_backward(m)
    assert any(not _is_zero_or_none(p) for p in m.neck.parameters())
    assert type(m.heads["density"]).__name__ == "Head"


def test_detached_density_head_structure_and_init():
    torch.manual_seed(0)
    m = DCCellNet(_detached_cfg("detached")).eval()
    d = m.heads["density"]
    assert d.out.bias.allclose(torch.tensor(-7.0))
    assert d.out.weight.std().item() == pytest.approx(0.01, rel=0.2)
    assert d.out.kernel_size == (1, 1) and d.out.out_channels == 3
    convs = [mod for mod in d.modules() if isinstance(mod, torch.nn.Conv2d)]
    assert [(c.kernel_size, c.in_channels, c.out_channels) for c in convs] == [
        ((3, 3), 128, 128), ((3, 3), 128, 128), ((1, 1), 128, 3)]
    assert sum(isinstance(mod, torch.nn.BatchNorm2d) for mod in d.modules()) == 2
    with torch.no_grad():
        out = m(torch.randn(2, 3, 128, 128))
    assert out["density"].shape == (2, 3, 32, 32)
    assert torch.isfinite(out["density"]).all() and out["density"].min() >= 0
    assert out["density"].dtype == torch.float32


def test_invalid_density_branch_raises():
    with pytest.raises(ValueError, match="density_branch"):
        DCCellNet(_detached_cfg("bogus"))


def test_density_branch_ignored_without_density_head():
    cfg = _detached_cfg("bogus")
    cfg["model"]["heads"]["density"] = False
    m = DCCellNet(cfg)
    assert "density" not in m.heads


def test_cc_loss_reaches_heatmap_head_in_detached_mode():
    from dccnet.losses import count_consistency_loss

    torch.manual_seed(0)
    m = DCCellNet(_detached_cfg("detached")).train()
    out = m(torch.randn(2, 3, 128, 128))
    loss = count_consistency_loss(out["heatmap"], out["density"], window=8, stop_grad_density=True, peak_thr=1e-3)
    loss.backward()
    hm = list(m.heads["heatmap"].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in hm)
