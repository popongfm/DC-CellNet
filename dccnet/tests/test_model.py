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
    torch.manual_seed(0)
    m = DCCellNet(_cfg()).train()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 256, 256))
    assert abs(out["heatmap"].mean().item() - 0.10) < 0.03


def test_head_init(model):
    assert model.heads["heatmap"].out.bias.allclose(torch.tensor(-2.19))
    assert model.heads["log_r"].out.bias.item() == pytest.approx(2.7725887, abs=1e-5)
    assert model.heads["offset"].out.bias.abs().max() == 0
    assert model.heads["density"].out.bias.allclose(torch.tensor(-9.0))
    assert model.heads["density"].out.weight.std().item() == pytest.approx(0.01, rel=0.1)


def test_density_softplus_keeps_gradient_for_negative_preactivations():
    # ReLU heads died on WBC/PLT channels (R17); softplus must pass gradient at bias -9.
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
