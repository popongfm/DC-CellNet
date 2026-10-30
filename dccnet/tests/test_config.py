import numpy as np
import pytest
import yaml

from dccnet.config import load_config
from dccnet.types import Sample


def _w(p, d):
    p.write_text(yaml.safe_dump(d))
    return p


def test_base_inherit(tmp_path):
    base = _w(tmp_path / "base.yaml", {"loss": {"lambda_cc": 0.5, "lambda_r": 0.1}, "train": {"seed": 0}})
    child = _w(tmp_path / "child.yaml", {"_base_": "base.yaml", "loss": {"lambda_cc": 0.0}})
    cfg = load_config(child)
    assert cfg["loss"]["lambda_cc"] == 0.0
    assert cfg["loss"]["lambda_r"] == 0.1
    assert cfg["train"]["seed"] == 0
    assert "_base_" not in cfg


def test_nested_merge(tmp_path):
    _w(tmp_path / "base.yaml", {"a": {"b": {"c": 1, "d": 2}, "lst": [1, 2, 3]}})
    child = _w(tmp_path / "child.yaml", {"_base_": "base.yaml", "a": {"b": {"c": 9}, "lst": [7]}})
    cfg = load_config(child)
    assert cfg["a"]["b"] == {"c": 9, "d": 2}
    assert cfg["a"]["lst"] == [7]


def test_chained_base_and_cycle(tmp_path):
    _w(tmp_path / "a.yaml", {"x": 1, "y": 1})
    _w(tmp_path / "b.yaml", {"_base_": "a.yaml", "y": 2})
    c = _w(tmp_path / "c.yaml", {"_base_": "b.yaml", "z": 3})
    assert load_config(c) == {"x": 1, "y": 2, "z": 3}
    _w(tmp_path / "p.yaml", {"_base_": "q.yaml"})
    _w(tmp_path / "q.yaml", {"_base_": "p.yaml"})
    with pytest.raises(ValueError):
        load_config(tmp_path / "p.yaml")


def test_base_yaml_constraints():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    assert cfg["train"]["batch"] == 8 and cfg["train"]["amp"] is True
    assert cfg["train"]["lr"] == 1.25e-4
    assert cfg["loss"]["cc"]["window"] == 32
    assert cfg["decode"]["nms"] == "none"
    assert cfg["data"]["img_size"] == 640 and cfg["data"]["stride"] == 4


def test_sample_dataclass(tmp_path):
    s = Sample(tmp_path / "a.png", "txl", 640, 480,
               np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
    assert s.boxes.shape == (0, 4) and s.labels.dtype == np.int64


def test_base_defaults_for_m2d_knobs():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    assert cfg["data"]["heatmap_sigma"] == "radius"
    assert cfg["model"]["out_stride"] == 4 == cfg["data"]["stride"]


def _flat(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, (*prefix, k)))
        else:
            out[(*prefix, k)] = v
    return out


@pytest.mark.parametrize("name, expected", [
    ("exp_m2_d_sigma.yaml", {("data", "heatmap_sigma"): "adaptive"}),
    ("exp_m2_d_s2.yaml", {("model", "out_stride"): 2, ("data", "stride"): 2}),
    ("exp_m2_d_both.yaml", {("model", "out_stride"): 2, ("data", "stride"): 2, ("data", "heatmap_sigma"): "adaptive"}),
])
def test_m2d_configs_differ_from_b_circlenet_only_in_their_keys(name, expected):
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "configs"
    base = _flat(load_config(d / "exp_m2_b_circlenet.yaml"))
    cand = _flat(load_config(d / name))
    assert set(base) == set(cand)
    diff = {k: cand[k] for k in cand if cand[k] != base[k]}
    # stride-2 arm may also lower train.batch for VRAM (see README M2d)
    diff.pop(("train", "batch"), None)
    assert diff == expected
    assert cand[("data", "split")] == "splits/synthetic.json" and cand[("model", "pretrained")] is True
    assert cand[("model", "heads", "density")] is False and cand[("loss", "cc", "enabled")] is False
    assert (cand[("train", "epochs_max")], cand[("train", "patience")]) == (120, 20)


def test_base_defaults_for_p3a_knobs():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    assert cfg["data"]["heatmap_divisor"] == 4
    assert cfg["data"]["heatmap_shape"] == "gaussian"


@pytest.mark.parametrize("name, expected", [
    ("exp_m2_p3a_agnostic.yaml", {("data", "heatmap_sigma"): "adaptive_agnostic"}),
    ("exp_m2_p3a_knn3.yaml", {("data", "heatmap_sigma"): "knn3"}),
    ("exp_m2_p3a_mcnn.yaml", {("data", "heatmap_sigma"): "mcnn"}),
    ("exp_m2_p3a_repel.yaml", {("data", "heatmap_sigma"): "radius", ("data", "heatmap_shape"): "repel"}),
    ("exp_m2_p3a_div2.yaml", {("data", "heatmap_divisor"): 2}),
    ("exp_m2_p3a_div8.yaml", {("data", "heatmap_divisor"): 8}),
])
def test_p3a_configs_differ_from_d_both_only_in_their_keys(name, expected):
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "configs"
    raw = yaml.safe_load((d / name).read_text())
    assert raw["_base_"] == "base.yaml"
    base = _flat(load_config(d / "exp_m2_d_both.yaml"))
    cand = _flat(load_config(d / name))
    assert set(base) == set(cand)
    assert {k: cand[k] for k in cand if cand[k] != base[k]} == expected
    assert cand[("data", "split")] == "splits/synthetic.json" and cand[("model", "pretrained")] is True
    assert cand[("model", "out_stride")] == 2 == cand[("data", "stride")]
    assert cand[("model", "heads", "density")] is False and cand[("loss", "cc", "enabled")] is False
    assert (cand[("train", "epochs_max")], cand[("train", "patience")]) == (120, 20)
