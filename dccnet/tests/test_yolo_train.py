import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from dccnet.datasets.yolo_export import write_yolo
from dccnet.types import Sample

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("yolo_train", ROOT / "scripts" / "baselines" / "yolo_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _samples(tmp_path):
    out = []
    for i, (boxes, labels) in enumerate([([[10, 20, 50, 60], [0, 0, 20, 10], [30, 30, 40, 50]], [1, 0, 2]), ([], [])]):
        p = tmp_path / f"im{i}.png"
        Image.new("RGB", (100, 80)).save(p)
        out.append(Sample(p, "PBC", 100, 80, np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                          np.asarray(labels, dtype=np.int64)))
    return out


def test_write_yolo_uses_split_labels(tmp_path):
    s = _samples(tmp_path)
    out = tmp_path / "yolo"
    yp = write_yolo({"train": s[:1], "val": s[1:], "test": []}, out)
    assert yp == out / "data.yaml"
    lines = (out / "labels" / "train" / "PBC_im0.txt").read_text().strip().splitlines()
    assert [ln.split()[0] for ln in lines] == ["1", "0", "2"]
    assert lines[0] == "1 0.300000 0.500000 0.400000 0.500000"
    assert (out / "labels" / "val" / "PBC_im1.txt").read_text() == ""
    cfg = yaml.safe_load(yp.read_text())
    assert cfg["names"] == {0: "RBC", 1: "WBC", 2: "Platelets"}
    assert Path(cfg["path"]).is_absolute()


def test_prepare_data_yaml_name_unchanged(tmp_path):
    yp = write_yolo({"train": _samples(tmp_path)[:1]}, tmp_path / "y", yaml_name="data_pooled.yaml")
    assert yp.name == "data_pooled.yaml" and yp.exists()


def test_best_epoch_from_results(tmp_path):
    mod = _load_module()
    csv = tmp_path / "results.csv"
    csv.write_text("                  epoch,   metrics/mAP50-95(B)\n0, 0.1\n1, 0.5\n2, 0.5\n3, 0.3\n")
    assert mod.best_epoch_from_results(csv) == 1
    assert mod.count_epochs(csv) == 4


def test_best_epoch_from_ckpt(tmp_path, caplog):
    mod = _load_module()
    csv = tmp_path / "results.csv"
    csv.write_text("epoch,metrics/mAP50-95(B)\n0,0.1\n1,0.5\n")
    pt = tmp_path / "best.pt"
    torch.save({"epoch": 7}, pt)
    assert mod.best_epoch_from_ckpt(pt) == 7
    torch.save({"other": 1}, pt)
    with caplog.at_level(logging.WARNING):
        assert mod.best_epoch_from_ckpt(pt, csv) == 1
    assert "falling back" in caplog.text


def test_write_summary(tmp_path):
    mod = _load_module()
    mod.write_summary(tmp_path, best_epoch=3, epochs_run=10, train_seconds=12.5, n_params=42)
    assert json.loads((tmp_path / "summary.json").read_text()) == {
        "family": "yolo", "best_epoch": 3, "select_from_epoch": 0, "epochs_run": 10,
        "config_stem": "yolo26m", "train_seconds": 12.5, "n_params": 42}


def test_yolo_train_imports_without_ultralytics():
    mod = _load_module()
    with pytest.raises(SystemExit) as e:
        mod.main(["--help"])
    assert e.value.code == 0


def test_best_epoch_from_ckpt_stripped_minus_one(tmp_path, caplog):
    mod = _load_module()
    csv = tmp_path / "results.csv"
    csv.write_text("epoch,metrics/mAP50-95(B)\n0,0.1\n1,0.5\n2,0.5\n")
    pt = tmp_path / "best.pt"
    torch.save({"epoch": -1}, pt)
    with caplog.at_level(logging.WARNING):
        assert mod.best_epoch_from_ckpt(pt, csv) == 1
    assert caplog.text == ""


def test_best_epoch_from_ckpt_minus_one_without_csv_raises(tmp_path):
    mod = _load_module()
    pt = tmp_path / "best.pt"
    torch.save({"epoch": -1}, pt)
    with pytest.raises(ValueError):
        mod.best_epoch_from_ckpt(pt)
    with pytest.raises(ValueError):
        mod.best_epoch_from_ckpt(pt, tmp_path / "nope.csv")


def test_main_wiring_with_stub_ultralytics(tmp_path, monkeypatch):
    import sys
    import types

    mod = _load_module()
    data_root = tmp_path / "data"
    (data_root / "images").mkdir(parents=True)
    samples = []
    for i in range(2):
        p = data_root / "images" / f"im{i}.png"
        Image.new("RGB", (100, 80)).save(p)
        samples.append({"image": f"images/im{i}.png", "source": "PBC", "width": 100, "height": 80,
                        "boxes": [[10, 20, 50, 60]], "labels": [1]})
    (data_root / "splits").mkdir()
    (data_root / "splits" / "pooled.json").write_text(
        json.dumps({"train": samples[:1], "val": samples[1:], "test": []}))
    old = data_root / "yolo_pooled"
    old.mkdir()
    (old / "data_pooled.yaml").write_text("keep")

    calls = {}

    class FakeYOLO:
        def __init__(self, model):
            calls["model"] = model
            self.model = torch.nn.Linear(2, 3)

        def train(self, **kw):
            calls.update(kw)
            run = Path(kw["project"]) / kw["name"]
            (run / "weights").mkdir(parents=True, exist_ok=True)
            (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n0,0.2\n1,0.4\n")
            torch.save({"epoch": -1}, run / "weights" / "best.pt")

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.chdir(tmp_path)
    assert mod.main(["--split", "splits/pooled.json", "--data-root", str(data_root), "--runs-dir", "runs",
                     "--run-name", "p3b_pooled_yolo26m_s2", "--seed", "2", "--epochs-max", "5",
                     "--patience", "3", "--batch", "4", "--imgsz", "320"]) == 0
    assert Path(calls["project"]).is_absolute() and calls["project"] == str((tmp_path / "runs").resolve())
    assert calls["name"] == "p3b_pooled_yolo26m_s2" and calls["exist_ok"] is True
    assert (calls["epochs"], calls["patience"], calls["batch"], calls["imgsz"], calls["seed"]) == (5, 3, 4, 320, 2)
    assert calls["deterministic"] is True
    assert Path(calls["data"]) == (data_root / "yolo_p3b_pooled" / "data.yaml").resolve()
    assert (old / "data_pooled.yaml").read_text() == "keep"
    s = json.loads((tmp_path / "runs" / "p3b_pooled_yolo26m_s2" / "summary.json").read_text())
    assert s["best_epoch"] == 1 and s["n_params"] == 9 and s["epochs_run"] == 2
