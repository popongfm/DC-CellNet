import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dccnet.datasets.splits import write_split
from dccnet.eval.export_json import save_records
from dccnet.types import Sample

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("yolo_predict", ROOT / "scripts" / "baselines" / "yolo_predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls


class _Result:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _Boxes(xyxy, conf, cls)


def test_xyxy_to_xywh_and_identity_class_map():
    mod = _load_module()
    r = _Result(np.array([[10, 20, 50, 80], [0, 0, 5, 5]], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32))
    recs = mod.yolo_results_to_records([r], ["images/a.jpg"], {0: 0, 1: 1, 2: 2})
    assert recs[0] == {"image": "images/a.jpg", "category_id": 1, "bbox": [10.0, 20.0, 40.0, 60.0],
                       "score": pytest.approx(0.9)}
    assert recs[1]["category_id"] == 2 and recs[1]["bbox"] == [0.0, 0.0, 5.0, 5.0]
    assert type(recs[0]["category_id"]) is int and type(recs[0]["score"]) is float


def test_class_remap_order_and_empty_results():
    mod = _load_module()
    r1 = _Result(np.array([[0, 0, 10, 10]]), np.array([0.5]), np.array([0]))
    r2 = _Result(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    r3 = _Result(np.array([[1, 2, 3, 4]]), np.array([0.7]), np.array([2]))
    recs = mod.yolo_results_to_records([r1, r2, r3], ["k1", "k2", "k3"], {0: 2, 1: 1, 2: 0})
    assert [(x["image"], x["category_id"]) for x in recs] == [("k1", 2), ("k3", 0)]
    with pytest.raises(ValueError):
        mod.yolo_results_to_records([r1], ["a", "b"], {0: 0})


def test_torch_tensors_accepted():
    torch = pytest.importorskip("torch")
    mod = _load_module()
    r = _Result(torch.tensor([[10.0, 20.0, 50.0, 80.0]]), torch.tensor([0.9]), torch.tensor([1.0]))
    recs = mod.yolo_results_to_records([r], ["k"], {0: 0, 1: 1, 2: 2})
    assert recs[0]["bbox"] == [10.0, 20.0, 40.0, 60.0] and recs[0]["category_id"] == 1


def test_sweep_thr_picks_known_best(tmp_path):
    s = Sample(tmp_path / "images" / "a.jpg", "T", 200, 100,
               np.array([[10, 10, 50, 50], [100, 20, 140, 60]], dtype=np.float32),
               np.array([0, 0], dtype=np.int64))
    split = tmp_path / "split.json"
    write_split({"train": [], "val": [s], "test": []}, split, tmp_path)
    preds = [
        {"image": "images/a.jpg", "category_id": 0, "bbox": [10, 10, 40, 40], "score": 0.8},
        {"image": "images/a.jpg", "category_id": 0, "bbox": [100, 20, 40, 40], "score": 0.8},
        {"image": "images/a.jpg", "category_id": 0, "bbox": [150, 60, 20, 20], "score": 0.32},  # FP
    ]
    pp = tmp_path / "preds.json"
    save_records(preds, pp)
    out = tmp_path / "sweep.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--split", str(split),
                        "--subset", "val", "--preds", str(pp), "--data-root", str(tmp_path),
                        "--out", str(out), "--sweep-thr"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    thrs = [e["thr"] for e in d["sweep"]]
    assert thrs == pytest.approx([round(0.05 * k, 2) for k in range(1, 20)])
    by = {round(e["thr"], 2): e for e in d["sweep"]}
    assert by[0.3]["precision"] == pytest.approx(2 / 3) and by[0.3]["recall"] == 1.0
    assert by[0.35]["f1"] == pytest.approx(1.0)
    assert by[0.8]["f1"] == pytest.approx(1.0)
    assert by[0.85]["recall"] == 0.0 and by[0.85]["f1"] == 0.0
    assert d["best_thr"] == pytest.approx(0.35)
    assert "0.35" in r.stdout
