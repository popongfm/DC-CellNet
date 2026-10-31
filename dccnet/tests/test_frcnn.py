"""Faster R-CNN baseline: dataset/targets, record conversion, CPU train/predict smoke, image-key parity."""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from dccnet.datasets.splits import read_split, write_split
from dccnet.datasets.synthetic import make_synthetic
from dccnet.eval.export_json import image_key, load_records
from dccnet.types import Sample

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "scripts" / "baselines"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def common():
    return _load("frcnn_common", BASELINES / "frcnn_common.py")


def _blank_image(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 200, np.uint8)
    img[:, : w // 2, 0] = 50  # asymmetric so a flip is detectable
    assert cv2.imwrite(str(path), img)


def test_dataset_labels_shift_and_degenerate_boxes(tmp_path, common):
    p = tmp_path / "a.png"
    _blank_image(p, 64, 48)
    s = Sample(p, "T", 64, 48,
               np.array([[1, 2, 11, 12], [20, 5, 30, 25], [40, 10, 40.5, 30], [50, 30, 60, 40]], np.float32),
               np.array([0, 1, 1, 2], np.int64))
    empty = Sample(p, "T", 64, 48, np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
    ds = common.SplitDetDataset([s, empty], train=False)
    img, tgt = ds[0]
    assert img.shape == (3, 48, 64) and img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert tgt["labels"].dtype == torch.int64 and tgt["labels"].tolist() == [1, 2, 3]
    assert tgt["boxes"].dtype == torch.float32
    assert tgt["boxes"].tolist() == [[1, 2, 11, 12], [20, 5, 30, 25], [50, 30, 60, 40]]
    _, tgt_e = ds[1]
    assert tuple(tgt_e["boxes"].shape) == (0, 4) and tuple(tgt_e["labels"].shape) == (0,)


def test_dataset_train_hflip(tmp_path, common, monkeypatch):
    p = tmp_path / "a.png"
    _blank_image(p, 64, 48)
    s = Sample(p, "T", 64, 48, np.array([[1, 2, 11, 12]], np.float32), np.array([0], np.int64))
    ds = common.SplitDetDataset([s], train=True)
    plain, _ = common.SplitDetDataset([s], train=False)[0]
    monkeypatch.setattr(common.random, "random", lambda: 0.0)  # always flip
    img, tgt = ds[0]
    assert torch.equal(img, plain.flip(-1))
    assert tgt["boxes"].tolist() == [[53, 2, 63, 12]]
    monkeypatch.setattr(common.random, "random", lambda: 0.99)  # never flip
    img, tgt = ds[0]
    assert torch.equal(img, plain) and tgt["boxes"].tolist() == [[1, 2, 11, 12]]


def test_outputs_to_records_roundtrip(common):
    outs = [{"boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 5.0, 6.0]]),
             "labels": torch.tensor([1, 3]), "scores": torch.tensor([0.9, 0.2])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64), "scores": torch.zeros(0)}]
    recs = common.outputs_to_records(outs, ["k1", "k2"])
    assert [r["category_id"] for r in recs] == [0, 2]
    assert recs[0] == {"image": "k1", "category_id": 0, "bbox": [10.0, 20.0, 40.0, 60.0],
                       "score": pytest.approx(0.9)}
    assert recs[1]["bbox"] == [0.0, 0.0, 5.0, 6.0]
    assert type(recs[0]["category_id"]) is int and type(recs[0]["score"]) is float
    with pytest.raises(ValueError):
        common.outputs_to_records(outs, ["k1"])


def test_build_frcnn_heads(common):
    m = common.build_frcnn(pretrained=False, min_size=128, max_size=128)
    assert m.roi_heads.box_predictor.cls_score.out_features == 4
    assert m.roi_heads.detections_per_img == 300 and m.roi_heads.score_thresh == 0.001
    assert m.transform.min_size == (128,) and m.transform.max_size == 128


def test_lr_factor_warmup_and_milestones(common):
    f = common.lr_factor
    assert f(0, 0, warmup_iters=10, milestones=(32, 42)) == pytest.approx(0.001)
    assert f(5, 0, warmup_iters=10, milestones=(32, 42)) == pytest.approx(0.001 + 0.999 * 0.5)
    assert f(10, 0, warmup_iters=10, milestones=(32, 42)) == pytest.approx(1.0)
    assert f(10_000, 31, warmup_iters=10, milestones=(32, 42)) == pytest.approx(1.0)
    assert f(10_000, 32, warmup_iters=10, milestones=(32, 42)) == pytest.approx(0.1)
    assert f(10_000, 42, warmup_iters=10, milestones=(32, 42)) == pytest.approx(0.01)
    assert f(0, 0, warmup_iters=0, milestones=(32, 42)) == pytest.approx(1.0)


def _tiny_split(tmp: Path) -> tuple[Path, Path]:
    """4 synthetic 128px samples, 2 of them with zero boxes; train = val = test = all 4."""
    data_root = tmp / "data"
    samples = make_synthetic(data_root / "syn" / "images", 4, seed=0, img_size=128)
    for s in samples[2:]:
        s.boxes = np.zeros((0, 4), np.float32)
        s.labels = np.zeros(0, np.int64)
    split = data_root / "splits" / "tiny.json"
    write_split({"train": samples, "val": samples, "test": samples}, split, data_root)
    return data_root, split


@pytest.fixture(scope="module")
def frcnn_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("frcnn")
    data_root, split = _tiny_split(tmp)
    runs = tmp / "runs"
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-W", "error", str(BASELINES / "frcnn_train.py"), "--split", "splits/tiny.json",
         "--data-root", str(data_root), "--runs-dir", str(runs), "--run-name", "p3b_tiny_frcnn_s0",
         "--seed", "0", "--no-pretrained", "--epochs-max", "2", "--patience", "10", "--batch", "2",
         "--min-size", "128", "--max-size", "128", "--device", "cpu"],
        capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    return {"tmp": tmp, "data_root": data_root, "split": split, "runs": runs,
            "run_dir": runs / "p3b_tiny_frcnn_s0", "elapsed": elapsed}


def test_frcnn_train_smoke(frcnn_run):
    run_dir = frcnn_run["run_dir"]
    # generous bound: only catches a hang; wall-clock under CPU load was observed at 138 s
    assert frcnn_run["elapsed"] < 300, f"smoke took {frcnn_run['elapsed']:.1f}s"
    assert {"best.pt", "summary.json", "metrics.jsonl"} <= {p.name for p in run_dir.iterdir()}
    assert not (run_dir / "best.pt.tmp").exists()
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 2 and [r["epoch"] for r in rows] == [0, 1]
    for r in rows:
        assert np.isfinite(r["loss_total"]) and "val_mAP50_95" in r and "lr" in r
    summ = json.loads((run_dir / "summary.json").read_text())
    assert set(summ) == {"family", "best_epoch", "select_from_epoch", "epochs_run", "val", "config_stem",
                         "train_seconds", "n_params", "batch", "accum_steps"}
    assert summ["family"] == "frcnn" and summ["config_stem"] == "frcnn"
    assert summ["select_from_epoch"] == 0 and summ["best_epoch"] in (0, 1) and summ["epochs_run"] == 2
    assert summ["batch"] == 2 and summ["accum_steps"] == 1
    assert summ["train_seconds"] > 0 and summ["n_params"] > 1_000_000
    assert summ["val"]["mAP50_95"] == rows[summ["best_epoch"]]["val_mAP50_95"]
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) == {"model", "epoch", "args"} and ckpt["epoch"] == summ["best_epoch"]
    assert ckpt["args"]["min_size"] == 128 and ckpt["args"]["max_size"] == 128


def _frcnn_predict(frcnn_run, out: Path):
    r = subprocess.run(
        [sys.executable, "-W", "error", str(BASELINES / "frcnn_predict.py"), "--run", str(frcnn_run["run_dir"]),
         "--split", str(frcnn_run["split"]), "--subset", "test", "--data-root", str(frcnn_run["data_root"]),
         "--out", str(out), "--device", "cpu", "--batch", "2"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return load_records(out)


def test_frcnn_predict_records(frcnn_run):
    recs = _frcnn_predict(frcnn_run, frcnn_run["tmp"] / "preds_test.json")
    samples = read_split(frcnn_run["split"], frcnn_run["data_root"])["test"]
    keys = {image_key(s, frcnn_run["data_root"]) for s in samples}
    for rec in recs:
        assert rec["image"] in keys and rec["category_id"] in (0, 1, 2)
        assert len(rec["bbox"]) == 4 and rec["score"] >= 0.001


def test_frcnn_train_refuses_existing_run(tmp_path):
    data_root, _ = _tiny_split(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")
    train = _load("frcnn_train", BASELINES / "frcnn_train.py")
    with pytest.raises(FileExistsError):
        train.main(["--split", "splits/tiny.json", "--data-root", str(data_root), "--runs-dir",
                    str(tmp_path / "runs"), "--run-name", "r", "--seed", "0", "--no-pretrained",
                    "--epochs-max", "1", "--device", "cpu"])
    (run_dir / "summary.json").unlink()
    (run_dir / "best.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        train.main(["--split", "splits/tiny.json", "--data-root", str(data_root), "--runs-dir",
                    str(tmp_path / "runs"), "--run-name", "r", "--seed", "0", "--no-pretrained",
                    "--epochs-max", "1", "--device", "cpu"])


def test_frcnn_metrics_truncated_on_retrain(tmp_path):
    """A run killed after its first metrics row but before best.pt must not duplicate that row on retrain."""
    data_root, _ = _tiny_split(tmp_path)
    runs = tmp_path / "runs"
    (runs / "r").mkdir(parents=True)
    (runs / "r" / "metrics.jsonl").write_text('{"epoch": 0, "stale": true}\n')
    train = _load("frcnn_train", BASELINES / "frcnn_train.py")
    assert train.main(["--split", "splits/tiny.json", "--data-root", str(data_root), "--runs-dir", str(runs),
                       "--run-name", "r", "--seed", "0", "--no-pretrained", "--epochs-max", "1", "--batch", "2",
                       "--min-size", "128", "--max-size", "128", "--device", "cpu"]) == 0
    rows = [json.loads(x) for x in (runs / "r" / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 1 and "stale" not in rows[0] and rows[0]["epoch"] == 0


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls


class _Result:
    def __init__(self, n):
        self.boxes = _Boxes(np.tile([[1.0, 1.0, 9.0, 9.0]], (n, 1)), np.full(n, 0.5), np.zeros(n))


def test_image_key_parity(frcnn_run):
    """FRCNN, DC-CellNet predict.py and the YOLO adapter key the same split identically, including the
    images with zero GT boxes."""
    tmp, data_root, split = frcnn_run["tmp"], frcnn_run["data_root"], frcnn_run["split"]
    samples = read_split(split, data_root)["test"]
    expected = {image_key(s, data_root) for s in samples}
    assert sum(len(s.boxes) == 0 for s in samples) == 2

    # Untrained models score every image above 0.001, so every image appears in the records.
    fr = _frcnn_predict(frcnn_run, tmp / "parity_frcnn.json")

    cfg = {"_base_": str(ROOT / "configs" / "base.yaml"), "model": {"pretrained": False},
           "train": {"epochs_max": 1, "batch": 2}, "data": {"img_size": 128, "split": "splits/tiny.json"}}
    cfg_path = tmp / "parity_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, "-W", "error", str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
                        "--seed", "0", "--device", "cpu", "--data-root", str(data_root), "--runs-dir",
                        str(tmp / "dc_runs"), "--run-name", "dc", "--num-workers", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-W", "error", str(ROOT / "scripts" / "predict.py"), "--run", str(tmp / "dc_runs" / "dc"),
                        "--split", str(split), "--subset", "test", "--data-root", str(data_root),
                        "--out", str(tmp / "parity_dc.json"), "--device", "cpu", "--batch", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dc = load_records(tmp / "parity_dc.json")

    yolo = _load("yolo_predict", BASELINES / "yolo_predict.py")
    yl = yolo.yolo_results_to_records([_Result(2) for _ in samples], [image_key(s, data_root) for s in samples])

    key_sets = {name: {r["image"] for r in recs} for name, recs in (("frcnn", fr), ("dccnet", dc), ("yolo", yl))}
    for name, ks in key_sets.items():
        assert ks == expected, name


def test_dataset_drops_box_under_1px_high(tmp_path, common):
    p = tmp_path / "a.png"
    _blank_image(p, 64, 48)
    s = Sample(p, "T", 64, 48, np.array([[1, 2, 11, 12], [20, 5, 30, 5.6]], np.float32), np.array([2, 0], np.int64))
    _, tgt = common.SplitDetDataset([s], train=False)[0]
    assert tgt["boxes"].tolist() == [[1, 2, 11, 12]] and tgt["labels"].tolist() == [3]


def test_lr_factor_warmup_only_in_epoch_0(common):
    assert common.lr_factor(0, 1, warmup_iters=10) == pytest.approx(1.0)
    assert common.lr_factor(3, 1, warmup_iters=10) == pytest.approx(1.0)
    assert common.lr_factor(3, 0, warmup_iters=10) == pytest.approx(0.001 + 0.999 * 0.3)


@pytest.fixture
def train_mod():
    return _load("frcnn_train_fix", BASELINES / "frcnn_train.py")


def test_window_size_and_should_step(train_mod):
    ws = train_mod.window_size
    assert [ws(b, 2, 5) for b in range(5)] == [2, 2, 2, 2, 1]
    assert [ws(b, 3, 4) for b in range(4)] == [3, 3, 3, 1]
    assert [ws(b, 1, 3) for b in range(3)] == [1, 1, 1]
    assert train_mod.should_step(1) and train_mod.should_step(3) and not train_mod.should_step(0)


class _ToyDet(torch.nn.Module):
    """Stand-in detector: loss = w * x (x from the image), NaN on the call indices in ``nan_calls``."""

    def __init__(self, nan_calls=()):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.nan_calls, self.calls, self.grads = set(nan_calls), 0, []

    def forward(self, images, targets):
        v = self.w * torch.stack([im.sum() for im in images]).sum()
        if self.calls in self.nan_calls:
            v = v * float("nan")
        self.calls += 1
        return {k: v / 4 for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")}


def _toy_loader(values):
    return [([torch.full((1,), float(v))], [{"labels": torch.zeros(0, dtype=torch.int64)}]) for v in values]


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, params, model):
        super().__init__(params, lr=0.005)
        self.model, self.lrs, self.grads = model, [], []

    def step(self, closure=None):
        self.lrs.append(self.param_groups[0]["lr"])
        self.grads.append(None if self.model.w.grad is None else float(self.model.w.grad))
        return super().step(closure)


def test_nonfinite_window_is_skipped_without_step(train_mod, caplog):
    model = _ToyDet(nan_calls={1})
    opt = _RecordingSGD(model.parameters(), model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # same code path as AMP, inert on CPU
    with caplog.at_level("WARNING", logger="frcnn_train"):
        out, step = train_mod.train_one_epoch(model, _toy_loader([1, 2, 3]), opt, scaler, torch.device("cpu"),
                                              epoch=0, step=0, warmup_iters=2, accum=1, amp=False)
    assert step == 2 and out["nonfinite_steps"] == 1 and len(opt.lrs) == 2
    assert "non-finite" in caplog.text and "optimizer step skipped" in caplog.text
    assert opt.lrs == pytest.approx([0.005 * 0.001, 0.005 * (0.001 * 0.5 + 0.5)])  # counter did not advance
    assert all(g is not None for g in opt.grads)


def test_partial_last_window_scaled_by_its_size(train_mod):
    model = _ToyDet()
    opt = _RecordingSGD(model.parameters(), model)
    _, step = train_mod.train_one_epoch(model, _toy_loader([1, 3, 10]), opt, None, torch.device("cpu"),
                                        epoch=1, step=0, warmup_iters=0, accum=2, amp=False)
    # loss = w * x -> dL/dw = x; window 1 = (1 + 3) / 2, window 2 = 10 / 1 (not 10 / 2)
    assert step == 2 and opt.grads == pytest.approx([2.0, 10.0])


def test_warmup_reaches_base_lr_at_end_of_epoch_0(train_mod):
    model = _ToyDet()
    opt = _RecordingSGD(model.parameters(), model)
    loader = _toy_loader([1, 1, 1, 1])
    warmup = train_mod.warmup_iters_for(steps_epoch0=len(loader))
    assert warmup == 3
    _, step = train_mod.train_one_epoch(model, loader, opt, None, torch.device("cpu"), 0, 0, warmup, 1, False)
    _, step = train_mod.train_one_epoch(model, loader, opt, None, torch.device("cpu"), 1, step, warmup, 1, False)
    assert opt.lrs[3] == pytest.approx(0.005) and opt.lrs[4:] == pytest.approx([0.005] * 4)
    assert opt.lrs[0] == pytest.approx(0.005 * 0.001)
    assert train_mod.warmup_iters_for(steps_epoch0=5000) == 1000 and train_mod.warmup_iters_for(1) == 0
