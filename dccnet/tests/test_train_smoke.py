"""Smoke tests for the training loop, scripts/train.py and scripts/predict.py (CPU, synthetic data)."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from dccnet.config import load_config
from dccnet.datasets.splits import read_split, write_split
from dccnet.datasets.synthetic import make_synthetic
from dccnet.eval.coco_eval import coco_map
from dccnet.eval.export_json import load_records
from dccnet.train.loop import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.yaml"
RUN_FILES = {"config.yaml", "git_hash.txt", "metrics.jsonl", "best.pt", "summary.json"}
LOSS_KEYS = ("loss_total", "loss_focal", "loss_r", "loss_ciou", "loss_o", "loss_d", "loss_cc", "loss_cc_bg")
METRIC_KEYS = {"epoch", *LOSS_KEYS, "lambda_cc", "val_mAP50", "val_mAP50_95", "val_rbc_count_mae",
               "lr", "nonfinite_steps", "epoch_time_s",
               *(f"val_{w}_count_{c}" for w in ("den", "gt") for c in ("RBC", "WBC", "Platelets"))}


def _smoke_config(tmp_path: Path, cc_enabled: bool) -> Path:
    cfg = {
        "_base_": str(BASE),
        "model": {"pretrained": False},
        "loss": {"cc": {"enabled": cc_enabled}},
        "train": {"epochs_max": 2, "batch": 2, "patience": 30},
        "data": {"img_size": 256, "split": "splits/syn.json"},
    }
    p = tmp_path / "smoke_cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Train 2 epochs on 6 synthetic 256px images (4 train / 2 val) through scripts/train.py."""
    tmp = tmp_path_factory.mktemp("smoke")
    data_root = tmp / "data"
    samples = make_synthetic(data_root / "synthetic" / "images", 6, seed=0, img_size=256)
    write_split({"train": samples[:4], "val": samples[4:], "test": samples[4:]},
                data_root / "splits" / "syn.json", data_root)
    cfg_path = _smoke_config(tmp, cc_enabled=False)
    runs = tmp / "runs"
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path), "--seed", "3",
         "--device", "cpu", "--data-root", str(data_root), "--runs-dir", str(runs), "--num-workers", "0"],
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    return {"tmp": tmp, "data_root": data_root, "run_dir": runs / "smoke_cfg_s3", "elapsed": elapsed}


def test_train_smoke_run_dir(smoke_run):
    run_dir = smoke_run["run_dir"]
    assert smoke_run["elapsed"] < 60, f"smoke training took {smoke_run['elapsed']:.1f}s"
    assert {p.name for p in run_dir.iterdir()} == RUN_FILES

    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [r["epoch"] for r in rows] == [0, 1]
    for row in rows:
        assert set(row) == METRIC_KEYS
        for k in LOSS_KEYS:
            assert math.isfinite(row[k]), (k, row[k])
        assert row["loss_cc"] == 0.0 and row["lambda_cc"] == 0.0 and row["loss_cc_bg"] == 0.0
        assert row["nonfinite_steps"] == 0
        assert row["val_mAP50_95"] is not None and 0.0 <= row["val_mAP50_95"] <= 1.0
        assert row["val_rbc_count_mae"] >= 0.0
        for c in ("RBC", "WBC", "Platelets"):
            assert row[f"val_den_count_{c}"] > 0.0  # softplus density never collapses to exactly 0
            assert row[f"val_gt_count_{c}"] >= 0.0
        assert row["val_gt_count_RBC"] > 0.0

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["train"]["seed"] == 3 and cfg["train"]["epochs_max"] == 2
    assert cfg["loss"]["cc"]["enabled"] is False and "_base_" not in cfg
    assert (run_dir / "git_hash.txt").read_text().strip()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["exp"] == "smoke_cfg_s3" and summary["config_stem"] == "smoke_cfg" and summary["seed"] == 3 and summary["test"] is None
    assert summary["best_epoch"] in (0, 1)
    assert set(summary["val"]) == {"mAP50", "mAP50_95", "rbc_count_mae"}
    best_row = rows[summary["best_epoch"]]
    assert summary["val"]["mAP50_95"] == best_row["val_mAP50_95"]
    # best = max mAP50_95, ties -> earlier epoch
    assert best_row["val_mAP50_95"] == max(r["val_mAP50_95"] for r in rows)
    if rows[0]["val_mAP50_95"] == rows[1]["val_mAP50_95"]:
        assert summary["best_epoch"] == 0

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) == {"model", "config", "epoch", "val"}
    assert ckpt["epoch"] == summary["best_epoch"]
    assert ckpt["config"]["train"]["seed"] == 3


def test_predict_cli_records_accepted_by_coco_map(smoke_run):
    data_root, run_dir = smoke_run["data_root"], smoke_run["run_dir"]
    split = data_root / "splits" / "syn.json"
    out = smoke_run["tmp"] / "preds_test.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "predict.py"), "--run", str(run_dir), "--split", str(split),
         "--subset", "test", "--data-root", str(data_root), "--out", str(out), "--device", "cpu", "--batch", "2"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    recs = load_records(out)
    samples = read_split(split, data_root)["test"]
    keys = {s.image_path.relative_to(data_root).as_posix() for s in samples}
    for rec in recs:
        assert rec["image"] in keys
        assert rec["category_id"] in (0, 1, 2)
        assert len(rec["bbox"]) == 4 and len(rec["circle"]) == 3
        assert rec["score"] >= 0.001
        x, y, w, h = rec["bbox"]
        cx, cy, rr = rec["circle"]
        assert w == pytest.approx(2 * rr, abs=1e-3) and x == pytest.approx(cx - rr, abs=1e-3)
    res = coco_map(samples, recs, data_root)
    assert res["mAP50_95"] is not None and 0.0 <= res["mAP50_95"] <= 1.0


def test_fit_with_cc_enabled_logs_finite_cc(tmp_path, caplog):
    from dccnet.train.loop import fit

    data_root = tmp_path / "data"
    samples = make_synthetic(data_root / "img", 3, seed=1, img_size=256)
    write_split({"train": samples[:2], "val": samples[2:]}, data_root / "splits" / "syn.json", data_root)
    cfg = load_config(_smoke_config(tmp_path, cc_enabled=True))
    cfg["train"]["epochs_max"] = 1
    cfg["decode"]["density_fill"] = True
    with caplog.at_level("WARNING", logger="dccnet.train.loop"):
        summary = fit(cfg, tmp_path / "run", torch.device("cpu"), data_root, num_workers=0)
    assert "density_fill" in caplog.text  # fill is a no-op at the 0.001 validation threshold
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().strip())
    assert math.isfinite(row["loss_cc"]) and row["lambda_cc"] == 0.0  # warm-up: computed, not weighted
    assert summary["best_epoch"] == 0
    assert {p.name for p in (tmp_path / "run").iterdir()} == RUN_FILES  # no best.pt.tmp left behind


def test_fit_without_density_head_omits_density_counts(tmp_path):
    from dccnet.train.loop import fit

    data_root = tmp_path / "data"
    samples = make_synthetic(data_root / "img", 2, seed=2, img_size=256)
    write_split({"train": samples[:1], "val": samples[1:]}, data_root / "splits" / "syn.json", data_root)
    cfg = load_config(_smoke_config(tmp_path, cc_enabled=False))
    cfg["model"]["heads"]["density"] = False
    cfg["train"]["epochs_max"] = 1
    fit(cfg, tmp_path / "run", torch.device("cpu"), data_root, num_workers=0)
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().strip())
    assert not any(k.startswith(("val_den_count", "val_gt_count")) for k in row)


def test_fit_refuses_existing_run(tmp_path):
    from dccnet.train.loop import fit

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError):
        fit(load_config(BASE), run, torch.device("cpu"), tmp_path, num_workers=0)


def test_early_stopping_ties_and_patience():
    es = EarlyStopping(patience=2)
    assert es.step(0.1, epoch=0) is True  # first epoch is always best
    assert es.step(0.1, epoch=1) is False  # tie -> earlier stays best
    assert not es.should_stop
    assert es.step(0.05, epoch=2) is False
    assert es.should_stop and es.best_epoch == 0
    es2 = EarlyStopping(patience=2)
    es2.step(None, epoch=0)  # no-GT split -> still a (worst) best so a checkpoint exists
    assert es2.best_epoch == 0
    assert es2.step(0.0, epoch=1) is True and es2.best_epoch == 1


def _fake_step_env(monkeypatch, nan_calls):
    """Linear model whose 'loss' is NaN on the call indices in ``nan_calls``."""
    import dccnet.train.loop as loop

    calls = {"n": 0}

    def fake_losses(outputs, targets, cfg, epoch):
        v = outputs["x"].sum()
        if calls["n"] in nan_calls:
            v = v * float("nan")
        calls["n"] += 1
        return {k: v for k in loop.LOSS_KEYS}

    monkeypatch.setattr(loop, "compute_losses", fake_losses)
    monkeypatch.setattr(torch.nn.Linear, "forward", lambda self, x: {"x": self.weight * x})
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    batches = [(torch.ones(1), {"t": torch.zeros(1)}, [{}])] * 3
    return loop, model, opt, batches


def test_nonfinite_loss_step_is_skipped(monkeypatch, caplog):
    loop, model, opt, batches = _fake_step_env(monkeypatch, nan_calls={0})
    with caplog.at_level("WARNING", logger="dccnet.train.loop"):
        out = loop.train_one_epoch(model, batches, opt, None, {"train": {"amp": True}}, torch.device("cpu"), 0)
    assert out["nonfinite_steps"] == 1
    assert math.isfinite(out["loss_total"])
    assert "non-finite" in caplog.text


def test_all_steps_nonfinite_aborts(monkeypatch):
    loop, model, opt, batches = _fake_step_env(monkeypatch, nan_calls={0, 1, 2})
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        loop.train_one_epoch(model, batches, opt, None, {"train": {}}, torch.device("cpu"), 0)
    assert torch.equal(model.weight, before)


class _FakeScaler:
    """GradScaler stand-in: halves the scale (= skipped step) on the update indices in ``skip``."""

    def __init__(self, skip):
        self.scale_v, self.skip, self.n = 1024.0, skip, 0

    def scale(self, t):
        return t

    def step(self, opt):
        if self.n not in self.skip:
            opt.step()

    def update(self):
        if self.n in self.skip:
            self.scale_v /= 2
        self.n += 1

    def get_scale(self):
        return self.scale_v


def test_gradscaler_skipped_steps_counted(monkeypatch):
    loop, model, opt, batches = _fake_step_env(monkeypatch, nan_calls=set())
    out = loop.train_one_epoch(model, batches, opt, _FakeScaler({1}), {"train": {}}, torch.device("cpu"), 0)
    assert out["nonfinite_steps"] == 1


def test_gradscaler_all_steps_skipped_aborts(monkeypatch):
    loop, model, opt, batches = _fake_step_env(monkeypatch, nan_calls=set())
    with pytest.raises(RuntimeError, match="non-finite"):
        loop.train_one_epoch(model, batches, opt, _FakeScaler({0, 1, 2}), {"train": {}}, torch.device("cpu"), 0)
