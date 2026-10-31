"""scripts/train.py CLI overrides (``apply_overrides``): --seed / --epochs-max / --split / --batch / --accum-steps."""
import argparse
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train as train_script  # noqa: E402

from dccnet.config import load_config  # noqa: E402

BASE = ROOT / "configs" / "base.yaml"


def _ns(**kw):
    d = {"seed": None, "epochs_max": None, "split": None, "batch": None, "accum_steps": None}
    d.update(kw)
    return argparse.Namespace(**d)


def test_apply_overrides_split():
    cfg = load_config(BASE)
    orig = copy.deepcopy(cfg)
    out = train_script.apply_overrides(cfg, _ns(split="splits/loso_PBC.json"))
    assert out["data"]["split"] == "splits/loso_PBC.json"
    assert cfg == orig  # input not mutated
    out = train_script.apply_overrides(cfg, _ns())
    assert out == orig  # nothing given -> unchanged
    out = train_script.apply_overrides(cfg, _ns(seed=2, epochs_max=5))
    assert out["train"]["seed"] == 2 and out["train"]["epochs_max"] == 5
    assert out["data"]["split"] == orig["data"]["split"]


def test_apply_overrides_batch_accum():
    cfg = load_config(BASE)
    out = train_script.apply_overrides(cfg, _ns(batch=4, accum_steps=2))
    assert out["train"]["batch"] == 4 and out["train"]["accum_steps"] == 2
    out = train_script.apply_overrides(cfg, _ns())
    assert out["train"]["batch"] == 8 and out["train"]["accum_steps"] == 1
    with pytest.raises(ValueError, match="accum"):
        train_script.apply_overrides(cfg, _ns(accum_steps=0))
    with pytest.raises(ValueError, match="batch"):
        train_script.apply_overrides(cfg, _ns(batch=0))


def test_apply_overrides_tolerates_namespace_without_new_flags():
    cfg = load_config(BASE)
    out = train_script.apply_overrides(cfg, argparse.Namespace(seed=1, epochs_max=None))
    assert out["train"]["seed"] == 1 and out["data"]["split"] == cfg["data"]["split"]


def test_cli_parses_new_flags(monkeypatch, tmp_path):
    seen = {}

    def fake_fit(cfg, run_dir, device, data_root, num_workers=0, config_stem=None):
        seen.update(cfg=cfg, run_dir=run_dir, config_stem=config_stem)
        return {"best_epoch": 0, "val": {}}

    monkeypatch.setattr(train_script, "fit", fake_fit)
    rc = train_script.main(["--config", str(BASE), "--seed", "1", "--split", "splits/loso_BCCD.json",
                            "--batch", "4", "--accum-steps", "2", "--device", "cpu",
                            "--runs-dir", str(tmp_path)])
    assert rc == 0
    assert seen["cfg"]["data"]["split"] == "splits/loso_BCCD.json"
    assert seen["cfg"]["train"]["batch"] == 4 and seen["cfg"]["train"]["accum_steps"] == 2
    assert seen["run_dir"] == tmp_path / "base_s1"  # default run name unchanged
    assert seen["config_stem"] == "base"
