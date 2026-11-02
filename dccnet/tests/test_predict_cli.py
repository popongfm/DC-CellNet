import argparse
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("predict_cli", Path(__file__).resolve().parents[1] / "scripts" / "predict.py")
predict_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_cli)


def test_apply_decode_overrides():
    cfg = {"decode": {"nms": "none", "score_thr": 0.3}}
    out = predict_cli.apply_decode_overrides(cfg, argparse.Namespace(nms="center", nms_center_ratio=0.25))
    assert out["decode"]["nms"] == "center" and out["decode"]["nms_center_ratio"] == 0.25
    keep = predict_cli.apply_decode_overrides({"decode": {"nms": "none"}}, argparse.Namespace(nms=None, nms_center_ratio=None))
    assert keep["decode"] == {"nms": "none"}


def test_predict_help_lists_nms():
    with pytest.raises(SystemExit) as e:
        predict_cli.main(["--help"])
    assert e.value.code == 0
