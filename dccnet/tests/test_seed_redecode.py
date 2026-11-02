import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("seed_redecode", Path(__file__).resolve().parents[1] / "scripts" / "seed_redecode.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def _run(src, tag, files):
    d = src / tag
    d.mkdir(parents=True)
    for f in files:
        p = d / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f)


def test_seed_copies_training_outputs_only_for_dcc_and_whole_baselines(tmp_path):
    src, dst = tmp_path / "runs", tmp_path / "runs_c"
    _run(src, "p3b_pooled_d_both_s0", ["best.pt", "config.yaml", "summary.json", "metrics.jsonl", "preds_test.json",
                                      "sweep_val.json", "summary_test.json", "preds_val.json"])
    _run(src, "p3b_pooled_yolo26m_s0", ["summary.json", "summary_test.json", "weights/best.pt", "preds_test.json"])
    assert sr.main(["--src", str(src), "--dst", str(dst), "--dcc-arms", "d_both", "--copy-arms", "yolo26m",
                    "--seeds", "0"]) == 0
    assert sorted(p.name for p in (dst / "p3b_pooled_d_both_s0").iterdir()) == \
        ["best.pt", "config.yaml", "metrics.jsonl", "summary.json"]
    assert (dst / "p3b_pooled_yolo26m_s0" / "summary_test.json").is_file()
    assert (dst / "p3b_pooled_yolo26m_s0" / "weights" / "best.pt").is_file()


def test_seed_never_overwrites_and_reports_missing(tmp_path, capsys):
    src, dst = tmp_path / "runs", tmp_path / "runs_c"
    _run(src, "p3b_pooled_d_both_s0", ["best.pt", "summary.json"])
    (dst / "p3b_pooled_d_both_s0").mkdir(parents=True)
    (dst / "p3b_pooled_d_both_s0" / "preds_test.json").write_text("mine")
    rc = sr.main(["--src", str(src), "--dst", str(dst), "--dcc-arms", "d_both", "--copy-arms", "", "--seeds", "0,1"])
    assert rc == 1 and "p3b_pooled_d_both_s1" in capsys.readouterr().err
    assert (dst / "p3b_pooled_d_both_s0" / "preds_test.json").read_text() == "mine"
    assert not (dst / "p3b_pooled_d_both_s1").exists()
