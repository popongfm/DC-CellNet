"""summarize_loso on fake P3b run dirs (tiny synthetic splits) whose numbers are known by construction."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dccnet.datasets.splits import write_split
from dccnet.types import Sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_loso  # noqa: E402

# three RBCs in a row (ends: bin 1, middle: bin 2) plus one isolated RBC (bin 0) -- as in test_compare_runs
CELLS = [(30, 30), (45, 30), (60, 30), (150, 70)]
MIDDLE = 1
POOLED_N = 6
FOLD_N = {"BCCD": 3, "BCDD": 4, "PBC": 5, "Raabin": 2}  # test images of each held-out source
ARMS = ("d_both", "circlenet", "frcnn")
SEEDS = (0, 1, 2)
FOLD_MAP = {"BCCD": 0.8, "BCDD": 0.7, "PBC": 0.6, "Raabin": 0.5}


def _box(cx, cy, r=10):
    return [cx - r, cy - r, cx + r, cy + r]


def _samples(data_root, source, n):
    return [Sample(data_root / source / f"im{i}.png", source, 200, 100,
                   np.array([_box(*c) for c in CELLS], dtype=np.float32), np.zeros(len(CELLS), dtype=np.int64))
            for i in range(n)]


def _preds(samples, data_root, drop_middle, score=0.9, hidden_score=0.4):
    out = []
    for s in samples:
        key = s.image_path.relative_to(data_root).as_posix()
        for j, c in enumerate(CELLS):
            x1, y1, x2, y2 = _box(*c)
            sc = hidden_score if (drop_middle and j == MIDDLE) else score
            out.append({"image": key, "category_id": 0, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
    return out


def _write_run(runs, split, arm, seed, preds, m_ap=0.5, thr=0.5, train_seconds=100.0, n_params=1000):
    d = runs / f"p3b_{split}_{arm}_s{seed}"
    d.mkdir(parents=True)
    summary = {"exp": d.name, "seed": seed, "best_epoch": 20, "select_from_epoch": 0}
    if train_seconds is not None:
        summary["train_seconds"] = train_seconds
    if n_params is not None:
        summary["n_params"] = n_params
    (d / "summary.json").write_text(json.dumps(summary))
    (d / "sweep_val.json").write_text(json.dumps({"sweep": [], "best_thr": thr}))
    test = {"coco": {"mAP50_95": m_ap, "mAP50": m_ap + 0.1,
                     "per_class": {"RBC": m_ap + 0.1, "WBC": None, "Platelets": None},
                     "per_class_50_95": {"RBC": m_ap, "WBC": None, "Platelets": None}},
            "counting": {"MAE": {"RBC": 0.0, "WBC": 0.0, "Platelets": 0.0}}}
    (d / "summary_test.json").write_text(json.dumps(test))
    (d / "preds_test.json").write_text(json.dumps(preds))
    return d


@pytest.fixture
def world(tmp_path):
    data_root = tmp_path / "data"
    tests = {"pooled": _samples(data_root, "mixed", POOLED_N)}
    for src, n in FOLD_N.items():
        tests[f"loso_{src}"] = _samples(data_root, src, n)
    for split, samples in tests.items():
        write_split({"train": [], "val": [], "test": samples}, data_root / "splits" / f"{split}.json", data_root)
    return {"data_root": data_root, "tests": tests, "runs": tmp_path / "runs", "out": tmp_path / "reports" / "p3b.md"}


def _fill(w, splits=None, arms=ARMS, seeds=SEEDS, cand_map=None):
    """d_both is perfect, every other arm misses the bin-2 middle cell (RBC MAE 1.0) -> d_both GO everywhere."""
    for split, samples in w["tests"].items():
        if splits is not None and split not in splits:
            continue
        for arm in arms:
            for seed in seeds:
                m = 0.4
                if arm == "d_both":
                    m = cand_map(split, seed) if cand_map else 0.6
                _write_run(w["runs"], split, arm, seed, _preds(samples, w["data_root"], drop_middle=arm != "d_both"),
                           m_ap=m, train_seconds=100.0 + seed)


def _summ(w, **kw):
    kw.setdefault("arms", ARMS)
    kw.setdefault("map_ci_n", 0)
    kw.setdefault("n_boot", 100)
    return summarize_loso.summarize(w["runs"], "p3b", w["data_root"], **kw)


def _cli(w, extra=()):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_loso.py"), "--runs-dir", str(w["runs"]),
                           "--prefix", "p3b", "--data-root", str(w["data_root"]), "--out", str(w["out"]),
                           "--arms", ",".join(ARMS), "--n-boot", "100", *extra], capture_output=True, text=True)


def test_c2_delta(world):
    def cmap(split, seed):
        return 0.9 if split == "pooled" else FOLD_MAP[split.split("_", 1)[1]] + (seed - 1) * 0.01

    _fill(world, cand_map=cmap)
    res = _summ(world)
    c2 = res["c2"]["d_both"]
    assert c2["pooled"] == pytest.approx(0.9)
    assert c2["folds"] == pytest.approx({"loso_BCCD": 0.8, "loso_BCDD": 0.7, "loso_PBC": 0.6, "loso_Raabin": 0.5})
    assert c2["loso_mean"] == pytest.approx(0.65)
    assert c2["delta"] == pytest.approx(0.25)
    assert res["c2"]["circlenet"]["delta"] == pytest.approx(0.0)
    md = summarize_loso.render(res)
    assert "0.6500" in md and "0.2500" in md


def test_loso_pooled_gate_concatenates_folds(world):
    _fill(world)
    res = _summ(world)
    g = res["comparisons"]["loso"]["circlenet"]
    assert g["verdict"] == "GO"
    assert g["n_images"] == sum(FOLD_N.values())
    assert g["seeds"] == [0, 1, 2]
    assert g["candidate_mae"] == pytest.approx(0.0) and g["baseline_mae"] == pytest.approx(1.0)
    assert res["comparisons"]["loso"]["frcnn"]["verdict"] == "GO"
    p = res["comparisons"]["pooled"]["circlenet"]
    assert p["verdict"] == "GO" and p["n_images"] == POOLED_N


def test_loso_gate_nogo_when_one_fold_worse_is_not_averaged_away(world):
    _fill(world)
    # circlenet perfect on every fold and seed too -> no improvement -> NO-GO (not GO)
    for split in ("loso_BCCD", "loso_BCDD", "loso_PBC", "loso_Raabin"):
        for seed in SEEDS:
            d = world["runs"] / f"p3b_{split}_circlenet_s{seed}"
            (d / "preds_test.json").write_text(json.dumps(_preds(world["tests"][split], world["data_root"], False)))
    res = _summ(world)
    assert res["comparisons"]["loso"]["circlenet"]["verdict"] == "NO-GO"


def test_missing_fold_inconclusive(world):
    _fill(world)
    shutil.rmtree(world["runs"] / "p3b_loso_PBC_circlenet_s1")
    (world["runs"] / "p3b_loso_PBC_circlenet_s1_aborted_42").mkdir()  # ignored by discovery
    res = _summ(world)
    g = res["comparisons"]["loso"]["circlenet"]
    assert g["verdict"] == "INCONCLUSIVE"
    assert any("p3b_loso_PBC_circlenet_s1" in why for why in g["reasons"])
    assert "diff_ci" not in g  # never a gate over fewer folds/seeds
    assert res["c2"]["circlenet"]["loso_mean"] is None and res["c2"]["circlenet"]["delta"] is None
    assert res["c2"]["d_both"]["loso_mean"] is not None
    assert any("p3b_loso_PBC_circlenet_s1" in w for w in res["warnings"])
    assert res["comparisons"]["pooled"]["circlenet"]["verdict"] == "GO"
    assert res["comparisons"]["loso"]["frcnn"]["verdict"] == "GO"
    assert res["stage1"] == "GO"
    md = summarize_loso.render(res)
    assert "p3b_loso_PBC_circlenet_s1" in md and "INCONCLUSIVE" in md


def test_incomplete_run_counts_as_missing(world):
    _fill(world)
    (world["runs"] / "p3b_pooled_circlenet_s2" / "summary_test.json").unlink()
    res = _summ(world)
    assert res["comparisons"]["pooled"]["circlenet"]["verdict"] == "INCONCLUSIVE"
    assert res["stage1"] == "INCONCLUSIVE"
    assert res["c2"]["circlenet"]["pooled"] is None
    assert any("p3b_pooled_circlenet_s2" in w and "summary_test.json" in w for w in res["warnings"])


def test_stage1_line_present(world):
    _fill(world, splits=("pooled",))
    r = _cli(world, ["--splits", "pooled", "--candidate", "frcnn"])
    assert r.returncode == 0, r.stderr
    line = "pooled stop/go: d_both vs circlenet = GO"
    assert line in r.stdout
    md = world["out"].read_text(encoding="utf-8")
    assert line in md
    data = json.loads(world["out"].with_suffix(".json").read_text())
    assert data["stage1"] == "GO" and data["candidate"] == "frcnn"
    # candidate frcnn is worse than d_both on pooled; no LOSO comparisons when no fold is requested
    assert data["comparisons"]["pooled"]["d_both"]["verdict"] == "NO-GO"
    assert data["comparisons"]["loso"] == {}
    assert data["c2"]["d_both"]["loso_mean"] is None


def test_stage1_inconclusive_without_runs(world):
    r = _cli(world)
    assert r.returncode == 0, r.stderr
    assert "pooled stop/go: d_both vs circlenet = INCONCLUSIVE" in r.stdout


def test_key_mismatch_inconclusive(world):
    _fill(world)
    d = world["runs"] / "p3b_pooled_circlenet_s0"
    preds = json.loads((d / "preds_test.json").read_text())
    preds.append({"image": "elsewhere/stray.png", "category_id": 0, "bbox": [1, 1, 5, 5], "score": 0.9})
    (d / "preds_test.json").write_text(json.dumps(preds))
    res = _summ(world)
    g = res["comparisons"]["pooled"]["circlenet"]
    assert g["verdict"] == "INCONCLUSIVE"
    assert any("elsewhere/stray.png" in why for why in g["reasons"])
    assert res["stage1"] == "INCONCLUSIVE"
    assert res["comparisons"]["pooled"]["frcnn"]["verdict"] == "GO"
    assert "elsewhere/stray.png" in summarize_loso.render(res)


def test_image_without_records_is_warning_not_inconclusive(world):
    _fill(world)
    d = world["runs"] / "p3b_pooled_circlenet_s0"
    preds = [p for p in json.loads((d / "preds_test.json").read_text()) if p["image"] != "mixed/im0.png"]
    (d / "preds_test.json").write_text(json.dumps(preds))
    res = _summ(world)
    assert res["comparisons"]["pooled"]["circlenet"]["verdict"] == "GO"
    assert any("p3b_pooled_circlenet_s0" in w and "1 test image" in w for w in res["warnings"])


def test_map_ci_zero_skips_ci(world, monkeypatch):
    _fill(world, splits=("pooled",))
    calls = []
    real = summarize_loso.coco_map_ci

    def spy(*a, **kw):
        calls.append(kw.get("n"))
        return real(*a, **kw)

    monkeypatch.setattr(summarize_loso, "coco_map_ci", spy)
    res = _summ(world, splits=("pooled",), map_ci_n=0)
    assert calls == []
    assert res["tables"]["pooled"]["d_both"]["map_ci"] is None
    assert "disabled" in summarize_loso.render(res)
    res = _summ(world, splits=("pooled",), map_ci_n=3)
    assert calls == [3] * len(ARMS)  # seed-0 run of each existing split x arm only
    ci = res["tables"]["pooled"]["d_both"]["map_ci"]
    assert ci["seed"] == 0 and len(ci["mAP50_95"]) == 2
    assert ci["mAP50_95"][0] == pytest.approx(1.0)  # perfect predictions
    assert "seed-0 run" in summarize_loso.render(res)


def test_split_table_and_efficiency(world):
    _fill(world, splits=("pooled",))
    (world["runs"] / "p3b_pooled_d_both_s0" / "latency.json").write_text(json.dumps({"fps": 25.0}))
    sm = world["runs"] / "p3b_pooled_frcnn_s0" / "summary.json"
    s = json.loads(sm.read_text())
    del s["train_seconds"], s["n_params"]
    sm.write_text(json.dumps(s))
    res = _summ(world, splits=("pooled",))
    t = res["tables"]["pooled"]["circlenet"]
    assert t["n_seeds"] == 3
    assert t["map_mean"] == pytest.approx(0.4) and t["map50_mean"] == pytest.approx(0.5)
    assert t["ap_class"]["RBC"]["mean"] == pytest.approx(0.4) and t["ap_class"]["WBC"]["mean"] is None
    assert t["mae_class"]["RBC"]["mean"] == pytest.approx(1.0)  # recomputed: middle cell missed
    assert t["recall"][2]["mean"] == pytest.approx(0.0) and t["recall"][1]["mean"] == pytest.approx(1.0)
    e = res["efficiency"]
    assert e["d_both"]["n_params"] == 1000 and e["d_both"]["train_seconds_mean"] == pytest.approx(101.0)
    assert e["d_both"]["fps"] == pytest.approx(25.0) and e["circlenet"]["fps"] is None
    assert e["frcnn"]["train_seconds_mean"] == pytest.approx(101.5)  # seeds 1, 2 only
    md = summarize_loso.render(res)
    assert "| d_both |" in md


def test_partial_loso_splits_inconclusive(world):
    _fill(world)
    res = _summ(world, splits=("pooled", "loso_BCCD"))
    assert res["comparisons"]["loso"]["circlenet"]["verdict"] == "INCONCLUSIVE"
    assert res["c2"]["d_both"]["loso_mean"] is None
