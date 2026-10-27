"""compare_runs on fake run dirs whose MAE difference and verdict are known by construction."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dccnet.datasets.splits import write_split
from dccnet.types import Sample

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_runs  # noqa: E402

N_IMG = 20
# three RBCs in a row (centres 15 px apart, r = 10): ends overlap 1 neighbour (bin 1), middle 2 (bin 2);
# plus one isolated RBC (bin 0).
CELLS = [(30, 30), (45, 30), (60, 30), (150, 70)]
MIDDLE = 1


def _box(cx, cy, r=10):
    return [cx - r, cy - r, cx + r, cy + r]


def _samples(data_root):
    return [Sample(data_root / "synthetic" / f"im{i}.png", "synthetic", 200, 100,
                   np.array([_box(*c) for c in CELLS], dtype=np.float32), np.zeros(len(CELLS), dtype=np.int64))
            for i in range(N_IMG)]


def _preds(samples, data_root, drop_middle, score=0.9, hidden_score=0.4):
    """Perfect RBC predictions; ``drop_middle`` -> the bin-2 cell is only predicted at ``hidden_score`` (< thr)."""
    out = []
    for s in samples:
        key = s.image_path.relative_to(data_root).as_posix()
        for j, c in enumerate(CELLS):
            x1, y1, x2, y2 = _box(*c)
            sc = hidden_score if (drop_middle and j == MIDDLE) else score
            out.append({"image": key, "category_id": 0, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
    return out


def _write_run(runs, arm, seed, preds, best_epoch=30, thr=0.5, m_ap=0.5, prefix="exp_m2_", select_from=None):
    d = runs / f"{prefix}{arm}_s{seed}"
    d.mkdir(parents=True)
    summary = {"exp": d.name, "config_stem": f"{prefix}{arm}", "seed": seed, "best_epoch": best_epoch,
               "val": {}, "test": None}
    if select_from is not None:
        summary["select_from_epoch"] = select_from
    (d / "summary.json").write_text(json.dumps(summary))
    (d / "sweep_val.json").write_text(json.dumps({"sweep": [], "best_thr": thr}))
    (d / "summary_test.json").write_text(json.dumps({"coco": {"mAP50_95": m_ap}}))
    (d / "preds_test.json").write_text(json.dumps(preds))


@pytest.fixture
def world(tmp_path):
    data_root = tmp_path / "data"
    samples = _samples(data_root)
    write_split({"train": [], "val": [], "test": samples}, data_root / "splits" / "synthetic.json", data_root)
    return {"data_root": data_root, "split": data_root / "splits" / "synthetic.json", "samples": samples,
            "runs": tmp_path / "runs", "out": tmp_path / "reports" / "m2.md"}


def _run(w, extra=()):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--runs-dir", str(w["runs"]),
                           "--prefix", "exp_m2_", "--split", str(w["split"]), "--data-root", str(w["data_root"]),
                           "--out", str(w["out"]), *extra], capture_output=True, text=True)


def _fill(w, cand_drops_middle, seeds=(0, 1, 2), cand_best_epoch=30, density_drops_middle=True,
          cand_select_from=None, density_seeds=None):
    s, dr = w["samples"], w["data_root"]
    for seed in seeds:
        _write_run(w["runs"], "circlenet", seed, _preds(s, dr, drop_middle=True), m_ap=0.40 + 0.01 * seed)
        if density_seeds is None or seed in density_seeds:
            _write_run(w["runs"], "density", seed, _preds(s, dr, drop_middle=density_drops_middle), m_ap=0.45)
        drops = cand_drops_middle(seed) if callable(cand_drops_middle) else cand_drops_middle
        _write_run(w["runs"], "dccell", seed, _preds(s, dr, drop_middle=drops),
                   best_epoch=cand_best_epoch, m_ap=0.60, select_from=cand_select_from)


def test_go_case(world):
    _fill(world, cand_drops_middle=False)
    r = _run(world)
    assert r.returncode == 0, r.stderr
    assert "VERDICT: GO" in r.stdout
    md = world["out"].read_text()
    assert "VERDICT: GO" in md
    # baseline misses the bin-2 cell on every image (MAE 1.0), candidate is exact (MAE 0.0)
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    g = res["gate"]
    assert g["verdict"] == "GO"
    assert g["baseline_mae"] == pytest.approx(1.0) and g["candidate_mae"] == pytest.approx(0.0)
    assert g["diff_ci"] == pytest.approx((-1.0, -1.0))
    assert g["baseline_recall_ge2"] == pytest.approx(0.0) and g["candidate_recall_ge2"] == pytest.approx(1.0)
    assert g["seeds"] == [0, 1, 2]
    arms = res["arms"]
    assert arms["circlenet"]["mae_mean"] == pytest.approx(1.0) and arms["circlenet"]["mae_std"] == pytest.approx(0.0)
    assert arms["circlenet"]["map_mean"] == pytest.approx(0.41)
    assert arms["circlenet"]["map_std"] == pytest.approx(np.std([0.40, 0.41, 0.42], ddof=1))
    assert arms["dccell"]["recall"][2]["mean"] == pytest.approx(1.0)
    assert arms["circlenet"]["recall"][2]["mean"] == pytest.approx(0.0)
    assert arms["circlenet"]["recall"][1]["mean"] == pytest.approx(1.0)
    assert arms["circlenet"]["n_seeds"] == 3
    # density-vs-circlenet diagnostic is identical predictions -> diff 0, not a GO
    assert res["diagnostic"]["verdict"] == "NO-GO"
    assert res["diagnostic"]["diff_ci"] == pytest.approx((0.0, 0.0))
    # second gate: dccell vs density (isolates L_cc), computed like the main gate
    ga = res["gate_ablation"]
    assert ga["verdict"] == "GO" and ga["diff_ci"] == pytest.approx((-1.0, -1.0))
    assert ga["baseline_recall_ge2"] == pytest.approx(0.0) and ga["candidate_recall_ge2"] == pytest.approx(1.0)
    # per-seed paired MAE differences
    assert g["per_seed_diff"] == pytest.approx({0: -1.0, 1: -1.0, 2: -1.0})
    assert ga["per_seed_diff"] == pytest.approx({0: -1.0, 1: -1.0, 2: -1.0})
    assert g["ok_seeds"] and ga["ok_seeds"]
    assert res["verdict"] == "GO" and res["inconclusive"] == []
    assert "dccell vs density" in md and "per-seed" in md


def test_nogo_when_no_improvement(world):
    _fill(world, cand_drops_middle=True)  # candidate identical to baseline
    r = _run(world)
    assert r.returncode == 0
    assert "VERDICT: NO-GO" in r.stdout
    assert "VERDICT: NO-GO" in world["out"].read_text()


def test_nogo_when_candidate_worse(world):
    s, dr = world["samples"], world["data_root"]
    for seed in (0, 1):
        _write_run(world["runs"], "circlenet", seed, _preds(s, dr, drop_middle=False))
        _write_run(world["runs"], "dccell", seed, _preds(s, dr, drop_middle=True))
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["gate"]["verdict"] == "NO-GO"
    assert res["gate"]["diff_ci"][0] > 0


def test_gate_uses_common_seeds_and_warns_early_best_epoch(world):
    s, dr = world["samples"], world["data_root"]
    _write_run(world["runs"], "circlenet", 0, _preds(s, dr, True))
    _write_run(world["runs"], "circlenet", 1, _preds(s, dr, True))
    _write_run(world["runs"], "dccell", 1, _preds(s, dr, False), best_epoch=10)
    _write_run(world["runs"], "dccell", 2, _preds(s, dr, False))
    r = _run(world)
    assert r.returncode == 0, r.stderr
    md = world["out"].read_text()
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["gate"]["seeds"] == [1]
    assert "selected before full L_cc weight" in md
    assert "dccell_s1" in md


def test_missing_arm_is_inconclusive_not_go(world):
    s, dr = world["samples"], world["data_root"]
    _write_run(world["runs"], "circlenet", 0, _preds(s, dr, True))
    r = _run(world)
    assert r.returncode == 0
    assert "VERDICT: INCONCLUSIVE" in r.stdout
    md = world["out"].read_text()
    assert "VERDICT: INCONCLUSIVE" in md and "dccell" in md and "density" in md


def test_nogo_when_dccell_not_better_than_density(world):
    # dccell beats circlenet, but density (no L_cc) is just as good -> the gain is not L_cc's
    _fill(world, cand_drops_middle=False, density_drops_middle=False)
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["gate"]["verdict"] == "GO"
    assert res["gate_ablation"]["verdict"] == "NO-GO"
    assert res["gate_ablation"]["diff_ci"] == pytest.approx((0.0, 0.0))
    assert res["verdict"] == "NO-GO"
    r = _run(world)
    assert "VERDICT: NO-GO" in r.stdout


def test_nogo_when_one_seed_not_better(world):
    # dccell seed 2 is no better than the baselines' seed 2; pooled CI still < 0 and pooled recall still higher
    _fill(world, cand_drops_middle=lambda seed: seed == 2)
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    for key in ("gate", "gate_ablation"):
        g = res[key]
        assert g["ok_mae"] and g["ok_recall"]
        assert g["diff_ci"][1] < 0
        assert g["per_seed_diff"] == pytest.approx({0: -1.0, 1: -1.0, 2: 0.0})
        assert not g["ok_seeds"] and g["verdict"] == "NO-GO"
    assert res["verdict"] == "NO-GO"
    md = compare_runs.render(res)[0]
    assert "s2 0.0000" in md


def test_inconclusive_when_a_seed_is_missing(world):
    _fill(world, cand_drops_middle=False, density_seeds=(0, 1))
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["gate"]["verdict"] == "GO" and res["gate_ablation"]["verdict"] == "GO"
    assert res["verdict"] == "INCONCLUSIVE"
    assert any("density" in why and "2" in why for why in res["inconclusive"])


def test_seeds_arg_sets_required_seeds(world):
    _fill(world, cand_drops_middle=False, seeds=(0, 1))
    assert compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])["verdict"] \
        == "INCONCLUSIVE"
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"], seeds=(0, 1))
    assert res["verdict"] == "GO"
    r = _run(world, ["--seeds", "0,1"])
    assert "VERDICT: GO" in r.stdout


def test_inconclusive_when_candidate_selected_before_select_from_epoch(world):
    _fill(world, cand_drops_middle=False, cand_best_epoch=10)  # no select_from_epoch in summary -> fallback 15
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["verdict"] == "INCONCLUSIVE"
    assert any("best_epoch 10" in why for why in res["inconclusive"])


def test_select_from_epoch_read_from_summary(world):
    _fill(world, cand_drops_middle=False, cand_best_epoch=10, cand_select_from=4)
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["verdict"] == "GO"
    _fill_dir = world["runs"]
    for d in _fill_dir.iterdir():
        if "dccell" in d.name:
            sm = json.loads((d / "summary.json").read_text())
            sm["select_from_epoch"] = 12
            (d / "summary.json").write_text(json.dumps(sm))
    assert compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])["verdict"] \
        == "INCONCLUSIVE"


def test_threshold_comes_from_sweep_val(world):
    s, dr = world["samples"], world["data_root"]
    # at thr 0.3 the hidden (0.4) middle cell counts -> baseline exact; at 0.5 it does not
    _write_run(world["runs"], "circlenet", 0, _preds(s, dr, True), thr=0.3)
    _write_run(world["runs"], "dccell", 0, _preds(s, dr, True), thr=0.5)
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert res["gate"]["baseline_mae"] == pytest.approx(0.0)
    assert res["gate"]["candidate_mae"] == pytest.approx(1.0)


def test_incomplete_run_is_skipped_with_warning(world):
    s, dr = world["samples"], world["data_root"]
    _write_run(world["runs"], "circlenet", 0, _preds(s, dr, True))
    _write_run(world["runs"], "dccell", 0, _preds(s, dr, False))
    (world["runs"] / "exp_m2_dccell_s0" / "summary_test.json").unlink()
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert "dccell" not in res["arms"]
    assert any("exp_m2_dccell_s0" in w for w in res["warnings"])


def test_split_relative_to_data_root(world):
    _fill(world, cand_drops_middle=False, seeds=(0,))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--runs-dir", str(world["runs"]),
                        "--split", "splits/synthetic.json", "--data-root", str(world["data_root"]),
                        "--out", str(world["out"]), "--seeds", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "VERDICT: GO" in r.stdout


def test_discover_ignores_aborted_folders(world):
    s, dr = world["samples"], world["data_root"]
    _write_run(world["runs"], "dccell", 0, _preds(s, dr, False))
    (world["runs"] / "exp_m2_dccell_s0_aborted_123").mkdir()
    (world["runs"] / "exp_m2_dccell_s1_aborted_77").mkdir()
    found = compare_runs.discover_runs(world["runs"], "exp_m2_")
    assert {a: sorted(v) for a, v in found.items()} == {"dccell": [0]}


def test_corrupt_json_run_is_skipped_with_warning(world):
    s, dr = world["samples"], world["data_root"]
    _write_run(world["runs"], "circlenet", 0, _preds(s, dr, True))
    _write_run(world["runs"], "dccell", 0, _preds(s, dr, False))
    (world["runs"] / "exp_m2_dccell_s0" / "preds_test.json").write_text('[{"image": ')
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"])
    assert "dccell" not in res["arms"]
    assert any("exp_m2_dccell_s0" in w and "unreadable" in w for w in res["warnings"])


def test_report_states_bootstrap_n(world):
    _fill(world, cand_drops_middle=False, seeds=(0,))
    res = compare_runs.compare(world["runs"], "exp_m2_", world["split"], world["data_root"], n_boot=50)
    assert "n=50, seed 0" in compare_runs.render(res)[0]


def test_discover_parses_underscore_arm_names(world):
    for name in ("exp_m2_b_dccell_s0", "exp_m2_b_circlenet_s12", "exp_m2_b_dccell_s0_aborted_5"):
        (world["runs"] / name).mkdir(parents=True)
    found = compare_runs.discover_runs(world["runs"], "exp_m2_")
    assert {a: sorted(v) for a, v in found.items()} == {"b_dccell": [0], "b_circlenet": [12]}
