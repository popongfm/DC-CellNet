import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dccnet.datasets.splits import write_split
from dccnet.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from dccnet.eval.coco_eval import build_coco_gt, coco_map, coco_map_ci
from dccnet.eval.counting import count_errors
from dccnet.eval.export_json import image_key, load_records, save_records
from dccnet.eval.overlap_bins import overlap_metrics
from dccnet.types import Sample

ROOT = Path(__file__).resolve().parents[1]


def mk(root, name, boxes, labels, w=200, h=100):
    return Sample(Path(root) / "images" / name, "T", w, h,
                  np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                  np.asarray(labels, dtype=np.int64))


def pred(root, name, box_xyxy, cat=0, score=1.0):
    x1, y1, x2, y2 = box_xyxy
    return {"image": f"images/{name}", "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": score}


def gt_preds(root, samples, score=1.0):
    out = []
    for s in samples:
        for b, c in zip(s.boxes, s.labels):
            out.append(pred(root, s.image_path.name, b.tolist(), int(c), score))
    return out


@pytest.fixture
def samples(tmp_path):
    return [
        mk(tmp_path, "a.jpg", [[10, 10, 50, 50], [100, 20, 140, 60], [150, 50, 170, 70]], [0, 0, 1]),
        mk(tmp_path, "b.jpg", [[20, 20, 60, 60], [80, 10, 90, 20]], [0, 2]),
    ]


# ---------------------------------------------------------------- COCO mAP

def test_perfect_preds_map_one_and_zero_count_error(tmp_path, samples):
    preds = gt_preds(tmp_path, samples)
    m = coco_map(samples, preds, data_root=tmp_path)
    assert m["mAP50"] == pytest.approx(1.0)
    assert m["mAP50_95"] == pytest.approx(1.0)
    assert m["per_class"] == {"RBC": pytest.approx(1.0), "WBC": pytest.approx(1.0),
                              "Platelets": pytest.approx(1.0)}
    ce = count_errors(samples, preds, 0.5, data_root=tmp_path)
    assert all(v == 0.0 for v in ce["MAE"].values())
    assert all(v == 0.0 for v in ce["rel"].values())


def test_empty_preds(tmp_path, samples):
    m = coco_map(samples, [], data_root=tmp_path)
    assert m["mAP50"] == 0.0 and m["mAP50_95"] == 0.0
    assert m["per_class"] == {"RBC": 0.0, "WBC": 0.0, "Platelets": 0.0}
    ce = count_errors(samples, [], 0.3, data_root=tmp_path)
    assert ce["per_image"]["RBC"] == [-2, -1]          # signed: pred - gt
    assert ce["MAE"]["RBC"] == pytest.approx(1.5)
    assert ce["MAE"]["WBC"] == pytest.approx(0.5)
    assert ce["rel"]["RBC"] == pytest.approx(1.0)
    lo, hi = coco_map_ci(samples, [], data_root=tmp_path, n=5)["mAP50"]
    assert lo == hi == 0.0


def test_class_without_gt_is_null_and_excluded_from_mean(tmp_path):
    s = [mk(tmp_path, "a.jpg", [[10, 10, 50, 50]], [0])]
    m = coco_map(s, gt_preds(tmp_path, s), data_root=tmp_path)
    assert m["per_class"] == {"RBC": pytest.approx(1.0), "WBC": None, "Platelets": None}
    assert m["mAP50"] == pytest.approx(1.0)
    # empty predictions: classes with GT -> 0.0, others still null
    m0 = coco_map(s, [], data_root=tmp_path)
    assert m0["per_class"] == {"RBC": 0.0, "WBC": None, "Platelets": None}
    ce = count_errors(s, [], 0.3, data_root=tmp_path)
    assert ce["rel"]["WBC"] is None and ce["MAE"]["WBC"] == 0.0


def test_false_positive_class_without_gt_does_not_leak(tmp_path):
    s = [mk(tmp_path, "a.jpg", [[10, 10, 50, 50]], [0])]
    preds = gt_preds(tmp_path, s) + [pred(tmp_path, "a.jpg", [60, 60, 80, 80], cat=1, score=0.9)]
    m = coco_map(s, preds, data_root=tmp_path)
    assert m["per_class"]["WBC"] is None
    assert m["mAP50"] == pytest.approx(1.0)


def test_coco_map_matches_hand_computation(tmp_path):
    # 2 images, 3 GT (all RBC): a1, a2 in image A; b1 in image B.
    # preds: a1 exact (0.9, TP), duplicate of a1 (0.8, FP: GT already matched), b1 exact (0.7, TP).
    # a2 is missed.  Score order: TP, FP, TP
    #   cum TP = 1,1,2 ; cum FP = 0,1,1 ; recall = 1/3, 1/3, 2/3 ; precision = 1, 1/2, 2/3
    # COCO: monotone envelope of precision, sampled at 101 recall points 0,.01,...,1:
    #   r <= 1/3     -> 34 points (0.00..0.33) at precision 1
    #   1/3 < r<=2/3 -> 33 points (0.34..0.66) at precision 2/3
    #   r > 2/3      -> 0
    #   AP = (34 + 33*2/3) / 101 = 56/101.  All boxes are exact so every IoU threshold gives the
    #   same value: AP50 = AP50:95 = 56/101.
    A = mk(tmp_path, "A.jpg", [[10, 10, 50, 50], [100, 20, 140, 60]], [0, 0])
    B = mk(tmp_path, "B.jpg", [[20, 20, 60, 60]], [0])
    preds = [pred(tmp_path, "A.jpg", [10, 10, 50, 50], 0, 0.9),
             pred(tmp_path, "A.jpg", [10, 10, 50, 50], 0, 0.8),
             pred(tmp_path, "B.jpg", [20, 20, 60, 60], 0, 0.7)]
    m = coco_map([A, B], preds, data_root=tmp_path)
    assert m["per_class"]["RBC"] == pytest.approx(56 / 101)
    assert m["mAP50"] == pytest.approx(56 / 101)
    assert m["mAP50_95"] == pytest.approx(56 / 101)


def test_coco_map_iou_threshold_slices(tmp_path):
    # One GT 100x100; pred 100x100 shifted by 20px in x: IoU = 80*100/(2*10000-8000) = 2/3.
    # TP for thresholds 0.50,0.55,0.60,0.65 (4 of 10) -> AP50 = 1, AP50:95 = 0.4
    s = [mk(tmp_path, "a.jpg", [[0, 0, 100, 100]], [0], w=300, h=300)]
    preds = [pred(tmp_path, "a.jpg", [20, 0, 120, 100], 0, 0.9)]
    m = coco_map(s, preds, data_root=tmp_path)
    assert m["mAP50"] == pytest.approx(1.0)
    assert m["mAP50_95"] == pytest.approx(0.4)


def test_more_than_100_dets_per_image(tmp_path):
    boxes = [[(i % 20) * 10, (i // 20) * 10, (i % 20) * 10 + 8, (i // 20) * 10 + 8] for i in range(150)]
    s = [mk(tmp_path, "a.jpg", boxes, [0] * 150, w=400, h=400)]
    m = coco_map(s, gt_preds(tmp_path, s), data_root=tmp_path)
    assert m["mAP50"] == pytest.approx(1.0)


def test_unknown_image_raises(tmp_path, samples):
    bad = [pred(tmp_path, f"zzz{i}.jpg", [0, 0, 5, 5]) for i in range(7)]
    with pytest.raises(ValueError) as e:
        coco_map(samples, bad, data_root=tmp_path)
    msg = str(e.value)
    assert "zzz0.jpg" in msg and "zzz4.jpg" in msg and "zzz5.jpg" not in msg
    with pytest.raises(ValueError):
        count_errors(samples, bad, 0.5, data_root=tmp_path)
    with pytest.raises(ValueError):
        overlap_metrics(samples, bad, 0.5, data_root=tmp_path)


def test_build_coco_gt_and_image_key(tmp_path, samples):
    assert image_key(samples[0], tmp_path) == "images/a.jpg"
    d = build_coco_gt(samples, data_root=tmp_path)
    assert len(d["images"]) == 2 and len(d["annotations"]) == 5
    a = d["annotations"][0]
    assert a["bbox"] == [10, 10, 40, 40] and a["area"] == 1600 and a["iscrowd"] == 0
    assert [c["id"] for c in d["categories"]] == [0, 1, 2]


def test_coco_map_ci(tmp_path, samples):
    preds = gt_preds(tmp_path, samples)
    ci = coco_map_ci(samples, preds, data_root=tmp_path, n=10, seed=1)
    assert ci["mAP50"] == pytest.approx((1.0, 1.0)) and ci["mAP50_95"] == pytest.approx((1.0, 1.0))
    # imperfect preds: interval is ordered and deterministic given the seed
    imperfect = preds[:3]
    c1 = coco_map_ci(samples, imperfect, data_root=tmp_path, n=20, seed=3)
    c2 = coco_map_ci(samples, imperfect, data_root=tmp_path, n=20, seed=3)
    assert c1 == c2 and c1["mAP50"][0] <= c1["mAP50"][1]


# ---------------------------------------------------------------- counting

def test_count_errors_threshold_and_sign(tmp_path, samples):
    preds = gt_preds(tmp_path, samples)
    preds += [pred(tmp_path, "a.jpg", [0, 80, 10, 90], 0, 0.2),   # below threshold
              pred(tmp_path, "b.jpg", [0, 80, 10, 90], 0, 0.6)]   # extra RBC in b
    ce = count_errors(samples, preds, 0.5, data_root=tmp_path)
    assert ce["per_image"]["RBC"] == [0, 1]
    assert ce["MAE"]["RBC"] == pytest.approx(0.5)
    assert ce["rel"]["RBC"] == pytest.approx(1 / 3)       # sum|err| / sum gt = 1/3
    ce2 = count_errors(samples, preds, 0.1, data_root=tmp_path)
    assert ce2["per_image"]["RBC"] == [1, 1]


# ---------------------------------------------------------------- overlap bins

def chain(root):
    # 4 equal circles (r=10) in a chain, centers x=50,65,80,95: adjacent overlap, others not.
    # overlap levels: A=1, B=2, C=2, D=1
    boxes = [[40, 40, 60, 60], [55, 40, 75, 60], [70, 40, 90, 60], [85, 40, 105, 60]]
    return mk(root, "c.jpg", boxes, [0, 0, 0, 0])


def test_overlap_bins_missed_cell_in_bin2(tmp_path):
    s = [chain(tmp_path)]
    A, B, C, D = s[0].boxes.tolist()
    preds = [pred(tmp_path, "c.jpg", b) for b in (A, B, D)]
    o = overlap_metrics(s, preds, 0.5, data_root=tmp_path)
    assert set(o) == {"0", "1", "2", "3"}
    assert o["2"]["n_gt"] == 2 and o["2"]["recall"] == pytest.approx(0.5)
    assert o["2"]["precision"] == pytest.approx(1.0)
    assert o["2"]["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert o["1"]["n_gt"] == 2 and o["1"]["recall"] == pytest.approx(1.0)
    for b in ("0", "3"):
        assert o[b] == {"recall": None, "precision": None, "f1": None, "n_gt": 0}


def test_overlap_fp_assignment(tmp_path):
    s = [chain(tmp_path), mk(tmp_path, "empty.jpg", [], [])]
    preds = gt_preds(tmp_path, [s[0]])
    # class-1 FP with no class-1 GT in image -> nearest any-class GT center (B, x=65, bin 2)
    preds.append(pred(tmp_path, "c.jpg", [64, 80, 68, 84], cat=1, score=0.9))
    # class-0 FP nearest to D (x=95, bin 1)
    preds.append(pred(tmp_path, "c.jpg", [93, 10, 97, 14], cat=0, score=0.9))
    # FP on an image with no GT -> bin 0
    preds.append(pred(tmp_path, "empty.jpg", [1, 1, 5, 5], cat=0, score=0.9))
    o = overlap_metrics(s, preds, 0.5, data_root=tmp_path)
    assert o["2"]["precision"] == pytest.approx(2 / 3)
    assert o["1"]["precision"] == pytest.approx(2 / 3)
    assert o["0"]["precision"] == 0.0 and o["0"]["recall"] is None and o["0"]["f1"] is None
    assert o["2"]["recall"] == 1.0


def test_overlap_greedy_one_to_one_and_score_order(tmp_path):
    s = [mk(tmp_path, "a.jpg", [[10, 10, 50, 50]], [0])]
    # duplicate pred -> one TP + one FP (both land in bin 0)
    preds = [pred(tmp_path, "a.jpg", [10, 10, 50, 50], 0, 0.9),
             pred(tmp_path, "a.jpg", [11, 10, 51, 50], 0, 0.8)]
    o = overlap_metrics(s, preds, 0.5, data_root=tmp_path)
    assert o["0"]["recall"] == 1.0 and o["0"]["precision"] == pytest.approx(0.5)
    # wrong class never matches
    o2 = overlap_metrics(s, [pred(tmp_path, "a.jpg", [10, 10, 50, 50], 1, 0.9)], 0.5, data_root=tmp_path)
    assert o2["0"]["recall"] == 0.0 and o2["0"]["precision"] == 0.0 and o2["0"]["f1"] == 0.0


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_constant():
    assert bootstrap_ci([3.0] * 10) == (3.0, 3.0)
    assert paired_bootstrap_diff([2.0] * 6, [1.0] * 6) == (1.0, 1.0)


def test_bootstrap_properties():
    rng = np.random.default_rng(0)
    v = rng.normal(5, 1, 200)
    lo, hi = bootstrap_ci(v, seed=1)
    assert lo < v.mean() < hi and hi - lo < 0.6
    assert bootstrap_ci(v, seed=1) == (lo, hi)
    assert bootstrap_ci(v, seed=2) != (lo, hi)
    a = v
    b = v + 0.5 + rng.normal(0, 0.1, 200)
    lo, hi = paired_bootstrap_diff(a, b, seed=0)
    assert hi < -0.3                                  # paired resampling keeps the strong pairing
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1])
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ratio_statistic():
    vals = np.array([[1, 2], [2, 2], [0, 2]])        # (tp, n_gt) per image
    lo, hi = bootstrap_ci(vals, stat=lambda v: v[:, 0].sum() / v[:, 1].sum(), n=200)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


# ---------------------------------------------------------------- export + CLI

def test_records_roundtrip(tmp_path):
    recs = [{"image": "images/a.jpg", "category_id": np.int64(0), "bbox": np.array([1.0, 2.0, 3.0, 4.0]),
             "score": np.float32(0.5), "circle": [2.0, 3.0, 2.0]}]
    p = tmp_path / "sub" / "p.json"
    save_records(recs, p)
    back = load_records(p)
    assert back[0]["bbox"] == [1.0, 2.0, 3.0, 4.0] and back[0]["category_id"] == 0
    p.write_text(json.dumps([{"image": "x"}]))
    with pytest.raises(ValueError):
        load_records(p)


def test_cli_end_to_end(tmp_path, samples):
    split = tmp_path / "split.json"
    write_split({"train": [], "val": [], "test": samples}, split, tmp_path)
    preds = gt_preds(tmp_path, samples)
    pp = tmp_path / "preds.json"
    save_records(preds, pp)
    out = tmp_path / "summary.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--split", str(split),
                        "--subset", "test", "--preds", str(pp), "--score-thr", "0.3",
                        "--data-root", str(tmp_path), "--out", str(out), "--map-ci-n", "5"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    s = json.loads(out.read_text())
    assert {"split", "subset", "n_images", "score_thr", "coco", "coco_ci", "counting", "overlap",
            "recall_ci_by_bin"} <= set(s)
    assert s["n_images"] == 2 and s["score_thr"] == 0.3 and s["subset"] == "test"
    assert s["coco"]["mAP50"] == pytest.approx(1.0)
    assert s["counting"]["MAE"]["RBC"] == 0.0 and s["counting"]["MAE_ci"]["RBC"] == [0.0, 0.0]
    assert s["coco_ci"]["mAP50"] == [pytest.approx(1.0)] * 2
    assert set(s["overlap"]) == {"0", "1", "2", "3"}
    assert s["recall_ci_by_bin"]["0"] == [1.0, 1.0]
    assert s["recall_ci_by_bin"]["3"] is None
    # --map-ci-n 0 disables the mAP CI
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--split", str(split),
                        "--preds", str(pp), "--score-thr", "0.3", "--data-root", str(tmp_path),
                        "--out", str(out), "--map-ci-n", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["coco_ci"] is None
