import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dccnet.datasets.relabel import from_yolo_lines, label_hash, merge_labels, to_yolo_lines

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prelabel_bccd.py"


def _a(x, dt=np.float32):
    return np.asarray(x, dtype=dt)


def test_merge_dedupes_same_class_only():
    # IoU of [0,0,10,10] and [0,0,10,12.5] = 0.8
    gb = _a([[0, 0, 10, 10], [0, 0, 10, 12.5]])
    b, l, o, st = merge_labels(gb, _a([0, 0], np.int64), np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0))
    assert len(b) == 1 and st["n_dupes_removed"] == 1 and st["n_gt"] == 2 and st["n_final"] == 1
    b, l, o, st = merge_labels(gb, _a([0, 1], np.int64), np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0))
    assert len(b) == 2 and st["n_dupes_removed"] == 0 and l.tolist() == [0, 1]


def test_merge_adds_only_unmatched_confident_preds():
    gb = _a([[0, 0, 10, 10]])
    gl = _a([0], np.int64)
    # IoU with GT: 0.2 -> [8,0,18,10]? inter=20, union=180 -> 0.11; build explicit values instead
    p_ok = [7.5, 0, 17.5, 10]      # inter 25, union 175 -> 0.143
    p_near = [4, 0, 14, 10]        # inter 60, union 140 -> 0.43
    p_far = [100, 100, 110, 110]
    pb = _a([p_ok, p_near, p_far, [200, 200, 210, 210], [202.5, 200, 212.5, 210]])
    pl = _a([1, 1, 2, 0, 0], np.int64)
    ps = _a([0.9, 0.9, 0.4, 0.95, 0.8])  # last two overlap each other (IoU 0.6)
    b, l, o, st = merge_labels(gb, gl, pb, pl, ps)
    assert o.tolist() == [0, 1, 1]
    assert l.tolist() == [0, 0, 1] or l.tolist() == [0, 1, 0]
    assert st == {"n_gt": 1, "n_dupes_removed": 0, "n_added": 2, "n_final": 3}
    assert o.dtype == np.int8 and b.dtype == np.float32 and l.dtype == np.int64
    # descending score order: 0.95 added before 0.9
    assert np.allclose(b[1], [200, 200, 210, 210]) and np.allclose(b[2], p_ok)


def test_merge_empty_inputs():
    b, l, o, st = merge_labels(np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0, np.int64),
                               np.zeros(0))
    assert b.shape == (0, 4) and len(l) == 0 and len(o) == 0
    assert st == {"n_gt": 0, "n_dupes_removed": 0, "n_added": 0, "n_final": 0}


def test_yolo_roundtrip_non_square():
    boxes = _a([[10.5, 20.25, 100.75, 90.5], [0, 0, 640, 480], [300.1, 200.2, 330.3, 260.4]])
    labels = _a([2, 0, 1], np.int64)
    lines = to_yolo_lines(boxes, labels, 640, 480)
    assert [ln.split()[0] for ln in lines] == ["2", "0", "1"]
    assert lines[1] == "0 0.500000 0.500000 1.000000 1.000000"
    b2, l2 = from_yolo_lines(lines, 640, 480)
    assert np.abs(b2 - boxes).max() < 0.01 and l2.tolist() == [2, 0, 1]
    b3, l3 = from_yolo_lines([], 640, 480)
    assert b3.shape == (0, 4) and len(l3) == 0


@pytest.mark.parametrize("line", ["3 0.5 0.5 0.1 0.1", "-1 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1", "a 0.5 0.5 0.1 0.1",
                                  "0 0.5 x 0.1 0.1", "0.5 0.5 0.5 0.1 0.1"])
def test_from_yolo_lines_rejects_bad_class_and_malformed(line):
    with pytest.raises(ValueError, match="0"):
        from_yolo_lines(["0 0.5 0.5 0.1 0.1", line], 640, 480)


def _setup(tmp_path):
    root = tmp_path / "data"
    imgdir = root / "bccd_full" / "BCCD" / "JPEGImages"
    imgdir.mkdir(parents=True)
    recs, items = [], []
    for i, boxes in enumerate([[[10, 10, 50, 50]], [[10, 10, 50, 50], [100, 100, 140, 140]]]):
        name = f"BloodImage_{i:05d}.jpg"
        Image.new("RGB", (640, 480)).save(imgdir / name)
        key = f"bccd_full/BCCD/JPEGImages/{name}"
        items.append({"image": key, "source": "BCCD", "width": 640, "height": 480, "boxes": boxes,
                      "labels": [0] * len(boxes)})
    (root / "splits").mkdir()
    (root / "splits" / "overlap_bccd.json").write_text(json.dumps({"test": items}))
    # image 0: one new confident pred; image 1: two new preds (one low-score)
    def rec(k, box, s, c=0):
        return {"image": items[k]["image"], "category_id": c, "bbox": [box[0], box[1], box[2] - box[0],
                box[3] - box[1]], "score": s}
    preds = [rec(0, [200, 200, 240, 240], 0.9), rec(1, [300, 300, 340, 340], 0.8), rec(1, [400, 100, 440, 140], 0.7),
             rec(1, [500, 300, 540, 340], 0.2)]
    (tmp_path / "preds.json").write_text(json.dumps(preds))
    return root


def _run(root, tmp_path, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--data-root", str(root), "--preds",
                           str(tmp_path / "preds.json"), *extra], capture_output=True, text=True)


def test_prelabel_cli_writes_export(tmp_path):
    root = _setup(tmp_path)
    r = _run(root, tmp_path)
    assert r.returncode == 0, r.stderr
    out = root / "bccd_relabel"
    assert (out / "images" / "BloodImage_00000.jpg").is_file()
    assert (out / "classes.txt").read_text() == "RBC\nWBC\nPlatelets\n"
    txt = (out / "labels" / "BloodImage_00001.txt").read_text()
    assert len(txt.strip().splitlines()) == 4
    rows = (out / "review_list.csv").read_text().strip().splitlines()
    assert rows[0] == "image,n_gt,n_dupes_removed,n_added,n_final,done"
    assert rows[1].startswith("BloodImage_00001") and rows[1].split(",")[3] == "2"
    assert rows[2].startswith("BloodImage_00000") and rows[1].endswith(",")
    man = json.loads((out / "prelabel_manifest.json").read_text())
    m = man["BloodImage_00001"]
    assert m["hash"] == hashlib.sha1(txt.encode()).hexdigest() == label_hash(txt)
    assert m["image"] == "bccd_full/BCCD/JPEGImages/BloodImage_00001.jpg" and m["origin"] == [0, 0, 1, 1]
    # second run without --force refuses and leaves labels untouched
    (out / "labels" / "BloodImage_00000.txt").write_text("edited\n")
    r2 = _run(root, tmp_path)
    assert r2.returncode != 0 and "--force" in (r2.stderr + r2.stdout)
    assert (out / "labels" / "BloodImage_00000.txt").read_text() == "edited\n"
    assert _run(root, tmp_path, "--force").returncode == 0
    assert (out / "labels" / "BloodImage_00000.txt").read_text() != "edited\n"


# ---- C2: compare_label_sets, strict parsing, import / progress ----
from dccnet.datasets.relabel import compare_label_sets  # noqa: E402

IMPORT = Path(__file__).resolve().parents[1] / "scripts" / "import_bccd_relabel.py"


def test_compare_label_sets_counts():
    old = _a([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])
    new = _a([[0, 0, 10, 11], [20, 20, 30, 30], [100, 100, 110, 110]])
    r = compare_label_sets(old, _a([0, 0, 0], np.int64), new, _a([0, 1, 0], np.int64))
    assert r == {"kept": 1, "added": 2, "removed": 2}  # box 2 changed class
    assert compare_label_sets(np.zeros((0, 4)), [], np.zeros((0, 4)), []) == {"kept": 0, "added": 0, "removed": 0}


@pytest.mark.parametrize("line", ["0 nan 0.5 0.1 0.1", "0 0.5 inf 0.1 0.1", "0 0.5 0.5 0 0.1", "0 0.5 0.5 0.1 -0.1",
                                  "0 1.2 0.5 0.1 0.1", "0 0.5 0.5 1.5 0.1", "0 -0.1 0.5 0.1 0.1"])
def test_from_yolo_lines_rejects_non_finite_degenerate_out_of_range(line):
    with pytest.raises(ValueError, match="bad YOLO line"):
        from_yolo_lines([line], 640, 480)


def test_from_yolo_lines_allows_blank_trailing_ws_and_tolerance():
    b, l = from_yolo_lines(["", "  ", "0 0.5 0.5 0.1 0.1   ", "1 0.5 0.5 1.0000005 0.1"], 640, 480)
    assert l.tolist() == [0, 1]


def _prelabel(root, tmp_path):
    r = _run(root, tmp_path)
    assert r.returncode == 0, r.stderr
    return root / "bccd_relabel"


def _import(root, *extra):
    return subprocess.run([sys.executable, str(IMPORT), "--data-root", str(root), *extra],
                          capture_output=True, text=True)


def test_import_roundtrip_keys_and_order(tmp_path):
    root = _setup(tmp_path)
    out = _prelabel(root, tmp_path)
    r = _import(root)
    assert r.returncode == 0, r.stderr
    v1 = json.loads((root / "splits" / "overlap_bccd.json").read_text())["test"]
    v2 = json.loads((root / "splits" / "bccd_overlap_v2.json").read_text())["test"]
    assert [x["image"] for x in v2] == [x["image"] for x in v1]
    for a, b in zip(v1, v2):
        assert list(a) == list(b)
        assert all(a[k] == b[k] for k in ("image", "source", "width", "height"))
    assert [len(x["boxes"]) for x in v2] == [2, 4]
    assert np.abs(np.array(v2[0]["boxes"][0]) - np.array([10, 10, 50, 50])).max() < 0.01
    assert [x["labels"] for x in v2] == [[0, 0], [0, 0, 0, 0]]
    st = json.loads((out / "import_stats.json").read_text())
    assert st["totals"]["n_images"] == 2 and st["totals"]["n_boxes"] == 6
    pi = st["per_image"]["BloodImage_00001"]
    assert pi["vs_original"] == {"kept": 2, "added": 2, "removed": 0}
    assert pi["yolo_origin_kept"] == 2
    assert st["per_image"]["BloodImage_00000"]["yolo_origin_kept"] == 1


def test_import_old_manifest_without_added_boxes_gives_null(tmp_path):
    root = _setup(tmp_path)
    out = _prelabel(root, tmp_path)
    man = json.loads((out / "prelabel_manifest.json").read_text())
    for m in man.values():
        del m["added_boxes"]
    (out / "prelabel_manifest.json").write_text(json.dumps(man))
    r = _import(root)
    assert r.returncode == 0 and "warning" in r.stderr.lower()
    st = json.loads((out / "import_stats.json").read_text())
    assert all(v["yolo_origin_kept"] is None for v in st["per_image"].values())


def test_import_reports_all_errors_and_writes_nothing(tmp_path):
    root = _setup(tmp_path)
    out = _prelabel(root, tmp_path)
    (out / "labels" / "BloodImage_00000.txt").unlink()
    (out / "labels" / "BloodImage_00001.txt").write_text("7 0.5 0.5 0.1 0.1\n")
    r = _import(root)
    assert r.returncode != 0
    assert "BloodImage_00000" in r.stderr and "BloodImage_00001" in r.stderr
    assert not (root / "splits" / "bccd_overlap_v2.json").exists()
    assert not (out / "import_stats.json").exists()


def test_import_empty_label_file_is_zero_boxes(tmp_path):
    root = _setup(tmp_path)
    out = _prelabel(root, tmp_path)
    (out / "labels" / "BloodImage_00000.txt").write_text("")
    assert _import(root).returncode == 0
    v2 = json.loads((root / "splits" / "bccd_overlap_v2.json").read_text())["test"]
    assert v2[0]["boxes"] == [] and v2[0]["labels"] == []


def test_progress_counts_edited_files_and_ignores_crlf(tmp_path):
    root = _setup(tmp_path)
    out = _prelabel(root, tmp_path)
    r = _import(root, "--progress")
    assert r.returncode == 0 and "0/2 label files edited since pre-label" in r.stdout
    p0 = out / "labels" / "BloodImage_00000.txt"
    p0.write_bytes(p0.read_bytes().replace(b"\n", b"\r\n"))  # CRLF only: not an edit
    r = _import(root, "--progress")
    assert "0/2 label files edited" in r.stdout
    p1 = out / "labels" / "BloodImage_00001.txt"
    p1.write_text("0 0.5 0.5 0.1 0.1\n", newline="\n")
    r = _import(root, "--progress")
    assert r.returncode == 0 and "1/2 label files edited" in r.stdout
    assert not (root / "splits" / "bccd_overlap_v2.json").exists()
    assert not (out / "import_stats.json").exists()


def test_prelabel_writes_lf_bytes_and_added_boxes(tmp_path):
    root = _setup(tmp_path)
    out = _prelabel(root, tmp_path)
    raw = (out / "labels" / "BloodImage_00001.txt").read_bytes()
    assert b"\r" not in raw
    man = json.loads((out / "prelabel_manifest.json").read_text())
    assert man["BloodImage_00001"]["hash"] == hashlib.sha1(raw).hexdigest()
    ab = man["BloodImage_00001"]["added_boxes"]
    assert len(ab) == 2 and len(ab[0]) == 5 and ab[0][4] == 0
    assert man["BloodImage_00000"]["added_boxes"] == [[200.0, 200.0, 240.0, 240.0, 0]]
