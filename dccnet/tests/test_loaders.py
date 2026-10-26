import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dccnet.datasets.bccd_voc import load_bccd_voc, load_bccd_voc_with_stats
from dccnet.datasets.splits import loso_split, pooled_split, read_split, write_split
from dccnet.datasets.txl import load_txl
from dccnet.types import Sample


def _mk_txl(root: Path, entries):
    """entries: list of (stem, split, source_name, label_lines or None-for-empty)."""
    for s in ("train", "val", "test"):
        (root / "images" / s).mkdir(parents=True)
        (root / "labels" / s).mkdir(parents=True)
    (root / "classes.txt").write_text("0 White Blood Cell (WBC)\n1 Red Blood Cell (RBC)\n2 Platelet")
    rows = []
    for stem, split, src, lines in entries:
        Image.new("RGB", (200, 100)).save(root / "images" / split / f"{stem}.png")
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines or []))
        rows.append({"Original_Image_Name": "o.jpg", "Source_Dataset": src,
                     "New_Image_Name": f"{stem}.png", "Source_URL": "u"})
    pd.DataFrame(rows).to_excel(root / "metadata_file.xlsx", index=False)


def test_load_txl_class_mapping_and_pixels(tmp_path):
    _mk_txl(tmp_path, [
        ("a", "train", "BCCD", ["0 0.5 0.5 0.5 0.5", "1 0.25 0.5 0.1 0.2", "2 0.75 0.5 0.1 0.4"]),
        ("b", "test", "Raabin-WBC", None),
    ])
    samples, official = load_txl(tmp_path)
    assert official == {"a": "train", "b": "test"}
    a = {s.image_path.stem: s for s in samples}["a"]
    assert a.source == "BCCD" and (a.width, a.height) == (200, 100)
    # TXL file order is WBC,RBC,Platelet -> ours RBC=0,WBC=1,Platelets=2
    assert a.labels.tolist() == [1, 0, 2]
    np.testing.assert_allclose(a.boxes[0], [50, 25, 150, 75])
    np.testing.assert_allclose(a.boxes[1], [40, 40, 60, 60])
    np.testing.assert_allclose(a.boxes[2], [140, 30, 160, 70])
    assert a.boxes.dtype == np.float32 and a.labels.dtype == np.int64
    b = {s.image_path.stem: s for s in samples}["b"]
    assert b.source == "Raabin" and b.boxes.shape == (0, 4) and b.labels.shape == (0,)


def test_load_txl_unmapped_raises(tmp_path):
    _mk_txl(tmp_path, [("a", "train", "BCCD", None)])
    Image.new("RGB", (10, 10)).save(tmp_path / "images" / "val" / "zzz.png")
    (tmp_path / "labels" / "val" / "zzz.txt").write_text("")
    with pytest.raises(ValueError, match="zzz"):
        load_txl(tmp_path)


def _mk_voc(root: Path):
    (root / "Annotations").mkdir(parents=True)
    (root / "JPEGImages").mkdir()
    def obj(n, x0, y0, x1, y1):
        return (f"<object><name>{n}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
                f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>")
    objs = obj("RBC", 10, 10, 50, 40) + obj("wbc", 5, 5, 20, 20) + obj("Platelets", 1, 1, 9, 9) \
        + obj("RBC", 30, 30, 30, 50) + obj("Foo", 1, 1, 5, 5) + "<object><name>RBC</name></object>"
    for i, o in enumerate([objs, ""]):
        (root / "Annotations" / f"im{i}.xml").write_text(
            f"<annotation><filename>im{i}.jpg</filename><size><width>64</width><height>48</height></size>{o}</annotation>")
        Image.new("RGB", (64, 48)).save(root / "JPEGImages" / f"im{i}.jpg")


def test_load_bccd_voc(tmp_path):
    _mk_voc(tmp_path)
    samples, stats = load_bccd_voc_with_stats(tmp_path)
    assert [s.source for s in samples] == ["BCCD_full"] * 2
    s0, s1 = samples
    assert s0.labels.tolist() == [0, 1, 2]
    np.testing.assert_allclose(s0.boxes[0], [10, 10, 50, 40])
    assert (s0.width, s0.height) == (64, 48)
    assert s1.boxes.shape == (0, 4)
    assert stats["skipped_boxes"] == 3
    assert len(load_bccd_voc(tmp_path)) == 2


def _samples():
    out = []
    for src, n in [("BCCD", 20), ("BCDD", 10), ("PBC", 30), ("Raabin", 40)]:
        for i in range(n):
            out.append(Sample(Path(f"/d/{src}/{i}.png"), src, 10, 10,
                              np.array([[1, 2, 3, 4]], np.float32), np.array([i % 3], np.int64)))
    return out


def test_loso_split():
    ss = _samples()
    sp = loso_split(ss, "PBC", 0.15, seed=0)
    assert {s.source for s in sp["test"]} == {"PBC"} and len(sp["test"]) == 30
    assert all(s.source != "PBC" for s in sp["train"] + sp["val"])
    for src, n in [("BCCD", 20), ("BCDD", 10), ("Raabin", 40)]:
        v = sum(s.source == src for s in sp["val"])
        assert abs(v - 0.15 * n) <= 1
        assert v + sum(s.source == src for s in sp["train"]) == n
    sp2 = loso_split(ss, "PBC", 0.15, seed=0)
    assert [s.image_path for s in sp["val"]] == [s.image_path for s in sp2["val"]]


def test_pooled_split():
    ss = _samples()
    official = {s.image_path.stem + s.source: "train" for s in ss}
    # stems collide across sources in this fixture; use unique stems
    ss = [Sample(Path(f"/d/{i}.png"), s.source, 10, 10, s.boxes, s.labels) for i, s in enumerate(ss)]
    official = {f"{i}": ("train", "val", "test")[i % 3] for i in range(len(ss))}
    sp = pooled_split(ss, official)
    assert sum(len(v) for v in sp.values()) == len(ss)
    assert all(official[s.image_path.stem] == "val" for s in sp["val"])


def test_write_read_roundtrip(tmp_path):
    root = tmp_path / "data"
    split = {"train": [], "val": [], "test": []}
    split["train"].append(Sample(root / "x" / "a.png", "PBC", 20, 10,
                                 np.array([[1.5, 2, 3, 4], [0, 0, 5, 5]], np.float32), np.array([0, 2], np.int64)))
    split["test"].append(Sample(root / "x" / "b.png", "BCCD", 20, 10,
                                np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)))
    p = tmp_path / "s.json"
    write_split(split, p, root)
    assert "x/a.png" in p.read_text() and "\\" not in p.read_text()
    out = read_split(p, root)
    assert set(out) == set(split)
    for k in split:
        assert len(out[k]) == len(split[k])
        for a, b in zip(split[k], out[k]):
            assert a.image_path == b.image_path and a.source == b.source
            assert (a.width, a.height) == (b.width, b.height)
            np.testing.assert_array_equal(a.boxes, b.boxes)
            np.testing.assert_array_equal(a.labels, b.labels)
            assert b.boxes.dtype == np.float32 and b.labels.dtype == np.int64 and b.boxes.shape == a.boxes.shape
    json.loads(p.read_text())
