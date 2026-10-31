"""Pre-label the BCCD overlap set (GT + confident YOLO predictions) and export for X-AnyLabeling.

Usage (from dccnet/):
    python scripts/prelabel_bccd.py --data-root data --preds runs/<yolo pooled run>/preds_bccd.json \
        [--split splits/overlap_bccd.json] [--out bccd_relabel] [--conf 0.5] [--force]

Writes <data-root>/<out>/{images/, labels/<stem>.txt, classes.txt, review_list.csv, prelabel_manifest.json}.
Refuses to touch an existing <out>/labels/ unless --force (it may hold manual corrections).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.relabel import label_hash, merge_labels, to_yolo_lines  # noqa: E402
from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.export_json import image_key, load_records  # noqa: E402

CLASS_NAMES = ["RBC", "WBC", "Platelets"]
README = """Relabel rule: label every RBC whose visible outline is more than half the cell,
including cells cut by the image border. Fix wrong classes, delete false boxes, add missing cells.
Classes (classes.txt): RBC, WBC, Platelets. Track progress in review_list.csv (column 'done').
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--split", default="splits/overlap_bccd.json")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", default="bccd_relabel")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    root = a.data_root
    out = root / a.out
    if (out / "labels").exists() and not a.force:
        print(f"error: {out / 'labels'} already exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 2
    samples = read_split(root / a.split, root)[a.subset]
    stems: dict[str, str] = {}
    for s in samples:
        st = Path(s.image_path).stem
        if st in stems:
            print(f"error: duplicate image stem {st!r}: {stems[st]} and {image_key(s, root)}", file=sys.stderr)
            return 2
        stems[st] = image_key(s, root)
    by_img: dict[str, list[dict]] = {}
    for r in load_records(a.preds):
        by_img.setdefault(r["image"], []).append(r)
    unknown = set(by_img) - set(stems.values())
    if unknown:
        print(f"error: {len(unknown)} prediction image keys not in split, e.g. {sorted(unknown)[:3]}", file=sys.stderr)
        return 2

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest, rows = {}, []
    tot = {"n_gt": 0, "n_dupes_removed": 0, "n_added": 0, "n_final": 0}
    for s in samples:
        stem, key = Path(s.image_path).stem, image_key(s, root)
        recs = by_img.get(key, [])
        pb = np.array([[r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
                       for r in recs], dtype=np.float32).reshape(-1, 4)
        boxes, labels, origin, st = merge_labels(
            s.boxes, s.labels, pb, np.array([r["category_id"] for r in recs], dtype=np.int64),
            np.array([r["score"] for r in recs], dtype=np.float64), conf=a.conf)
        lines = to_yolo_lines(boxes, labels, s.width, s.height)
        txt = "".join(ln + "\n" for ln in lines)
        (out / "labels" / f"{stem}.txt").write_text(txt)
        shutil.copyfile(s.image_path, out / "images" / Path(s.image_path).name)
        manifest[stem] = {"hash": label_hash(txt), "image": key, "origin": [int(o) for o in origin]}
        rows.append((stem, st))
        for k in tot:
            tot[k] += st[k]
    (out / "classes.txt").write_text("".join(c + "\n" for c in CLASS_NAMES))
    (out / "prelabel_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.txt").write_text(README)
    rows.sort(key=lambda r: (-r[1]["n_added"], r[0]))
    with open(out / "review_list.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image", "n_gt", "n_dupes_removed", "n_added", "n_final", "done"])
        for stem, st in rows:
            w.writerow([stem, st["n_gt"], st["n_dupes_removed"], st["n_added"], st["n_final"], ""])
    print(f"{len(samples)} images -> {out}  " + "  ".join(f"{k}={v}" for k, v in tot.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
