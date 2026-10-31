"""Import hand-corrected YOLO labels (from X-AnyLabeling) as the bccd_overlap_v2 split.

Usage (from dccnet/):
    python scripts/import_bccd_relabel.py --data-root data [--relabel-dir bccd_relabel] \
        [--split splits/overlap_bccd.json] [--subset test] [--out splits/bccd_overlap_v2.json] [--progress]

--progress: print how many label files differ from the pre-label export (newline-normalised hash) and exit 0.
Default: parse every labels/<stem>.txt; on any error list them all and write nothing; else write the v2 split
(same image keys / order as --split) and <relabel-dir>/import_stats.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.relabel import (compare_label_sets, from_yolo_lines, label_hash,  # noqa: E402
                                     normalize_label_text)
from dccnet.datasets.splits import read_split, write_split  # noqa: E402
from dccnet.eval.export_json import image_key  # noqa: E402


def _read_norm(path: Path) -> str:
    return normalize_label_text(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--relabel-dir", default="bccd_relabel")
    ap.add_argument("--split", default="splits/overlap_bccd.json")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", default="splits/bccd_overlap_v2.json")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args(argv)

    root = a.data_root
    rdir = root / a.relabel_dir
    samples = read_split(root / a.split, root)[a.subset]
    mpath = rdir / "prelabel_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}

    if a.progress:
        changed = missing = 0
        for s in samples:
            stem = Path(s.image_path).stem
            p = rdir / "labels" / f"{stem}.txt"
            if not p.is_file():
                missing += 1
                changed += 1
            elif stem not in manifest or label_hash(_read_norm(p)) != manifest[stem]["hash"]:
                changed += 1
        extra = f" ({missing} missing)" if missing else ""
        print(f"{changed}/{len(samples)} label files edited since pre-label{extra}")
        return 0

    errors, new_samples = [], []
    for s in samples:
        stem = Path(s.image_path).stem
        p = rdir / "labels" / f"{stem}.txt"
        if not p.is_file():
            errors.append(f"{p}: missing label file")
            continue
        try:
            boxes, labels = from_yolo_lines(p.read_text(encoding="utf-8").splitlines(), s.width, s.height)
        except ValueError as e:
            errors.append(f"{p}: {e}")
            continue
        new_samples.append(dataclasses.replace(s, boxes=boxes, labels=labels))
    if errors:
        print(f"error: {len(errors)} label file(s) invalid; nothing written:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1

    per_image, tot = {}, {"n_images": len(samples), "n_boxes": 0, "kept": 0, "added": 0, "removed": 0}
    warned = False
    for s, n in zip(samples, new_samples):
        stem = Path(s.image_path).stem
        cmp = compare_label_sets(s.boxes, s.labels, n.boxes, n.labels)
        ab = manifest.get(stem, {}).get("added_boxes")
        if ab is None:
            okept = None
            warned = True
        else:
            ab = [list(map(float, b)) for b in ab]
            okept = compare_label_sets([b[:4] for b in ab], [int(b[4]) for b in ab], n.boxes, n.labels)["kept"]
        per_image[stem] = {"image": image_key(s, root), "n_boxes": int(len(n.boxes)), "vs_original": cmp,
                           "n_prelabel_added": None if ab is None else len(ab), "yolo_origin_kept": okept}
        tot["n_boxes"] += len(n.boxes)
        for k in ("kept", "added", "removed"):
            tot[k] += cmp[k]
    if warned:
        print("warning: manifest lacks 'added_boxes' (old export); yolo_origin_kept is null for those images",
              file=sys.stderr)
    write_split({a.subset: new_samples}, root / a.out, root)
    (rdir / "import_stats.json").write_text(
        json.dumps({"totals": tot, "per_image": per_image}, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(new_samples)} images -> {root / a.out}  " + "  ".join(f"{k}={v}" for k, v in tot.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
