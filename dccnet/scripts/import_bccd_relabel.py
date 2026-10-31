"""Import hand-corrected YOLO labels (from X-AnyLabeling) as the bccd_overlap_v2 split.

Usage (from dccnet/):
    python scripts/import_bccd_relabel.py --data-root data [--relabel-dir bccd_relabel] \
        [--split splits/overlap_bccd.json] [--subset test] [--out splits/bccd_overlap_v2.json] [--progress]

--progress: print how many label files differ from the pre-label export (newline-normalised hash) and exit 0.
Default: parse every labels/<stem>.txt; on any error list them all and write nothing; else write the v2 split
(same image keys / order as --split) and <relabel-dir>/import_stats.json.

Label-noise recheck (P3c step 7):
--sample-recheck N [--seed S]: copy N deterministic random images (numpy default_rng(S).choice without replacement over
the sorted stems) to <relabel-dir>/recheck/images/ with EMPTY label files in recheck/labels/ and classes.txt; refuses when
recheck/labels/ exists (never overwrites).
--recheck: compare the re-labelled recheck/labels/<stem>.txt of the sampled stems (recheck/sample.json) with the v2 split boxes (--out) of the same images, print the
per-class F1@IoU0.5 and store it under "recheck" in import_stats.json (other keys are kept).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.relabel import (compare_label_sets, from_yolo_lines, label_hash,  # noqa: E402
                                     normalize_label_text, per_class_f1, sample_stems)
from dccnet.datasets.splits import read_split, write_split  # noqa: E402
from dccnet.eval.export_json import image_key  # noqa: E402


def _read_norm(path: Path) -> str:
    return normalize_label_text(path.read_text(encoding="utf-8"))


def _sample_recheck(samples, rdir: Path, n: int, seed: int) -> int:
    labels_dir = rdir / "recheck" / "labels"
    if labels_dir.exists():
        print(f"error: {labels_dir} exists; refusing to overwrite a recheck pass (delete the recheck folder to redo it)",
              file=sys.stderr)
        return 1
    by_stem = {Path(s.image_path).stem: s for s in samples}
    try:
        stems = sample_stems(by_stem, n, seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    missing = [st for st in stems if not (rdir / "images" / by_stem[st].image_path.name).is_file()]
    if missing:
        print(f"error: {len(missing)} image(s) missing from {rdir / 'images'} (run prelabel_bccd.py first): "
              f"{missing[:5]}", file=sys.stderr)
        return 1
    (rdir / "recheck" / "images").mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True)
    for st in stems:
        name = by_stem[st].image_path.name
        shutil.copyfile(rdir / "images" / name, rdir / "recheck" / "images" / name)
        (labels_dir / f"{st}.txt").write_text("", encoding="utf-8", newline="\n")
    (rdir / "recheck" / "sample.json").write_text(json.dumps({"seed": seed, "n": n, "stems": stems}, indent=1),
                                                  encoding="utf-8", newline="\n")
    (rdir / "recheck" / "classes.txt").write_text("RBC\nWBC\nPlatelets\n", encoding="utf-8", newline="\n")
    print(f"{n} images -> {rdir / 'recheck'} (seed {seed}); re-label them from scratch in X-AnyLabeling "
          f"(Rule: every RBC whose visible outline is more than half the cell), then run --recheck")
    return 0


def _recheck(root: Path, rdir: Path, out: str, subset: str) -> int:
    labels_dir = rdir / "recheck" / "labels"
    sample = rdir / "recheck" / "sample.json"
    if not sample.is_file():
        print(f"error: {sample} not found; run --sample-recheck first", file=sys.stderr)
        return 1
    files = [labels_dir / f"{st}.txt" for st in json.loads(sample.read_text(encoding="utf-8"))["stems"]]
    if not (root / out).is_file():
        print(f"error: {root / out} not found; run the import first", file=sys.stderr)
        return 1
    v2 = {Path(s.image_path).stem: s for s in read_split(root / out, root)[subset]}
    errors, pairs = [], []
    for p in files:
        if not p.is_file():
            errors.append(f"{p}: missing label file")
            continue
        s = v2.get(p.stem)
        if s is None:
            errors.append(f"{p}: image {p.stem!r} is not in {out}")
            continue
        try:
            nb, nl = from_yolo_lines(p.read_text(encoding="utf-8").splitlines(), s.width, s.height)
        except ValueError as e:
            errors.append(f"{p}: {e}")
            continue
        pairs.append((s.boxes, s.labels, nb, nl))
    if errors:
        print(f"error: {len(errors)} recheck label file(s) invalid; nothing written:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    f1 = per_class_f1(pairs)
    rec = {"n_images": len(pairs), "f1_iou50": f1, "n_boxes_v2": int(sum(len(p[0]) for p in pairs)),
           "n_boxes_recheck": int(sum(len(p[2]) for p in pairs))}
    sp = rdir / "import_stats.json"
    stats = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
    stats["recheck"] = rec
    sp.write_text(json.dumps(stats, indent=1), encoding="utf-8", newline="\n")
    print(f"recheck of {len(pairs)} images: per-class F1@IoU0.5 (v2 vs second pass) "
          + "  ".join(f"{k}={'n/a' if v is None else format(v, '.4f')}" for k, v in f1.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--relabel-dir", default="bccd_relabel")
    ap.add_argument("--split", default="splits/overlap_bccd.json")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", default="splits/bccd_overlap_v2.json")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--sample-recheck", type=int, default=None, metavar="N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recheck", action="store_true")
    a = ap.parse_args(argv)

    root = a.data_root
    rdir = root / a.relabel_dir
    samples = read_split(root / a.split, root)[a.subset]
    mpath = rdir / "prelabel_manifest.json"
    stems: dict[str, str] = {}
    for s in samples:
        st = Path(s.image_path).stem
        if st in stems:
            print(f"error: duplicate image stem {st!r}: {stems[st]} and {image_key(s, root)}", file=sys.stderr)
            return 2
        stems[st] = image_key(s, root)
    if mpath.is_file():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        manifest = {}
        print(f"warning: {mpath} missing; progress/stats cannot compare against the pre-label", file=sys.stderr)

    if a.sample_recheck is not None:
        return _sample_recheck(samples, rdir, a.sample_recheck, a.seed)
    if a.recheck:
        return _recheck(root, rdir, a.out, a.subset)

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

    per_image, tot = {}, {"n_images": len(samples), "n_boxes": 0, "kept": 0, "added": 0, "removed": 0,
                      "n_prelabel_added": 0, "yolo_origin_kept": 0}
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
        for k, v in (("n_prelabel_added", None if ab is None else len(ab)), ("yolo_origin_kept", okept)):
            tot[k] = None if v is None or tot[k] is None else tot[k] + v
        for k in ("kept", "added", "removed"):
            tot[k] += cmp[k]
    if warned:
        print("warning: manifest lacks 'added_boxes' (old export); yolo_origin_kept is null for those images",
              file=sys.stderr)
    write_split({a.subset: new_samples}, root / a.out, root)
    old = rdir / "import_stats.json"
    prev = json.loads(old.read_text(encoding="utf-8")) if old.is_file() else {}
    keep = {"recheck": prev["recheck"]} if "recheck" in prev else {}  # a normal import keeps the recheck result
    old.write_text(
        json.dumps({"totals": tot, "per_image": per_image, **keep}, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(new_samples)} images -> {root / a.out}  " + "  ".join(f"{k}={v}" for k, v in tot.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
