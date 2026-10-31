"""Build splits, overlap set, dedupe report and a YOLO-format copy for baselines.

Usage:
    python scripts/prepare_data.py --txl-root data/txl_pbc --bccd-root data/bccd_full/BCCD --data-root data
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.bccd_voc import load_bccd_voc  # noqa: E402
from dccnet.datasets.dedupe import hamming_matrix, hash_paths  # noqa: E402
from dccnet.datasets.splits import loso_split, pooled_split, write_split  # noqa: E402
from dccnet.datasets.txl import load_txl  # noqa: E402
from dccnet.datasets.yolo_export import write_yolo  # noqa: E402
from dccnet.types import Sample  # noqa: E402

MAX_DIST = 6
LOSO_SOURCES = ("BCCD", "BCDD", "PBC", "Raabin")
EXTS = {".jpg", ".jpeg", ".png"}


def _rel(p: Path, base: Path) -> str:
    try:
        return Path(p).relative_to(base).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _pairs(ha, hb, same: bool):
    """Index pairs (i, j, d) with d <= MAX_DIST; if same, only i < j."""
    d = hamming_matrix(ha, hb)
    ii, jj = np.nonzero(d <= MAX_DIST)
    return [(i, j, int(d[i, j])) for i, j in zip(ii.tolist(), jj.tolist()) if not same or i < j]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txl-root", type=Path, required=True)
    ap.add_argument("--bccd-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("dccnet/data"))
    ap.add_argument("--extra-dedupe-dir", type=Path, action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data_root: Path = args.data_root
    splits_dir = data_root / "splits"

    txl, official = load_txl(args.txl_root)
    bccd = load_bccd_voc(args.bccd_root)
    print(f"TXL total: {len(txl)}  sources: {dict(sorted(Counter(s.source for s in txl).items()))}")
    print(f"BCCD_full total: {len(bccd)}")

    pooled = pooled_split(txl, official)
    write_split(pooled, splits_dir / "pooled.json", data_root)
    print("pooled:", {k: len(v) for k, v in pooled.items()})
    for src in LOSO_SOURCES:
        sp = loso_split(txl, src, seed=args.seed)
        write_split(sp, splits_dir / f"loso_{src}.json", data_root)
        print(f"loso_{src}:", {k: len(v) for k, v in sp.items()})

    # ---- dedupe ----
    extra: list[Path] = []
    for d in args.extra_dedupe_dir:
        extra += sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    txl_p = [s.image_path for s in txl]
    bccd_p = [s.image_path for s in bccd]
    h_txl, h_bccd, h_extra = hash_paths(txl_p), hash_paths(bccd_p), hash_paths(extra)

    rows = []

    def add(pa, sa, pb, sb, pairs, A, B):
        for i, j, d in pairs:
            rows.append((_rel(A[i], data_root), sa, _rel(B[j], data_root), sb, d))

    tb = _pairs(h_txl, h_bccd, False)
    add(None, "TXL", None, "BCCD_full", tb, txl_p, bccd_p)
    tt = _pairs(h_txl, h_txl, True)
    add(None, "TXL", None, "TXL", tt, txl_p, txl_p)
    if extra:
        add(None, "BCCD_full", None, "extra", _pairs(h_bccd, h_extra, False), bccd_p, extra)
        add(None, "TXL", None, "extra", _pairs(h_txl, h_extra, False), txl_p, extra)
    data_root.mkdir(parents=True, exist_ok=True)
    with open(data_root / "dedupe_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_a", "set_a", "path_b", "set_b", "hamming"])
        w.writerows(rows)

    matched = {j for _, j, _ in tb}
    overlap = [s for k, s in enumerate(bccd) if k not in matched]
    write_split({"test": overlap}, splits_dir / "overlap_bccd.json", data_root)
    print(f"BCCD_full images matched to TXL: {len(matched)}")
    print(f"N_overlap (BCCD_full with no TXL duplicate): {len(overlap)}")
    print(f"TXL<->TXL duplicate pairs: {len(tt)}")
    print(f"dedupe report rows: {len(rows)} (extra dirs: {len(extra)} images)")

    write_yolo(pooled, data_root / "yolo_pooled", yaml_name="data_pooled.yaml")
    print(f"YOLO copy written to {data_root / 'yolo_pooled'}")


if __name__ == "__main__":
    main()
