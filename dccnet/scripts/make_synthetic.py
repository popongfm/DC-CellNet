"""Generate the synthetic overlapping-disc dataset (400 train / 100 val / 200 test).

Usage:
    python scripts/make_synthetic.py --out data/synthetic --data-root data --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.splits import write_split  # noqa: E402
from dccnet.datasets.synthetic import make_synthetic  # noqa: E402

SIZES = {"train": 400, "val": 100, "test": 200}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--clump-prob", type=float, default=0.5)
    args = ap.parse_args()

    split, start = {}, 0
    for name, n in SIZES.items():
        split[name] = make_synthetic(args.out / "images", n, args.seed, args.img_size,
                                     args.clump_prob, start_index=start)
        start += n
        print(f"{name}: {n} images")
    path = args.data_root / "splits" / "synthetic.json"
    write_split(split, path, args.data_root)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
