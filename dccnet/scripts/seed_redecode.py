"""Seed a new runs dir for re-decoding finished P3b runs without retraining.

Usage (from dccnet/):
    python scripts/seed_redecode.py --src runs --dst runs_c --prefix p3b --splits pooled \
        [--dcc-arms d_both,d_sigma,circlenet,repel] [--copy-arms yolo26m,frcnn] [--seeds 0,1,2]

DC-CellNet run dirs get only the training outputs (best.pt, config.yaml, summary.json, metrics.jsonl,
git_hash.txt), so ``run_exp`` skips training (summary.json present) and redoes predict val -> sweep -> predict test
-> evaluate with ``EXP_DCC_PREDICT_ARGS`` (e.g. ``--nms center``). Baseline run dirs are copied whole (finished,
so the runner skips them) and the summary report sees every arm in one place. Existing destination run dirs are
never overwritten. Missing source runs are listed and the exit code is 1 (nothing is half-copied per run).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DCC_KEEP = ("best.pt", "config.yaml", "summary.json", "metrics.jsonl", "git_hash.txt")


def seed(src: Path, dst: Path, prefix: str, splits, dcc_arms, copy_arms, seeds) -> tuple[list[str], list[str]]:
    done, missing = [], []
    dst.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for arm in list(dcc_arms) + list(copy_arms):
            for s in seeds:
                tag = f"{prefix}_{split}_{arm}_s{s}"
                a, b = src / tag, dst / tag
                if b.exists():
                    continue
                if not (a / "summary.json").is_file():
                    missing.append(tag)
                    continue
                if arm in dcc_arms:
                    if not (a / "best.pt").is_file():
                        missing.append(tag)
                        continue
                    tmp = dst / (tag + ".partial")
                    shutil.rmtree(tmp, ignore_errors=True)
                    tmp.mkdir()
                    for f in DCC_KEEP:
                        if (a / f).is_file():
                            shutil.copy2(a / f, tmp / f)
                    tmp.rename(b)
                else:
                    shutil.copytree(a, b)
                done.append(tag)
    return done, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--prefix", default="p3b")
    ap.add_argument("--splits", default="pooled")
    ap.add_argument("--dcc-arms", default="d_both,d_sigma,circlenet,repel")
    ap.add_argument("--copy-arms", default="yolo26m,frcnn")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    sp = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    done, missing = seed(a.src, a.dst, a.prefix, sp(a.splits), sp(a.dcc_arms), sp(a.copy_arms), sp(a.seeds))
    print(f"seeded {len(done)} run dirs into {a.dst}")
    if missing:
        print("missing or unfinished source runs: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
