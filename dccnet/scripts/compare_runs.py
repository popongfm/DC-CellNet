"""M2 go/no-go report: compare experiment arms (e.g. circlenet / density / dccell) over seeds.

Usage (from dccnet/):
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json \
        --data-root data --out reports/m2_go_no_go.md [--baseline circlenet --candidate dccell]

Run dirs are ``<runs-dir>/<prefix><arm>_s<seed>`` and must hold summary.json, sweep_val.json (best_thr),
summary_test.json and preds_test.json (see scripts/run_m2.bat). Per-image RBC |count error| and per-overlap-bin
TP/n_gt are recomputed from preds_test.json at each run's own best_thr.

Gate (candidate vs baseline, only seeds present in both arms): per test image the RBC |count error| is averaged
over those seeds; the 95% paired-bootstrap CI of mean(candidate) - mean(baseline) must lie entirely below 0 AND the
pooled recall in overlap bins 2+3 (TP / n_gt, summed over seeds) must be higher for the candidate. Else NO-GO.
The exit code is always 0; the verdict line is printed and written to the report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from dccnet.eval.counting import count_errors  # noqa: E402
from dccnet.eval.export_json import load_records  # noqa: E402
from dccnet.eval.overlap_bins import N_BINS, overlap_counts  # noqa: E402

EARLY_EPOCH = 15  # dccell runs selected before this epoch never saw the full L_cc weight
ARM_ORDER = ("circlenet", "density", "dccell")
NAN = float("nan")


def discover_runs(runs_dir: Path, prefix: str) -> dict[str, dict[int, Path]]:
    """{arm: {seed: run_dir}} from directory names ``<prefix><arm>_s<seed>``."""
    pat = re.compile(rf"^{re.escape(prefix)}(?P<arm>.+)_s(?P<seed>\d+)$")
    found: dict[str, dict[int, Path]] = {}
    for d in sorted(Path(runs_dir).iterdir()) if Path(runs_dir).is_dir() else []:
        m = pat.match(d.name)
        if d.is_dir() and m:
            found.setdefault(m["arm"], {})[int(m["seed"])] = d
    return found


def _ratio(v):
    return v[:, 0].sum() / v[:, 1].sum() if v[:, 1].sum() > 0 else NAN


def load_run(run_dir: Path, samples, data_root: Path) -> dict:
    """Read one run's artefacts and recompute the per-image statistics. Raises FileNotFoundError if incomplete."""
    summary = json.loads((run_dir / "summary.json").read_text())
    thr = float(json.loads((run_dir / "sweep_val.json").read_text())["best_thr"])
    test = json.loads((run_dir / "summary_test.json").read_text())
    preds = load_records(run_dir / "preds_test.json")
    err = np.asarray(count_errors(samples, preds, thr, data_root)["per_image"]["RBC"], dtype=np.float64)
    counts = overlap_counts(samples, preds, thr, data_root)  # (images, bins, [TP, FP, n_gt])
    tot = counts.sum(axis=0)
    recall = [float(tot[b, 0] / tot[b, 2]) if tot[b, 2] > 0 else NAN for b in range(N_BINS)]
    return {"name": run_dir.name, "best_epoch": summary.get("best_epoch"), "thr": thr,
            "map": (test.get("coco") or {}).get("mAP50_95"), "abs_err": np.abs(err), "counts": counts,
            "recall": recall, "mae": float(np.abs(err).mean()),
            "summary_mae": ((test.get("counting") or {}).get("MAE") or {}).get("RBC")}


def _ms(vals) -> tuple[float, float]:
    v = np.array([x if x is not None else NAN for x in vals], dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return NAN, NAN
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else NAN


def summarize_arm(runs: dict[int, dict]) -> dict:
    mm, ms = _ms([r["map"] for r in runs.values()])
    ma, sa = _ms([r["mae"] for r in runs.values()])
    be, bs = _ms([r["best_epoch"] for r in runs.values()])
    rec = []
    for b in range(N_BINS):
        m, s = _ms([r["recall"][b] for r in runs.values()])
        rec.append({"mean": m, "std": s})
    return {"n_seeds": len(runs), "seeds": sorted(runs), "map_mean": mm, "map_std": ms, "mae_mean": ma,
            "mae_std": sa, "recall": rec, "best_epoch_mean": be, "best_epoch_std": bs,
            "best_epochs": {s: r["best_epoch"] for s, r in sorted(runs.items())},
            "thr": {s: r["thr"] for s, r in sorted(runs.items())}}


def gate(cand: dict[int, dict], base: dict[int, dict], n_boot: int = 1000) -> dict | None:
    seeds = sorted(set(cand) & set(base))
    if not seeds:
        return None
    c_err = np.mean([cand[s]["abs_err"] for s in seeds], axis=0)
    b_err = np.mean([base[s]["abs_err"] for s in seeds], axis=0)
    lo, hi = paired_bootstrap_diff(c_err, b_err, n=n_boot, seed=0)

    def pooled(runs):  # per image (TP, n_gt) in bins 2+3 summed over seeds
        return sum(runs[s]["counts"][:, 2:, :][:, :, [0, 2]].sum(axis=1) for s in seeds).astype(np.float64)

    pc, pb = pooled(cand), pooled(base)
    rc, rb = _ratio(pc), _ratio(pb)
    ci_c = bootstrap_ci(pc, stat=_ratio, n=n_boot, seed=0) if pc[:, 1].sum() > 0 else (NAN, NAN)
    ci_b = bootstrap_ci(pb, stat=_ratio, n=n_boot, seed=0) if pb[:, 1].sum() > 0 else (NAN, NAN)
    ok_mae = bool(hi < 0)
    ok_rec = bool(rc > rb)  # False when either is NaN
    return {"seeds": seeds, "n_images": len(c_err), "candidate_mae": float(c_err.mean()),
            "baseline_mae": float(b_err.mean()), "diff": float(c_err.mean() - b_err.mean()), "diff_ci": (lo, hi),
            "candidate_recall_ge2": rc, "baseline_recall_ge2": rb,
            "candidate_recall_ci": ci_c, "baseline_recall_ci": ci_b,
            "n_gt_ge2": int(pc[:, 1].sum()), "ok_mae": ok_mae, "ok_recall": ok_rec,
            "verdict": "GO" if ok_mae and ok_rec else "NO-GO"}


def _resolve_split(split: Path, data_root: Path) -> Path:
    """``--split`` is relative to --data-root (like data.split in the configs); a path valid as given also works."""
    rel = Path(data_root) / split
    return rel if rel.exists() or not Path(split).exists() else Path(split)


def compare(runs_dir: Path, prefix: str, split: Path, data_root: Path, baseline: str = "circlenet",
            candidate: str = "dccell", n_boot: int = 1000) -> dict:
    samples = read_split(_resolve_split(split, data_root), data_root)["test"]
    warnings: list[str] = []
    loaded: dict[str, dict[int, dict]] = {}
    for arm, dirs in discover_runs(runs_dir, prefix).items():
        for seed, d in dirs.items():
            try:
                loaded.setdefault(arm, {})[seed] = load_run(d, samples, data_root)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                why = (f"missing {Path(e.filename).name}" if isinstance(e, FileNotFoundError) and e.filename
                       else f"unreadable file: {e}")
                warnings.append(f"{d.name}: incomplete run skipped ({why})")
    for arm, runs in loaded.items():
        for seed, r in runs.items():
            if r["summary_mae"] is not None and abs(r["summary_mae"] - r["mae"]) > 1e-6:
                warnings.append(f"{r['name']}: recomputed RBC MAE {r['mae']:.4f} != summary_test.json "
                                f"{r['summary_mae']:.4f}")
            if arm == candidate and r["best_epoch"] is not None and r["best_epoch"] < EARLY_EPOCH:
                warnings.append(f"{r['name']}: best_epoch {r['best_epoch']} < {EARLY_EPOCH} - "
                                f"selected before full L_cc weight")
    arms = {a: summarize_arm(r) for a, r in loaded.items()}
    res = {"arms": arms, "baseline": baseline, "candidate": candidate, "warnings": warnings,
           "gate": None, "diagnostic": None, "n_boot": n_boot, "n_test_images": len(samples)}
    missing = [a for a in (baseline, candidate) if a not in loaded]
    if missing:
        warnings.append(f"gate not evaluated: no complete runs for arm(s) {missing}")
        res["gate"] = {"verdict": "NO-GO", "reason": f"missing arm(s) {missing}"}
    else:
        res["gate"] = gate(loaded[candidate], loaded[baseline], n_boot)
        if res["gate"] is None:
            warnings.append(f"gate not evaluated: {candidate} and {baseline} share no seed")
            res["gate"] = {"verdict": "NO-GO", "reason": "no common seeds"}
    if "density" in loaded and baseline in loaded:
        res["diagnostic"] = gate(loaded["density"], loaded[baseline], n_boot)
    return res


def _f(x, nd=4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _pm(m, s, nd=4) -> str:
    return "n/a" if m is None or math.isnan(m) else (f"{m:.{nd}f}" if s is None or math.isnan(s)
                                                       else f"{m:.{nd}f} ± {s:.{nd}f}")


def _gate_block(g: dict, cand: str, base: str, n_boot: int) -> list[str]:
    if "diff_ci" not in g:
        return [f"- not evaluated: {g.get('reason')}"]
    return [
        f"- seeds used: {g['seeds']}, test images: {g['n_images']}",
        f"- RBC |count error| (per-image, mean over seeds): {cand} {_f(g['candidate_mae'])} vs {base} "
        f"{_f(g['baseline_mae'])}; difference {_f(g['diff'])}",
        f"- paired bootstrap 95% CI of the difference (n={n_boot}, seed 0): [{_f(g['diff_ci'][0])}, "
        f"{_f(g['diff_ci'][1])}] -> upper < 0: **{'yes' if g['ok_mae'] else 'no'}**",
        f"- pooled recall, overlap bins 2+3 (n_gt = {g['n_gt_ge2']} over seeds): {cand} "
        f"{_f(g['candidate_recall_ge2'])} (95% CI [{_f(g['candidate_recall_ci'][0])}, "
        f"{_f(g['candidate_recall_ci'][1])}]) vs {base} {_f(g['baseline_recall_ge2'])} "
        f"(95% CI [{_f(g['baseline_recall_ci'][0])}, {_f(g['baseline_recall_ci'][1])}]) -> candidate higher: "
        f"**{'yes' if g['ok_recall'] else 'no'}**",
    ]


def render(res: dict) -> tuple[str, str]:
    cand, base = res["candidate"], res["baseline"]
    order = [a for a in ARM_ORDER if a in res["arms"]] + sorted(a for a in res["arms"] if a not in ARM_ORDER)
    L = ["# M2 go/no-go: does L_cc help separate overlapping cells?", "",
         f"Test images: {res['n_test_images']}. Values are mean ± std (sample std) over seeds; thresholds are each "
         f"run's own val-swept best_thr.", "", "## Per-arm results", "",
         "| arm | n_seeds | test mAP50-95 | RBC count MAE | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 "
         "| best_epoch |", "|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        s = res["arms"][a]
        L.append(f"| {a} | {s['n_seeds']} | {_pm(s['map_mean'], s['map_std'])} | {_pm(s['mae_mean'], s['mae_std'])} | "
                 + " | ".join(_pm(r["mean"], r["std"]) for r in s["recall"])
                 + f" | {_pm(s['best_epoch_mean'], s['best_epoch_std'], 1)} |")
    L += ["", "Overlap bin = number of other GT circles a cell overlaps (0..3+). Recall at IoU 0.5.", "",
          f"## Gate: {cand} vs {base}", ""]
    g = res["gate"]
    L += _gate_block(g, cand, base, res["n_boot"])
    verdict = f"VERDICT: {g['verdict']} ({cand} vs {base})"
    L += ["", f"**{verdict}**", ""]
    if res["diagnostic"]:
        L += ["## Diagnostic: density vs " + base + " (not part of the gate)", ""]
        L += _gate_block(res["diagnostic"], "density", base, res["n_boot"])
        L.append("")
    if res["warnings"]:
        L += ["## Warnings", ""] + [f"- ⚠ {w}" for w in res["warnings"]] + [""]
    return "\n".join(L), verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--prefix", default="exp_m2_")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", default="circlenet")
    ap.add_argument("--candidate", default="dccell")
    args = ap.parse_args(argv)

    res = compare(args.runs_dir, args.prefix, args.split, args.data_root, args.baseline, args.candidate)
    md, verdict = render(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    for w in res["warnings"]:
        print(f"WARNING: {w}")
    print(f"wrote {args.out}")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
