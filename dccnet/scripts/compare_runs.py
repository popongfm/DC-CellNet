"""M2 go/no-go report: compare experiment arms (e.g. circlenet / density / dccell) over seeds.

Usage (from dccnet/):
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json \
        --data-root data --out reports/m2_go_no_go.md [--baseline circlenet --candidate dccell --ablation density]
        [--seeds 0,1,2]

Run dirs are ``<runs-dir>/<prefix><arm>_s<seed>`` and must hold summary.json, sweep_val.json (best_thr),
summary_test.json and preds_test.json (see scripts/run_m2.bat). Per-image RBC |count error| and per-overlap-bin
TP/n_gt are recomputed from preds_test.json at each run's own best_thr.

Two comparisons, each computed the same way (only requested seeds present in both arms): dccell vs circlenet (main
gate) and dccell vs density (isolates L_cc). A comparison passes iff
  * per test image the RBC |count error| is averaged over the common seeds; the 95% paired-bootstrap CI of
    mean(candidate) - mean(baseline) lies entirely below 0, AND
  * the pooled recall in overlap bins 2+3 (TP / n_gt, summed over seeds) is higher for the candidate, AND
  * seed robustness: for every common seed s, mean over images of |err|(candidate s) - |err|(baseline s) < 0.
Verdict:
  * INCONCLUSIVE if any of the three arms lacks a complete run for any requested seed (--seeds, default 0,1,2), or
    any candidate run has best_epoch < its select_from_epoch (summary.json; fallback 15 for older runs);
  * else GO iff both comparisons pass; else NO-GO.
density vs circlenet is reported as a diagnostic only. The exit code is always 0; the verdict line is printed and
written to the report.
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

EARLY_EPOCH = 15  # fallback select_from_epoch (end of the L_cc ramp) for runs whose summary.json predates R20
DEFAULT_SEEDS = (0, 1, 2)
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
            "select_from": int(summary.get("select_from_epoch", EARLY_EPOCH)),
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


def gate(cand: dict[int, dict], base: dict[int, dict], n_boot: int = 1000, seeds=None) -> dict | None:
    """One comparison (candidate vs baseline) over the common seeds (restricted to ``seeds`` if given)."""
    seeds = sorted(set(cand) & set(base) & (set(seeds) if seeds is not None else set(cand)))
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
    per_seed = {s: float(np.mean(cand[s]["abs_err"] - base[s]["abs_err"])) for s in seeds}
    ok_seeds = all(v < 0 for v in per_seed.values())  # False on NaN
    return {"seeds": seeds, "n_images": len(c_err), "candidate_mae": float(c_err.mean()),
            "baseline_mae": float(b_err.mean()), "diff": float(c_err.mean() - b_err.mean()), "diff_ci": (lo, hi),
            "candidate_recall_ge2": rc, "baseline_recall_ge2": rb,
            "candidate_recall_ci": ci_c, "baseline_recall_ci": ci_b,
            "n_gt_ge2": int(pc[:, 1].sum()), "ok_mae": ok_mae, "ok_recall": ok_rec,
            "per_seed_diff": per_seed, "ok_seeds": ok_seeds,
            "verdict": "GO" if ok_mae and ok_rec and ok_seeds else "NO-GO"}


def _resolve_split(split: Path, data_root: Path) -> Path:
    """``--split`` is relative to --data-root (like data.split in the configs); a path valid as given also works."""
    rel = Path(data_root) / split
    return rel if rel.exists() or not Path(split).exists() else Path(split)


def _comparison(loaded: dict, cand: str, base: str, n_boot: int, seeds, warnings: list[str]) -> dict:
    missing = [a for a in (base, cand) if a not in loaded]
    if missing:
        warnings.append(f"{cand} vs {base} not evaluated: no complete runs for arm(s) {missing}")
        return {"verdict": "NO-GO", "reason": f"missing arm(s) {missing}"}
    g = gate(loaded[cand], loaded[base], n_boot, seeds)
    if g is None:
        warnings.append(f"{cand} vs {base} not evaluated: no common seed among {sorted(seeds)}")
        return {"verdict": "NO-GO", "reason": "no common seeds"}
    return g


def compare(runs_dir: Path, prefix: str, split: Path, data_root: Path, baseline: str = "circlenet",
            candidate: str = "dccell", n_boot: int = 1000, ablation: str = "density",
            seeds=DEFAULT_SEEDS) -> dict:
    seeds = tuple(sorted(int(x) for x in seeds))
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
    inconclusive: list[str] = []
    for arm, runs in loaded.items():
        for seed, r in runs.items():
            if r["summary_mae"] is not None and abs(r["summary_mae"] - r["mae"]) > 1e-6:
                warnings.append(f"{r['name']}: recomputed RBC MAE {r['mae']:.4f} != summary_test.json "
                                f"{r['summary_mae']:.4f}")
            early = r["best_epoch"] is None or r["best_epoch"] < r["select_from"]
            if early:
                warnings.append(f"{r['name']}: best_epoch {r['best_epoch']} < select_from_epoch {r['select_from']} - "
                                f"selected before full L_cc weight")
                if arm == candidate and seed in seeds:
                    inconclusive.append(f"{r['name']}: best_epoch {r['best_epoch']} < select_from_epoch "
                                        f"{r['select_from']}")
    for arm in (baseline, ablation, candidate):
        miss = [s for s in seeds if s not in loaded.get(arm, {})]
        if miss:
            inconclusive.append(f"arm {arm}: no complete run for seed(s) {miss}")
    arms = {a: summarize_arm(r) for a, r in loaded.items()}
    res = {"arms": arms, "baseline": baseline, "candidate": candidate, "ablation": ablation, "seeds": list(seeds),
           "warnings": warnings, "diagnostic": None, "n_boot": n_boot, "n_test_images": len(samples)}
    res["gate"] = _comparison(loaded, candidate, baseline, n_boot, seeds, warnings)
    res["gate_ablation"] = _comparison(loaded, candidate, ablation, n_boot, seeds, warnings)
    if ablation in loaded and baseline in loaded:
        res["diagnostic"] = gate(loaded[ablation], loaded[baseline], n_boot, seeds)
    res["inconclusive"] = inconclusive
    if inconclusive:
        res["verdict"] = "INCONCLUSIVE"
    elif res["gate"]["verdict"] == "GO" and res["gate_ablation"]["verdict"] == "GO":
        res["verdict"] = "GO"
    else:
        res["verdict"] = "NO-GO"
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
        "- per-seed paired RBC |count error| difference (mean over images, " + cand + " seed s - " + base
        + " seed s): " + ", ".join(f"s{s} {_f(v)}" for s, v in sorted(g["per_seed_diff"].items()))
        + f" -> all < 0: **{'yes' if g['ok_seeds'] else 'no'}**",
        f"- comparison: **{'pass' if g['verdict'] == 'GO' else 'fail'}**",
    ]


def render(res: dict) -> tuple[str, str]:
    cand, base, abl = res["candidate"], res["baseline"], res["ablation"]
    order = [a for a in ARM_ORDER if a in res["arms"]] + sorted(a for a in res["arms"] if a not in ARM_ORDER)
    L = ["# M2 go/no-go: does L_cc help separate overlapping cells?", "",
         f"Test images: {res['n_test_images']}. Required seeds: {res['seeds']}. Values are mean ± std (sample std) "
         f"over seeds; thresholds are each run's own val-swept best_thr.", "", "## Per-arm results", "",
         "| arm | n_seeds | test mAP50-95 | RBC count MAE | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 "
         "| best_epoch |", "|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        s = res["arms"][a]
        L.append(f"| {a} | {s['n_seeds']} | {_pm(s['map_mean'], s['map_std'])} | {_pm(s['mae_mean'], s['mae_std'])} | "
                 + " | ".join(_pm(r["mean"], r["std"]) for r in s["recall"])
                 + f" | {_pm(s['best_epoch_mean'], s['best_epoch_std'], 1)} |")
    L += ["", "Overlap bin = number of other GT circles a cell overlaps (0..3+). Recall at IoU 0.5.", "",
          f"## Gate 1: {cand} vs {base}", ""]
    L += _gate_block(res["gate"], cand, base, res["n_boot"])
    L += ["", f"## Gate 2: {cand} vs {abl} (isolates L_cc)", ""]
    L += _gate_block(res["gate_ablation"], cand, abl, res["n_boot"])
    verdict = f"VERDICT: {res['verdict']} ({cand} vs {base} and {cand} vs {abl})"
    L += ["", "## Verdict", "",
          "GO requires both gates to pass; INCONCLUSIVE when a required seed run is missing for any arm or a "
          f"{cand} run picked its checkpoint before its select_from_epoch.", ""]
    if res["inconclusive"]:
        L += [f"- inconclusive: {why}" for why in res["inconclusive"]] + [""]
    L += [f"**{verdict}**", ""]
    if res["diagnostic"]:
        L += [f"## Diagnostic: {abl} vs {base} (not part of the gate)", ""]
        L += _gate_block(res["diagnostic"], abl, base, res["n_boot"])
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
    ap.add_argument("--ablation", default="density", help="second baseline: candidate minus L_cc")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds every arm must have")
    args = ap.parse_args(argv)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    res = compare(args.runs_dir, args.prefix, args.split, args.data_root, args.baseline, args.candidate,
                  ablation=args.ablation, seeds=seeds)
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
