"""P3b report: per-split tables, C2 (pooled vs leave-one-source-out) and paired comparisons over all arms.

Usage (from dccnet/):
    python scripts/summarize_loso.py --runs-dir runs --prefix p3b --data-root data --out reports/p3b.md
        [--splits pooled,loso_BCCD,loso_BCDD,loso_PBC,loso_Raabin] [--arms d_both,d_sigma,circlenet,yolo26m,frcnn]
        [--candidate d_both] [--seeds 0,1,2] [--map-ci-n 1000] [--n-boot 1000]

Run dirs are ``<runs-dir>/<prefix>_<split>_<arm>_s<seed>`` (scripts/run_exp.*); test images come from
``<data-root>/splits/<split>.json``. Every run is read with ``compare_runs.load_run`` (thresholds = each run's own
val-swept best_thr). Writes the markdown report to ``--out`` and the same data to ``--out`` with a ``.json`` suffix.

Report:
  1. per split x arm: mAP50-95 / mAP50 / per-class AP50-95 (summary_test.json) mean ± sd over seeds, a 95% image
     bootstrap CI of mAP (``coco_map_ci``, first requested seed's run only; ``--map-ci-n 0`` disables), per-class
     count MAE and recall per overlap bin (recomputed from preds_test.json);
  2. C2 per arm: pooled mAP50-95, LOSO mean (mean over the 4 folds of each fold's seed mean), delta = pooled - LOSO;
     a split missing any requested seed leaves that value blank (never averaged over fewer seeds/folds);
  3. paired comparisons ``--candidate`` vs every other arm with ``compare_runs.gate``: (a) pooled split, (b) LOSO
     pooled -- per seed the per-image arrays of the 4 folds are concatenated (BCCD, BCDD, PBC, Raabin). Verdict
     GO / NO-GO, or INCONCLUSIVE when any requested seed or fold is missing for either arm, or a run's predictions
     name images outside the split's test subset (the offending keys are listed);
  4. efficiency per arm: n_params, mean train_seconds per split (summary.json), FPS from ``<run>/latency.json`` if present;
  5. stage-1 line ``pooled stop/go: d_both vs circlenet = <verdict>`` (spec section 3.2), whatever ``--candidate``.
The exit code is always 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_runs  # noqa: E402
from compare_runs import discover_runs, gate, load_run, summarize_arm  # noqa: E402
from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.coco_eval import coco_map_ci  # noqa: E402
from dccnet.eval.common import CLASS_NAMES  # noqa: E402
from dccnet.eval.counting import count_errors  # noqa: E402
from dccnet.eval.export_json import image_key, load_records  # noqa: E402
from dccnet.eval.overlap_bins import N_BINS  # noqa: E402

FOLDS = ("loso_BCCD", "loso_BCDD", "loso_PBC", "loso_Raabin")  # fixed concatenation order
SPLITS = ("pooled",) + FOLDS
ARMS = ("d_both", "d_sigma", "circlenet", "yolo26m", "frcnn")
STAGE1 = ("d_both", "circlenet")
NAN = float("nan")


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _ms(vals) -> dict:
    m, s = compare_runs._ms(vals)
    return {"mean": _num(m), "std": _num(s)}


def _run_name(prefix: str, split: str, arm: str, seed: int) -> str:
    return f"{prefix}_{split}_{arm}_s{seed}"


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _protocol_warnings(name: str, arm: str, run_dir: Path, summary: dict) -> list[str]:
    """One line when a run used a non-protocol batch / accumulation (DC-CellNet: batch 8, accum 1; FRCNN:
    batch 4, accum 1; YOLO skipped). Missing or unreadable info gives no warning."""
    try:
        if arm == "frcnn":
            batch, accum, ref = summary.get("batch"), summary.get("accum_steps"), 4
        elif arm == "yolo26m":
            return []
        else:
            import yaml
            tr = (yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}).get("train") or {}
            batch, accum, ref = tr.get("batch"), tr.get("accum_steps"), 8
    except Exception:
        return []
    bad_b = batch is not None and batch != ref
    bad_a = accum not in (None, 1)
    if not (bad_b or bad_a):
        return []
    return [f"{name}: non-protocol batch/accum (batch {batch}, accum_steps {accum}; protocol batch {ref}, "
            f"accum 1)"]


def load_split(runs_dir: Path, prefix: str, split: str, data_root: Path, arms, seeds, warnings: list[str]) -> dict:
    """Load every requested run of one split.

    Returns ``{"samples", "runs": {arm: {seed: run}}, "problems": {arm: {seed: reason}}}``; a run with a problem
    (no dir, incomplete, predictions outside the subset) is absent from ``runs``.
    """
    out = {"samples": None, "runs": {}, "problems": {}}

    def problem(arm, seed, why):
        out["problems"].setdefault(arm, {})[seed] = why
        warnings.append(why)

    split_path = Path(data_root) / "splits" / f"{split}.json"
    if not split_path.exists():
        warnings.append(f"split {split}: {split_path} not found; all its runs treated as missing")
        for arm in arms:
            for seed in seeds:
                out["problems"].setdefault(arm, {})[seed] = f"{_run_name(prefix, split, arm, seed)}: split missing"
        return out
    samples = read_split(split_path, data_root)["test"]
    out["samples"] = samples
    keys = {image_key(s, data_root) for s in samples}
    found = discover_runs(runs_dir, f"{prefix}_{split}_")
    for arm in arms:
        for seed in seeds:
            name = _run_name(prefix, split, arm, seed)
            d = found.get(arm, {}).get(seed)
            if d is None:
                problem(arm, seed, f"{name}: no run dir")
                continue
            try:
                preds = load_records(d / "preds_test.json")
                pred_keys = {p["image"] for p in preds}
                extra = sorted(pred_keys - keys)
                if extra:
                    problem(arm, seed, f"{name}: {len(extra)} prediction image key(s) not in the {split} test subset; "
                                       f"first up to 5: {extra[:5]}")
                    continue
                if len(keys - pred_keys):
                    warnings.append(f"{name}: {len(keys - pred_keys)} test image(s) without prediction records "
                                    f"(scored as zero detections)")
                r = load_run(d, samples, data_root)
                summary = json.loads((d / "summary.json").read_text())
                test = json.loads((d / "summary_test.json").read_text())
                latency = _read_json(d / "latency.json")
            except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                why = (f"missing {Path(e.filename).name}" if isinstance(e, FileNotFoundError) and e.filename
                       else f"unreadable: {e}")
                problem(arm, seed, f"{name}: incomplete run skipped ({why})")
                continue
            if r["best_epoch"] is None or r["best_epoch"] < r["select_from"]:
                problem(arm, seed, f"{name}: invalid best_epoch {r['best_epoch']} (select_from_epoch "
                                   f"{r['select_from']}); run excluded")
                continue
            if r["summary_mae"] is not None and abs(r["summary_mae"] - r["mae"]) > 1e-6:
                warnings.append(f"{name}: recomputed RBC MAE {r['mae']:.4f} != summary_test.json "
                                f"{r['summary_mae']:.4f}")
            warnings.extend(_protocol_warnings(name, arm, d, summary))
            ce = count_errors(samples, preds, r["thr"], data_root)
            coco = test.get("coco") or {}
            r.update({"dir": d, "map50": coco.get("mAP50"), "ap_class": coco.get("per_class_50_95") or {},
                      "mae_class": ce["MAE"], "train_seconds": summary.get("train_seconds"),
                      "n_params": summary.get("n_params"),
                      "fps": (latency or {}).get("fps") if isinstance(latency, dict) else None})
            out["runs"].setdefault(arm, {})[seed] = r
    return out


def _table_row(runs: dict[int, dict], samples, data_root: Path, map_ci_n: int, ci_seed: int, label: str) -> dict:
    base = summarize_arm(runs)
    row = {"n_seeds": base["n_seeds"], "seeds": base["seeds"],
           "map_mean": _num(base["map_mean"]), "map_std": _num(base["map_std"]),
           "map50_mean": _ms([r["map50"] for r in runs.values()])["mean"],
           "map50_std": _ms([r["map50"] for r in runs.values()])["std"],
           "ap_class": {c: _ms([r["ap_class"].get(c) for r in runs.values()]) for c in CLASS_NAMES},
           "mae_class": {c: _ms([r["mae_class"].get(c) for r in runs.values()]) for c in CLASS_NAMES},
           "recall": [{"mean": _num(x["mean"]), "std": _num(x["std"])} for x in base["recall"]],
           "map_ci": None}
    if map_ci_n > 0 and ci_seed in runs:
        t0 = time.time()
        ci = coco_map_ci(samples, load_records(runs[ci_seed]["dir"] / "preds_test.json"), data_root, n=map_ci_n,
                         seed=0)
        print(f"  mAP CI {label} seed {ci_seed}: {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        row["map_ci"] = {"seed": ci_seed, "n": map_ci_n, "mAP50_95": [_num(x) for x in ci["mAP50_95"]],
                         "mAP50": [_num(x) for x in ci["mAP50"]]}
    return row


def _missing(split_data: dict, arm: str, seeds) -> list[str]:
    probs = split_data["problems"].get(arm, {})
    return [probs[s] for s in seeds if s in probs]


def _compare(cand_runs, base_runs, reasons: list[str], n_boot: int, seeds) -> dict:
    if reasons:
        return {"verdict": "INCONCLUSIVE", "reasons": reasons}
    g = gate(cand_runs, base_runs, n_boot, seeds)
    g["reasons"] = []
    return g


def _pooled_cmp(splits_data, cand, base, n_boot, seeds) -> dict:
    sd = splits_data["pooled"]
    reasons = _missing(sd, cand, seeds) + _missing(sd, base, seeds)
    return _compare(sd["runs"].get(cand, {}), sd["runs"].get(base, {}), reasons, n_boot, seeds)


def _loso_cmp(splits_data, cand, base, n_boot, seeds) -> dict:
    reasons = [f"{f} not in --splits" for f in FOLDS if f not in splits_data]
    for f in FOLDS:
        if f in splits_data:
            reasons += _missing(splits_data[f], cand, seeds) + _missing(splits_data[f], base, seeds)
    if reasons:
        return {"verdict": "INCONCLUSIVE", "reasons": reasons}

    def combine(arm):  # per seed, per-image arrays of the 4 folds concatenated in FOLDS order
        return {s: {k: np.concatenate([splits_data[f]["runs"][arm][s][k] for f in FOLDS], axis=0)
                    for k in ("abs_err", "counts")} for s in seeds}

    return _compare(combine(cand), combine(base), [], n_boot, seeds)


def summarize(runs_dir: Path, prefix: str, data_root: Path, splits=SPLITS, arms=ARMS, candidate: str = "d_both",
              seeds=(0, 1, 2), map_ci_n: int = 1000, n_boot: int = 1000) -> dict:
    seeds = sorted(int(s) for s in seeds)
    splits = list(splits)
    arms = list(arms)
    load_arms = arms + [a for a in (candidate, *STAGE1) if a not in arms]
    warnings: list[str] = []
    data = {sp: load_split(runs_dir, prefix, sp, data_root, load_arms, seeds, warnings) for sp in splits}
    ci_seed = seeds[0] if seeds else 0

    tables = {}
    for sp in splits:
        tables[sp] = {}
        for arm in arms:
            runs = data[sp]["runs"].get(arm)
            if runs:
                tables[sp][arm] = _table_row(runs, data[sp]["samples"], data_root, map_ci_n, ci_seed, f"{sp} {arm}")

    def split_map(sp, arm):  # seed mean, only when every requested seed is present with a non-null mAP
        if sp not in data or _missing(data[sp], arm, seeds) or not seeds:
            return None
        runs = data[sp]["runs"][arm]
        null = [runs[s]["name"] for s in seeds if runs[s]["map"] is None]
        if null:
            warnings.append(f"C2 {arm} {sp}: blank, no mAP50_95 in summary_test.json of {null}")
            return None
        return _ms([runs[s]["map"] for s in seeds])["mean"]

    c2 = {}
    for arm in arms:
        folds = {f: split_map(f, arm) for f in FOLDS}
        pooled = split_map("pooled", arm)
        loso = float(np.mean(list(folds.values()))) if all(v is not None for v in folds.values()) else None
        if loso is None and any(f in data for f in FOLDS):
            warnings.append(f"C2 {arm}: LOSO mean blank (needs every requested seed on all 4 folds; missing: "
                            + ", ".join(f for f, v in folds.items() if v is None) + ")")
        c2[arm] = {"pooled": pooled, "folds": folds, "loso_mean": loso,
                   "delta": pooled - loso if pooled is not None and loso is not None else None}

    comparisons = {"pooled": {}, "loso": {}}
    for base in arms:
        if base == candidate:
            continue
        if "pooled" in data:
            comparisons["pooled"][base] = _pooled_cmp(data, candidate, base, n_boot, seeds)
        if any(f in data for f in FOLDS):
            comparisons["loso"][base] = _loso_cmp(data, candidate, base, n_boot, seeds)
    if "pooled" in data:
        s1 = (comparisons["pooled"][STAGE1[1]] if candidate == STAGE1[0] and STAGE1[1] in comparisons["pooled"]
              else _pooled_cmp(data, *STAGE1, n_boot, seeds))
    else:
        s1 = {"verdict": "INCONCLUSIVE", "reasons": ["pooled split not in --splits"]}

    efficiency = {}
    for arm in arms:
        runs = [data[sp]["runs"][arm][s] for sp in splits for s in data[sp]["runs"].get(arm, {})]
        train_s = {}
        for sp in splits:
            t = [r["train_seconds"] for r in data[sp]["runs"].get(arm, {}).values() if r["train_seconds"] is not None]
            train_s[sp] = {"mean": float(np.mean(t)) if t else None, "n": len(t)}
        n_params = sorted({r["n_params"] for r in runs if r["n_params"] is not None})
        if len(n_params) > 1:
            warnings.append(f"{arm}: n_params differs between runs {n_params}")
        fps = [r["fps"] for r in runs if r["fps"] is not None]
        efficiency[arm] = {"n_params": n_params[0] if len(n_params) == 1 else (n_params or None),
                           "train_seconds": train_s,
                           "fps": float(np.mean(fps)) if fps else None}

    return {"prefix": prefix, "splits": splits, "arms": arms, "candidate": candidate, "seeds": seeds,
            "map_ci_n": map_ci_n, "n_boot": n_boot,
            "n_test_images": {sp: (len(d["samples"]) if d["samples"] is not None else None) for sp, d in data.items()},
            "tables": tables, "c2": c2, "comparisons": comparisons, "stage1_detail": s1, "stage1": s1["verdict"],
            "efficiency": efficiency, "warnings": warnings}


# ---------------------------------------------------------------------------------------------------- rendering
_f = compare_runs._f


def _pm(d: dict, nd=4) -> str:
    return compare_runs._pm(NAN if d["mean"] is None else d["mean"], NAN if d["std"] is None else d["std"], nd)


def _blank(x, nd=4) -> str:
    return "" if x is None else _f(x, nd)


def stage1_line(res: dict) -> str:
    return f"pooled stop/go: {STAGE1[0]} vs {STAGE1[1]} = {res['stage1']}"


def _cmp_row(scope: str, cand: str, base: str, g: dict) -> str:
    if "diff_ci" not in g:
        why = "; ".join(g.get("reasons", []))
        return f"| {scope} | {cand} vs {base} | **{g['verdict']}** | | | | | | | {why} |"
    per_seed = ", ".join(f"s{s} {_f(v)}" for s, v in sorted(g["per_seed_diff"].items()))
    return (f"| {scope} | {cand} vs {base} | **{g['verdict']}** | {g['seeds']} | {g['n_images']} | "
            f"{_f(g['candidate_mae'])} vs {_f(g['baseline_mae'])} | [{_f(g['diff_ci'][0])}, {_f(g['diff_ci'][1])}] "
            f"({'yes' if g['ok_mae'] else 'no'}) | {_f(g['candidate_recall_ge2'])} vs {_f(g['baseline_recall_ge2'])} "
            f"({'yes' if g['ok_recall'] else 'no'}) | {per_seed} ({'yes' if g['ok_seeds'] else 'no'}) | |")


def render(res: dict) -> str:
    cand, seeds = res["candidate"], res["seeds"]
    L = ["# P3b: TXL-PBC pooled + leave-one-source-out summary", "",
         f"Splits: {', '.join(res['splits'])}. Arms: {', '.join(res['arms'])}. Required seeds: {seeds}. "
         f"Candidate: {cand}. Values are mean ± sample std over seeds; thresholds are each run's own val-swept "
         f"best_thr. Test images: " + ", ".join(f"{sp} {n if n is not None else 'n/a'}"
                                               for sp, n in res["n_test_images"].items()) + ".", "",
         f"**{stage1_line(res)}**", ""]
    reasons = res["stage1_detail"].get("reasons", [])
    L += [f"- stage 1 inconclusive: {r}" for r in reasons] + ([""] if reasons else [])
    L += ["## 1. Per split x arm", ""]
    if res["map_ci_n"] > 0:
        L += [f"mAP50-95 95% CI: image bootstrap (`coco_map_ci`, n={res['map_ci_n']}) of the seed-{seeds[0] if seeds else 0} "
              f"run only, not over seeds.", ""]
    else:
        L += ["mAP bootstrap CI disabled (`--map-ci-n 0`).", ""]
    ap_h = " | ".join(f"AP50-95 {c}" for c in CLASS_NAMES)
    mae_h = " | ".join(f"count MAE {c}" for c in CLASS_NAMES)
    rec_h = " | ".join(f"recall bin {b}" for b in range(N_BINS))
    ncol = 5 + 2 * len(CLASS_NAMES) + N_BINS
    for sp in res["splits"]:
        L += [f"### {sp}", "", f"| arm | n_seeds | mAP50-95 | mAP50-95 CI | mAP50 | {ap_h} | {mae_h} | {rec_h} |",
              "|" + "---|" * ncol]
        for arm in res["arms"]:
            t = res["tables"][sp].get(arm)
            if t is None:
                L.append(f"| {arm} | 0 |" + " |" * (ncol - 2))
                continue
            ci = t["map_ci"]
            ci_s = f"[{_f(ci['mAP50_95'][0])}, {_f(ci['mAP50_95'][1])}] (s{ci['seed']})" if ci else ""
            L.append(f"| {arm} | {t['n_seeds']} | {_pm({'mean': t['map_mean'], 'std': t['map_std']})} | {ci_s} | "
                     f"{_pm({'mean': t['map50_mean'], 'std': t['map50_std']})} | "
                     + " | ".join(_pm(t["ap_class"][c]) for c in CLASS_NAMES) + " | "
                     + " | ".join(_pm(t["mae_class"][c]) for c in CLASS_NAMES) + " | "
                     + " | ".join(_pm(x) for x in t["recall"]) + " |")
        L.append("")
    L += ["Overlap bin = number of other GT circles a cell overlaps (0..3+); recall at IoU 0.5.", "",
          "## 2. C2: pooled vs leave-one-source-out (mAP50-95)", "",
          "LOSO mean = mean over the 4 folds of each fold's seed mean; blank when any fold misses a required seed.", "",
          "| arm | pooled | " + " | ".join(FOLDS) + " | LOSO mean | Δ = pooled − LOSO |",
          "|" + "---|" * (4 + len(FOLDS))]
    for arm in res["arms"]:
        c = res["c2"][arm]
        L.append(f"| {arm} | {_blank(c['pooled'])} | " + " | ".join(_blank(c["folds"][f]) for f in FOLDS)
                 + f" | {_blank(c['loso_mean'])} | {_blank(c['delta'])} |")
    L += ["", f"## 3. Paired comparisons: {cand} vs every other arm", "",
          "`compare_runs.gate` per comparison: RBC |count error| per image averaged over seeds, paired bootstrap 95% CI "
          f"(n={res['n_boot']}) of candidate − baseline upper < 0; pooled recall in overlap bins 2+3 higher; every "
          "per-seed mean difference < 0. LOSO pooled = the 4 folds' per-image arrays concatenated per seed (BCCD, BCDD, "
          "PBC, Raabin). INCONCLUSIVE = a required seed/fold is missing or a run's predictions name images outside "
          "the test subset; no gate is computed then.", "",
          "| scope | comparison | verdict | seeds | images | RBC MAE cand vs base | diff CI (upper<0) | "
          "recall bins 2+3 (higher) | per-seed diff (all<0) | inconclusive because |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for scope, label in (("pooled", "pooled"), ("loso", "LOSO pooled")):
        for base, g in res["comparisons"][scope].items():
            L.append(_cmp_row(label, cand, base, g))
    if not res["comparisons"]["loso"]:
        L += ["", "LOSO comparisons not run (no LOSO split in --splits)."]
    L += ["", "## 4. Efficiency", "",
          "Mean train_seconds per split over that split's seeds (n runs timed); splits are not mixed because their "
          "train sets differ in size.", "",
          "| arm | n_params | " + " | ".join(f"train_seconds {sp}" for sp in res["splits"]) + " | inference FPS |",
          "|" + "---|" * (3 + len(res["splits"]))]
    for arm in res["arms"]:
        e = res["efficiency"][arm]
        ts = " | ".join(f"{_blank(e['train_seconds'][sp]['mean'], 0)} (n={e['train_seconds'][sp]['n']})"
                        for sp in res["splits"])
        L.append(f"| {arm} | {'' if e['n_params'] is None else e['n_params']} | {ts} | {_blank(e['fps'], 1)} |")
    L += ["", "FPS only from `<run>/latency.json` (`fps`); blank otherwise.", "", "## Verdicts", "",
          f"- **{stage1_line(res)}**"]
    for scope, label in (("pooled", "pooled"), ("loso", "LOSO pooled")):
        for base, g in res["comparisons"][scope].items():
            L.append(f"- {label}: {cand} vs {base} = {g['verdict']}")
    L.append("")
    if res["warnings"]:
        L += ["## Warnings", ""] + [f"- ⚠ {w}" for w in res["warnings"]] + [""]
    return "\n".join(L)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--prefix", default="p3b")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--candidate", default="d_both")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--map-ci-n", type=int, default=1000, help="image-bootstrap draws for the mAP CI; 0 disables")
    ap.add_argument("--n-boot", type=int, default=1000, help="paired-bootstrap draws for the comparisons")
    args = ap.parse_args(argv)
    if not _csv(args.seeds):
        ap.error("--seeds must name at least one seed")

    res = summarize(args.runs_dir, args.prefix, args.data_root, _csv(args.splits), _csv(args.arms), args.candidate,
                    [int(x) for x in _csv(args.seeds)], args.map_ci_n, args.n_boot)
    md = render(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    for w in res["warnings"]:
        print(f"WARNING: {w}")
    print(f"wrote {args.out} and {json_out}")
    for scope, label in (("pooled", "pooled"), ("loso", "LOSO pooled")):
        for base, g in res["comparisons"][scope].items():
            print(f"{label}: {res['candidate']} vs {base} = {g['verdict']}")
    print(stage1_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
