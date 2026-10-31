# dccnet

DC-CellNet research code for blood-smear cell detection (design spec and plan in `docs/`).

## Setup
    pip install -r dccnet/requirements.txt
    python dccnet/scripts/check_env.py   # expects CUDA True and the GPU name

## Scripts
- `scripts/check_env.py`: prints torch version, CUDA availability and GPU name; exit 1 if no CUDA.
- `scripts/disk_report.py <root> --top 40 --out report.md`: read-only disk usage report
  (largest files, per-folder totals, zip-next-to-folder duplicates). Standard library only.
- `scripts/train.py --config C [--seed S] [--device cuda] [--data-root data] [--runs-dir runs] [--run-name NAME]
  [--epochs-max N] [--num-workers 0]`: trains DC-CellNet on `<data-root>/<data.split>` ("train"/"val") and writes
  `runs/<run-name>/` (default `<config stem>_s<seed>`): `config.yaml, git_hash.txt, metrics.jsonl, best.pt, summary.json`.
- `scripts/predict.py --run runs/<exp> --split data/splits/pooled.json --subset test --data-root data --out preds.json
  [--batch 8]`: evaluator records from `best.pt` (score >= 0.001) for `scripts/evaluate.py`.

`dccnet/data/` and `dccnet/runs/` are git-ignored.

## M1 — YOLO26m baseline (run on Windows)
Anaconda Prompt, from `C:\Research\dccnet`:

    conda activate dccnet
    pip install ultralytics

1. Prepare splits and the YOLO copy:

       python scripts/prepare_data.py --txl-root data/TXL-PBC_Dataset/TXL-PBC --bccd-root data/bccd_full/BCCD --data-root data --extra-dedupe-dir ..\blood-cell-detection-datatset

2. Train:

       yolo detect train model=yolo26m.pt data=data/yolo_pooled/data_pooled.yaml epochs=150 patience=100 imgsz=640 batch=8 seed=0 amp=True project=runs name=yolo26m_pooled

3. Predict on val and test (original images from the split JSON):

       python scripts/baselines/yolo_predict.py --weights runs/yolo26m_pooled/weights/best.pt --split data/splits/pooled.json --subset val --data-root data --conf 0.001 --imgsz 640 --batch 8 --device 0 --max-det 1000 --out runs/yolo26m_pooled/preds_val.json
       python scripts/baselines/yolo_predict.py --weights runs/yolo26m_pooled/weights/best.pt --split data/splits/pooled.json --subset test --data-root data --conf 0.001 --imgsz 640 --batch 8 --device 0 --max-det 1000 --out runs/yolo26m_pooled/preds_test.json

4. Choose the score threshold on val (max micro-F1 at IoU 0.5; prints `best_thr`):

       python scripts/evaluate.py --split data/splits/pooled.json --subset val --preds runs/yolo26m_pooled/preds_val.json --data-root data --out runs/yolo26m_pooled/sweep_val.json --sweep-thr

5. Evaluate on test with that threshold (replace `0.35` with `best_thr`):

       python scripts/evaluate.py --split data/splits/pooled.json --subset test --preds runs/yolo26m_pooled/preds_test.json --score-thr 0.35 --data-root data --out runs/yolo26m_pooled/summary_test.json

M1 gate: test mAP50 about 0.95-0.99 and mAP50-95 about 0.80-0.90 (TXL-PBC published YOLO range); outside that,
debug the evaluator / class mapping before M2.

## Data notes
- `scripts/prepare_data.py --txl-root <TXL-PBC> --bccd-root <BCCD folder with Annotations/JPEGImages> --data-root data`
  writes `data/splits/*.json`, `data/dedupe_report.csv` and the YOLO copy `data/yolo_pooled/` (idempotent).
- **N_overlap = 204, not 172.** BCCD_full has 364 images; pHash (8x8, Hamming <= 6) matches exactly 160 of them to TXL
  (all 160 TXL "BCCD" images, every one at Hamming 0, i.e. exact pixel content). 364 - 160 = 204 images have no TXL
  duplicate and form `splits/overlap_bccd.json`. The 172 figure is not derivable from the dedupe: in
  `BCCD_selection.xlsx`, 172 is the length of the `RBC_noboundary` column (a list of BCCD images rejected for RBCs lacking
  boundaries, none of which are among TXL's 160 selected images); the selected set is 160 (`selected_img`).
- TXL internal near-duplicates: 6 pairs with Hamming <= 6 (4 to 6), listed in `dedupe_report.csv`; 2 of them cross
  train/val splits. They are reported only, not removed.
- TXL's own label files use WBC=0, RBC=1, Platelet=2; the YOLO copy is re-written in our order (0 RBC, 1 WBC, 2 Platelets).

## M2 — synthetic go/no-go (run on Windows)
Question: does the local count-consistency loss `L_cc` help separate overlapping cells? Three arms
(`configs/exp_m2_circlenet.yaml`: no density head; `exp_m2_density.yaml`: density head, no `L_cc`;
`exp_m2_dccell.yaml`: density head + `L_cc`) x seeds 0, 1, 2 = 9 runs on the synthetic overlapping-disc set
(400 train / 100 val / 200 test; generated automatically if `data\splits\synthetic.json` or
`data\synthetic\VERSION` is missing, or `VERSION` is not `2`).

**Synthetic data must be regenerated (version 2).** Version 1 drew every disc half a pixel off its GT centre
(cv2 pixel-centre vs our edge convention); version 2 draws at sub-pixel precision so the drawn centroid equals the GT
centre. `run_m2.bat` regenerates `data\synthetic` + `data\splits\synthetic.json` by itself when `VERSION` is missing
or not `2` (the images are overwritten in place). Any run folders made on version-1 data (e.g. an earlier
`runs_smoke\`) are stale: delete them before re-running.

Anaconda Prompt:

    conda activate dccnet
    cd /d C:\Research\dccnet
    scripts\run_m2.bat

**Windows smoke test first** (about 10-20 minutes; checks the whole pipeline on 1 seed and 1 epoch before the real
run). It writes to `runs_smoke\`, so the real `runs\` stays clean:

    set M2_SEEDS=0
    set M2_TRAIN_ARGS=--epochs-max 1
    set M2_RUNS_DIR=runs_smoke
    scripts\run_m2.bat

If it ends with `ALL DONE`, clear the knobs (same Anaconda Prompt, or just open a new one) and start the real run:

    set M2_SEEDS=
    set M2_TRAIN_ARGS=
    set M2_RUNS_DIR=
    scripts\run_m2.bat

Per run it trains (`runs\exp_m2_<arm>_s<seed>\`), predicts val, sweeps the score threshold on val, predicts test and
evaluates test at the swept `best_thr`; progress goes to the console and `runs\m2_runner.log`.

- **Resume:** just re-run `scripts\run_m2.bat`. Runs with `summary_test.json` are skipped; a run with `summary.json`
  skips training. It stops at the first failing command. If a run was killed mid-training (a run folder with
  `best.pt` but no `summary.json`), the script renames that folder to `exp_m2_<arm>_s<seed>_aborted_<n>` and trains
  it fresh; `compare_runs` ignores such folders and you may delete them manually later.
- **Duration:** 9 runs x at most 60 epochs (best checkpoint / early stopping only from epoch 15, the end of the
  `L_cc` ramp, then patience 20) on the 5060 Ti; plan for several hours and leave the machine alone. The first command
  output tells you per-epoch time. Test evaluation skips the slow mAP bootstrap CI (`--map-ci-n 0`), so each run's
  evaluate step takes seconds; Claude computes the mAP CIs offline later from `preds_test.json`.
- **Send back:** nothing. Claude reads `runs\` from the shared folder and runs
  `python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/m2_go_no_go.md`
  (`--seeds 0,1,2` by default). Verdict is GO / NO-GO / INCONCLUSIVE. GO iff dccell beats **both** circlenet and
  density (density isolates `L_cc`), each comparison requiring: the paired-bootstrap 95% CI of the RBC |count error|
  difference entirely below 0, higher pooled recall in overlap bins 2+3, and a negative paired MAE difference for every
  seed. INCONCLUSIVE if any arm lacks a seed-0/1/2 run or a dccell run's `best_epoch` is before its
  `select_from_epoch`.
- `scripts/run_m2.sh` is the bash twin (used for CPU smoke tests). Optional env knobs, off by default: `M2_SEEDS`,
  `M2_CONFIGS`, `M2_DEVICE`, `M2_TRAIN_ARGS`, `M2_PREDICT_ARGS`, `M2_DATA_ROOT`, `M2_RUNS_DIR`, `M2_CONFIG_DIR`, `M2_PYTHON`.

## M2b — detached density branch (run on Windows)
M2 showed `L_cc` helps (dccell beat the density-only arm on every seed) but the shared density head cost the detector
mAP50-95 (0.790 -> 0.743), so dccell could not beat plain CircleNet. M2b sets `model.density_branch: detached`
(base default `shared`, so M2 stays reproducible): the density head reads `f.detach()` (no gradient into backbone/neck)
and is deeper (conv3x3+BN+ReLU x2, conv1x1; final conv init as before, bias -7). `L_cc` still trains the heatmap head.
Arms: `configs/exp_m2_b_circlenet.yaml`, `exp_m2_b_density.yaml`, `exp_m2_b_dccell.yaml` (120 epochs max, patience 20).
Run folders are `runs\exp_m2_b_<arm>_s<seed>\`; the arm names are `b_circlenet`, `b_density`, `b_dccell`.

Smoke test first (writes to `runs_smoke_b\`):

    set M2_CONFIGS=b_circlenet b_density b_dccell
    set M2_SEEDS=0
    set M2_TRAIN_ARGS=--epochs-max 1
    set M2_RUNS_DIR=runs_smoke_b
    scripts\run_m2.bat

Then the real run (keep `M2_CONFIGS`, clear the other knobs):

    set M2_SEEDS=
    set M2_TRAIN_ARGS=
    set M2_RUNS_DIR=
    scripts\run_m2.bat

Compare:

    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/m2b_go_no_go.md --baseline b_circlenet --candidate b_dccell --ablation b_density

Duration: 9 runs x at most 120 epochs, about 10.5 h at most on the 5060 Ti.

## M2c — oracle L_cc test (run on Windows)
M2b (density head detached) showed no benefit from `L_cc`. This is the decisive test of the idea with a perfect target:
`loss.cc.target: oracle` (base default `density`) computes `N_den` from the ground-truth per-tile center counts
(`oracle_tile_counts`, same tiling as `L_cc`) instead of the density head, so the `b_oracle` arm
(`configs/exp_m2_b_oracle.yaml`) has no density head and is architecturally identical to `b_circlenet` plus `L_cc`.
If even this cannot beat `b_circlenet`, the `L_cc` idea is dead. Run folders: `runs\exp_m2_b_oracle_s<seed>\`.

Smoke test first (writes to `runs_smoke_c\`):

    set M2_CONFIGS=b_oracle
    set M2_SEEDS=0
    set M2_TRAIN_ARGS=--epochs-max 1
    set M2_RUNS_DIR=runs_smoke_c
    scripts\run_m2.bat

Then the real run (keep `M2_CONFIGS=b_oracle`, clear the other knobs):

    set M2_SEEDS=
    set M2_TRAIN_ARGS=
    set M2_RUNS_DIR=
    scripts\run_m2.bat

Compare:

    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/m2c_oracle.md --baseline b_circlenet --candidate b_oracle --ablation b_circlenet

Duration: 3 runs x at most 120 epochs, about 3-3.5 h on the 5060 Ti; the `b_circlenet` runs from M2b are reused.

## M2d — merged-peak fixes (run on Windows)
M2/M2b/M2c traced the remaining counting errors on the synthetic overlap set to clumps of same-class cells whose
predicted heatmap collapses into a single peak. M2d tests two fixes, each against `b_circlenet` (M2b runs reused):

* `d_sigma` (`configs/exp_m2_d_sigma.yaml`): `data.heatmap_sigma: adaptive` (base default `radius`, unchanged).
  Each cell's heatmap Gaussian is shrunk by the distance `d_nn` (input px) to the nearest other center of the SAME
  class: `sigma_h = max(min(r / (3*stride), d_nn / (4*stride)), 0.5)` cells. Density targets are unchanged.
* `d_s2` (`configs/exp_m2_d_s2.yaml`): `model.out_stride: 2` + `data.stride: 2` (base default 4, unchanged). The
  stride-4 neck output is upsampled x2 (bilinear) and refined by conv3x3+BN+ReLU; all heads run on the 320x320 map
  (640 px input). `data.stride` must equal `model.out_stride` (ValueError otherwise). Train batch stays 8: estimated
  ~4.5 GiB of activations at batch 8 with AMP (vs ~3.4 GiB at stride 4). If the smoke test runs out of GPU memory,
  set `train: batch: 4` in `exp_m2_d_s2.yaml` and say so in the report.

Both arms differ from `exp_m2_b_circlenet.yaml` only in those keys. Run folders: `runs\exp_m2_d_sigma_s<seed>\`,
`runs\exp_m2_d_s2_s<seed>\`.

Smoke test first (writes to `runs_smoke_d\`; the `d_s2` epoch is also the 8 GB memory check):

    set M2_CONFIGS=d_sigma d_s2
    set M2_SEEDS=0
    set M2_TRAIN_ARGS=--epochs-max 1
    set M2_RUNS_DIR=runs_smoke_d
    scripts\run_m2.bat

Then the real run (keep `M2_CONFIGS=d_sigma d_s2`, clear the other knobs):

    set M2_SEEDS=
    set M2_TRAIN_ARGS=
    set M2_RUNS_DIR=
    scripts\run_m2.bat

Compare (one report per fix):

    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/m2d_sigma.md --baseline b_circlenet --candidate d_sigma --ablation b_circlenet
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/m2d_s2.md --baseline b_circlenet --candidate d_s2 --ablation b_circlenet

Duration: 6 runs x at most 120 epochs, about 8-9 h on the 5060 Ti (the stride-2 runs are a little slower per epoch).

## P3a — target ablation (run on Windows)
Which part of the adaptive-sigma heatmap target in `d_both` matters, and how does it compare with prior-art target
rules? Six arms, each differs from `configs/exp_m2_d_both.yaml` only in the heatmap-target keys (base defaults
`data.heatmap_sigma: radius`, `data.heatmap_divisor: 4`, `data.heatmap_shape: gaussian` keep every older config
bit-identical). Sigma in cells, `s` = stride, `r` = cell radius (px), all floored at 0.5; density targets unchanged.

| arm | config keys | heatmap target |
|---|---|---|
| `p3a_agnostic` | `heatmap_sigma: adaptive_agnostic` | `min(r/(3s), d_nn/(4s))`, `d_nn` over ALL classes |
| `p3a_knn3` | `heatmap_sigma: knn3` | `min(r/(3s), mean 3-NN same-class dist/(4s))` |
| `p3a_mcnn` | `heatmap_sigma: mcnn` | MCNN/CSRNet `0.3 * mean 3-NN same-class dist / s`, no cap (`r/(3s)` if no neighbour) |
| `p3a_repel` | `heatmap_sigma: radius`, `heatmap_shape: repel` | repel code `1/(1 + 0.8 * d1 (1 + d1/d2)^2)` within `r/s` cells of each center |
| `p3a_div2` | `heatmap_divisor: 2` | adaptive, `min(r/(3s), d_nn/(2s))` |
| `p3a_div8` | `heatmap_divisor: 8` | adaptive, `min(r/(3s), d_nn/(8s))` |

Run folders: `runs\exp_m2_p3a_<arm>_s<seed>\`. The `d_both` runs (seeds 0, 1, 2) are reused as the reference.

Smoke test first (writes to `runs_smoke_p3a\`):

    set M2_CONFIGS=p3a_agnostic p3a_knn3 p3a_mcnn p3a_repel p3a_div2 p3a_div8
    set M2_SEEDS=0
    set M2_TRAIN_ARGS=--epochs-max 1
    set M2_RUNS_DIR=runs_smoke_p3a
    scripts\run_m2.bat

Then the real run, in two parts. First the four rule arms with all 3 seeds:

    set M2_CONFIGS=p3a_agnostic p3a_knn3 p3a_mcnn p3a_repel
    set M2_SEEDS=
    set M2_TRAIN_ARGS=
    set M2_RUNS_DIR=
    scripts\run_m2.bat

Then the two divisor arms with seed 0 only:

    set M2_CONFIGS=p3a_div2 p3a_div8
    set M2_SEEDS=0
    scripts\run_m2.bat

Compare (one report per arm; `d_both` is the candidate, the arm is the baseline and the ablation):

    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/p3a_agnostic.md --baseline p3a_agnostic --candidate d_both --ablation p3a_agnostic --seeds 0,1,2
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/p3a_knn3.md --baseline p3a_knn3 --candidate d_both --ablation p3a_knn3 --seeds 0,1,2
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/p3a_mcnn.md --baseline p3a_mcnn --candidate d_both --ablation p3a_mcnn --seeds 0,1,2
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/p3a_repel.md --baseline p3a_repel --candidate d_both --ablation p3a_repel --seeds 0,1,2
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/p3a_div2.md --baseline p3a_div2 --candidate d_both --ablation p3a_div2 --seeds 0
    python scripts/compare_runs.py --runs-dir runs --prefix exp_m2_ --split splits/synthetic.json --data-root data --out reports/p3a_div8.md --baseline p3a_div8 --candidate d_both --ablation p3a_div8 --seeds 0

Duration: 14 stride-2 runs (12 + 2) x at most 120 epochs, about 24-28 h on the 5060 Ti.

## P3b — TXL-PBC pooled + LOSO (run on Windows)

Five arms (`d_both`, `d_sigma`, `circlenet`, `yolo26m`, `frcnn`) x five splits (`pooled`, `loso_BCCD`, `loso_BCDD`, `loso_PBC`, `loso_Raabin`) x seeds 0 1 2, all driven by one runner, `scripts\run_exp.bat`. Run dirs are `runs\p3b_<split>_<arm>_s<seed>`; the log is `runs\exp_runner.log`. The runner is resumable (re-run the same command after any stop), skips finished runs (`summary_test.json`), renames a killed run (weights present, no `summary.json`) to `..._aborted_<random>` and retrains it, and stops at the first failure. It never regenerates data or splits: `data\splits\<split>.json` must already exist (made by `prepare_data.py`), otherwise it logs the missing file and fails.

Setup (once, in the dccnet conda env, from `C:\Research\dccnet`):

    git pull dccnet dccnet
    pip install ultralytics
    pip show ultralytics

`pip install ultralytics` takes whatever version pip resolves today; copy the `Version:` line of `pip show ultralytics` into your notes / the chat so the report can state it. The YOLO arm exports the split to `data\yolo_p3b_<split>\` (labels taken from the split JSON in our class order 0 RBC, 1 WBC, 2 Platelets); it never touches `data\yolo_pooled\`.

Smoke test (all 5 arms, pooled, seed 0, 1 epoch each; roughly 30-60 min, separate runs dir):

    set EXP_SEEDS=0
    set EXP_TRAIN_ARGS=--epochs-max 1
    set EXP_RUNS_DIR=runs_smoke_p3b
    scripts\run_exp.bat

Then one extra DC-CellNet run to exercise AMP plus gradient accumulation on CUDA (batch 4 x accum 2, own runs dir):

    set EXP_ARMS=d_both
    set EXP_SEEDS=0
    set EXP_TRAIN_ARGS=--epochs-max 1 --batch 4 --accum-steps 2
    set EXP_RUNS_DIR=runs_smoke_p3b_accum
    scripts\run_exp.bat

Check that every `runs_smoke_p3b\p3b_pooled_*_s0` has `summary_test.json`, and tell Claude if anything failed.

Then exercise the `.bat` resume branches (they must work before the unattended launch). Use the first smoke variables again (`set EXP_ARMS=`, `set EXP_SEEDS=0`, `set EXP_TRAIN_ARGS=--epochs-max 1`, `set EXP_RUNS_DIR=runs_smoke_p3b`):

1. Re-run `scripts\run_exp.bat` unchanged. Expected: one "done already, skipping" line per run and no training at all.
2. Delete one run's result, e.g. `del runs_smoke_p3b\p3b_pooled_frcnn_s0\summary_test.json`, and re-run. Expected: for that run only, "summary.json exists, skipping train" followed by predict/evaluate (no training), and `summary_test.json` is back; all other runs print "done already, skipping".

If either expectation fails, stop, do not start the real run, and report the log `runs_smoke_p3b\exp_runner.log`. Reset the variables before the real run (`set EXP_ARMS=`, `set EXP_SEEDS=`, `set EXP_TRAIN_ARGS=`, `set EXP_RUNS_DIR=`).

Out of memory (repeat any per-family recipe below for stage 2 too, see there): never shrink the effective batch (the runner skips finished runs, so re-running with only the OOM'd arm is safe). DC-CellNet arms: `set EXP_ARMS=d_both d_sigma circlenet` and `set EXP_TRAIN_ARGS=--batch 4 --accum-steps 2`. Faster R-CNN (protocol batch is already 4): `set EXP_ARMS=frcnn` as its own pass with `set EXP_TRAIN_ARGS=--batch 2 --accum-steps 2` (not `--batch 4 --accum-steps 2`, which would double the effective batch). YOLO: `set EXP_ARMS=yolo26m` and lower `--batch` only (e.g. `--batch 4`); Ultralytics `nbs` keeps the effective batch.

Before the real run (3-5 days): set Windows power to never sleep and pause Windows Update for the whole duration.

Real run, stage 1: pooled split, 5 arms x 3 seeds = 15 runs:

    set EXP_SPLITS=pooled
    scripts\run_exp.bat

Then summarise and apply the stop/go rule (spec section 3.2: `d_both` vs `circlenet` on pooled; continue to stage 2 only on GO):

    python scripts/summarize_loso.py --runs-dir runs --prefix p3b --data-root data --out reports/p3b_stage1.md --splits pooled

Stage 2: the four leave-one-source-out folds (60 more runs). If stage 1 needed a per-family OOM recipe (separate `EXP_ARMS` passes with their own `EXP_TRAIN_ARGS`), repeat exactly the same passes for stage 2 with `EXP_SPLITS` set to the LOSO folds; otherwise the default args OOM again and the runner halts.

    set EXP_SPLITS=loso_BCCD loso_BCDD loso_PBC loso_Raabin
    scripts\run_exp.bat

    python scripts/summarize_loso.py --runs-dir runs --prefix p3b --data-root data --out reports/p3b.md

Each summarize command writes the markdown report plus the same numbers as `.json` next to it (`reports/p3b_stage1.json`, `reports/p3b.json`) and prints the verdict lines; the last line is `pooled stop/go: d_both vs circlenet = GO / NO-GO / INCONCLUSIVE`. INCONCLUSIVE means a run (seed or LOSO fold) is missing or incomplete; the report's Warnings section names it. The per-run mAP bootstrap CI (`--map-ci-n 1000`, seed-0 run of every split x arm) is the slow part, roughly 2 min per pooled run and up to ~8 min per LOSO fold run on a CPU; add `--map-ci-n 0` for a quick look.

Send the results back. First check `runs\p3b_pooled_d_both_s0\preds_test.json` (and after stage 2 a LOSO one): DC-CellNet can write up to 1000 records per image, so LOSO predictions may reach 100+ MB per run, and the tar may be too large to send as is. Then (Git Bash or WSL, from `dccnet`):

    tar czf runs_p3b.tgz --ignore-failed-read --exclude="*_aborted_*" data/splits/pooled.json data/splits/loso_*.json runs/p3b_*/{summary.json,summary_test.json,sweep_val.json,preds_test.json,metrics.jsonl,results.csv} reports/p3b*.md reports/p3b*.json

YOLO runs have no `metrics.jsonl` (they write `results.csv`) and DC-CellNet/FRCNN runs have no `results.csv`; `--ignore-failed-read` turns those missing-file errors into warnings, and `--exclude` keeps renamed `*_aborted_*` run dirs out of the archive.

ETA (RTX 5060 Ti): DC-CellNet arms roughly 1.5-2 h per run at 150 epochs, YOLO26m roughly 1-1.5 h, Faster R-CNN roughly 2-3 h; stage 1 is about 25-35 h, stage 2 about 4x that. The estimate is rough; the smoke run's per-epoch times give the real figure.

## P3c — BCCD overlap relabel

1. Predict on the 204-image overlap set with the pooled YOLO26m weights: `python scripts\baselines\yolo_predict.py --weights <path to the pooled best.pt> --split data\splits\overlap_bccd.json --subset test --data-root data --conf 0.001 --out <path>\preds_bccd.json`.
2. `python scripts\prelabel_bccd.py --data-root data --preds <run>\preds_bccd.json` (merges GT with predictions of score >= 0.5; exports `data\bccd_relabel\`; refuses to overwrite existing `labels\` without `--force`; WARNING: `--force` regenerates all labels and destroys your manual edits).
3. X-AnyLabeling: open `data\bccd_relabel\images`, import the YOLO labels with `classes.txt`, work through `review_list.csv` top-down, and save as YOLO. Rule: label every RBC whose visible outline is more than half the cell, including cells cut by the image border.
4. Any time: `python scripts\import_bccd_relabel.py --data-root data --progress` prints `<changed>/<total> label files edited since pre-label`.
5. When done: `python scripts\import_bccd_relabel.py --data-root data` writes `data\splits\bccd_overlap_v2.json` (same image keys and order as `overlap_bccd.json`) and `data\bccd_relabel\import_stats.json`. Any missing or malformed label file is listed and nothing is written.
6. Evaluate the trained arms on the original (`overlap_bccd`) and relabelled (`bccd_overlap_v2`) BCCD overlap sets. Needs the `pooled` runs and the `loso_BCCD` runs (P3b stage 2: `loso_BCCD` never saw a BCCD image, so it gives the unseen-source overlap result) to exist first; the runner skips training and the main-split predictions of finished runs and only predicts and scores the extra splits (`preds_<x>.json`, `summary_<x>.json` in each run dir, scored at the run's own val-swept `best_thr`; finished extras are skipped on a re-run). A missing `data\splits\<x>.json` is a failure:

        set EXP_SPLITS=pooled loso_BCCD
        set EXP_EXTRA_SPLITS=overlap_bccd bccd_overlap_v2
        scripts\run_exp.bat

        python scripts/summarize_loso.py --runs-dir runs --prefix p3b --data-root data --out reports/p3c.md --splits pooled,loso_BCCD --extra overlap_bccd,bccd_overlap_v2

   Section 5 of the report has, per extra split and training split, the per-arm table, the `d_both`-vs-arm paired comparisons, and a label-bias table (mAP50-95 and RBC MAE on v1 vs v2 and v2 minus v1, plus the `yolo_origin_kept` / `n_prelabel_added` totals from `import_stats.json`).
7. Optional label-noise check: re-label 20 random v2 images from scratch. `python scripts\import_bccd_relabel.py --data-root data --sample-recheck 20 --seed 0` copies them to `data\bccd_relabel\recheck\images` with empty label files in `recheck\labels` (it refuses if `recheck\labels` exists; it never overwrites). Label them in X-AnyLabeling without looking at v2 (same rule as step 3), save as YOLO, then `python scripts\import_bccd_relabel.py --data-root data --recheck` prints the per-class F1 at IoU 0.5 between the two passes and stores it under `recheck` in `data\bccd_relabel\import_stats.json`.
