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
