# DC-CellNet M0–M2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the workspace (M0), build the data pipeline + unified evaluator and re-evaluate YOLO26m through it (M1), and run the synthetic go/no-go experiment comparing CircleNet vs CircleNet+density vs CircleNet+density+L_cc (M2).

**Architecture:** A new `dccnet/` package inside `C:\Research` (repo `mickieppf/Blood-Cell-Detection`, branch `dccnet`). Pure, separately-tested modules for geometry, GT target generation, decoding, losses and evaluation; one CenterNet/CircleNet-style model with optional density head; one config-driven training script. Code + CPU tests are written/run from the Linux VM; GPU training runs on Windows via `python scripts/train.py --config ...`.

**Tech Stack:** Python 3.11, PyTorch ≥2.3 + torchvision, numpy, opencv-python, pycocotools, imagehash, PyYAML, pytest, ultralytics (baselines only).

**Spec:** `docs/specs/2026-10-26-dccellnet-design.md`

## Global Constraints

- Class indices everywhere in `dccnet`: `0 = RBC, 1 = WBC, 2 = Platelets` (old repo/TXL use other orders → map on load by name).
- Input 640×640 letterbox; output stride 4 → 160×160 maps.
- Box→circle: center = box center, `r = sqrt(w*h)/2`; box within 2 px of the image edge → `radius_valid = False`.
- Overlap bins per GT cell: number of other GT circles with `d < r_i + r_j` → 0 / 1 / 2 / 3 (3 means ≥3).
- Density GT: per-cell Gaussian, σ = r/3 (px), normalized so each cell sums to 1 **after** clipping to the map.
- Heatmap GT: Gaussian peak 1.0 at center cell, σ_h = max(r / (3·stride), 0.5) cells.
- L_cc: window `w = 32` cells, non-overlapping; stop-grad on density by default; SmoothL1.
- Loss weights default: `λ_r = 0.1, λ_o = 1, λ_d = 1, λ_cc = 0.5`; λ_cc = 0 for first 10 epochs then linear ramp over 5 epochs.
- Training: early stop when val mAP50-95 does not improve for 30 epochs; max 300 epochs.
- Augmentation default: hflip, vflip, rot90, scale ±20%, translate ±10%; stain/HSV aug off unless `aug.stain: true`.
- Dedupe: pHash 64-bit, duplicate if Hamming ≤ 6.
- LOSO validation: 15% of training-source images, stratified by source.
- Bootstrap: 1,000 resamples over test images, 95% percentile CI, seed 0.
- Every run dir `dccnet/runs/<exp>/` contains `config.yaml, git_hash.txt, metrics.jsonl, best.pt, summary.json`; only `best.pt` is kept (no `last.pt`).
- No file deletion in `C:\Research` without Mickie's explicit approval of the specific list.
- Thresholds / λ are tuned on validation only; test is evaluated once per final config.

## Review Focus

1. **Images with no cells of a class** (common: PBC/Raabin crops have 1 WBC, 0 platelets) — density, L_cc, counting and COCO eval must return finite values, not NaN/crash. → tests in Tasks 5, 8, 10.
2. **Cells cut by the image border** — no radius/cIoU loss, still counted; density still sums to 1 per cell. → tests in Tasks 2, 8, 10.
3. **Non-square originals (BCCD 640×480)** — letterbox must map circles back to original coordinates exactly before evaluation. → test in Task 12.
4. **Zero predictions for an image or a whole split** — evaluator returns 0 AP / full count error, not an exception. → test in Task 5.
5. **More objects than the per-image target buffer** (synthetic clumps) — `max_objs = 512`; exceeding raises `ValueError` rather than silently dropping cells. → test in Task 8.

---

## File Structure

```
dccnet/
  dccnet/__init__.py
  dccnet/config.py                 # YAML load with _base_ inheritance
  dccnet/types.py                  # Sample dataclass
  dccnet/datasets/geometry.py      # box↔circle, radius_valid, overlap levels, circle IoU
  dccnet/datasets/txl.py           # TXL-PBC loader (+ source metadata)
  dccnet/datasets/bccd_voc.py      # BCCD full VOC loader
  dccnet/datasets/splits.py        # pooled / LOSO splits, split-file IO
  dccnet/datasets/dedupe.py        # pHash + duplicate search
  dccnet/datasets/synthetic.py     # toy overlapping-disc generator (M2)
  dccnet/datasets/torch_dataset.py # CellDataset: letterbox + aug + targets
  dccnet/datasets/targets.py       # GT heatmap/radius/offset/density
  dccnet/models/{backbone,neck,heads,model}.py
  dccnet/losses/{focal,radius_ciou,density,count_consistency}.py
  dccnet/decode.py
  dccnet/eval/{export_json,coco_eval,counting,overlap_bins,bootstrap}.py
  dccnet/train/loop.py             # train/val loop, early stop, λ_cc schedule, run-dir writing
  configs/base.yaml, configs/exp_m2_*.yaml
  scripts/{check_env,prepare_data,train,predict,evaluate,compare_runs,disk_report}.py
  scripts/baselines/yolo_predict.py
  tests/…                          # pytest, CPU only
  requirements.txt, README.md
  data/ runs/                      # gitignored
```

---

## M0 — Workspace

### Task 0: Repo hygiene, env check, disk report

**Files:**
- Modify: `.gitignore`
- Create: `dccnet/requirements.txt`, `dccnet/scripts/check_env.py`, `dccnet/scripts/disk_report.py`, `dccnet/README.md`

**Interfaces:**
- Produces: `dccnet/scripts/check_env.py` (prints torch version, `cuda.is_available()`, GPU name; exit code 1 if CUDA unavailable); `disk_report.py <root> --top 40` → `dccnet/disk_report.md`.

- [ ] **Step 1:** Append to `.gitignore`: `dccnet/data/`, `dccnet/runs/`, `*.pt`, `*.zip`, `*.pdf`, `__pycache__/`, `*.cache`, `BCCD/dataset-master/`, `BCCD/dataset2-master/`, `BCCD/**/dataset/`, `BCCD/**/runs/`, `BCCD/YOLO/yolo_dataset/`.
- [ ] **Step 2:** Commit pre-existing work as its own snapshot: `git add -A` (now filtered by .gitignore), check `git status` shows only code/markdown/json/csv/png ≤ 5 MB each, then `git commit -m "chore: snapshot pre-dccnet work"`.
- [ ] **Step 3:** Write `requirements.txt` (versions as installed on Windows env `dccnet`; floor `torch>=2.3`) and `check_env.py`.
- [ ] **Step 4:** Write `disk_report.py` (walk tree, list largest files and folder-vs-zip duplicates such as `train_results_*.zip` next to the extracted folder). Run on `C:\Research`; send `disk_report.md` to Mickie. **Delete nothing**; deletion happens only on Mickie's approved list.
- [ ] **Step 5:** In the VM: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` plus the remaining requirements (minus ultralytics). Run `python -c "import torch, pycocotools, imagehash"` → no error.
- [ ] **Step 6 (Mickie, Windows):** `python dccnet/scripts/check_env.py` → prints `True` and `RTX 4070`.
- [ ] **Step 7:** Commit (`.gitignore`, `dccnet/requirements.txt`, scripts, README).

---

## M1 — Data pipeline and unified evaluator

### Task 1: Package skeleton, config, Sample type

**Files:** Create `dccnet/dccnet/{__init__,config,types}.py`, `dccnet/configs/base.yaml`, `dccnet/tests/test_config.py`, `dccnet/pytest.ini` (`testpaths = tests`, `pythonpath = .`)

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict` — if YAML has `_base_: other.yaml` (relative path), deep-merge child over base; `Sample` dataclass: `image_path: Path, source: str, width: int, height: int, boxes: np.ndarray (N,4 float32, xyxy px), labels: np.ndarray (N, int64)`.
- `base.yaml` holds every Global Constraint value under keys: `model.{backbone, pretrained, heads.density, repr}`, `loss.{lambda_r, lambda_o, lambda_d, lambda_cc, cc.{enabled, scope, window, stop_grad_density, warmup_epochs, ramp_epochs}}`, `decode.{score_thr, density_fill, low_thr, nms}`, `train.{epochs_max, patience, batch, lr, seed}`, `aug.{hflip, vflip, rot90, scale, translate, stain}`, `data.{img_size, stride}`.

- [ ] **Step 1:** Tests: `test_base_inherit` (child overrides `loss.lambda_cc: 0.0`, other keys inherited), `test_nested_merge` (dict merged, list replaced).
- [ ] **Step 2:** Run `pytest tests/test_config.py -v` → FAIL (import error).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(dccnet): config loader and Sample type`.

### Task 2: Geometry

**Files:** Create `dccnet/dccnet/datasets/geometry.py`, `dccnet/tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `boxes_to_circles(boxes: np.ndarray, img_w: int, img_h: int, edge_px: float = 2.0) -> tuple[np.ndarray (N,3) cx,cy,r, np.ndarray (N,) bool radius_valid]`
  - `circles_to_boxes(circles: np.ndarray) -> np.ndarray (N,4) xyxy` (`[cx−r, cy−r, cx+r, cy+r]`)
  - `overlap_levels(circles: np.ndarray) -> np.ndarray (N,) int in {0,1,2,3}`
  - `circle_iou(a: Tensor (...,3), b: Tensor (...,3)) -> Tensor (...)` closed form, differentiable, works on torch tensors.

- [ ] **Step 1:** Tests:
  - box `[10,10,30,50]` in 640×640 → circle `(20, 30, sqrt(20*40)/2 ≈ 14.142)`, valid True.
  - box `[0,100,20,120]` → valid False; box `[618,0,639,10]` in 640 wide → valid False (x2 ≥ 640−2, y1 ≤ 2).
  - `overlap_levels` for 4 circles r=10 at x=0,15,30,100 (y=0) → `[1,2,1,0]`; 5 mutually overlapping circles → all 3.
  - `circle_iou`: identical → 1.0; disjoint → 0.0; concentric r=1 and r=2 → 0.25 (±1e-6); gradient finite at identical circles.
  - zero circles input → empty outputs, no error.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. `circle_iou` uses the lens-area formula with clamps for containment/disjoint cases (`torch.where`), `eps=1e-7` inside `acos`/`sqrt`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(dccnet): circle geometry`.

### Task 3: Dataset loaders and splits

**Files:** Create `dccnet/dccnet/datasets/{txl,bccd_voc,splits}.py`, `dccnet/tests/test_loaders.py`, `dccnet/tests/fixtures/` (tiny fake datasets built in `tmp_path` by the tests)

**Interfaces:**
- Consumes: `Sample` (Task 1).
- Produces:
  - `load_txl(root: Path) -> tuple[list[Sample], dict[str, str]]` — reads YOLO labels, maps class names → indices, `source` ∈ `{"BCCD","BCDD","PBC","Raabin"}` from the dataset's metadata file; second return value = official split (image stem → `"train"|"val"|"test"`).
  - `load_bccd_voc(root: Path) -> list[Sample]` — `source="BCCD_full"`; names `RBC/WBC/Platelets` (case-insensitive).
  - `pooled_split(samples, official: dict[str,str]) -> dict[str, list[Sample]]`
  - `loso_split(samples, held_out: str, val_frac: float = 0.15, seed: int = 0) -> dict[str, list[Sample]]` — test = all of `held_out`; val stratified by source among the rest.
  - `write_split(split: dict[str, list[Sample]], path: Path)` / `read_split(path) -> dict[str, list[Sample]]` (JSON; image paths relative to `dccnet/data`).

- [ ] **Step 0 (inspect, no code):** List `dccnet/data/txl_pbc/` and `dccnet/data/bccd_full/`; record the actual layout (folders, class-name file, metadata file name and columns) at the top of `txl.py` docstring. If TXL has no per-image source column, stop and report.
- [ ] **Step 1:** Tests on fixture datasets: class mapping (`"Platelets"` in TXL → 2); YOLO-normalized → pixel xyxy exact; `loso_split(held_out="PBC")` → no PBC in train/val, all PBC in test, val fraction within ±1 image of 15% per source, deterministic for fixed seed; image with empty label file → Sample with N=0; split write/read round-trip equal.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(dccnet): TXL-PBC/BCCD loaders and splits`.

### Task 4: Dedupe and overlap-set construction

**Files:** Create `dccnet/dccnet/datasets/dedupe.py`, `dccnet/scripts/prepare_data.py`, `dccnet/tests/test_dedupe.py`

**Interfaces:**
- Produces: `phash_file(path: Path) -> int`; `find_duplicates(a: list[Path], b: list[Path], max_dist: int = 6) -> list[tuple[Path, Path, int]]`.
- `prepare_data.py` writes to `dccnet/data/splits/`: `pooled.json`, `loso_{BCCD,BCDD,PBC,Raabin}.json`, `overlap_bccd.json` (BCCD_full images with **no** duplicate in any TXL split), plus `dccnet/data/dedupe_report.csv` (all duplicate pairs across TXL, BCCD_full, old Roboflow set `C:\Research\blood-cell-detection-datatset`) and prints counts. Also writes Ultralytics `data_pooled.yaml` + txt image lists for baselines (class order 0 RBC, 1 WBC, 2 Platelets).

- [ ] **Step 1:** Tests: image vs. its JPEG-recompressed 0.9× resize → distance ≤ 6; two different synthetic images → > 6; `find_duplicates` returns the pair once.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `python scripts/prepare_data.py` on real data. Expected printout: TXL total 1,260 with sources 160/100/500/500; BCCD_full 364; overlap set size N_overlap. If N_overlap ≠ 172, record the number and the reason (from dedupe report) in `dccnet/README.md` — do not force 172.
- [ ] **Step 6:** Commit `feat(dccnet): dedupe and split preparation`.

### Task 5: Unified evaluator

**Files:** Create `dccnet/dccnet/eval/{export_json,coco_eval,counting,overlap_bins,bootstrap}.py`, `dccnet/scripts/evaluate.py`, `dccnet/tests/test_eval.py`

**Interfaces:**
- Consumes: `Sample`, `read_split`, geometry (Tasks 1–3).
- Produces:
  - Prediction record (JSON list): `{"image": str (relative path), "category_id": int, "bbox": [x, y, w, h], "score": float, "circle": [cx, cy, r] (optional)}`, coords in original image pixels.
  - `build_coco_gt(samples) -> dict`; `coco_map(samples, preds) -> {"mAP50": float, "mAP50_95": float, "per_class": {"RBC": float, "WBC": float, "Platelets": float}}`
  - `count_errors(samples, preds, score_thr: float) -> {"per_image": {cls: list[int signed error]}, "MAE": {cls: float}, "rel": {cls: float}}` (rel = Σ|err| / Σ gt, per class)
  - `overlap_metrics(samples, preds, score_thr, iou_thr=0.5) -> {bin: {"recall","precision","f1","n_gt"}}` — greedy score-ordered matching per class; each TP counts in its GT's bin; each FP is assigned to the bin of the nearest GT center of the same class (any class if none).
  - `bootstrap_ci(values: list[float], stat=np.mean, n=1000, seed=0) -> tuple[float, float]`; `paired_bootstrap_diff(a, b, ...) -> tuple[float, float]`
  - CLI: `python scripts/evaluate.py --split data/splits/pooled.json --subset test --preds preds.json --score-thr 0.3 --out summary.json` → summary with all of the above + CIs.

- [ ] **Step 1:** Tests: preds == GT (score 1) → mAP50 = mAP50_95 = 1.0, MAE 0; empty preds → mAP 0.0 and MAE = gt count, no exception; class with zero GT in split → its AP reported as `null`, mean over classes with GT; overlap bins on a hand-built 4-circle case (one missed cell in bin 2 → bin-2 recall 0.5); bootstrap of constant list → (c, c).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (pycocotools `COCOeval`, `iouType="bbox"`, maxDets `[1, 100, 1000]` because RBC counts exceed 100). **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(dccnet): unified evaluator`.

### Task 6: YOLO26m baseline through the evaluator (M1 gate)

**Files:** Create `dccnet/scripts/baselines/yolo_predict.py`, `dccnet/tests/test_yolo_adapter.py`

**Interfaces:**
- Consumes: prediction record format and `evaluate.py` (Task 5); `data_pooled.yaml` (Task 4).
- Produces: `yolo_results_to_records(results, image_rel_paths, class_map: dict[int,int]) -> list[dict]`; CLI `python scripts/baselines/yolo_predict.py --weights W --split S --subset test --conf 0.001 --out preds.json`.

- [ ] **Step 1:** Test `yolo_results_to_records` with a stub results object (xyxy → xywh, class remap) .
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5 (Mickie, Windows):** `yolo detect train model=yolo26m.pt data=dccnet/data/splits/data_pooled.yaml epochs=150 patience=100 imgsz=640 batch=16 seed=0 project=dccnet/runs name=yolo26m_pooled`.
- [ ] **Step 6 (Mickie):** run `yolo_predict.py` on val and test, then `evaluate.py` with score-thr chosen on val (max F1).
- [ ] **Step 7 (gate):** Claude checks `summary.json`: test mAP50 within roughly 0.95–0.99 and mAP50-95 roughly 0.80–0.90 (TXL-PBC published YOLO range). Outside that → debug evaluator/class mapping before M2.
- [ ] **Step 8:** Commit adapter + note results in `dccnet/README.md`.

---

## M2 — Model and synthetic go/no-go

### Task 7: Synthetic overlapping-disc generator

**Files:** Create `dccnet/dccnet/datasets/synthetic.py`, `dccnet/scripts/make_synthetic.py`, `dccnet/tests/test_synthetic.py`

**Interfaces:**
- Produces: `make_synthetic(out_dir: Path, n_images: int, seed: int, img_size: int = 640, clump_prob: float = 0.5) -> list[Sample]` with `source="SYN"`. Per image: 20–60 RBC discs (r ~ U(18, 28) px, pale red, darker rim), 0–2 WBC (r ~ U(35, 45), purple nucleus blob), 0–6 platelets (r ~ U(5, 9)); with probability `clump_prob` a cell is placed so its center is 0.6–1.4·r from an existing RBC (creates overlap levels 1–3+); Gaussian blur σ=1 + noise. Script writes `data/synthetic/` and `data/splits/synthetic.json` (400 train / 100 val / 200 test, seed 0).

- [ ] **Step 1:** Tests: same seed → identical image bytes and boxes; `overlap_levels` over 50 images has every bin 0–3 non-empty; all boxes inside image.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 8: GT targets

**Files:** Create `dccnet/dccnet/datasets/targets.py`, `dccnet/tests/test_targets.py`

**Interfaces:**
- Consumes: geometry (Task 2).
- Produces: `build_targets(circles: np.ndarray (N,3) in input-image px, labels: np.ndarray (N,), radius_valid: np.ndarray (N,), img_size: int = 640, stride: int = 4, num_classes: int = 3, max_objs: int = 512) -> dict[str, torch.Tensor]` with keys `heatmap (C,H,W)`, `density (C,H,W)`, `ind (K,) long` (flattened center index), `mask (K,) bool`, `log_r (K,)`, `offset (K,2)`, `r_valid (K,) bool`, `cls (K,) long`, `circle (K,3)`.

- [ ] **Step 1:** Tests:
  - single circle at (100.5, 200.25, r=20), class 0 → `heatmap[0]` max = 1.0 at cell (y=50, x=25); offset = (0.125, 0.0625); log_r = ln 20.
  - density per class sums to number of cells of that class (±1e-3), including a circle centered at (1, 1) (clipped Gaussian renormalized).
  - two overlapping circles same class → heatmap takes element-wise max (two separate peaks of 1.0).
  - zero circles → all-zero maps, mask all False.
  - 513 circles → `ValueError` mentioning `max_objs`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 9: Decoding

**Files:** Create `dccnet/dccnet/decode.py`, `dccnet/tests/test_decode.py`

**Interfaces:**
- Consumes: `build_targets` (tests only).
- Produces: `decode(heatmap: Tensor (B,C,H,W), log_r: Tensor (B,1,H,W), offset: Tensor (B,2,H,W), density: Tensor | None = None, score_thr: float = 0.3, stride: int = 4, density_fill: bool = False, window: int = 32, low_thr: float = 0.1, nms: str = "none", max_dets: int = 1000) -> list[np.ndarray (M,5)]` rows `[cls, cx, cy, r, score]` in input-image px.
  - Peaks: `heatmap == max_pool2d(heatmap, 3, 1, 1)` and `> score_thr`.
  - `density_fill`: per (class, window) where `#peaks < round(density.sum())`, add highest remaining local maxima with value `> low_thr` until equal.
  - `nms` ∈ `{"none","circle","box"}` (IoU 0.5; used only for ablation).

- [ ] **Step 1:** Tests:
  - round-trip: targets from 30 random non-identical circles → dense maps (write `log_r`/`offset` at `ind`) → `decode` recovers all centers within 1 px and r within 5%.
  - density fill: two peaks (0.9, 0.8) and a third local max 0.2 in one window, density sum 3.0, `score_thr=0.3` → 2 dets without fill, 3 with fill.
  - all-zero heatmap → empty arrays per image.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 10: Losses

**Files:** Create `dccnet/dccnet/losses/{focal,radius_ciou,density,count_consistency}.py`, `dccnet/tests/test_losses.py`

**Interfaces:**
- Consumes: `circle_iou` (Task 2), target dict (Task 8).
- Produces:
  - `focal_loss(pred: Tensor, gt: Tensor, alpha=2, beta=4) -> Tensor` (CenterNet penalty-reduced; normalized by #positives, clamp ≥1)
  - `radius_ciou_loss(pred_log_r, pred_offset, targets, stride=4) -> tuple[Tensor l1, Tensor ciou]` (only `mask & r_valid`; returns 0 if none)
  - `offset_loss(pred_offset, targets) -> Tensor`
  - `density_loss(pred: Tensor, gt: Tensor) -> Tensor` = pixel MSE + mean over (B,C) of |Σpred − Σgt|
  - `count_consistency_loss(heatmap: Tensor, density: Tensor, window: int = 32, scope: str = "local", stop_grad_density: bool = True) -> Tensor` — SmoothL1 between per-window Σ(H·peak_mask) and Σ D; `scope="global"` = one window per map.
  - `cc_weight(epoch: int, target: float, warmup: int = 10, ramp: int = 5) -> float`

- [ ] **Step 1:** Tests:
  - focal: pred == gt (clamped 1e-4) → near 0; all-zero gt → finite.
  - count consistency: heatmap with exactly 3 peaks of 1.0 in window 0, density summing to 3 → loss ≈ 0; density summing to 5 → loss > 0 and `heatmap.grad` non-zero at the 3 peak locations; with `stop_grad_density=True` → `density.grad` is None or all zeros; `scope="global"` on 160×160 equals local loss with `window=160`; maps with zero cells → loss 0, finite.
  - `cc_weight`: epochs 0–9 → 0; 10 → 0.0; 12 → 0.2 (target 0.5); ≥15 → 0.5.
  - radius loss with no valid objects → 0.0 (no NaN).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 11: Model

**Files:** Create `dccnet/dccnet/models/{backbone,neck,heads,model}.py`, `dccnet/tests/test_model.py`

**Interfaces:**
- Consumes: config keys `model.*` (Task 1).
- Produces: `DCCellNet(cfg: dict)` (nn.Module); `forward(x: Tensor (B,3,640,640)) -> dict` with `heatmap` (sigmoid, clamped [1e-4, 1−1e-4]), `log_r`, `offset`, `density` (ReLU; key absent when `model.heads.density` is false). Backbone: torchvision ResNet-50 (C2–C5), `pretrained` flag; neck: FPN-lite 128 channels, top-down fused to stride 4; heads: conv3×3(128)→ReLU→conv1×1. Heatmap bias init −2.19. `repr: box` and `backbone: convnext_t` are **not** in this plan (M3) — raise `NotImplementedError` with that message.

- [ ] **Step 1:** Tests (`pretrained=False`, CPU, input 1×3×640×640): output shapes `heatmap (1,3,160,160)`, `log_r (1,1,160,160)`, `offset (1,2,160,160)`, `density (1,3,160,160)`; density absent when disabled; one backward pass of the summed outputs gives finite grads.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 12: Torch dataset with letterbox + augmentation

**Files:** Create `dccnet/dccnet/datasets/torch_dataset.py`, `dccnet/tests/test_torch_dataset.py`

**Interfaces:**
- Consumes: `Sample`, geometry, `build_targets`.
- Produces: `CellDataset(samples: list[Sample], img_size=640, aug: dict | None = None, stride=4)`; `__getitem__ -> (image Tensor (3,640,640) float [0,1], targets dict, meta dict {"scale": float, "pad": (px, py), "orig_size": (w, h), "image": str})`; `unletterbox(circles: np.ndarray, meta) -> np.ndarray` (back to original px).

- [ ] **Step 1:** Tests: 640×480 image with box `[100,100,140,140]` → letterboxed circle center (120, 200) (pad_y = 80) and `unletterbox` returns (120, 120, 20) exactly; rot90 keeps radii and maps centers correctly; no aug → deterministic output; `aug.stain=False` leaves pixel values unchanged except resize.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 13: Training loop and CLI

**Files:** Create `dccnet/dccnet/train/loop.py`, `dccnet/scripts/train.py`, `dccnet/scripts/predict.py`, `dccnet/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `python scripts/train.py --config C [--seed S] [--device cuda]` → run dir per Global Constraints; `metrics.jsonl` one line per epoch `{epoch, loss_total, loss_focal, loss_r, loss_ciou, loss_o, loss_d, loss_cc, lambda_cc, val_mAP50, val_mAP50_95, val_rbc_count_mae}`; `summary.json` `{exp, seed, git_hash, best_epoch, val: {...}, test: null}`. `scripts/predict.py --run R --split S --subset test --score-thr T --out preds.json` writes evaluator records (unletterboxed). Optimizer AdamW lr 1.25e-4, wd 1e-4, cosine schedule, AMP on CUDA; total loss per spec §2.4; val every epoch with `decode` + `coco_map`; early stop per Global Constraints.

- [ ] **Step 1:** Smoke test: 6 synthetic images (Task 7), `epochs_max=2`, batch 2, CPU, `pretrained=False` → run dir contains the 5 files, `metrics.jsonl` has 2 lines, all losses finite; with `loss.cc.enabled=false` → `loss_cc` logged as 0.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit.

### Task 14: M2 experiment and go/no-go report

**Files:** Create `dccnet/configs/exp_m2_circlenet.yaml` (`heads.density: false`, `cc.enabled: false`), `exp_m2_density.yaml` (density on, cc off), `exp_m2_dccell.yaml` (density on, cc on); all `_base_: base.yaml`, data `data/splits/synthetic.json`, `pretrained: true`. Create `dccnet/scripts/compare_runs.py`.

**Interfaces:**
- Consumes: run dirs, `evaluate.py` summaries.
- Produces: `compare_runs.py --runs runs/exp_m2_* --out reports/m2_go_no_go.md` → table per config (mean over seeds): test mAP50-95, RBC count MAE with bootstrap CI, recall per overlap bin, plus paired bootstrap of RBC |count error| (dccell − circlenet).

- [ ] **Step 1:** Test `compare_runs` on two fake run dirs with known summaries → table values match.
- [ ] **Step 2–4:** FAIL → implement → PASS. Commit.
- [ ] **Step 5 (Mickie, Windows):** for each of the 3 configs and seeds 0,1,2: `python scripts/train.py --config configs/exp_m2_X.yaml --seed S`, then `predict.py` + `evaluate.py` on test (score-thr from val).
- [ ] **Step 6 (gate):** Claude runs `compare_runs.py`. **GO** if `exp_m2_dccell` has lower RBC count MAE than `exp_m2_circlenet` with the paired-bootstrap 95% CI of the difference entirely below 0 **and** higher recall in bin ≥2. Otherwise **NO-GO**: write findings and stop for discussion before M3.
- [ ] **Step 7:** Commit report; write summary to Project doc.
