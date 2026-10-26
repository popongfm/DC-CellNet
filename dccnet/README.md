# dccnet

DC-CellNet research code for blood-smear cell detection (design spec and plan in `docs/`).

## Setup
    pip install -r dccnet/requirements.txt
    python dccnet/scripts/check_env.py   # expects CUDA True and the GPU name

## Scripts
- `scripts/check_env.py`: prints torch version, CUDA availability and GPU name; exit 1 if no CUDA.
- `scripts/disk_report.py <root> --top 40 --out report.md`: read-only disk usage report
  (largest files, per-folder totals, zip-next-to-folder duplicates). Standard library only.

`dccnet/data/` and `dccnet/runs/` are git-ignored.

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
