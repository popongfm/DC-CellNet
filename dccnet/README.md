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
