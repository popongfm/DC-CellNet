"""Run a trained Ultralytics YOLO model over a split subset and write evaluator prediction records.

Predicts on the ORIGINAL sample images listed in the split JSON (so records key via ``image_key``).

Usage:
    python scripts/baselines/yolo_predict.py --weights runs/yolo26m_pooled/weights/best.pt \
        --split data/splits/pooled.json --subset test --data-root data --conf 0.001 \
        --imgsz 640 --batch 8 --device 0 --max-det 1000 --out runs/yolo26m_pooled/preds_test.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.export_json import image_key, save_records  # noqa: E402

DEFAULT_CLASS_MAP = {0: 0, 1: 1, 2: 2}


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def yolo_results_to_records(results, image_keys: list[str], class_map: dict[int, int] | None = None) -> list[dict]:
    """Convert Ultralytics results (``results[i]`` <-> ``image_keys[i]``) to prediction records.

    Reads ``r.boxes.xyxy`` / ``.conf`` / ``.cls`` (torch tensors or numpy); emits xywh in original pixels.
    """
    class_map = DEFAULT_CLASS_MAP if class_map is None else class_map
    if len(results) != len(image_keys):
        raise ValueError(f"{len(results)} results for {len(image_keys)} image keys")
    records: list[dict] = []
    for r, key in zip(results, image_keys):
        xyxy = _to_numpy(r.boxes.xyxy).reshape(-1, 4)
        conf = _to_numpy(r.boxes.conf).reshape(-1)
        cls = _to_numpy(r.boxes.cls).reshape(-1)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            records.append({"image": key, "category_id": int(class_map[int(k)]),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(c)})
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--subset", choices=["train", "val", "test"], required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from ultralytics import YOLO  # lazy: not needed by the unit tests

    split = read_split(args.split, args.data_root)
    if args.subset not in split:
        ap.error(f"subset {args.subset!r} not in split (has {sorted(split)})")
    samples = split[args.subset]
    model = YOLO(str(args.weights))
    records: list[dict] = []
    for i in range(0, len(samples), args.batch):
        chunk = samples[i:i + args.batch]
        results = model.predict([str(s.image_path) for s in chunk], conf=args.conf, imgsz=args.imgsz,
                                device=args.device, max_det=args.max_det, verbose=False)
        records += yolo_results_to_records(results, [image_key(s, args.data_root) for s in chunk],
                                           DEFAULT_CLASS_MAP)
        print(f"{min(i + args.batch, len(samples))}/{len(samples)}", end="\r", flush=True)
    save_records(records, args.out)
    print(f"\nwrote {len(records)} detections for {len(samples)} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
