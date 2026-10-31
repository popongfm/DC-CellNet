"""Export a split (our class order, split-JSON labels) as an Ultralytics YOLO dataset."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from dccnet.types import Sample

CLASS_NAMES = {0: "RBC", 1: "WBC", 2: "Platelets"}


def write_yolo(split: dict[str, list[Sample]], out: Path, yaml_name: str = "data.yaml") -> Path:
    if out.exists():
        shutil.rmtree(out)
    for part, items in split.items():
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        for s in items:
            name = f"{s.source}_{s.image_path.name}"
            shutil.copy2(s.image_path, out / "images" / part / name)
            lines = []
            for (x0, y0, x1, y1), c in zip(np.asarray(s.boxes).reshape(-1, 4), s.labels):
                cx, cy = (x0 + x1) / 2 / s.width, (y0 + y1) / 2 / s.height
                bw, bh = (x1 - x0) / s.width, (y1 - y0) / s.height
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / part / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": CLASS_NAMES}
    yaml_path = out / yaml_name
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return yaml_path
