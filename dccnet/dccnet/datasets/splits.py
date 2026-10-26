from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dccnet.types import Sample


def pooled_split(samples: list[Sample], official: dict[str, str]) -> dict[str, list[Sample]]:
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        out[official[s.image_path.stem]].append(s)
    return out


def loso_split(samples: list[Sample], held_out: str, val_frac: float = 0.15,
               seed: int = 0) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    out["test"] = [s for s in samples if s.source == held_out]
    if not out["test"]:
        raise ValueError(f"No samples with source {held_out!r}")
    for src in sorted({s.source for s in samples} - {held_out}):
        group = [s for s in samples if s.source == src]
        n_val = round(val_frac * len(group))
        val_idx = set(rng.permutation(len(group))[:n_val].tolist())
        for i, s in enumerate(group):
            out["val" if i in val_idx else "train"].append(s)
    return out


def write_split(split: dict[str, list[Sample]], path: Path, data_root: Path) -> None:
    data_root = Path(data_root)
    obj = {}
    for name, items in split.items():
        obj[name] = [{
            "image": Path(s.image_path).relative_to(data_root).as_posix(),
            "source": s.source, "width": s.width, "height": s.height,
            "boxes": np.asarray(s.boxes, dtype=np.float32).reshape(-1, 4).tolist(),
            "labels": np.asarray(s.labels).tolist(),
        } for s in items]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj))


def read_split(path: Path, data_root: Path) -> dict[str, list[Sample]]:
    obj = json.loads(Path(path).read_text())
    return {name: [Sample(Path(data_root) / d["image"], d["source"], d["width"], d["height"],
                          np.asarray(d["boxes"], dtype=np.float32).reshape(-1, 4),
                          np.asarray(d["labels"], dtype=np.int64)) for d in items]
            for name, items in obj.items()}
