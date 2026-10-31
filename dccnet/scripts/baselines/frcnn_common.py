"""Shared pieces of the torchvision Faster R-CNN baseline (model, dataset, records, schedule).

Labels: our classes 0 RBC, 1 WBC, 2 Platelets map to torchvision labels 1..3 (0 = background).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dccnet.eval.export_json import image_key  # noqa: E402
from dccnet.types import Sample  # noqa: E402

NUM_CLASSES = 4  # background + RBC, WBC, Platelets
MIN_BOX_PX = 1.0
WARMUP_FACTOR = 0.001
WARMUP_ITERS = 1000
MILESTONES = (32, 42)
GAMMA = 0.1


def build_frcnn(pretrained: bool = True, min_size: int = 800, max_size: int = 1333):
    """``fasterrcnn_resnet50_fpn`` (COCO weights if ``pretrained``, else fully random and download-free)
    with a 4-way box predictor."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, weights_backbone=None,
                                    min_size=min_size, max_size=max_size,
                                    box_detections_per_img=300, box_score_thresh=0.001)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)


class SplitDetDataset(torch.utils.data.Dataset):
    """``(image float [3,H,W] in 0..1, {"boxes": float32 (N,4) xyxy, "labels": int64 = class + 1})``.

    Boxes narrower or shorter than 1 px are dropped (torchvision rejects non-positive sizes); images
    without boxes yield ``boxes`` of shape (0, 4). ``train=True`` adds a horizontal flip with p=0.5.
    """

    def __init__(self, samples: list[Sample], train: bool) -> None:
        self.samples = list(samples)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        rgb = _load_rgb(s.image_path)
        boxes = np.asarray(s.boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(s.labels, dtype=np.int64).reshape(-1)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_PX) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_PX)
        boxes, labels = boxes[keep], labels[keep]
        if self.train and random.random() < 0.5:
            w = rgb.shape[1]
            rgb = rgb[:, ::-1]
            boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], axis=1).reshape(-1, 4)
        target = {"boxes": torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4),
                  "labels": torch.from_numpy(labels + 1)}
        return _to_tensor(rgb), target


def collate(batch):
    return tuple(zip(*batch))


def outputs_to_records(outputs: list[dict], keys: list[str]) -> list[dict]:
    """torchvision detection outputs (``outputs[i]`` <-> ``keys[i]``) -> evaluator records (xywh, our ids)."""
    if len(outputs) != len(keys):
        raise ValueError(f"{len(outputs)} outputs for {len(keys)} image keys")
    records: list[dict] = []
    for out, key in zip(outputs, keys):
        boxes = out["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
        labels = out["labels"].detach().cpu().numpy().reshape(-1)
        scores = out["scores"].detach().float().cpu().numpy().reshape(-1)
        for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
            records.append({"image": key, "category_id": int(lab) - 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(sc)})
    return records


@torch.no_grad()
def predict_split(model, samples: list[Sample], data_root: Path, device, batch: int = 4) -> list[dict]:
    """Records for ``samples`` (original pixels; score >= the model's box_score_thresh)."""
    device = torch.device(device)
    model.eval()
    ds = SplitDetDataset(samples, train=False)
    records: list[dict] = []
    for i in range(0, len(samples), batch):
        idx = range(i, min(i + batch, len(samples)))
        images = [ds[j][0].to(device) for j in idx]
        outputs = model(images)
        records += outputs_to_records(outputs, [image_key(samples[j], data_root) for j in idx])
    return records


def lr_factor(step: int, epoch: int, warmup_iters: int = WARMUP_ITERS, milestones=MILESTONES,
              gamma: float = GAMMA, warmup_factor: float = WARMUP_FACTOR) -> float:
    """LR multiplier after ``step`` optimizer steps in (0-based) ``epoch``: LinearLR warmup from
    ``warmup_factor`` to 1 over ``warmup_iters`` steps in epoch 0 only (torchvision reference), times
    MultiStepLR ``gamma`` per milestone reached."""
    w = 1.0
    if epoch == 0 and step < warmup_iters:
        alpha = step / warmup_iters
        w = warmup_factor * (1.0 - alpha) + alpha
    return w * gamma ** sum(int(epoch) >= m for m in milestones)
