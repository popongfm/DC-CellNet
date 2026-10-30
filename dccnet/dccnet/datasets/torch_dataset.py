"""Torch dataset for DC-CellNet: letterbox to a square, augmentation, GT targets, collate.

Conventions (binding for training and evaluation):

* Letterbox: ``s = img_size / max(w, h)``; resize to ``(round(w*s), round(h*s))`` (INTER_LINEAR),
  pad CENTERED with (114, 114, 114): ``pad = ((S - new_w)//2, (S - new_h)//2)``.
  ``meta = {"scale", "pad", "orig_size": (w, h), "image"}``.
* ``radius_valid`` is computed on the ORIGINAL image, then circles are mapped
  ``cx' = cx*s + pad_x, cy' = cy*s + pad_y, r' = r*s``.
* Augmentation (only when ``aug`` is given): optional stain (HSV) jitter, hflip / vflip (p=0.5),
  rot90 (k uniform in 0..3, CCW), then a random affine (scale about the center + translation).
  Cells whose center ends up outside ``[0, S)`` are dropped; cells whose circle crosses the image
  edge get ``radius_valid=False``.
* RNG: ``np.random.default_rng([torch.initial_seed() % 2**32, epoch, index])`` per ``__getitem__``;
  call ``set_epoch(epoch)`` each epoch so augmentation differs across epochs (default epoch 0).
* Cells with ``r' < 0.5`` px (after letterbox / augmentation) are dropped, so zero-area boxes never
  reach ``build_targets``.
* Targets: ``build_targets(..., stride=stride, heatmap_sigma=..., heatmap_divisor=..., heatmap_shape=...)``
  (output stride, heatmap width rule / divisor / shape; the training loop passes ``data.stride`` /
  ``data.heatmap_sigma`` / ``data.heatmap_divisor`` / ``data.heatmap_shape``). Called on the circles that
  survive augmentation and the small-radius drop, so neighbour statistics (adaptive / knn3 / mcnn / repel)
  only see cells actually present in the augmented image.
* ``unletterbox`` inverts only the letterbox mapping (evaluation never augments).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dccnet.datasets.geometry import boxes_to_circles
from dccnet.datasets.targets import build_targets, check_heatmap_divisor, check_heatmap_shape, check_heatmap_sigma
from dccnet.types import Sample

_PAD_VALUE = 114
_MIN_RADIUS = 0.5


def unletterbox(circles: np.ndarray, meta: dict) -> np.ndarray:
    """Map (cx, cy, r) circles from letterboxed input px back to original image px."""
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    s = float(meta["scale"])
    px, py = meta["pad"]
    out = np.stack([(c[:, 0] - px) / s, (c[:, 1] - py) / s, c[:, 2] / s], axis=1)
    return out.astype(np.float32)


def _hsv_jitter(img_bgr: np.ndarray, rng, h: float = 0.015, s: float = 0.4, v: float = 0.3) -> np.ndarray:
    """YOLO-style multiplicative HSV jitter on a uint8 BGR image."""
    r = rng.uniform(-1.0, 1.0, 3) * np.array([h, s, v]) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge([cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class CellDataset(Dataset):
    def __init__(self, samples: list[Sample], img_size: int = 640, aug: dict | None = None, stride: int = 4,
                 heatmap_sigma: str = "radius", heatmap_divisor: float = 4.0, heatmap_shape: str = "gaussian"):
        self.samples = list(samples)
        self.img_size = int(img_size)
        self.aug = aug
        self.stride = int(stride)
        self.heatmap_sigma = check_heatmap_sigma(heatmap_sigma)
        self.heatmap_divisor = check_heatmap_divisor(heatmap_divisor)
        self.heatmap_shape = check_heatmap_shape(heatmap_shape)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _make_rng(self, index: int):
        return np.random.default_rng([torch.initial_seed() % 2**32, int(self.epoch), int(index)])

    def __getitem__(self, index: int):
        sample = self.samples[index]
        S = self.img_size
        img = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"cannot read image: {sample.image_path}")
        h, w = img.shape[:2]

        # radius_valid on the ORIGINAL image, then letterbox the circles
        circles, valid = boxes_to_circles(sample.boxes, w, h)
        labels = np.asarray(sample.labels, dtype=np.int64).reshape(-1)

        s = S / max(w, h)
        new_w, new_h = round(w * s), round(h * s)
        pad_x, pad_y = (S - new_w) // 2, (S - new_h) // 2
        canvas = np.full((S, S, 3), _PAD_VALUE, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        c = circles.astype(np.float64)
        c = np.stack([c[:, 0] * s + pad_x, c[:, 1] * s + pad_y, c[:, 2] * s], axis=1)
        valid = valid.copy()

        if self.aug is not None:
            canvas, c, labels, valid = self._augment(canvas, c, labels, valid, self._make_rng(index))

        small = c[:, 2] < _MIN_RADIUS
        if small.any():
            c, labels, valid = c[~small], labels[~small], valid[~small]

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0)
        targets = build_targets(c, labels, valid, img_size=S, stride=self.stride,
                                heatmap_sigma=self.heatmap_sigma, heatmap_divisor=self.heatmap_divisor,
                                heatmap_shape=self.heatmap_shape)
        meta = {
            "scale": float(s),
            "pad": (int(pad_x), int(pad_y)),
            "orig_size": (int(w), int(h)),
            "image": str(sample.image_path),
        }
        return image, targets, meta

    def _augment(self, img, c, labels, valid, rng):
        aug, S = self.aug, self.img_size
        if aug.get("stain", False):
            img = _hsv_jitter(img, rng)
        if aug.get("hflip", False) and rng.random() < 0.5:
            img = img[:, ::-1]
            c = np.stack([S - c[:, 0], c[:, 1], c[:, 2]], axis=1)
        if aug.get("vflip", False) and rng.random() < 0.5:
            img = img[::-1]
            c = np.stack([c[:, 0], S - c[:, 1], c[:, 2]], axis=1)
        if aug.get("rot90", False):
            k = int(rng.integers(0, 4))
            img = np.rot90(img, k)  # CCW; continuous coords (x, y) -> (y, S - x) per quarter turn
            for _ in range(k):
                c = np.stack([c[:, 1], S - c[:, 0], c[:, 2]], axis=1)
        img = np.ascontiguousarray(img)

        scale_amp = float(aug.get("scale", 0.0))
        trans_amp = float(aug.get("translate", 0.0))
        sc = float(rng.uniform(1.0 - scale_amp, 1.0 + scale_amp))
        tx = float(rng.uniform(-trans_amp, trans_amp)) * S
        ty = float(rng.uniform(-trans_amp, trans_amp)) * S
        if sc != 1.0 or tx != 0.0 or ty != 0.0:
            half = S / 2.0
            # circles live in pixel-edge coordinates; cv2 uses pixel-center coordinates (shift 0.5)
            M = np.array(
                [
                    [sc, 0.0, (1 - sc) * half + tx + 0.5 * sc - 0.5],
                    [0.0, sc, (1 - sc) * half + ty + 0.5 * sc - 0.5],
                ],
                dtype=np.float64,
            )
            img = cv2.warpAffine(
                img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
            )
            c = np.stack([sc * (c[:, 0] - half) + half + tx, sc * (c[:, 1] - half) + half + ty, c[:, 2] * sc], axis=1)

        # drop cells whose center left the image; flag cells crossing the edge
        keep = (c[:, 0] >= 0) & (c[:, 0] < S) & (c[:, 1] >= 0) & (c[:, 1] < S)
        c, labels, valid = c[keep], labels[keep], valid[keep]
        crossing = (c[:, 0] - c[:, 2] < 0) | (c[:, 1] - c[:, 2] < 0) | (c[:, 0] + c[:, 2] > S) | (c[:, 1] + c[:, 2] > S)
        valid = valid & ~crossing
        return img, c, labels, valid


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    metas = [b[2] for b in batch]
    return images, targets, metas
