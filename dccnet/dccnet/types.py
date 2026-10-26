from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    """One annotated image. Classes: 0=RBC, 1=WBC, 2=Platelets."""

    image_path: Path
    source: str
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) float32, xyxy in original pixels
    labels: np.ndarray  # (N,) int64
