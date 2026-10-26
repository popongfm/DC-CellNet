"""Percentile bootstrap over images (95% CI)."""
from __future__ import annotations

import numpy as np


def _ci(draws: np.ndarray) -> tuple[float, float]:
    draws = np.asarray(draws, dtype=np.float64)
    if np.all(np.isnan(draws)):
        return (float("nan"), float("nan"))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(values, stat=np.mean, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of ``stat`` over images resampled with replacement.

    ``values`` is indexed along axis 0 (one entry/row per image), so a 2-D array can carry
    several per-image quantities for ratio statistics.
    """
    v = np.asarray(values)
    if len(v) == 0:
        raise ValueError("bootstrap_ci: empty values")
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        draws[i] = stat(v[rng.integers(0, len(v), size=len(v))])
    return _ci(draws)


def paired_bootstrap_diff(a, b, stat=np.mean, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """CI of stat(a) - stat(b), resampling the same image indices for both."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError(f"paired_bootstrap_diff: length mismatch {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("paired_bootstrap_diff: empty values")
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(a), size=len(a))
        draws[i] = stat(a[idx]) - stat(b[idx])
    return _ci(draws)
