"""Local count-consistency loss L_cc (DC-CellNet contribution C1, spec §2.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_SCOPES = ("local", "global")


def _tile_sums(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """Sum (B,C,H,W) over non-overlapping wh x ww tiles from (0,0); bottom/right are
    zero-padded to the next multiple, so the last tile holds the partial remainder.
    Returns (B,C,nH*nW)."""
    B, C, H, W = x.shape
    ph = (-H) % wh
    pw = (-W) % ww
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    nH, nW = (H + ph) // wh, (W + pw) // ww
    return x.reshape(B, C, nH, wh, nW, ww).sum(dim=(3, 5)).reshape(B, C, nH * nW)


def count_consistency_loss(
    heatmap: torch.Tensor,
    density: torch.Tensor,
    window: int = 32,
    scope: str = "local",
    stop_grad_density: bool = True,
) -> torch.Tensor:
    """SmoothL1 between per-window soft detection counts and density counts.

    For image b, class c, tile k:
      * ``P = 1[H == maxpool3x3(H)]`` -- peak mask, computed under no_grad (no gradient);
      * ``N_det = sum_{x in k} H(x) * P(x)`` -- soft count; gradient flows only into
        the heatmap values at peaks;
      * ``N_den = sum_{x in k} D(x)``, detached when ``stop_grad_density`` (default:
        D is supervised by its own GT; L_cc only pushes H to match it);
      * ``L_cc = mean_{b,c,k} SmoothL1(N_det, N_den; beta=1)``.

    Tiles are non-overlapping ``window`` x ``window`` cells starting at (0,0); if H or W
    is not divisible the map is zero-padded bottom/right (the last tile is the partial
    remainder). ``scope="global"`` uses a single tile covering the whole map.

    Note: every local maximum counts, including background maxima (and plateaus, e.g.
    cells sitting at the sigmoid clamp floor). They contribute their (small) heatmap
    values to N_det. This is intended: the loss pushes spurious peaks down and pushes
    under-separated clumps (fewer peaks than density mass) to split into more peaks.

    ``heatmap`` (B,C,H,W) sigmoided, ``density`` (B,C,H,W) >= 0; both cast to float32.
    Returns a 0-dim float32 tensor.
    """
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    heatmap = heatmap.float()
    density = density.float()
    H, W = heatmap.shape[-2:]
    if scope == "global":
        wh, ww = H, W
    else:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        wh = ww = int(window)

    with torch.no_grad():
        peak = (heatmap == F.max_pool2d(heatmap, 3, stride=1, padding=1)).float()
    n_det = _tile_sums(heatmap * peak, wh, ww)
    n_den = _tile_sums(density, wh, ww)
    if stop_grad_density:
        n_den = n_den.detach()
    return F.smooth_l1_loss(n_det, n_den, beta=1.0)


def cc_weight(epoch: int, target: float, warmup: int = 10, ramp: int = 5) -> float:
    """λ_cc schedule: 0 for ``epoch < warmup``; then ``target * min(1, (epoch - warmup) / ramp)``
    (epoch 10 -> 0, 12 -> 0.4*target, >= 15 -> target with the defaults)."""
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(target)
    return float(target) * min(1.0, (epoch - warmup) / ramp)
