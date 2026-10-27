"""Local count-consistency loss L_cc (DC-CellNet contribution C1, spec §2.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_SCOPES = ("local", "global")
_COUNT_FNS = ("sigmoid", "linear")


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


def _tile_grid(H: int, W: int, window: int, scope: str) -> tuple[int, int, int, int]:
    """(wh, ww, nTy, nTx) of the L_cc tiling (shared by the loss and ``oracle_tile_counts``)."""
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    if scope == "global":
        return H, W, 1, 1
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    w = int(window)
    return w, w, -(-H // w), -(-W // w)


def oracle_tile_counts(
    targets: dict[str, torch.Tensor],
    num_classes: int,
    H: int,
    W: int,
    window: int,
    scope: str,
) -> torch.Tensor:
    """Ground-truth center counts per class per tile, float32 (B, C, nTy, nTx).

    Uses ``targets["ind"]`` (``iy = ind // W``, ``ix = ind % W``), ``targets["mask"]`` (slots
    with mask False are ignored) and ``targets["cls"]``. Tiling is identical to
    ``count_consistency_loss``: non-overlapping ``window`` tiles from (0,0), the remainder
    tile is partial; ``scope="global"`` is a single tile.
    """
    wh, ww, nty, ntx = _tile_grid(H, W, window, scope)
    ind, cls, mask = targets["ind"].long(), targets["cls"].long(), targets["mask"].bool()
    B = ind.shape[0]
    tile = (ind // W // wh) * ntx + (ind % W) // ww
    flat = (cls * (nty * ntx) + tile).clamp_(0, num_classes * nty * ntx - 1)
    out = torch.zeros(B, num_classes * nty * ntx, dtype=torch.float32, device=ind.device)
    out.scatter_add_(1, flat, mask.float())
    return out.view(B, num_classes, nty, ntx)


def count_consistency_loss(
    heatmap: torch.Tensor,
    density: torch.Tensor | None = None,
    window: int = 32,
    scope: str = "local",
    stop_grad_density: bool = True,
    peak_thr: float = 0.01,
    count_fn: str = "sigmoid",
    count_center: float = 0.3,
    count_temp: float = 0.05,
    target_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """SmoothL1 between per-window soft detection counts and density counts.

    For image b, class c, tile k:
      * ``P = 1[(H == maxpool3x3(H)) & (H > peak_thr)]`` -- peak mask, computed under
        no_grad (no gradient);
      * ``N_det = sum_{x in k} n(H(x)) * P(x)`` -- soft count; gradient flows only into
        the heatmap values at peaks. Per-peak contribution ``n`` (R21):
        ``count_fn="sigmoid"`` (default): ``n(H) = sigmoid((H - count_center) / count_temp)``;
        ``count_fn="linear"``: ``n(H) = H`` (the pre-R21 behaviour).
        Why sigmoid: L_cc must count *detections*, not sum peak heights. With ``n = H`` a
        tile of three confident cells (H = 0.95) counts 2.85 against a density of 3, so
        L_cc keeps pushing correct peaks toward 1 and, worse, a candidate at 0.25 counts as
        a quarter cell and the upward pressure is spread over every peak in the tile. With a
        steep sigmoid around the decode threshold (center 0.3, temp 0.05) a confident peak
        counts ~1 and is left alone (saturated, ~zero gradient), while a candidate near the
        threshold sits on the steep part and receives almost all of the gradient -- exactly
        the split candidates L_cc exists to grow (or, in an over-counted tile, suppress).
      * ``N_den = sum_{x in k} D(x)``, detached when ``stop_grad_density`` (default:
        D is supervised by its own GT; L_cc only pushes H to match it);
      * ``L_cc = mean_{b,c,k} SmoothL1(N_det, N_den; beta=1)``.

    Tiles are non-overlapping ``window`` x ``window`` cells starting at (0,0); if H or W
    is not divisible the map is zero-padded bottom/right (the last tile is the partial
    remainder). ``scope="global"`` uses a single tile covering the whole map.

    Peak threshold ``peak_thr`` (tau, default 0.01): without it every background local
    maximum -- including whole plateaus of cells at the 1e-4 sigmoid clamp floor --
    adds its value to N_det. Summed over a 32x32 tile this background mass is far from
    negligible (~0.1 at the floor, ~0.2-1.7 for background logits of -7..-5) and it
    biases the gradient on the real peaks even when the real counts match. Local maxima
    with ``H <= tau`` are therefore excluded (they receive no gradient and add nothing;
    see ``cc_background_mass`` for a logged diagnostic). tau is deliberately small
    (0.01) rather than e.g. 0.1: split candidates inside an under-separated clump start
    out weak, below ``decode.low_thr``, and must still receive the upward gradient that
    lets L_cc grow them into detectable peaks. Peaks above tau that are spurious are
    still pushed down where a tile is over-counted.

    ``heatmap`` (B,C,H,W) sigmoided, ``density`` (B,C,H,W) >= 0; both cast to float32.
    Instead of ``density`` the target may be given as precomputed per-tile counts
    ``target_counts`` (B,C,nTy,nTx), e.g. from ``oracle_tile_counts`` (M2c oracle test);
    exactly one of ``density`` / ``target_counts`` must be passed. Returns a 0-dim float32 tensor.
    """
    if (density is None) == (target_counts is None):
        raise ValueError("pass exactly one of density / target_counts")
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    if count_fn not in _COUNT_FNS:
        raise ValueError(f"count_fn must be one of {_COUNT_FNS}, got {count_fn!r}")
    if count_fn == "sigmoid" and not count_temp > 0:
        raise ValueError(f"count_temp must be > 0, got {count_temp}")
    heatmap = heatmap.float()
    H, W = heatmap.shape[-2:]
    wh, ww, nty, ntx = _tile_grid(H, W, window, scope)

    with torch.no_grad():
        local_max = heatmap == F.max_pool2d(heatmap, 3, stride=1, padding=1)
        peak = (local_max & (heatmap > peak_thr)).float()
    if count_fn == "sigmoid":
        contrib = torch.sigmoid((heatmap - float(count_center)) / float(count_temp))
    else:
        contrib = heatmap
    n_det = _tile_sums(contrib * peak, wh, ww)
    if target_counts is not None:
        n_den = target_counts.float().detach().reshape(heatmap.shape[0], heatmap.shape[1], nty * ntx)
    else:
        n_den = _tile_sums(density.float(), wh, ww)
        if stop_grad_density:
            n_den = n_den.detach()
    return F.smooth_l1_loss(n_det, n_den, beta=1.0)


@torch.no_grad()
def cc_background_mass(heatmap: torch.Tensor, peak_thr: float = 0.01) -> torch.Tensor:
    """Diagnostic (logging only): mean over (B,C) of the summed heatmap value of local
    maxima with ``H <= peak_thr`` -- the background mass excluded from L_cc. Detached
    0-dim float32 tensor."""
    heatmap = heatmap.detach().float()
    local_max = heatmap == F.max_pool2d(heatmap, 3, stride=1, padding=1)
    bg = heatmap * (local_max & (heatmap <= peak_thr)).float()
    return bg.sum(dim=(-2, -1)).mean()


def cc_weight(epoch: int, target: float, warmup: int = 10, ramp: int = 5) -> float:
    """λ_cc schedule: 0 for ``epoch < warmup``; then ``target * min(1, (epoch - warmup) / ramp)``
    (epoch 10 -> 0, 12 -> 0.4*target, >= 15 -> target with the defaults)."""
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(target)
    return float(target) * min(1.0, (epoch - warmup) / ramp)
