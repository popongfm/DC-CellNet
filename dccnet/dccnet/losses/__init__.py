"""DC-CellNet losses and the total-loss combiner (spec §2.4)."""
from __future__ import annotations

import torch

from dccnet.losses.count_consistency import cc_background_mass, cc_weight, count_consistency_loss, oracle_tile_counts
from dccnet.losses.density import density_loss
from dccnet.losses.focal import focal_loss
from dccnet.losses.radius_ciou import offset_loss, radius_ciou_loss

_CC_TARGETS = ("density", "oracle")

__all__ = [
    "cc_background_mass",
    "cc_weight",
    "compute_losses",
    "count_consistency_loss",
    "density_loss",
    "focal_loss",
    "offset_loss",
    "oracle_tile_counts",
    "radius_ciou_loss",
]


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    cfg: dict,
    epoch: int,
) -> dict[str, torch.Tensor]:
    """Total loss per spec §2.4:
    ``focal + λ_r (l1 + ciou) + λ_o offset + λ_d density + λ_cc(epoch) cc``.

    ``outputs``: model dict (``heatmap``, ``log_r``, ``offset``, optional ``density``).
    ``targets``: batched ``build_targets`` dict. ``cfg``: config with ``loss.*`` and
    ``data.stride`` (as in configs/base.yaml).

    The density term is 0 when ``outputs`` has no ``density``; with ``loss.cc.target: density`` (default)
    the cc term (and ``lambda_cc``) is then 0 too, while ``target: oracle`` (M2c) builds the cc target
    from the GT center counts and needs no density head (invalid target -> ValueError);
    the cc term and ``lambda_cc`` are 0 when ``loss.cc.enabled`` is false. During cc
    warm-up ``loss_cc`` is still computed and logged, with ``lambda_cc = 0``.
    ``loss_cc_bg`` (detached diagnostic, never in the total) is the background mass of
    local maxima at or below ``loss.cc.peak_thr`` (0 when cc is off).
    All values are 0-dim float32 tensors; only ``loss_total`` is meant for backward.
    """
    lc = cfg["loss"]
    cc_cfg = lc["cc"]
    stride = int(cfg["data"]["stride"])

    hm = outputs["heatmap"]
    l_focal = focal_loss(hm, targets["heatmap"])
    l_r, l_ciou = radius_ciou_loss(outputs["log_r"], outputs["offset"], targets, stride=stride)
    l_o = offset_loss(outputs["offset"], targets)

    zero = torch.zeros((), dtype=torch.float32, device=hm.device)
    l_d, l_cc, l_cc_bg, lam_cc = zero, zero, zero, 0.0
    density = outputs.get("density")
    cc_target = str(cc_cfg.get("target", "density"))
    if cc_target not in _CC_TARGETS:
        raise ValueError(f"loss.cc.target must be one of {_CC_TARGETS}, got {cc_target!r}")
    if density is not None:
        l_d = density_loss(density, targets["density"])
    if cc_cfg["enabled"] and (density is not None or cc_target == "oracle"):
        peak_thr = float(cc_cfg["peak_thr"])
        window, scope = int(cc_cfg["window"]), cc_cfg["scope"]
        kw = dict(
            window=window,
            scope=scope,
            stop_grad_density=bool(cc_cfg["stop_grad_density"]),
            peak_thr=peak_thr,
            count_fn=str(cc_cfg.get("count_fn", "sigmoid")),
            count_center=float(cc_cfg.get("count_center", 0.3)),
            count_temp=float(cc_cfg.get("count_temp", 0.05)),
        )
        if cc_target == "oracle":
            # M2c: GT per-tile center counts replace the density head's counts
            counts = oracle_tile_counts(targets, hm.shape[1], hm.shape[-2], hm.shape[-1], window, scope)
            l_cc = count_consistency_loss(hm, target_counts=counts, **kw)
        else:
            l_cc = count_consistency_loss(hm, density, **kw)
        l_cc_bg = cc_background_mass(hm, peak_thr)
        lam_cc = cc_weight(
            epoch,
            float(lc["lambda_cc"]),
            warmup=int(cc_cfg["warmup_epochs"]),
            ramp=int(cc_cfg["ramp_epochs"]),
        )

    total = (
        l_focal
        + float(lc["lambda_r"]) * (l_r + l_ciou)
        + float(lc["lambda_o"]) * l_o
        + float(lc["lambda_d"]) * l_d
        + lam_cc * l_cc
    )
    return {
        "loss_focal": l_focal,
        "loss_r": l_r,
        "loss_ciou": l_ciou,
        "loss_o": l_o,
        "loss_d": l_d,
        "loss_cc": l_cc,
        "loss_cc_bg": l_cc_bg,
        "lambda_cc": torch.tensor(lam_cc, dtype=torch.float32, device=hm.device),
        "loss_total": total,
    }
