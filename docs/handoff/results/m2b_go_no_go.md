# M2 go/no-go: does L_cc help separate overlapping cells?

Test images: 200. Required seeds: [0, 1, 2]. Values are mean ± std (sample std) over seeds; thresholds are each run's own val-swept best_thr.

## Per-arm results

| arm | n_seeds | test mAP50-95 | RBC count MAE | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 | best_epoch |
|---|---|---|---|---|---|---|---|---|
| b_circlenet | 3 | 0.8365 ± 0.0018 | 2.9450 ± 0.0397 | 0.9996 ± 0.0003 | 0.9928 ± 0.0017 | 0.9766 ± 0.0021 | 0.8391 ± 0.0024 | 108.0 ± 14.7 |
| b_dccell | 3 | 0.8385 ± 0.0045 | 2.9617 ± 0.0401 | 0.9993 ± 0.0003 | 0.9922 ± 0.0006 | 0.9748 ± 0.0012 | 0.8379 ± 0.0019 | 95.3 ± 12.2 |
| b_density | 3 | 0.8388 ± 0.0054 | 2.9650 ± 0.0180 | 0.9995 ± 0.0000 | 0.9926 ± 0.0008 | 0.9755 ± 0.0016 | 0.8385 ± 0.0004 | 109.0 ± 15.6 |

Overlap bin = number of other GT circles a cell overlaps (0..3+). Recall at IoU 0.5.

## Gate 1: b_dccell vs b_circlenet

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): b_dccell 2.9617 vs b_circlenet 2.9450; difference 0.0167
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.0417, 0.0717] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): b_dccell 0.8712 (95% CI [0.8589, 0.8828]) vs b_circlenet 0.8725 (95% CI [0.8602, 0.8835]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, b_dccell seed s - b_circlenet seed s): s0 0.1000, s1 -0.0550, s2 0.0050 -> all < 0: **no**
- comparison: **fail**

## Gate 2: b_dccell vs b_density (isolates L_cc)

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): b_dccell 2.9617 vs b_density 2.9650; difference -0.0033
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.0500, 0.0400] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): b_dccell 0.8712 (95% CI [0.8589, 0.8828]) vs b_density 0.8718 (95% CI [0.8599, 0.8832]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, b_dccell seed s - b_density seed s): s0 0.0300, s1 -0.0600, s2 0.0200 -> all < 0: **no**
- comparison: **fail**

## Verdict

GO requires both gates to pass; INCONCLUSIVE when a required seed run is missing for any arm or a b_dccell run picked its checkpoint before its select_from_epoch.

**VERDICT: NO-GO (b_dccell vs b_circlenet and b_dccell vs b_density)**

## Diagnostic: b_density vs b_circlenet (not part of the gate)

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): b_density 2.9650 vs b_circlenet 2.9450; difference 0.0200
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.0350, 0.0783] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): b_density 0.8718 (95% CI [0.8599, 0.8832]) vs b_circlenet 0.8725 (95% CI [0.8602, 0.8835]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, b_density seed s - b_circlenet seed s): s0 0.0700, s1 0.0050, s2 -0.0150 -> all < 0: **no**
- comparison: **fail**
