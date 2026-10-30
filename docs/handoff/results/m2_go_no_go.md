# M2 go/no-go: does L_cc help separate overlapping cells?

Test images: 200. Required seeds: [0, 1, 2]. Values are mean ± std (sample std) over seeds; thresholds are each run's own val-swept best_thr.

## Per-arm results

| arm | n_seeds | test mAP50-95 | RBC count MAE | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 | best_epoch |
|---|---|---|---|---|---|---|---|---|
| circlenet | 3 | 0.7895 ± 0.0017 | 3.2267 ± 0.0539 | 0.9987 ± 0.0006 | 0.9910 ± 0.0018 | 0.9716 ± 0.0016 | 0.8237 ± 0.0026 | 54.0 ± 4.4 |
| density | 3 | 0.7433 ± 0.0037 | 3.3283 ± 0.0454 | 0.9964 ± 0.0023 | 0.9893 ± 0.0017 | 0.9683 ± 0.0033 | 0.8167 ± 0.0033 | 46.3 ± 8.4 |
| dccell | 3 | 0.7427 ± 0.0065 | 3.2400 ± 0.0709 | 0.9995 ± 0.0009 | 0.9897 ± 0.0014 | 0.9703 ± 0.0016 | 0.8207 ± 0.0033 | 50.0 ± 7.9 |

Overlap bin = number of other GT circles a cell overlaps (0..3+). Recall at IoU 0.5.

## Gate 1: dccell vs circlenet

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): dccell 3.2400 vs circlenet 3.2267; difference 0.0133
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.0567, 0.0833] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): dccell 0.8571 (95% CI [0.8450, 0.8691]) vs circlenet 0.8597 (95% CI [0.8468, 0.8720]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, dccell seed s - circlenet seed s): s0 0.0150, s1 0.0300, s2 -0.0050 -> all < 0: **no**
- comparison: **fail**

## Gate 2: dccell vs density (isolates L_cc)

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): dccell 3.2400 vs density 3.3283; difference -0.0883
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.1500, -0.0367] -> upper < 0: **yes**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): dccell 0.8571 (95% CI [0.8450, 0.8691]) vs density 0.8535 (95% CI [0.8405, 0.8657]) -> candidate higher: **yes**
- per-seed paired RBC |count error| difference (mean over images, dccell seed s - density seed s): s0 -0.0150, s1 -0.0750, s2 -0.1750 -> all < 0: **yes**
- comparison: **pass**

## Verdict

GO requires both gates to pass; INCONCLUSIVE when a required seed run is missing for any arm or a dccell run picked its checkpoint before its select_from_epoch.

**VERDICT: NO-GO (dccell vs circlenet and dccell vs density)**

## Diagnostic: density vs circlenet (not part of the gate)

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): density 3.3283 vs circlenet 3.2267; difference 0.1017
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [0.0233, 0.1750] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): density 0.8535 (95% CI [0.8405, 0.8657]) vs circlenet 0.8597 (95% CI [0.8468, 0.8720]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, density seed s - circlenet seed s): s0 0.0300, s1 0.1050, s2 0.1700 -> all < 0: **no**
- comparison: **fail**
