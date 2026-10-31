# M2 go/no-go: does L_cc help separate overlapping cells?

Test images: 200. Required seeds: [0]. Values are mean ± std (sample std) over seeds; thresholds are each run's own val-swept best_thr.

## Per-arm results

| arm | n_seeds | test mAP50-95 | RBC count MAE | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 | best_epoch |
|---|---|---|---|---|---|---|---|---|
| d_both | 3 | 0.8516 ± 0.0024 | 1.4800 ± 0.1587 | 1.0000 ± 0.0000 | 0.9967 ± 0.0000 | 0.9917 ± 0.0024 | 0.9062 ± 0.0072 | 107.0 ± 5.6 |
| d_s2 | 3 | 0.8423 ± 0.0109 | 2.4867 ± 0.0153 | 1.0000 ± 0.0000 | 0.9948 ± 0.0012 | 0.9789 ± 0.0021 | 0.8594 ± 0.0015 | 99.0 ± 11.1 |
| p3a_agnostic | 3 | 0.8592 ± 0.0033 | 1.4233 ± 0.0785 | 1.0000 ± 0.0000 | 0.9972 ± 0.0006 | 0.9914 ± 0.0028 | 0.9091 ± 0.0044 | 100.0 ± 4.0 |
| p3a_div2 | 1 | 0.8502 | 2.0650 | 1.0000 | 0.9956 | 0.9852 | 0.8791 | 101.0 |
| p3a_div8 | 1 | 0.8444 | 1.4600 | 1.0000 | 0.9967 | 0.9914 | 0.9066 | 109.0 |
| p3a_knn3 | 3 | 0.8473 ± 0.0032 | 1.9033 ± 0.1678 | 1.0000 ± 0.0000 | 0.9967 ± 0.0000 | 0.9836 ± 0.0008 | 0.8873 ± 0.0056 | 110.3 ± 0.6 |
| p3a_mcnn | 3 | 0.8217 ± 0.0006 | 2.1350 ± 0.0757 | 1.0000 ± 0.0000 | 0.9954 ± 0.0008 | 0.9823 ± 0.0020 | 0.8751 ± 0.0059 | 106.7 ± 18.8 |
| p3a_repel | 3 | 0.8551 ± 0.0092 | 1.3500 ± 0.1704 | 1.0000 ± 0.0000 | 0.9969 ± 0.0006 | 0.9925 ± 0.0005 | 0.9116 ± 0.0082 | 110.0 ± 1.0 |

Overlap bin = number of other GT circles a cell overlaps (0..3+). Recall at IoU 0.5.

## Gate 1: d_both vs p3a_div8

- seeds used: [0], test images: 200
- RBC |count error| (per-image, mean over seeds): d_both 1.4200 vs p3a_div8 1.4600; difference -0.0400
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.1600, 0.0750] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 5275 over seeds): d_both 0.9270 (95% CI [0.9174, 0.9362]) vs p3a_div8 0.9272 (95% CI [0.9161, 0.9371]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, d_both seed s - p3a_div8 seed s): s0 -0.0400 -> all < 0: **yes**
- comparison: **fail**

## Gate 2: d_both vs p3a_div8 (isolates L_cc)

- seeds used: [0], test images: 200
- RBC |count error| (per-image, mean over seeds): d_both 1.4200 vs p3a_div8 1.4600; difference -0.0400
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.1600, 0.0750] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 5275 over seeds): d_both 0.9270 (95% CI [0.9174, 0.9362]) vs p3a_div8 0.9272 (95% CI [0.9161, 0.9371]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, d_both seed s - p3a_div8 seed s): s0 -0.0400 -> all < 0: **yes**
- comparison: **fail**

## Verdict

GO requires both gates to pass; INCONCLUSIVE when a required seed run is missing for any arm or a d_both run picked its checkpoint before its select_from_epoch.

**VERDICT: NO-GO (d_both vs p3a_div8 and d_both vs p3a_div8)**

## Diagnostic: p3a_div8 vs p3a_div8 (not part of the gate)

- seeds used: [0], test images: 200
- RBC |count error| (per-image, mean over seeds): p3a_div8 1.4600 vs p3a_div8 1.4600; difference 0.0000
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [0.0000, 0.0000] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 5275 over seeds): p3a_div8 0.9272 (95% CI [0.9161, 0.9371]) vs p3a_div8 0.9272 (95% CI [0.9161, 0.9371]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, p3a_div8 seed s - p3a_div8 seed s): s0 0.0000 -> all < 0: **no**
- comparison: **fail**
