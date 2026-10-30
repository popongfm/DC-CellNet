# M2 go/no-go: does L_cc help separate overlapping cells?

Test images: 200. Required seeds: [0, 1, 2]. Values are mean ± std (sample std) over seeds; thresholds are each run's own val-swept best_thr.

## Per-arm results

| arm | n_seeds | test mAP50-95 | RBC count MAE | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 | best_epoch |
|---|---|---|---|---|---|---|---|---|
| b_circlenet | 3 | 0.8365 ± 0.0018 | 2.9450 ± 0.0397 | 0.9996 ± 0.0003 | 0.9928 ± 0.0017 | 0.9766 ± 0.0021 | 0.8391 ± 0.0024 | 108.0 ± 14.7 |
| b_dccell | 3 | 0.8385 ± 0.0045 | 2.9617 ± 0.0401 | 0.9993 ± 0.0003 | 0.9922 ± 0.0006 | 0.9748 ± 0.0012 | 0.8379 ± 0.0019 | 95.3 ± 12.2 |
| b_density | 3 | 0.8388 ± 0.0054 | 2.9650 ± 0.0180 | 0.9995 ± 0.0000 | 0.9926 ± 0.0008 | 0.9755 ± 0.0016 | 0.8385 ± 0.0004 | 109.0 ± 15.6 |
| b_oracle | 3 | 0.8388 ± 0.0006 | 2.9400 ± 0.0265 | 0.9993 ± 0.0003 | 0.9922 ± 0.0010 | 0.9742 ± 0.0008 | 0.8376 ± 0.0008 | 107.3 ± 9.3 |
| d_s2 | 3 | 0.8423 ± 0.0109 | 2.4867 ± 0.0153 | 1.0000 ± 0.0000 | 0.9948 ± 0.0012 | 0.9789 ± 0.0021 | 0.8594 ± 0.0015 | 99.0 ± 11.1 |
| d_sigma | 3 | 0.8412 ± 0.0020 | 2.3033 ± 0.0058 | 1.0000 ± 0.0000 | 0.9948 ± 0.0006 | 0.9794 ± 0.0012 | 0.8681 ± 0.0004 | 114.0 ± 6.2 |

Overlap bin = number of other GT circles a cell overlaps (0..3+). Recall at IoU 0.5.

## Gate 1: d_s2 vs b_circlenet

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): d_s2 2.4867 vs b_circlenet 2.9450; difference -0.4583
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.5483, -0.3733] -> upper < 0: **yes**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): d_s2 0.8884 (95% CI [0.8771, 0.8989]) vs b_circlenet 0.8725 (95% CI [0.8602, 0.8835]) -> candidate higher: **yes**
- per-seed paired RBC |count error| difference (mean over images, d_s2 seed s - b_circlenet seed s): s0 -0.4000, s1 -0.5050, s2 -0.4700 -> all < 0: **yes**
- comparison: **pass**

## Gate 2: d_s2 vs b_circlenet (isolates L_cc)

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): d_s2 2.4867 vs b_circlenet 2.9450; difference -0.4583
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [-0.5483, -0.3733] -> upper < 0: **yes**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): d_s2 0.8884 (95% CI [0.8771, 0.8989]) vs b_circlenet 0.8725 (95% CI [0.8602, 0.8835]) -> candidate higher: **yes**
- per-seed paired RBC |count error| difference (mean over images, d_s2 seed s - b_circlenet seed s): s0 -0.4000, s1 -0.5050, s2 -0.4700 -> all < 0: **yes**
- comparison: **pass**

## Verdict

GO requires both gates to pass; INCONCLUSIVE when a required seed run is missing for any arm or a d_s2 run picked its checkpoint before its select_from_epoch.

**VERDICT: GO (d_s2 vs b_circlenet and d_s2 vs b_circlenet)**

## Diagnostic: b_circlenet vs b_circlenet (not part of the gate)

- seeds used: [0, 1, 2], test images: 200
- RBC |count error| (per-image, mean over seeds): b_circlenet 2.9450 vs b_circlenet 2.9450; difference 0.0000
- paired bootstrap 95% CI of the difference (n=1000, seed 0): [0.0000, 0.0000] -> upper < 0: **no**
- pooled recall, overlap bins 2+3 (n_gt = 15825 over seeds): b_circlenet 0.8725 (95% CI [0.8602, 0.8835]) vs b_circlenet 0.8725 (95% CI [0.8602, 0.8835]) -> candidate higher: **no**
- per-seed paired RBC |count error| difference (mean over images, b_circlenet seed s - b_circlenet seed s): s0 0.0000, s1 0.0000, s2 0.0000 -> all < 0: **no**
- comparison: **fail**
