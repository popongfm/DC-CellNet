# P3b: TXL-PBC pooled + leave-one-source-out summary

Splits: pooled. Arms: d_both, d_sigma, circlenet, repel, yolo26m, frcnn. Required seeds: [0, 1, 2]. Candidate: d_both. Values are mean ± sample std over seeds; thresholds are each run's own val-swept best_thr. Test images: pooled 126.

**pooled stop/go: d_both vs circlenet = NO-GO**

## 1. Per split x arm

mAP bootstrap CI disabled (`--map-ci-n 0`).

### pooled

| arm | n_seeds | mAP50-95 | mAP50-95 CI | mAP50 | AP50-95 RBC | AP50-95 WBC | AP50-95 Platelets | count MAE RBC | count MAE WBC | count MAE Platelets | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_both | 3 | 0.6382 ± 0.0056 |  | 0.9186 ± 0.0042 | 0.5835 ± 0.0020 | 0.6369 ± 0.0096 | 0.6941 ± 0.0176 | 1.0820 ± 0.0947 | 1.0317 ± 0.0550 | 0.0265 ± 0.0046 | 0.9325 ± 0.0056 | 0.8970 ± 0.0027 | 0.9230 ± 0.0072 | 0.9415 ± 0.0047 |
| d_sigma | 3 | 0.6505 ± 0.0025 |  | 0.9419 ± 0.0021 | 0.5897 ± 0.0026 | 0.6542 ± 0.0096 | 0.7075 ± 0.0159 | 1.0582 ± 0.1162 | 0.9101 ± 0.1101 | 0.0159 ± 0.0079 | 0.9416 ± 0.0156 | 0.9092 ± 0.0156 | 0.9206 ± 0.0059 | 0.9510 ± 0.0041 |
| circlenet | 3 | 0.6546 ± 0.0138 |  | 0.9392 ± 0.0126 | 0.5950 ± 0.0014 | 0.6608 ± 0.0378 | 0.7081 ± 0.0199 | 1.0317 ± 0.0210 | 0.8016 ± 0.1457 | 0.0159 ± 0.0079 | 0.9416 ± 0.0088 | 0.9160 ± 0.0111 | 0.9371 ± 0.0116 | 0.9537 ± 0.0062 |
| repel | 3 | 0.6724 ± 0.0046 |  | 0.9613 ± 0.0037 | 0.5706 ± 0.0015 | 0.7273 ± 0.0098 | 0.7195 ± 0.0046 | 1.8545 ± 0.1290 | 0.1825 ± 0.0275 | 0.0212 ± 0.0046 | 0.9044 ± 0.0024 | 0.8609 ± 0.0021 | 0.9025 ± 0.0036 | 0.9279 ± 0.0103 |
| yolo26m | 3 | 0.8595 ± 0.0044 |  | 0.9759 ± 0.0020 | 0.8809 ± 0.0073 | 0.8737 ± 0.0062 | 0.8239 ± 0.0022 | 1.2116 ± 0.2039 | 0.0265 ± 0.0121 | 0.0556 ± 0.0159 | 0.9662 ± 0.0073 | 0.9318 ± 0.0114 | 0.9332 ± 0.0150 | 0.9469 ± 0.0071 |
| frcnn | 3 | 0.8315 ± 0.0023 |  | 0.9826 ± 0.0010 | 0.8572 ± 0.0014 | 0.8295 ± 0.0047 | 0.8079 ± 0.0125 | 0.8280 ± 0.0046 | 0.0238 ± 0.0079 | 0.0423 ± 0.0092 | 0.9768 ± 0.0042 | 0.9571 ± 0.0034 | 0.9709 ± 0.0014 | 0.9728 ± 0.0085 |

Overlap bin = number of other GT circles a cell overlaps (0..3+); recall at IoU 0.5.

## 2. C2: pooled vs leave-one-source-out (mAP50-95)

LOSO mean = mean over the 4 folds of each fold's seed mean; blank when any fold misses a required seed.

| arm | pooled | loso_BCCD | loso_BCDD | loso_PBC | loso_Raabin | LOSO mean | Δ = pooled − LOSO |
|---|---|---|---|---|---|---|---|
| d_both | 0.6382 |  |  |  |  |  |  |
| d_sigma | 0.6505 |  |  |  |  |  |  |
| circlenet | 0.6546 |  |  |  |  |  |  |
| repel | 0.6724 |  |  |  |  |  |  |
| yolo26m | 0.8595 |  |  |  |  |  |  |
| frcnn | 0.8315 |  |  |  |  |  |  |

## 3. Paired comparisons: d_both vs every other arm

`compare_runs.gate` per comparison: RBC |count error| per image averaged over seeds, paired bootstrap 95% CI (n=1000) of candidate − baseline upper < 0; pooled recall in overlap bins 2+3 higher; every per-seed mean difference < 0. LOSO pooled = the 4 folds' per-image arrays concatenated per seed (BCCD, BCDD, PBC, Raabin). INCONCLUSIVE = a required seed/fold is missing or a run's predictions name images outside the test subset; no gate is computed then.

| scope | comparison | verdict | seeds | images | RBC MAE cand vs base | diff CI (upper<0) | recall bins 2+3 (higher) | per-seed diff (all<0) | inconclusive because |
|---|---|---|---|---|---|---|---|---|---|
| pooled | d_both vs d_sigma | **NO-GO** | [0, 1, 2] | 126 | 1.0820 vs 1.0582 | [-0.0847, 0.1217] (no) | 0.9297 vs 0.9317 (no) | s0 -0.2063, s1 0.1190, s2 0.1587 (no) | |
| pooled | d_both vs circlenet | **NO-GO** | [0, 1, 2] | 126 | 1.0820 vs 1.0317 | [-0.0504, 0.1482] (no) | 0.9297 vs 0.9432 (no) | s0 -0.0635, s1 0.1111, s2 0.1032 (no) | |
| pooled | d_both vs repel | **GO** | [0, 1, 2] | 126 | 1.0820 vs 1.8545 | [-0.9894, -0.5661] (yes) | 0.9297 vs 0.9118 (yes) | s0 -0.9921, s1 -0.7222, s2 -0.6032 (yes) | |
| pooled | d_both vs yolo26m | **NO-GO** | [0, 1, 2] | 126 | 1.0820 vs 1.2116 | [-0.3095, 0.0635] (no) | 0.9297 vs 0.9382 (no) | s0 -0.0476, s1 -0.0238, s2 -0.3175 (yes) | |
| pooled | d_both vs frcnn | **NO-GO** | [0, 1, 2] | 126 | 1.0820 vs 0.8280 | [0.1084, 0.3995] (no) | 0.9297 vs 0.9716 (no) | s0 0.1429, s1 0.3333, s2 0.2857 (no) | |

LOSO comparisons not run (no LOSO split in --splits).

## 4. Efficiency

Mean train_seconds per split over that split's seeds (n runs timed); splits are not mixed because their train sets differ in size.

| arm | n_params | train_seconds pooled | inference FPS |
|---|---|---|---|
| d_both | 24739014 | 7513 (n=3) |  |
| d_sigma | 24591302 | 5305 (n=3) |  |
| circlenet | 24591302 | 5055 (n=3) |  |
| repel | 24739014 | 9953 (n=3) |  |
| yolo26m | 21777514 | 3213 (n=3) |  |
| frcnn | 41309411 | 1970 (n=3) |  |

FPS only from `<run>/latency.json` (`fps`); blank otherwise.

## Verdicts

- **pooled stop/go: d_both vs circlenet = NO-GO**
- pooled: d_both vs d_sigma = NO-GO
- pooled: d_both vs circlenet = NO-GO
- pooled: d_both vs repel = GO
- pooled: d_both vs yolo26m = NO-GO
- pooled: d_both vs frcnn = NO-GO
