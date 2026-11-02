# P3b: TXL-PBC pooled + leave-one-source-out summary

Splits: pooled. Arms: d_both, d_sigma, circlenet, repel, yolo26m, frcnn. Required seeds: [0, 1, 2]. Candidate: d_both. Values are mean ± sample std over seeds; thresholds are each run's own val-swept best_thr. Test images: pooled 126.

**pooled stop/go: d_both vs circlenet = NO-GO**

## 1. Per split x arm

mAP bootstrap CI disabled (`--map-ci-n 0`).

### pooled

| arm | n_seeds | mAP50-95 | mAP50-95 CI | mAP50 | AP50-95 RBC | AP50-95 WBC | AP50-95 Platelets | count MAE RBC | count MAE WBC | count MAE Platelets | recall bin 0 | recall bin 1 | recall bin 2 | recall bin 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_both | 3 | 0.6719 ± 0.0067 |  | 0.9712 ± 0.0045 | 0.5935 ± 0.0009 | 0.7207 ± 0.0073 | 0.7014 ± 0.0139 | 0.8995 ± 0.0391 | 0.0397 ± 0.0210 | 0.0238 ± 0.0079 | 0.9543 ± 0.0053 | 0.9223 ± 0.0021 | 0.9442 ± 0.0036 | 0.9537 ± 0.0047 |
| d_sigma | 3 | 0.6708 ± 0.0021 |  | 0.9746 ± 0.0017 | 0.5924 ± 0.0023 | 0.7124 ± 0.0165 | 0.7075 ± 0.0159 | 0.9630 ± 0.0528 | 0.0344 ± 0.0046 | 0.0212 ± 0.0046 | 0.9466 ± 0.0074 | 0.9137 ± 0.0092 | 0.9253 ± 0.0049 | 0.9524 ± 0.0024 |
| circlenet | 3 | 0.6755 ± 0.0085 |  | 0.9736 ± 0.0048 | 0.5973 ± 0.0009 | 0.7211 ± 0.0111 | 0.7082 ± 0.0200 | 0.9550 ± 0.0242 | 0.0344 ± 0.0092 | 0.0132 ± 0.0046 | 0.9564 ± 0.0068 | 0.9255 ± 0.0098 | 0.9426 ± 0.0098 | 0.9578 ± 0.0024 |
| repel | 3 | 0.6757 ± 0.0024 |  | 0.9688 ± 0.0020 | 0.5714 ± 0.0018 | 0.7362 ± 0.0025 | 0.7195 ± 0.0045 | 1.7063 ± 0.0496 | 0.0873 ± 0.0159 | 0.0212 ± 0.0046 | 0.9044 ± 0.0024 | 0.8595 ± 0.0021 | 0.9025 ± 0.0036 | 0.9279 ± 0.0103 |
| yolo26m | 3 | 0.8595 ± 0.0044 |  | 0.9759 ± 0.0020 | 0.8809 ± 0.0073 | 0.8737 ± 0.0062 | 0.8239 ± 0.0022 | 1.2116 ± 0.2039 | 0.0265 ± 0.0121 | 0.0556 ± 0.0159 | 0.9662 ± 0.0073 | 0.9318 ± 0.0114 | 0.9332 ± 0.0150 | 0.9469 ± 0.0071 |
| frcnn | 3 | 0.8315 ± 0.0023 |  | 0.9826 ± 0.0010 | 0.8572 ± 0.0014 | 0.8295 ± 0.0047 | 0.8079 ± 0.0125 | 0.8280 ± 0.0046 | 0.0238 ± 0.0079 | 0.0423 ± 0.0092 | 0.9768 ± 0.0042 | 0.9571 ± 0.0034 | 0.9709 ± 0.0014 | 0.9728 ± 0.0085 |

Overlap bin = number of other GT circles a cell overlaps (0..3+); recall at IoU 0.5.

## 2. C2: pooled vs leave-one-source-out (mAP50-95)

LOSO mean = mean over the 4 folds of each fold's seed mean; blank when any fold misses a required seed.

| arm | pooled | loso_BCCD | loso_BCDD | loso_PBC | loso_Raabin | LOSO mean | Δ = pooled − LOSO |
|---|---|---|---|---|---|---|---|
| d_both | 0.6719 |  |  |  |  |  |  |
| d_sigma | 0.6708 |  |  |  |  |  |  |
| circlenet | 0.6755 |  |  |  |  |  |  |
| repel | 0.6757 |  |  |  |  |  |  |
| yolo26m | 0.8595 |  |  |  |  |  |  |
| frcnn | 0.8315 |  |  |  |  |  |  |

## 3. Paired comparisons: d_both vs every other arm

`compare_runs.gate` per comparison: RBC |count error| per image averaged over seeds, paired bootstrap 95% CI (n=1000) of candidate − baseline upper < 0; pooled recall in overlap bins 2+3 higher; every per-seed mean difference < 0. LOSO pooled = the 4 folds' per-image arrays concatenated per seed (BCCD, BCDD, PBC, Raabin). INCONCLUSIVE = a required seed/fold is missing or a run's predictions name images outside the test subset; no gate is computed then.

| scope | comparison | verdict | seeds | images | RBC MAE cand vs base | diff CI (upper<0) | recall bins 2+3 (higher) | per-seed diff (all<0) | inconclusive because |
|---|---|---|---|---|---|---|---|---|---|
| pooled | d_both vs d_sigma | **NO-GO** | [0, 1, 2] | 126 | 0.8995 vs 0.9630 | [-0.1667, 0.0239] (no) | 0.9477 vs 0.9352 (yes) | s0 -0.0635, s1 -0.1032, s2 -0.0238 (yes) | |
| pooled | d_both vs circlenet | **NO-GO** | [0, 1, 2] | 126 | 0.8995 vs 0.9550 | [-0.1455, 0.0423] (no) | 0.9477 vs 0.9482 (no) | s0 -0.0159, s1 -0.0556, s2 -0.0952 (yes) | |
| pooled | d_both vs repel | **GO** | [0, 1, 2] | 126 | 0.8995 vs 1.7063 | [-1.0054, -0.6085] (yes) | 0.9477 vs 0.9118 (yes) | s0 -0.7778, s1 -0.8730, s2 -0.7698 (yes) | |
| pooled | d_both vs yolo26m | **GO** | [0, 1, 2] | 126 | 0.8995 vs 1.2116 | [-0.4974, -0.1269] (yes) | 0.9477 vs 0.9382 (yes) | s0 -0.0794, s1 -0.3095, s2 -0.5476 (yes) | |
| pooled | d_both vs frcnn | **NO-GO** | [0, 1, 2] | 126 | 0.8995 vs 0.8280 | [-0.0582, 0.1958] (no) | 0.9477 vs 0.9716 (no) | s0 0.1111, s1 0.0476, s2 0.0556 (no) | |

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
- pooled: d_both vs yolo26m = GO
- pooled: d_both vs frcnn = NO-GO
