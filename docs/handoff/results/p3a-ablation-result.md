# P3a ablation result (synthetic v2, stride 2, 200 test images)

Runs: device commit ef78cec (P3a = eeb5f06). Gate = compare_runs (paired bootstrap CI of per-image RBC |count err| < 0, pooled recall bins 2+3 higher, every seed). Full reports: `results/p3a/`.

| arm | seeds | mAP50-95 | RBC MAE | recall bin 3 | d_both vs arm |
|---|---|---|---|---|---|
| d_both (adaptive same-class, D=4) | 3 | 0.852 ± 0.002 | 1.48 ± 0.16 | 0.906 | — |
| d_s2 (radius σ) | 3 | 0.842 ± 0.011 | 2.49 ± 0.02 | 0.859 | **GO** |
| knn3 | 3 | 0.847 ± 0.003 | 1.90 ± 0.17 | 0.887 | **GO** |
| mcnn (0.3·mean 3-NN) | 3 | 0.822 ± 0.001 | 2.14 ± 0.08 | 0.875 | **GO** |
| agnostic (d_nn any class) | 3 | 0.859 ± 0.003 | 1.42 ± 0.08 | 0.909 | NO-GO (tie: CI [−0.008, 0.120]) |
| repel (Liang proximity coding) | 3 | 0.855 ± 0.009 | 1.35 ± 0.17 | 0.912 | NO-GO (repel better on average: d_both − repel CI [0.045, 0.218]; seed 1 favours d_both) |
| div2 | 1 | 0.850 | 2.07 | 0.879 | (info) worse |
| div8 | 1 | 0.844 | 1.46 | 0.907 | (info) ≈ d_both |

## Reading
- Nearest-neighbour capping beats a fixed radius σ, mean-of-3-NN (knn3) and the crowd-counting MCNN rule — the *nearest* neighbour is what matters.
- Same-class restriction does not matter on synthetic (agnostic ties, slightly better mAP). Claim must not say "class-specific cap is necessary".
- Repel coding (prior work) is at least as good as d_both on synthetic: lower count error on average (significant on the seed-averaged per-image bootstrap) but not on every seed. So the adaptive-σ rule is NOT shown to beat the strongest prior target.
- D: 2 too tight, 4 ≈ 8 → robust above 4.

## Consequence for claims
"Novel σ rule beats prior targets" is not supported. Supported: neighbour-aware targets (adaptive σ or repel) + stride 2 halve count error vs CircleNet-style radius σ; nearest-neighbour > k-NN mean / MCNN rule. Real-data comparison (P3b) must include repel as an arm.
