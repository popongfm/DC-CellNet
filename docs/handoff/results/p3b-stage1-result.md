# P3b stage 1 (TXL-PBC pooled, 6 arms × 3 seeds) — result + diagnosis

Report: `results/p3b/p3b_stage1.md` (made with `--map-ci-n 0`; the device split keys start with `TXL-PBC_Dataset/`, so use the device's `pooled.json`).

| arm | mAP50-95 | mAP50 | RBC MAE | WBC MAE | recall bins 2+3 |
|---|---|---|---|---|---|
| d_both | 0.638 | 0.919 | 1.08 | 1.03 | 0.930 |
| d_sigma | 0.651 | 0.942 | 1.06 | 0.91 | 0.932 |
| circlenet | 0.655 | 0.939 | 1.03 | 0.80 | 0.943 |
| repel | 0.672 | 0.961 | 1.85 | 0.18 | 0.912 |
| yolo26m | 0.860 | 0.976 | 1.21 | 0.03 | 0.938 |
| frcnn | 0.832 | 0.983 | 0.83 | 0.02 | 0.972 |

**Stop/go (spec §3.2): d_both vs circlenet = NO-GO.** d_both beats repel (GO); it loses to frcnn on RBC count and ties d_sigma, circlenet and yolo26m.

## Diagnosis
1. **Duplicate WBC peaks (decoder problem, every DC-CellNet arm).** About 2 detections per GT WBC. Every extra one lies within 0.16·R of the GT centre, and none fall on an image without a WBC. A 3×3 max-pool cannot suppress near-ties on large cells. Offline per-class box NMS (IoU 0.5) on the seed-0 test predictions at the same threshold:
   - d_both: WBC MAE 1.00 → 0.016, mAP50 0.918 → 0.968, mAP50-95 0.641 → 0.674
   - circlenet: WBC MAE 0.97 → 0.016, mAP50 0.925 → 0.974, mAP50-95 0.642 → 0.672
   - RBC MAE changes by less than 0.1.
2. **mAP50-95 gap to YOLO/FRCNN is mostly representation.** TXL GT boxes are rectangles (RBCs cut by the border, elongated cells). The best square box fitted to the GT has mean IoU 0.835 for RBC, and 26% of RBCs cannot reach IoU 0.75. A circle detector is therefore capped at high IoU thresholds whatever the target rule. Compare AP50 and count, or add a w/h head.
3. **Little real overlap in TXL** (the risk in spec §5): bins 2+3 recall is ≈ 0.93–0.97 for every arm. The overlap claim has to be tested on BCCD (P3c).
