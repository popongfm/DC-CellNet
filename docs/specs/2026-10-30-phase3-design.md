# DC-CellNet Phase 3 — Design Spec

**วันที่:** 2026-10-30 · **สถานะ:** Draft — รอ review
**ต่อจาก:** `docs/specs/2026-10-26-dccellnet-design.md` (M0–M2) และผล M2/M2b/M2c/M2d (`dccnet/reports/`)
**อ้างอิง novelty:** Project doc `claude/novelty-check-adaptive-sigma.md`

---

## 1. บริบทและเป้าหมาย

### 1.1 สิ่งที่รู้แล้ว (จาก M2 series บน synthetic, 3 seeds, paired bootstrap)
- L_cc (local count-consistency) ไม่ช่วยแยกเซลล์ซ้อน แม้ใช้เป้าแบบ oracle (M2c) → ปิดแนวทาง
- ปัญหาจริง = peak ของ heatmap หลอมรวมในกลุ่มเซลล์คลาสเดียวกันที่ชิดกัน
- **Adaptive heatmap σ** (σ = max(min(r/(3s), d_nn/(4s)), 0.5) cells, d_nn = ระยะถึงเซลล์คลาสเดียวกันที่ใกล้สุด) + **stride 2** = `d_both`: RBC count MAE 2.95 → 1.48 (−50%), recall bin ≥3 0.839 → 0.906, mAP50-95 0.837 → 0.852

### 1.2 คำถามของ Phase 3
1. **P3a (novelty):** ส่วนไหนของกฎ σ ที่ทำให้ได้ผล และดีกว่ากฎ/target ที่มีในงานก่อนหน้า (LDC-Net, TopoCount, MCNN/CSRNet, repel coding) หรือไม่
2. **P3b (C2):** บนข้อมูลจริง TXL-PBC ทั้ง pooled และ leave-one-source-out — `d_both`/`d_sigma` ดีกว่า CircleNet, YOLO26m, Faster R-CNN หรือไม่ และทนต่อการข้ามแหล่งข้อมูลแค่ไหน
3. **P3c (C4):** บนชุดเซลล์ซ้อนจริง (BCCD 204 ภาพ, label แก้แล้ว) — ผลข้อ 2 ยังคงอยู่หรือไม่

### 1.3 Claims ที่จะอ้าง (ขึ้นกับผล)
- "To our knowledge" — per-instance Gaussian width ของ class-specific center heatmap ถูก cap ด้วยระยะถึงเพื่อนบ้านคลาสเดียวกัน ใน CenterNet/CircleNet-style detector สำหรับ blood smear
- LOSO benchmark บน TXL-PBC ครั้งแรก + การวัดตามระดับการซ้อน + count error
- Negative result: L_cc (รวม oracle) ไม่ช่วย

---

## 2. P3a — Ablation บน synthetic (stride 2)

### 2.1 Config ใหม่ (ค่า default คงพฤติกรรมเดิม)
- `data.heatmap_sigma`: `radius` (default) | `adaptive` | `adaptive_agnostic` | `knn3` | `mcnn`
  - `adaptive`: σ = max(min(r/(3s), d_nn/(D·s)), 0.5), d_nn เพื่อนบ้านคลาสเดียวกันที่ใกล้สุด
  - `adaptive_agnostic`: เหมือน `adaptive` แต่ d_nn จากทุกคลาส
  - `knn3`: เหมือน `adaptive` แต่ใช้ค่าเฉลี่ยระยะถึงเพื่อนบ้านคลาสเดียวกัน 3 ตัว (น้อยกว่า 3 ตัว → ใช้เท่าที่มี; ไม่มี → inf)
  - `mcnn`: σ = max(0.3 × mean 3-NN (คลาสเดียวกัน) / s, 0.5); ไม่มีเพื่อนบ้าน → σ = r/(3s) (fallback, ระบุในเปเปอร์)
- `data.heatmap_divisor`: D (default 4)
- `data.heatmap_shape`: `gaussian` (default) | `repel`
  - `repel` (Liang et al.): ที่ cell (x,y) ภายในรัศมี r_repel = r/s รอบเซลล์ i: d1 = ระยะถึง center ใกล้สุด, d2 = ใกล้อันดับสอง (คลาสเดียวกัน; ไม่มี → d2 = ∞), D′ = d1·(1 + d1/d2)², value = 1/(1 + α·D′), α = 0.8 (ค่าจาก Xie/Liang สำหรับ proximity map; ระบุใน README); peak = 1 ที่ center; รวม per class ด้วย max

### 2.2 แขน (prefix `exp_m2_`, ข้อมูล `splits/synthetic.json`, stride 2, density off, cc off, 120 epochs, patience 20)
| แขน | ค่าที่ต่างจาก d_both | seeds |
|---|---|---|
| `d_s2` (มีแล้ว) | heatmap_sigma radius | 3 |
| `d_both` (มีแล้ว) | — | 3 |
| `p3a_agnostic` | adaptive_agnostic | 3 |
| `p3a_knn3` | knn3 | 3 |
| `p3a_mcnn` | mcnn | 3 |
| `p3a_repel` | heatmap_shape repel | 3 |
| `p3a_div2` | divisor 2 | 1 |
| `p3a_div8` | divisor 8 | 1 |

ใหม่ 14 runs (~24–28 ชม.)

### 2.3 การตัดสิน
- `compare_runs --baseline <arm> --candidate d_both --ablation <arm>` ต่อแต่ละแขน; เกณฑ์เดิม (paired CI < 0 + recall bin≥2 สูงกว่า + ทุก seed)
- รายงานตารางรวมทุกแขน (mAP50-95, RBC MAE, recall bin 0–3)
- div2/div8 เป็นข้อมูลประกอบ (1 seed) ไม่เข้าเกณฑ์

---

## 3. P3b — TXL-PBC เต็มรูปแบบ

### 3.1 แขน
| แขน | ตระกูล | config |
|---|---|---|
| `d_both` | DC-CellNet | adaptive σ + stride 2 |
| `d_sigma` | DC-CellNet | adaptive σ, stride 4 |
| `circlenet` | DC-CellNet | radius σ, stride 4 |
| `yolo26m` | Ultralytics | defaults, imgsz 640 |
| `frcnn` | torchvision | fasterrcnn_resnet50_fpn (COCO pretrained) |

### 3.2 Splits
- `pooled` (882/252/126) + `loso_{BCCD,BCDD,PBC,Raabin}` (มีแล้วจาก `prepare_data.py`)
- 5 แขน × 5 splits × 3 seeds = 75 runs; รันลำดับ pooled ก่อน (15 runs) → หยุดคุยถ้า d_both ไม่ดีกว่า circlenet บน pooled → LOSO ทีละ fold

### 3.3 งบการเทรน
- DC-CellNet: epochs_max 150, patience 30, select_from_epoch 0, batch 8, AMP, aug เดิม
- YOLO26m: epochs 150, patience 30, batch 8, imgsz 640, seed ตาม run
- Faster R-CNN: epochs 50, patience 10 (val mAP50-95), batch 4, SGD ตาม torchvision reference; ระบุความต่างของงบในเปเปอร์
- threshold: ทุกแขน `evaluate.py --sweep-thr` บน val

### 3.4 การวัดผล
- evaluator กลาง: mAP50, mAP50-95, per-class AP, count MAE ต่อคลาส, recall ตาม overlap bin
- C2: mean mAP50-95 ข้าม 4 LOSO folds, Δ = pooled − LOSO mean, ต่อแขน
- การเทียบแขน: paired bootstrap ต่อภาพ (pool ทุก fold และ seed); mAP CI ด้วย `coco_map_ci` offline

### 3.5 เครื่องมือใหม่
- `scripts/run_exp.bat` / `run_exp.sh`: runner ทั่วไป — ENV: `EXP_PREFIX`, `EXP_SPLITS` (รายชื่อ split), `EXP_ARMS`, `EXP_SEEDS`; ต่อแขนเลือกตระกูล (dccnet / yolo / frcnn) ตามชื่อ config; resumable + rename-aside เหมือน run_m2
- configs `configs/p3b_*.yaml` (DC-CellNet) — split ส่งผ่าน CLI override ของ train.py (`--split`)
- `scripts/baselines/yolo_train.py`: เขียน Ultralytics data YAML ต่อ split (ภาพเดิม, label ลำดับคลาสของเรา) แล้วเทรน
- `scripts/baselines/frcnn_train.py` / `frcnn_predict.py`: เทรนจาก split JSON, export records
- `scripts/summarize_loso.py`: ตารางรวม pooled/LOSO/Δ + paired comparisons

---

## 4. P3c — BCCD overlap set (204 ภาพ)

1. `scripts/prelabel_bccd.py`: YOLO26m (pooled best.pt) conf ≥ 0.5 → merge กับ label เดิม: dedupe GT ที่ IoU > 0.7, เพิ่ม prediction ที่ IoU < 0.3 กับ GT ทุกกล่อง; export `data/bccd_relabel/{images,labels}` (YOLO txt) + `classes.txt` (RBC, WBC, Platelets) + `review_list.csv` (เรียงตามจำนวนกล่องที่เพิ่ม) + `progress` helper
2. Mickie แก้ใน X-AnyLabeling (กติกา: ทุก RBC ที่เห็นขอบ > ครึ่งเซลล์ รวมที่ถูกขอบภาพตัด)
3. `scripts/import_bccd_relabel.py` → `splits/bccd_overlap_v2.json` + สถิติ (กล่องเพิ่ม/ลบ/คงเดิม)
4. วัดทุกแขนที่เทรนบน pooled บนทั้ง label เดิมและ v2; รายงาน bias ของ pre-label จาก YOLO

---

## 5. ความเสี่ยง
| ความเสี่ยง | การรับมือ |
|---|---|
| d_both ไม่ดีบนข้อมูลจริง (TXL ตัดภาพซ้อนออก) | ดู pooled ก่อน; P3c เป็นตัวทดสอบหลักเรื่องการซ้อน |
| เวลา GPU 3–5 วัน | runner resumable; ลำดับ pooled → LOSO; หยุดคุยได้หลัง pooled |
| stride 2 OOM บนภาพจริง | smoke test ก่อน; ถ้า OOM ใช้ gradient accumulation (batch 4 × 2) แทนการลด batch จริง |
| pre-label bias เข้าทาง YOLO | Mickie ตรวจทุกภาพ; รายงานทั้งสอง label |
| novelty: PAC-MAP / Höfener ยังไม่ได้อ่าน | Mickie ดาวน์โหลด PDF → Claude อ่านก่อนเขียนเปเปอร์ |
| Faster R-CNN งบต่าง | ระบุในเปเปอร์; เป็น baseline อ้างอิง ไม่ใช่คู่แข่งหลัก |
