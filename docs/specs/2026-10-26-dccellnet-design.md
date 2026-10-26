# DC-CellNet — Design Spec

**วันที่:** 2026-10-26  
**ผู้เขียน:** Mickie (ร่วมกับ Claude)  
**สถานะ:** Draft — รอ review  
**อ้างอิงพื้นฐาน:** `claude/novelty-check-cell-detector.md` (Project) — รายงานตรวจ novelty

---

## 1. เป้าหมายและขอบเขต

### 1.1 ความตั้งใจ
พัฒนาอัลกอริทึม CNN ใหม่สำหรับ object detection ในภาพ blood smear (RBC / WBC / Platelets) ที่ออกแบบจากธรรมชาติของโดเมน — เซลล์เกือบกลม ขนาดค่อนข้างคงที่ และ RBC ซ้อนทับกัน — โดยไม่มี deadline

### 1.2 Contributions ที่จะอ้าง (ตกลงแล้ว)
| ID | Claim | สถานะในรอบนี้ |
|---|---|---|
| **C1** | Local count-consistency loss ที่บังคับให้จำนวน instance (heatmap peaks) สอดคล้องกับอินทิกรัลของ density map ในแต่ละหน้าต่าง + decode แบบไม่ใช้ NMS | แกนหลัก |
| **C2** | Leave-one-source-out (LOSO) evaluation บน TXL-PBC ครั้งแรก | แกนหลัก |
| **C4** | การประเมินแยกตามระดับการซ้อนทับ + counting error | แกนหลัก |
| C3 | การใช้ circle representation กับ peripheral blood smear (domain application ไม่ใช่ representation ใหม่) | framing รอง |
| C5 | Hierarchical WBC-subtype head + long-tail ภายใน detector (Raabin-WBC) | **นอกขอบเขต** — spec แยกรอบหน้า |

### 1.3 Claims ที่ห้ามอ้าง
- "circle/center-heatmap detection ใหม่" (CircleNet, arXiv 2006.02474, 2020)
- "NMS-free blood cell detector ตัวแรก" (DINO-DETR / RT-DETR บน TXL-PBC แล้ว)
- "density-aware blood cell detection ตัวแรก" (BloodContourNet, Diagnostics 2026)
- "count-regularized cell detection ตัวแรก" (ConCORDe-Net, arXiv 1908.00907)

### 1.4 ข้อจำกัด (constraints)
- ข้อมูล: public datasets เท่านั้น
- Compute: RTX 4070 (12 GB) ตัวเดียวบน Windows; ไดรฟ์ C: เหลือ ~31 GB ณ วันที่เขียน
- การแบ่งงาน: Claude เขียนโค้ด/เทสต์ (CPU VM, ไม่มี GPU) และวิเคราะห์ผล; Mickie สั่งรันเทรนบน Windows

---

## 2. สถาปัตยกรรมโมเดล

### 2.1 Backbone + neck
- ResNet-50 (ImageNet pretrained) + FPN แบบเบา → upsample รวมเป็น feature map เดียวที่ **stride 4**
- input 640×640 → output 160×160
- Ablation (C2): ConvNeXt-T (LayerNorm) แทน ResNet-50 (BatchNorm) — มีหลักฐานว่า BN ทำให้ cross-source generalization แย่ลง

### 2.2 Heads (ใช้ feature ร่วม; แต่ละหัวเป็น conv 3×3 → ReLU → conv 1×1)
| Head | Shape | ความหมาย | Loss |
|---|---|---|---|
| Center heatmap `H` | 3×160×160 | ความน่าจะเป็นจุดศูนย์กลาง ต่อคลาส | penalty-reduced focal loss (CenterNet) |
| Radius `R` | 1×160×160 | log(r) ในหน่วยพิกเซลของภาพ input | L1 บน log(r) + cIoU loss ที่ตำแหน่ง GT center |
| Offset `O` | 2×160×160 | sub-pixel offset ของ center | L1 ที่ตำแหน่ง GT center |
| Density `D` | 3×160×160 (ReLU) | density ต่อคลาส; แต่ละเซลล์อินทิเกรตได้ 1 | pixel MSE + L1 ของ count ต่อคลาสต่อภาพ |

Ground truth:
- Heatmap: Gaussian ที่ center, σ แปรตามรัศมี (แบบ CenterNet)
- Density: Gaussian normalized ให้ผลรวม = 1 ต่อเซลล์, σ = r/3 (ปรับได้ใน config)

### 2.3 Local Count-Consistency Loss (C1)
แบ่ง feature map เป็นหน้าต่าง `w×w` (default `w = 32` cells บน 160×160 → 5×5 = 25 หน้าต่าง; stride = w ไม่ซ้อน; ablation: แบบซ้อน stride w/2)

สำหรับแต่ละภาพ b, คลาส c, หน้าต่าง k:
- `P = 1[H == maxpool3x3(H)]` (mask peaks; ไม่มี gradient)
- `N_det(b,c,k) = Σ_{x∈k} H(x)·P(x)` — soft count; gradient ไหลผ่านค่า H
- `N_den(b,c,k) = Σ_{x∈k} D(x)`
- `L_cc = mean_{b,c,k} SmoothL1(N_det − N_den)`

ทิศทาง gradient: default คือ **stop-gradient ที่ N_den** (D ถูกสอนโดย GT อยู่แล้ว; L_cc มีหน้าที่ดัน H ให้แตก peak ตามจำนวนที่ D บอก) — ablation: two-way gradient

Warm-up: `λ_cc = 0` ใน `E_warm` epochs แรก (default 10) แล้วเพิ่มเชิงเส้นถึงค่าเป้าใน 5 epochs เพื่อไม่ให้ D ที่ยังไม่แม่นดึง H ผิดทาง

### 2.4 Total loss
`L = L_focal + λ_r·L_radius + λ_o·L_offset + λ_d·L_density + λ_cc·L_cc`
ค่าเริ่มต้น: `λ_r = 0.1, λ_o = 1, λ_d = 1, λ_cc = 0.5` (จะ sweep บน validation เท่านั้น)

### 2.5 Decoding (ไม่มี NMS)
1. Peaks: `H == maxpool3x3(H)` และ `H > τ` (τ เลือกบน validation)
2. แต่ละ peak → (class, cx+ox, cy+oy, r, score)
3. **Density-guided fill (optional, ablation):** ในหน้าต่างใดที่ `#peaks_c < round(N_den)` ให้รับ peak ที่ score สูงสุดถัดไปที่ `H > τ_low` จนครบหรือหมด
4. Export: วงกลม + กล่องที่ครอบวงกลม (`[cx−r, cy−r, 2r, 2r]`) สำหรับ evaluator กลาง

### 2.6 Ablations ที่โค้ดต้องรองรับ (ผ่าน config flags)
| Flag | ค่า |
|---|---|
| `heads.density` | on / off |
| `loss.cc.enabled` | on / off |
| `loss.cc.scope` | local / global (ทั้งภาพ) |
| `loss.cc.stop_grad_density` | true / false |
| `decode.nms` | none / circle-NMS / box-NMS |
| `decode.density_fill` | on / off |
| `repr` | circle / box (box mode = CenterNet: wh head แทน radius) |
| `backbone` | resnet50 / convnext_t |
| `aug.stain` | on / off |

หมายเหตุ: `density=off, cc=off, repr=circle, nms=none` = **CircleNet re-implementation** (ใช้เป็น baseline); `repr=box` = **CenterNet baseline**

---

## 3. ข้อมูล

### 3.1 Datasets
| Dataset | ใช้ทำอะไร | หมายเหตุ |
|---|---|---|
| **TXL-PBC** (1,260 ภาพ, YOLO format, 3 คลาส, มี metadata แหล่งที่มา) | train/val/test หลัก — pooled + LOSO | license ไม่ระบุในเปเปอร์ → ต้องเช็กหน้า GitHub/Figshare ก่อนเผยแพร่ |
| **BCCD full** (Shenggan, 364 ภาพ, Pascal VOC) | **Overlap test set** = เฉพาะ 172 ภาพที่ TXL-PBC ตัดทิ้ง | ห้ามใช้เทรน; label เดิมขาด RBC บางตัว → ต้อง audit |
| Chula RBC-12 (center-point labels) | counting / point-localization test (optional) | ใช้เฉพาะถ้า license อนุญาต; ไม่วัด mAP |

### 3.2 Splits
- **Pooled:** ใช้ split ทางการของ TXL-PBC (882/252/126) — 5 seeds
- **LOSO:** 4 folds (held-out: BCCD 160 / BCDD 100 / PBC 500 / Raabin 500); validation = 15% สุ่มจากแหล่งที่ใช้เทรน (stratified ตามแหล่ง); held-out source ไม่ถูกใช้ทั้ง train, val, หรือ tuning
- **Overlap set:** 172 ภาพ BCCD — test-only สำหรับทุกโมเดลที่เทรนบน pooled split

### 3.3 Dedupe (กัน leakage)
- pHash (64-bit) ทุกภาพจาก TXL-PBC, BCCD full, Chula, และ dataset Roboflow เดิมใน `C:\Research\blood-cell-detection-datatset`
- คู่ที่ Hamming distance ≤ 6 ถือว่าซ้ำ → รายงานใน `data/dedupe_report.csv`
- ถ้าภาพใน overlap set ซ้ำกับ TXL-PBC train → ตัดออกจาก overlap set และบันทึกจำนวนที่ตัด

### 3.4 Box → circle
- center = จุดกลางกล่อง, `r = √(w·h)/2`
- **Truncated cells:** กล่องที่ห่างขอบภาพ ≤ 2 px → `radius_valid = 0` (ยังเทรน heatmap/offset/density ตามปกติ แต่ไม่คิด radius/cIoU loss และไม่นับใน circle-mAP)

### 3.5 Overlap level (สำหรับ C4)
สำหรับแต่ละ GT cell: `k = จำนวน GT cell อื่นที่วงกลมตัดกัน (d < r_i + r_j)` → bins: 0 / 1 / 2 / ≥3

### 3.6 BCCD overlap-set audit
1. สุ่ม 30 ภาพ นับ RBC ที่ขาด label ด้วยตา (Mickie + Claude ช่วยดูภาพ)
2. ถ้าภาพที่ขาด ≥ 1 ตัวมีมากกว่า 10% → แก้ label ทั้ง 172 ภาพ (CVAT; บันทึกเป็น version ใหม่ `bccd_overlap_v2`)
3. รายงานทั้งผลบน label เดิมและ label ที่แก้แล้ว

### 3.7 Preprocessing / augmentation
- Letterbox resize เป็น 640×640
- Default aug: horizontal/vertical flip, rot90, scale jitter ±20%, translate ±10%
- **ไม่เปิด** stain/HSV augmentation เป็นค่า default — เป็นตัวแปร ablation (`aug.stain`)

---

## 4. การประเมิน

### 4.1 Evaluator กลาง
- ทุกโมเดล (ของเรา + baselines) export predictions เป็น COCO-style JSON เดียวกัน (`bbox`, `score`, `category_id`; ของเราเพิ่ม `circle: [cx, cy, r]`)
- วัดด้วย **pycocotools** สคริปต์เดียว (`dccnet/eval/coco_eval.py`) — ห้ามใช้ evaluator ภายในของแต่ละ framework ในตารางผล

### 4.2 Metrics
| กลุ่ม | Metrics |
|---|---|
| Detection | mAP50, **mAP50-95**, per-class AP; circle-mAP (cIoU) เป็นค่ารอง |
| Counting | per-image count MAE และ relative error ต่อคลาส (ใช้ score threshold ที่เลือกบน val) |
| Overlap | Recall / Precision / F1 ต่อ overlap bin (match ที่ IoU ≥ 0.5) |
| Cross-source | mAP50-95 ต่อ LOSO fold, ค่าเฉลี่ย, และ Δ = pooled − LOSO |
| Speed | FPS (batch 1, 640×640, RTX 4070, FP16) |

### 4.3 Baselines
| Baseline | ที่มา |
|---|---|
| YOLO26m | Ultralytics; config เดียวกับงานเดิม (150 epochs, patience 100) |
| Faster R-CNN | `CNN/train_detector.py` เดิม (ปรับ epochs ให้ยุติธรรม — ดู 4.5) |
| CenterNet (box) | โมเดลเรา `repr=box, density=off, cc=off` |
| CircleNet | โมเดลเรา `repr=circle, density=off, cc=off` |
| RT-DETR | Ultralytics (NMS-free) |
| YOLOv10 one-to-one | Ultralytics (NMS-free) |
| Cellpose | overlap set เท่านั้น, counting เท่านั้น |
| BloodContourNet | ตัวเลขที่รายงาน (pooled TXL-PBC: 87.8 mAP50-95) — ไม่มีโค้ด |

### 4.4 สถิติ
- Pooled: 5 seeds → mean ± std
- ทุก test set: bootstrap 95% CI จากภาพ (1,000 resamples) + paired comparison ต่อภาพ

### 4.5 ความยุติธรรมของการเทียบ
- ทุกโมเดลใช้ input 640, augmentation default เดียวกัน (เท่าที่ framework รองรับ — ระบุส่วนที่ต่างไว้ในเปเปอร์)
- Training budget: ทุกโมเดลเทรนจนกว่า validation mAP50-95 ไม่ดีขึ้น 30 epochs (early stopping) หรือสูงสุด 300 epochs
- Threshold / λ ทุกตัว tune บน validation เท่านั้น — test ถูกรันครั้งเดียวต่อ config สุดท้าย

### 4.6 เกณฑ์ความสำเร็จ (pre-registered)
1. **Primary:** บน overlap set — RBC count MAE ของ DC-CellNet ต่ำกว่า baseline ที่ดีที่สุด โดย bootstrap CI ไม่ทับซ้อน **และ** recall ใน bin ≥2 สูงกว่า baseline ที่ดีที่สุด
2. **Secondary:** LOSO mean mAP50-95 ≥ baseline ที่ดีที่สุด และ Δ(pooled − LOSO) น้อยกว่า
3. **Non-goal:** ไม่จำเป็นต้องชนะ 87.8 บน pooled — ยอมรับได้ถ้าห่าง ≤ ~1 จุด
4. **Negative result policy:** ถ้า L_cc ไม่ช่วย ให้รายงานตามจริงใน ablation แล้วทบทวนแกน C1 ก่อนเดินต่อ

---

## 5. โครงสร้างโค้ด

ตำแหน่ง: `C:\Research\dccnet\` ภายใน repo `mickieppf/Blood-Cell-Detection`

```
dccnet/
  dccnet/
    datasets/  txl.py · bccd_voc.py · splits.py · targets.py · dedupe.py
    models/    backbone.py · neck.py · heads.py · model.py
    losses/    focal.py · radius_ciou.py · density.py · count_consistency.py
    decode.py
    eval/      export_json.py · coco_eval.py · counting.py · overlap_bins.py · bootstrap.py
  configs/     base.yaml · exp_*.yaml
  scripts/     prepare_data.py · train.py · predict.py · evaluate.py · baselines/
  tests/       pytest (CPU, synthetic images)
  data/        (gitignored)
  runs/        (gitignored) <exp>/{config.yaml, git_hash.txt, metrics.jsonl, best.pt, summary.json}
  requirements.txt · README.md
```

หลักการ:
- แต่ละไฟล์มีหน้าที่เดียว; `targets.py` (สร้าง GT) และ `decode.py` ต้องเป็นฟังก์ชัน pure ที่เทสต์แยกได้
- 1 การทดลอง = 1 config; run directory เก็บ config + git hash เพื่อ reproducibility
- `summary.json` มี schema คงที่ เพื่อให้ Claude อ่านและเปรียบเทียบได้อัตโนมัติ

### 5.1 Tests ที่ต้องมี
- `targets` → `decode` round-trip: วงกลม GT บนภาพสังเคราะห์ถอดกลับได้ครบ (ตำแหน่ง ±1 px, รัศมี ±5%)
- Density GT: ผลรวมต่อภาพ = จำนวนเซลล์ (±1e-3), รวมเซลล์ที่โดนขอบตัด
- `count_consistency`: N_det ถูกต้องบน heatmap ที่รู้จำนวน peak; gradient ไม่เป็นศูนย์ที่ peaks
- Evaluator: mAP = 1.0 เมื่อ prediction = GT; ผลตรงกับตัวอย่าง pycocotools มาตรฐาน
- Overlap bins: กรณีสังเคราะห์ที่รู้คำตอบ

---

## 6. วิธีทำงานและ Milestones

**การแบ่งงาน:** Claude เขียนโค้ด + เทสต์ (CPU) และอ่าน `runs/` เพื่อวิเคราะห์/เขียน config รอบถัดไป; Mickie สร้าง conda env `dccnet` บน Windows ครั้งเดียว แล้วรัน `python scripts/train.py --config configs/exp_XX.yaml`

| Milestone | เนื้องาน | Gate |
|---|---|---|
| **M0** | Commit งานค้างแยก branch; ตรวจ/ล้างพื้นที่ดิสก์ (เฉพาะไฟล์ที่ Mickie อนุมัติ); ตั้ง env | env รัน `torch.cuda.is_available()` = True |
| **M1** | Data pipeline, splits, dedupe, overlap bins, evaluator กลาง; วัด YOLO26m ใหม่ผ่าน evaluator | YOLO26m pooled ผ่าน evaluator กลางได้ตัวเลขสมเหตุสมผล |
| **M2** | Toy proof: ภาพสังเคราะห์วงกลมซ้อนกัน — CircleNet vs CircleNet + density + L_cc | **Go/No-go:** L_cc ต้องลด count error / เพิ่ม recall ใน bin ≥2 บนข้อมูลสังเคราะห์ ไม่งั้นหยุดทบทวน |
| **M3** | DC-CellNet เต็มบน TXL-PBC pooled + วัด overlap set (+ BCCD audit) | ผลเทียบ CircleNet/CenterNet ได้ครบ |
| **M4** | LOSO 4 folds + baselines ทั้งหมด | ตารางหลักครบ |
| **M5** | Ablations ตาม 2.6 + เขียนเปเปอร์ | เกณฑ์ 4.6 ประเมินแล้ว |

C5 (WBC subtype) → spec แยกหลัง M4

---

## 7. ความเสี่ยงและการรับมือ
| ความเสี่ยง | การรับมือ |
|---|---|
| Crowd-counting ล่าสุด (STEERER, CLTR, PET, APGCC, arXiv 1904.03373) อาจมีกลไกเดียวกับ C1 | ตรวจ literature เพิ่มก่อน M3; ถ้าซ้ำ ปรับ claim เป็น "นำมาใช้กับ detection บน blood smear + C2/C4" |
| ยังไม่ได้อ่านตารางผล BloodContourNet และ AIP Advances 2025 (overlap) | Mickie ดาวน์โหลด PDF ไว้ใน `C:\Research\references\` ให้ Claude อ่าน |
| Label BCCD ไม่ครบ | Audit ตาม 3.6 |
| L_cc ทำให้เทรนไม่เสถียร | warm-up + stop-grad (2.3); ถ้ายังไม่เสถียร ลด λ_cc / ใช้ global scope ก่อน |
| ดิสก์เต็ม | ไม่เก็บ last.pt ทุก run (เก็บ best.pt เท่านั้น); ตรวจพื้นที่ใน M0 |
| License ของ TXL-PBC / Chula ไม่ชัด | ตรวจก่อนเผยแพร่โค้ดหรือ label ที่แก้ไข |

---

## 8. แหล่งอ้างอิงหลัก
- CircleNet — arXiv 2006.02474 (MICCAI 2020)
- BloodContourNet — Diagnostics 2026, doi 10.3390/diagnostics16152362
- TXL-PBC — arXiv 2407.13214; Scientific Data 2025
- ConCORDe-Net — arXiv 1908.00907
- Zheng et al. (decoupling counting & localization) — arXiv 2503.13989 (MICCAI 2024)
- Raabin-WBC — Scientific Reports 2022 (PMC8782871)
- Naruenatthanaset et al. (overlapping RBC, Chula RBC-12) — arXiv 2012.01321

---

## 9. Amendments ระหว่าง implementation (2026-10-27)

การตัดสินใจระหว่างลงมือ M0–M2 (บันทึกครบใน ledger) ที่เปลี่ยนจากข้อความข้างบน:

| ID | ส่วนที่เปลี่ยน | เดิม | ปัจจุบัน | เหตุผล |
|---|---|---|---|---|
| R13 | §2.3 peak mask | `P = 1[H == maxpool(H)]` | `P = 1[H == maxpool(H) ∧ H > 0.01]` (`loss.cc.peak_thr`) | local maxima ในพื้นหลังไปปน N_det และกด gradient ของ peak จริง |
| R21 | §2.3 soft count | `N_det = Σ H·P` | `N_det = Σ P·sigmoid((H − 0.3)/0.05)` (`loss.cc.count_fn: sigmoid`; `linear` = แบบเดิม ใช้ทำ ablation) | ให้นับจำนวน detection ไม่ใช่ความสูงของ peak; gradient ไปลงที่ peak ใกล้เส้นตัดสินใจ |
| R17/R18 | §2.2 density head | ReLU | softplus, bias init −7.0; count term normalize ด้วย max(Σgt, 1) | ReLU + count-L1 ทำให้ช่อง WBC/PLT ตายภายในราว 10 step |
| R14 | §2.2 heads | — | sigmoid/clamp/softplus คำนวณใน fp32; focal ทำ re-clamp ซ้ำ | กัน inf ตอนใช้ AMP |
| — | §2.1 input | — | normalize input ด้วย mean/std ของ ImageNet ภายในโมเดล | ให้ใช้ pretrained ResNet-50 ได้เต็มประสิทธิภาพ |
| R20 | §4.5 checkpoint | เลือก best ตั้งแต่ epoch 0 | `train.select_from_epoch = 15` (= warm-up + ramp) ทุกแขน; patience เริ่มนับจาก epoch นั้น | กันกรณีเลือก checkpoint ของแขน dccell ที่ยังไม่ได้เทรนด้วย L_cc |
| R16 | §4.2 decode for mAP | ใช้ score_thr ของ config | decode ที่ 0.001 / 1000 dets (เท่ากับ YOLO conf 0.001); operating threshold เลือกจาก val (max micro-F1) | ให้ mAP ยุติธรรมระหว่างโมเดล |
| R19 | §6 M2 | 300 / 30 | M2 synthetic: epochs_max 60, patience 20 | M2 เป็นการพิสูจน์กลไกบน toy data |
| — | §4.6 / §6 gate M2 | dccell vs circlenet | GO ต้องผ่านทั้ง dccell<circlenet **และ** dccell<density (paired bootstrap CI ของ RBC |err| < 0 + recall bin≥2 สูงกว่า + ชนะครบทุก seed); ข้อมูลไม่ครบ → INCONCLUSIVE | แยกผลของ L_cc ออกจากผลของ density head |
| — | §3 / §3.6 | overlap set = 172 ภาพ | 204 ภาพ (BCCD full ที่ไม่ซ้ำกับ TXL-PBC); 172 คือภาพที่ถูกคัดออกเพราะ `RBC_noboundary` → label น่าจะไม่ครบ ต้อง audit ก่อน M3 | จากการตรวจ `BCCD_selection.xlsx` |
| — | §1.4 GPU | RTX 4070 | RTX 5060 Ti 8 GB (sm_120, ต้องใช้ torch cu128); batch 8 + AMP | ฮาร์ดแวร์จริง |
