# ความใหม่อยู่ที่การผสาน ไม่ใช่ที่ชิ้นส่วนใด

**คำตอบสั้น:** ทั้งระบบยังใหม่ แต่แทบทุกชิ้นส่วนเคยมีคนทำแล้ว สูตรที่เสนอประกอบด้วย center heatmap + circle radius, density-map count-consistency loss, NMS-free decoding, การเน้น RBC ที่ซ้อนทับกัน และ leave-one-source-out (LOSO) บน TXL-PBC ซึ่งยังไม่พบในงาน blood smear ชิ้นใดที่อ่านมา ส่วนชิ้นส่วนย่อยนั้นมีงานก่อนหน้าแล้ว ได้แก่ CircleNet (2020) ที่ทำ center + radius + decoding แบบไม่ใช้ NMS ไว้แล้ว ([arXiv 2006.02474](https://arxiv.org/pdf/2006.02474)), ConCORDe-Net ที่ใส่ count loss ให้ cell detector ([ar5iv](https://ar5iv.arxiv.org/html/1908.00907)), ตระกูล DETR ที่เป็น NMS-free บน TXL-PBC อยู่แล้ว และ BloodContourNet (2026) ซึ่งเป็นงานที่ใกล้ที่สุด ใช้ "density-aware" reweighting กับ Bézier contour แต่ยังคงใช้ NMS ([MDPI](https://www.mdpi.com/2075-4418/16/15/2362)) ดังนั้นสิ่งที่อ้างได้จริงมีสามเรื่อง (1) loss ที่บังคับให้ **detection ระดับ instance ต้องสอดคล้องกับ integral ของ density map** (2) **การประเมินแบบ cross-source/LOSO** ที่ยังไม่มีใครทำบน TXL-PBC และ (3) **detector ที่มี hierarchical WBC-subtype head พร้อมการจัดการ long-tail** ข้อควรระวังสำคัญคือ TXL-PBC ตัดภาพ BCCD ที่มี RBC ซ้อนกันทิ้งไปแล้ว และ mAP50 อิ่มตัวที่ราว 97–99% ([arXiv 2407.13214v2](https://arxiv.org/html/2407.13214v2)) เพราะฉะนั้น claim เรื่อง RBC ซ้อนทับต้องพิสูจน์ด้วยข้อมูลชุดอื่น และใช้ metric ที่ละเอียดกว่า mAP50

> **แก้ citation:** CircleNet คือ **arXiv 2006.02474** (Yang ... Huo, MICCAI 2020) และฉบับ journal คือ arXiv 2110.12093 ส่วน **arXiv 2012.01321 เป็นอีกงานหนึ่ง** คือ Naruenatthanaset et al. (จุฬาฯ, Chula RBC-12, concave points + ellipse fitting) ([arXiv](https://arxiv.org/abs/2012.01321)) งานนี้ยังควรอ้างอิงในหัวข้อการแยก RBC ซ้อนทับ

## แต่ละชิ้นส่วนมีคนทำแล้วหรือยัง

| ชิ้นส่วน | สถานะ | งานที่ต้องอ้างอิง |
|---|---|---|
| Center heatmap + radius + offset | **ทำแล้ว** (โดเมน pathology) | CircleNet ([arXiv](https://arxiv.org/pdf/2006.02474)), CircleSnake ([arXiv 2403.11507](https://arxiv.org/pdf/2403.11507)), EllipseNet ([ar5iv](https://ar5iv.labs.arxiv.org/html/2109.12474)) |
| CenterNet บน blood smear | **ทำแล้ว** (WBC, box) | TFF-CenterNet: mAP@0.5 = 98.98% บน BCCD ([JOCA](https://www.joca.cn/EN/abstract/abstract25798.shtml)) และ CenterNet เป็น baseline ใน BloodContourNet |
| Circle บน **peripheral blood smear** (RBC/WBC/PLT) | **ยังไม่พบ** | — (มีงาน circular anchors + RetinaNet สำหรับ hematopoietic cells ([IEEE](https://ieeexplore.ieee.org/document/9098398/)) ที่ยังไม่ได้อ่านรายละเอียด) |
| Count loss บน detector ของเซลล์ | **มีงานใกล้เคียง** | ConCORDe-Net (count penalty บน probability map ไม่ใช่ density integral), TopoCount (บังคับให้มี c modes ต่อ patch) ([arXiv](https://arxiv.org/pdf/2012.12482)) |
| Density integral ↔ instance detections + NMS-free | **ยังไม่พบ** (ในงานที่ตรวจแล้ว) | DecideNet ใช้ fuse ไม่ได้ constrain ([ar5iv](https://ar5iv.arxiv.org/html/1712.06679)) และ Zheng et al. เสนอให้ *แยก* count กับ localization ออกจากกัน ([arXiv 2503.13989](https://arxiv.org/html/2503.13989v1)) |
| NMS-free บน blood cells | **ทำแล้ว** | DINO-DETR และ RT-DETR บน TXL-PBC (baseline ของ BloodContourNet), MSRT-DETR, GGC-DETR ([Springer](https://link.springer.com/article/10.1007/s10278-026-02292-5)) |
| "Density-aware" สำหรับเซลล์หนาแน่น | **ทำแล้วในรูปแบบอื่น** | BloodContourNet DARH ใช้จำนวนเพื่อนบ้านเพื่อถ่วงน้ำหนัก cls loss ไม่ได้ใช้ density map |
| LOSO / cross-source detection บน TXL-PBC | **ยังไม่พบ** | dataset paper และ BloodContourNet ใช้เพียง pooled split |
| Hierarchical + long-tail WBC subtype **ภายใน detector** | **ยังไม่พบ** | งานใกล้เคียงทั้งหมดเป็น crop classification ([arXiv 2604.23271](https://arxiv.org/pdf/2604.23271), [arXiv 2603.16249](https://arxiv.org/html/2603.16249v1)) |

## งานก่อนหน้าที่ใกล้ที่สุด และจุดที่เราต่างออกไป

- **BloodContourNet** (Diagnostics 2026, 16(15):2362) ([MDPI](https://www.mdpi.com/2075-4418/16/15/2362))
  - โครงสร้างคือ YOLOv26-L + HAM + DARH + BCRH (Bézier contour แบบ weakly supervised จาก ellipse prior ของ box)
  - **ยังใช้ class-wise NMS** แล้วตรวจ contour-IoU เพิ่มเฉพาะคู่ที่ box-IoU อยู่ระหว่าง 0.45–0.75
  - "Density" ในงานนี้คือจำนวน GT centers ภายในรัศมี 1.5·√(wh) ที่ใช้ถ่วง classification loss อย่างเดียว ไม่มี density map และไม่มี count loss
  - ผลลัพธ์บน TXL-PBC คือ **99.1 mAP50 / 87.8 mAP50-95** (เฉลี่ย 5 seeds, 95% CI 87.5–88.0) ไม่มี cross-dataset และประเมินเฉพาะ box
  - **จุดที่เราต่าง:** มี count regression, ไม่ใช้ NMS และใช้ radius แทน Bézier แต่ reviewer น่าจะถามว่า "radius เป็น shape prior ที่ง่ายกว่า BCRH หรือไม่" จึงควรเตรียมคำตอบไว้
- **CircleNet / CircleSnake / EllipseNet**
  - CircleNet-DLA ได้ AP 0.647 เทียบกับ CenterNet-DLA ที่ 0.598 บน glomeruli และนิยาม cIoU แบบ closed form ([arXiv](https://arxiv.org/pdf/2006.02474))
  - CircleSnake ยอมรับข้อจำกัดเองว่า circle ไม่เหมาะกับวัตถุทรงรี และ feature แบบ single-scale ทำงานแย่เมื่อขนาดวัตถุต่างกันมาก ([arXiv 2403.11507](https://arxiv.org/pdf/2403.11507)) ปัญหานี้ตรงกับกรณี RBC/WBC/PLT ที่ขนาดต่างกันมาก
  - EllipseNet พบว่าถ้าฝึกด้วย IoU loss อย่างเดียว ellipse จะยุบกลายเป็น circle ([ar5iv](https://ar5iv.labs.arxiv.org/html/2109.12474))
- **ConCORDe-Net** (2019): Dice + K·count-error โดย K=0.3 ผ่าน counter ที่ pretrain แล้ว decode ด้วย threshold + centroid ทำให้ F1 ขึ้นจาก 0.845 เป็น 0.873 โดย recall ขึ้นแต่ precision ลง ([ar5iv](https://ar5iv.arxiv.org/html/1908.00907)) ผลนี้เตือนว่า count loss อาจทำให้เกิด false positive เพิ่ม
- **TopoCount** (AAAI 2021) ใช้ persistence loss เพื่อบังคับให้มี c modes ต่อ patch ส่วน **P2PNet** (ICCV 2021) ได้ count = จำนวน points ผ่าน Hungarian matching โดยไม่ต้องใช้ density branch ([arXiv](https://arxiv.org/pdf/2107.12746)) ทั้งสองงานใกล้เคียงในเชิงแนวคิด แต่ไม่ได้ผูก detection เข้ากับ density integral
- **Zheng et al. "Decoupling Counting and Localization"** (MICCAI 2024, LNCS 15004; arXiv 2503.13989 ขึ้นปี 2025) ([MICCAI](https://papers.miccai.org/miccai-2024/655-Paper2995.html))
  - งานนี้เสนอแนวทาง**ตรงข้าม**กับเรา คือแยก counter ออกจาก localizer
  - ได้ MAE: DCC 0.8, MBM 1.4 และผู้เขียนยอมรับว่าเซลล์ที่ซ้อนกันแยกได้ยาก
  - **ต้องตอบงานนี้โดยตรง** ด้วย ablation ที่แสดงว่าการ couple ดีกว่าการ decouple ในกรณี RBC ซ้อนทับ
- **ผลของ NMS-free ในฉากหนาแน่น**
  - YOLOv10 แบบ o2o ด้อยกว่า o2m+NMS อยู่ 0–1 AP บน COCO ([arXiv](https://arxiv.org/html/2405.14458v2))
  - ใน crowd ตัว query-based detector ยังให้ duplicate predictions ต่อวัตถุเดียว ([ar5iv 2203.07669](https://ar5iv.labs.arxiv.org/html/2203.07669))
  - ยังไม่พบงานที่วัดความล้มเหลวของ NMS-free บนเซลล์ซ้อนทับ ช่องว่างนี้เป็นโอกาสให้ทำ experiment ได้
- **การแยก RBC ซ้อนทับแบบเดิม**
  - Watershed ใช้ได้เฉพาะ "เมื่อ centers อยู่ห่างกัน" และ concave-point ล้มเหลวเมื่อมี concave point เดียว Naruenatthanaset แยกได้ถูก 0.889 จาก 277 contours ([ar5iv](https://ar5iv.arxiv.org/html/2012.01321))
  - Moallem et al. (SPIE 2018) ใช้ mean-shift หา center ก่อน แล้วจึงหาขอบเขต ได้ F1 95% ([NLM](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/Detectingandsegmentingoverlappingredbloodcellsinmicroscopicimagesofthinbloodsmears.html)) ดังนั้น **แนวคิด "หา center ก่อน" ไม่ใหม่**
  - ในตัวอย่างที่เซลล์จับกลุ่ม (199 cells) Cellpose นับได้เพียง 113, Mask R-CNN 172 และ YOLOv8 183 ([SciTePress](https://www.scitepress.org/Papers/2024/124550/124550.pdf)) ผลนี้ใช้เป็น motivation ได้ดี

## Claim ที่อ้างได้ วิธีเขียน และสิ่งที่ห้ามอ้าง

**Claim ที่อ้างได้ (เรียงจากแข็งแรงที่สุด):**

- **C1 – Density-integral consistency สำหรับ instance detections:** "we tie the *instance-level* detection set (e.g., Σ peak confidences / number of decoded centers) to the integral of a jointly predicted density map via a differentiable consistency loss, enabling NMS-free decoding in dense RBC clusters"
  - ต้องอ้าง ConCORDe-Net, TopoCount, P2PNet และ DecideNet ในประโยคเดียวกัน แล้วอธิบายว่าต่างกันอย่างไร
- **C2 – First cross-source (LOSO) detection benchmark บน TXL-PBC:** TXL-PBC มี metadata ระบุที่มาของแต่ละภาพ จึงทำ LOSO ได้ ([arXiv 2407.13214v2](https://arxiv.org/html/2407.13214v2)) ขณะที่ทั้ง dataset paper และ BloodContourNet ใช้ pooled split
  - เขียนว่า "to our knowledge, the first leave-one-source-out evaluation on TXL-PBC"
  - หลักฐานว่าเรื่องนี้สำคัญ: งาน WBC cross-lab รายงาน mAP@0.5 ตกจาก **0.945 → 0.346** ([arXiv 2409.06742](https://arxiv.org/html/2409.06742v1))
- **C3 – Circle-based detection สำหรับ peripheral blood smear แบบ multi-class:** อ้างได้ในฐานะ *การนำมาใช้กับโดเมนใหม่พร้อมปรับแก้* ไม่ใช่ representation ใหม่
  - เขียนว่า "we adapt the circle representation [CircleNet] to multi-scale RBC/WBC/PLT detection in blood smears"
- **C4 – Overlap-stratified + counting evaluation:** ยังไม่พบ blood detector ที่รายงาน count MAE หรือแยก metric ตามระดับความซ้อนทับ จึงอ้างได้ว่าเป็น contribution ด้านการประเมิน
- **C5 (ระยะต่อไป) – Hierarchical cell-type → WBC-subtype head ภายใน single-stage detector + long-tail loss:** ยังไม่พบงานที่ทำแบบนี้ ([arXiv 2604.23271](https://arxiv.org/pdf/2604.23271) และ MFDS-DETR ไม่มีทั้งสองอย่าง)

**สิ่งที่ห้ามอ้าง:**

- ห้ามอ้างว่า "เป็นคนแรกที่ใช้ circle/radius ในการตรวจจับเซลล์" เพราะ CircleNet ทำไว้แล้วตั้งแต่ 2020
- ห้ามอ้างว่า "เป็นคนแรกที่ใช้ CenterNet กับ blood cells" เพราะมี TFF-CenterNet และ CenterNet เป็น baseline ใน BloodContourNet
- ห้ามอ้างว่า "เป็น NMS-free blood detector ตัวแรก" เพราะมี DETR family อยู่แล้ว
- ห้ามอ้างว่า "เป็นคนแรกที่ทำ density-aware blood detection" เพราะคำนี้ BloodContourNet ใช้ไปแล้ว ให้ใช้ "density-*integral* consistency" แทน
- ห้ามอ้างว่า "เป็นคนแรกที่ใช้ count-regularized cell detection" เพราะมี ConCORDe-Net
- ห้ามอ้างว่า "เป็นคนแรกที่ตรวจหา center ของ RBC ที่ซ้อนกัน" เพราะมี Moallem 2018 และ seed prediction 2016 ให้เขียนว่า *learned end-to-end jointly with class* แทน
- ห้ามอ้าง SOTA จาก mAP50 บน TXL-PBC เพราะค่าอิ่มตัวแล้ว
- ห้ามอ้างว่าแก้ปัญหา overlap ได้ หากทดสอบบน TXL-PBC เพียงชุดเดียว

## Protocol การประเมินที่แนะนำ

**Metrics:**

- mAP50-95 (ตัวหลัก) + per-class AP โดยเฉพาะ Platelets ซึ่งเป็นคลาสที่อ่อนที่สุด (mAP50-95 ราว 0.76–0.77 สำหรับ YOLOv8s/v11s) ([arXiv 2407.13214v2](https://arxiv.org/html/2407.13214v2))
- รายงาน **mAP50 ไว้เพื่ออ้างอิงเท่านั้น**
- **Count MAE/MAPE ต่อภาพ** (แยกตามคลาส)
- **Metrics แบบแบ่งตามระดับความซ้อนทับ**: isolated / 2-cell / 3-cell / >3 ตามแบบของ Naruenatthanaset หรือแบ่งตาม neighbor-count ตามนิยามของ BloodContourNet เพื่อให้เทียบกันได้
- cIoU-based AP (ตาม CircleNet) ควบคู่กับ box AP
- **Multi-seed (≥5) + 95% CI + paired test** เพราะ BloodContourNet ชนะ baseline เพียง +0.7–1.4 mAP50-95 ([MDPI](https://www.mdpi.com/2075-4418/16/15/2362))

**Splits:**

- In-distribution: pooled 7:2:1 (882/252/126, seed 0) เพื่อเทียบตรงกับ BloodContourNet ข้อควรระวังคือ dataset paper รายงาน split ไม่ตรงกันเอง (มีทั้ง 882/252/126 และ 1,008/288/144)
- **LOSO 4 folds** (BCCD / BCDD / PBC / Raabin) ให้รายงาน per-class AP ต่อ source ที่ hold out เพราะสัดส่วนคลาสต่างกันมาก ตัวอย่างเช่นภาพจาก PBC มักมี WBC เพียงหนึ่งตัวอยู่กลางภาพ
- Overlap-rich test: BCCD ฉบับเต็ม 364 ภาพ (รวมภาพที่ TXL-PBC ตัดทิ้ง) และ/หรือ Chula RBC-12

**Baselines ที่ต้องมี:**

- YOLOv8/v11/v26 (+NMS), **YOLOv10 o2o (NMS-free)**, RT-DETR และ DINO-DETR (NMS-free)
- **Vanilla CenterNet** และ **CircleNet** (มี code ที่ hrlblab/CircleNet)
- BloodContourNet ถ้า code เปิดเผย ถ้าไม่เปิดก็อ้างตัวเลขที่ตีพิมพ์บน split เดียวกัน
- Cellpose, watershed และ circle Hough สำหรับงาน counting ([SciTePress](https://www.scitepress.org/Papers/2024/124550/124550.pdf)) และควรเพิ่ม StarDist
- **สำหรับ cross-source:** stain normalization/augmentation (Macenko/HED) และ backbone แบบ GroupNorm หรือ frozen-BN เพราะ ResNet50-BN ตกเหลือ 23% บน LISC ขณะที่ GN ได้ 73% ([arXiv 2303.01777](https://ar5iv.labs.arxiv.org/html/2303.01777))

**Ablations ที่จำเป็น:**

- (a) มี/ไม่มี count-consistency loss (ตรวจ precision drop แบบที่เกิดใน ConCORDe-Net)
- (b) NMS-free เทียบกับ NMS บน decoder เดียวกัน
- (c) Coupled เทียบกับ decoupled counting (ตอบ Zheng et al.)
- (d) Circle เทียบกับ box เทียบกับ ellipse head
- (e) Max-pool kernel ของ peak extraction เทียบกับระยะห่างระหว่าง RBC centers ที่ซ้อนกัน (เป็นจุดอ่อนที่อนุมานจากการ decode แบบ 3×3 ของ CircleNet แต่ยังไม่มีงานทดสอบ)

## Dataset: TXL-PBC ไม่พอสำหรับ claim เรื่อง overlap

- **TXL-PBC** (Scientific Data 2025) ([Nature](https://www.nature.com/articles/s41597-025-05980-z))
  - 1,260 ภาพ / 18,143 boxes (RBC 16,302, WBC 1,298, PLT 543) มาจาก BCCD 160, BCDD 100, PBC 500 และ Raabin 500
  - **มีการตัดภาพ BCCD ที่มี RBC ซ้อนกันออก 172 ภาพ** ([arXiv 2407.13214v2](https://arxiv.org/html/2407.13214v2))
  - Baselines ได้ mAP50 0.969–0.984 และ mAP50-95 0.786–0.864 ส่วน BloodContourNet ได้ 87.8 ซึ่งเป็นเพดานในปัจจุบัน
  - Dataset ไม่ระบุ license จึงต้องตรวจสอบก่อนตีพิมพ์
- **BCCD (364 ภาพเต็ม):** ใช้เป็นชุดทดสอบ overlap แต่ split ต่างกันในแต่ละ paper (CST-YOLO 92.7 mAP50, TE-YOLOF 58.4 AP) จึงต้องรันทุกอย่างใหม่บน split เดียวกัน ([arXiv](https://arxiv.org/abs/2108.12313))
- **Chula RBC-12** (งานจากไทย): 706 ภาพ, 20,875 cells, 12 คลาส มี overlap จริง ([arXiv 2012.01321](https://arxiv.org/abs/2012.01321)) แต่บาง release มีเพียง point labels ([arXiv 2106.00389](https://arxiv.org/pdf/2106.00389)) ซึ่งเข้ากับ center-heatmap ได้ดี
- **สำหรับ WBC subtype:**
  - **Raabin-WBC** มี boxes, 5 subtypes, CC BY 4.0, 17,965 cells (Neutrophil 10,862 ต่อ Basophil 301) ([PMC8782871](https://pmc.ncbi.nlm.nih.gov/articles/PMC8782871/))
  - **ข้อควรระวัง:** Test-B มีเพียง Neutrophil/Lymphocyte จึงวัด rare class ไม่ได้ (CNN ได้ 99% บน Test-A แต่ได้เพียง 4–32% บน Test-B) ให้ใช้ **LISC ที่ relabel เป็น box** (250 ภาพ, สัดส่วนคลาสสมดุล) เป็น cross-dataset target แทน
  - WBCDD (EOS 14, BAS 13 boxes) เป็น extreme-tail test ขนาดเล็ก ([arXiv 2401.00926](https://arxiv.org/pdf/2401.00926))
- **สำหรับส่วนขยาย hierarchical:** หลักฐานจาก WBCBench บ่งว่า hierarchy ให้ผล +0.05–0.08 macro-F1 ขณะที่ long-tail loss ทั่วไป (oversampling, LDAM, decoupling) ให้เพียง ±0.01 ([arXiv 2604.23271](https://arxiv.org/pdf/2604.23271), [arXiv 2603.16249](https://arxiv.org/html/2603.16249v1)) จึงควรให้น้ำหนักกับ hierarchy มากกว่าการปรับ loss และใช้ PBC (1,218 basophils) เป็นแหล่ง crop สำหรับทำ copy-paste augmentation

## สิ่งที่ยังต้องตรวจสอบก่อนเขียน claim

- **ตาราง results ของ BloodContourNet ฉบับเต็ม** (ตัวเลข CenterNet/RT-DETR, per-class, ablation และมี dense-subset หรือ counting metric หรือไม่) ตอนนี้ดึงได้เพียงส่วน methods เพราะ PMC ติด reCAPTCHA ต้องเปิดอ่านเอง
- **Full text ที่ยังไม่ได้อ่าน:**
  - GGC-DETR, MSRT-DETR, ASBHDETR
  - AIP Advances 2025 "small targets and overlapping areas" ([AIP](https://pubs.aip.org/aip/adv/article/15/9/095033/3364445/Construction-of-an-efficient-blood-cell-detection)) ซึ่งเสี่ยงที่สุดต่อ claim เรื่อง overlap
  - Frontiers in Medicine 2026 ([Frontiers](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1934762/full))
  - งาน circular anchors RetinaNet สำหรับ hematopoietic cells
  - U-Net++ + YOLOv5 counting
- **Density + detection ในงาน crowd ระหว่าง 2023–2026** (STEERER, CLTR, PET, APGCC) และ arXiv 1904.03373 อาจมี density-integral term อยู่แล้ว **งานกลุ่มนี้คุกคาม C1 มากที่สุด**
- ค้นใน venue hematology โดยตรง (JBHI, CBM, MedIA) และ Google Scholar ด้วยคำค้น "blood smear density map detection", "CircleNet blood" และ "RBC heatmap"
- นิยาม L_qua ของ DecideNet ซึ่งข้อความถูกตัด จึงยังตัดความเป็นไปได้ที่มี consistency แบบอ่อนออกไปไม่ได้
- Venue ของ ConCORDe-Net (น่าจะเป็น MICCAI 2019) และ CircleNet journal (น่าจะเป็น IEEE TMI) ซึ่งยังไม่ได้ยืนยัน
- License ของ TXL-PBC, WBCDD และ LISC ฉบับ relabel
- SCL-YOLO บน TXL-PBC ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5165561)) ซึ่งยังไม่ได้ตรวจตัวเลข

## Conclusion

การตรวจครั้งนี้ทำให้ต้องเปลี่ยนจุดยืนของงาน งานนี้ไม่ควรขายตัวเองว่าเป็น "detector แบบใหม่" แต่ควรเป็น**งานที่ทดสอบสมมติฐานเฉพาะ** คือ "ถ้าบังคับให้ detections สอดคล้องกับ density integral จะลด miss/duplicate ในกลุ่ม RBC ที่ซ้อนกันได้ เมื่อไม่มี NMS มาช่วย" โดยมี LOSO และ overlap-stratified metrics เป็นหลักฐาน ข้อดีของกรอบนี้คือ แม้ mAP50-95 แบบ pooled จะไม่ชนะ 87.8 ของ BloodContourNet งานก็ยังมีคุณค่า เพราะ contribution ด้าน cross-source, counting และ overlap ยังไม่มีใครครอบครอง

ลำดับงานที่แนะนำ (ไม่มี deadline จึงทำทีละขั้นได้):

1. ปิดช่องว่างจากรายการตรวจสอบข้างต้น โดยเฉพาะงาน crowd ช่วง 2023–2026 และตาราง BloodContourNet
2. สร้าง LOSO benchmark ด้วย baseline ที่มีอยู่แล้ว ขั้นนี้เป็นผลงานที่ตีพิมพ์ได้ด้วยตัวเอง และความเสี่ยงต่ำที่สุด
3. ทำ CircleNet บน blood smear + count-consistency loss พร้อม ablation เทียบกับ Zheng et al.
4. ทำ hierarchical head บน Raabin-WBC โดยใช้ LISC เป็น target ภายนอก
