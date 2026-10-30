# ความใหม่ของ adaptive sigma อยู่ที่การรวมกฎ ไม่ใช่ไอเดีย

**คำตอบสั้น:** ไอเดีย "ให้ความกว้างของเป้าหมายขึ้นกับระยะถึงเพื่อนบ้าน" **ไม่ใหม่** เพราะมีใช้แล้วใน crowd counting/localization แต่ **รูปแบบเฉพาะของเรา** คือ per-cell Gaussian sigma ใน CenterNet/CircleNet-style heatmap ที่ cap ด้วย `min(r/(3·stride), d_nn/(4·stride))` โดย `d_nn` นับเฉพาะเพื่อนบ้าน same-class และใช้กับ blood-smear cells **ยังไม่พบในงานที่ตรวจมา** ผล search ทุกชุดให้ข้อสรุปตรงกันว่า ไม่มีงานใน CenterNet-family ที่ cap Gaussian width ของ class heatmap ด้วยระยะถึง same-class neighbour และไม่มีงาน blood smear ที่ปรับ target ตามเพื่อนบ้าน งานที่ใกล้ที่สุดคือ **repel coding** (Liang et al.) ที่ใช้ระยะถึงเพื่อนบ้านปรับ *decay* ของ proximity map, **LDC-Net** ที่ใช้ `min(c, 0.25·d_nn)` กำหนดขนาด blob (ไม่ใช่ Gaussian) และ **geometry-adaptive kernel** (MCNN/CSRNet) ที่ใช้ sigma = 0.3 × ค่าเฉลี่ยระยะ 3 เพื่อนบ้านที่ใกล้สุด ข้อควรระวังคือหลักฐาน "ไม่พบ" มาจากการค้นที่จำกัด (ราว 17–25 tool calls ต่อหัวข้อ) และผลทดลองของเรายังเป็น synthetic เท่านั้น ดังนั้นควรเขียน claim แคบๆ ว่า "to our knowledge" และเป็น **ข้อเสนอเชิงประสบการณ์ที่มี ablation รองรับ** ไม่ใช่ "first"

---

## 1. เทียบสูตรของเรากับงานที่ใกล้ที่สุด

สูตรของเรามีสองส่วน ส่วนแรก `r/(3·stride)` คือกฎ "รัศมีวัตถุ ≈ 3σ" ซึ่งตรงกับแนวปฏิบัติของ CenterNet อยู่แล้ว (σ ผูกกับขนาด box ของตัวเอง) ส่วนที่ใหม่จริงมีแค่ **ส่วนที่สอง** คือ cap ด้วย `d_nn/(4·stride)` ([notes: detectors.md Inferences](https://arxiv.org/pdf/1904.07850))

### ตารางเทียบ

| งาน | สิ่งที่ทำ | ต่างจากเราอย่างไร |
|---|---|---|
| **Repel coding** (Liang et al., [arXiv 1904.08864](https://arxiv.org/pdf/1904.08864)) | `D' = dist1·(1 + dist1/dist2)²`, `C = 1/(1+αD')` ใช้ระยะถึงเพื่อนบ้านที่ใกล้ที่สุดและที่ใกล้เป็นอันดับสอง | ใกล้ที่สุด ปรับ *decay* ของ proximity map ไม่ใช่ σ ต่อเซลล์ ใช้ α และ r แบบ global ไม่มี own-size term ไม่ใช่ peak=1 แบบ CenterNet/focal loss เป็น arXiv (ไม่พบ journal) F1 ดีขึ้นสูงสุด ~14 จุดบน HBM กับ net อ่อน และลดลงเมื่อ net แรงขึ้น |
| **LDC-Net** ([arXiv 2110.04727](https://arxiv.org/pdf/2110.04727)) | `w = h = min(c, r·d_nn)`, r = 0.25, c = cap ต่อ dataset | **ใกล้ที่สุดในเชิงโครงสร้าง** (เป็น cap แบบ min เหมือนกัน และ 0.25 ตรงกับ `/4` ของเรา) แต่เป็น dilated blob ไม่ใช่ Gaussian, class-agnostic (head points), c เป็นค่าคงที่ต่อ dataset ไม่ใช่ per-object radius, ไม่มี ablation แยกเฉพาะการ sizing |
| **TopoCount** ([ar5iv 2012.12482](https://ar5iv.labs.arxiv.org/html/2012.12482)) | dot radius ≤ ½·d_nn, box data ใช้ `max(7, w/2, h/2)` | cap ด้วย NN distance เหมือนกัน แต่เป็น binary disk ตัวหาร 2 ไม่ใช่ Gaussian ในหน่วยของเรา 3σ = 0.75·d_nn ซึ่งเกิน ½·d_nn (อนุมานเอง) |
| **IIM** ([arXiv 2012.04164](https://arxiv.org/pdf/2012.04164)) | dilate dot mask (สูงสุด 31 px) ไม่ให้ทับกัน, ย่อ box จนระยะห่าง > ¼ ของความกว้าง | ¼ เป็น margin แยก box ไม่ใช่ตัวหาร σ, เป็น binary label |
| **Geometry-adaptive kernel** (MCNN, CSRNet: [arXiv 1802.10062](https://arxiv.org/pdf/1802.10062)) | σᵢ = β·d̄ᵢ, β = 0.3, d̄ = ค่าเฉลี่ย k เพื่อนบ้านที่ใกล้สุด (k = 3 ตาม CSRNet) | **Gaussian ที่ขึ้นกับเพื่อนบ้านจริง** แต่ใช้ mean-k ไม่ใช่ nearest, ไม่มี own-size cap (วัตถุโดดเดี่ยวได้ σ ใหญ่มาก), เป้าหมายคือ density map ที่ integral = count ไม่ใช่ peak localization, class-agnostic |
| **FIDT** ([arXiv 2102.07925](https://arxiv.org/pdf/2102.07925)) | distance transform `I = 1/(P^(αP+β)+C)` | แก้ปัญหา Gaussian ทับกันด้วยการเปลี่ยน target; NN เข้าทาง P ระดับ pixel ไม่ใช่ความกว้างต่อ object |
| **LSC-CNN** ([arXiv 1906.07538](https://arxiv.org/pdf/1906.07538)) | ขนาด box = ระยะถึง NN | ใช้ k=1 เหมือนเรา แต่กำหนด *ขนาด* ผ่าน classification ไม่ใช่ความกว้าง kernel |
| **SC-CNN / Xie** ([PMC5226438](https://pmc.ncbi.nlm.nih.gov/articles/PMC5226438)) | proximity map, r และ α คงที่ (Xie: r=5, α=0.8) | ความกว้างเท่ากันทุกเซลล์; กันการรวมด้วย patch/architecture |
| **PAC-MAP** ([bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.07.18.602066)) | 3D U-Net ทำนาย centroid + proximity ต่อ nuclei อื่น | **ยังไม่ได้อ่านสูตร** ห้ามระบุความต่างเกินกว่า "ทำนาย proximity เป็น output" |
| **CenterNet / CornerNet / mmdet `gaussian_radius`** ([arXiv 1904.07850](https://arxiv.org/pdf/1904.07850)) | radius จาก IoU กับขนาด box ของตัวเองเท่านั้น; overlap ใช้ element-wise max | size-adaptive แต่ไม่ neighbour-adaptive; สูตร radius ที่เห็นมาจาก mirror ของ mmdet |
| **TTFNet** ([arXiv 1909.00700](https://arxiv.org/pdf/1909.00700)) | σx = αw/6, σy = αh/6, α = 0.54 | own-box เท่านั้น ablation β ชี้ว่า target เล็กเกินและใหญ่เกินก็แย่ (27.2 → 32.5 → 29.5 AP) |
| **Solar Corona** ([arXiv 2202.06565](https://arxiv.org/pdf/2202.06565)) | เปลี่ยนรูป kernel ตามด้านของ oriented box | ยอมรับปัญหาเดียวกัน (kernel ใหญ่กด peak เพื่อนบ้าน) แต่แก้ที่รูป kernel ไม่ใช่ neighbour term |
| **SWAHR** ([ar5iv 2012.15175](https://ar5iv.labs.arxiv.org/html/2012.15175)) | σ = σ₀·s โดย s เป็น output ของ network | adaptive ฝั่ง **output (learned)** ของเราเป็นฝั่ง **label (คำนวณจาก GT geometry)** ไม่เกี่ยวกับ neighbour |
| **CircleNet** ([arXiv 2006.02474](https://arxiv.org/pdf/2006.02474)) | Gaussian ตาม CornerNet/CenterNet, radius head แยก | paper ไม่ระบุว่า σ คำนวณอย่างไรและไม่เชื่อมกับ neighbour; ต้องตรวจโค้ด hrlblab/CircleNet ก่อนเขียน baseline |

### สิ่งที่ตารางบอก

- **ไม่มีงานไหนทำครบสามอย่างพร้อมกัน:** (a) Gaussian ต่อเซลล์ (b) cap ด้วย `min(own-size, d_nn/k)` (c) same-class ใน CenterNet-style heatmap
- **ความกังวลหลัก:** reviewer ที่รู้ LDC-Net หรือ TopoCount จะมองว่าเป็น "นำกฎ NN-cap ไปใช้กับ Gaussian heatmap" ดังนั้น novelty อยู่ที่ *การย้ายโดเมนและตรวจสอบเชิงประจักษ์* มากกว่าตัวสูตร
- **ตัวเลข 0.25 / ¼ ตรงกับ LDC-Net และ IIM** เป็นความบังเอิญที่ต้องบอกเองในบทความ ไม่ปล่อยให้ reviewer ค้นพบ
- **หลักฐาน "narrow target ช่วยแยก peak" ยังบาง:** notes ไม่พบการทดลองควบคุมในงาน CenterNet-family; ผลของเราจึงเป็น evidence ใหม่ในจุดนี้ แต่ LDC-Net ก็ไม่ได้ ablate ส่วน sizing แยกออกมา

### ตรวจสอบทางคณิตศาสตร์ (คำนวณเอง ไม่ใช่ผลจาก paper)

- เมื่อเพื่อนบ้านห่างกัน `d_nn` ใน feature map จะห่างกัน 4σ ค่า saddle ที่จุดกึ่งกลางภายใต้ max-merge = exp(−2) ≈ **0.135** ซึ่งต่ำพอให้ 3×3 max-pool แยก peak ได้
- ค่า floor σ = 0.5 cells จะทำงานเมื่อ `d_nn < 2·stride` นั่นคือเพื่อนบ้านใกล้เกินกว่าที่ grid ความละเอียดนี้แยกได้ น่าจะเป็นเหตุผลที่ stride-2 ให้ผลดี (−16%) แต่เป็นข้อสันนิษฐาน ต้องทดสอบ

---

## 2. Claim ส่วนไหนปกป้องได้ และควรเขียนอย่างไร

### ระดับความแข็งแรง

| ส่วนของ claim | สถานะ | หมายเหตุ |
|---|---|---|
| "ความกว้าง target ขึ้นกับเพื่อนบ้าน" | **อ้างความใหม่ไม่ได้** | MCNN/CSRNet, LDC-Net, TopoCount, repel coding |
| "σ ≈ r/3" | **อ้างไม่ได้** | แนวปฏิบัติ CenterNet |
| `min(own-size, NN-cap)` กับ **Gaussian peak heatmap** | **อ้างได้ แต่แคบ** | ใกล้ LDC-Net/TopoCount ต้องยอมรับ |
| Same-class neighbour | **อ้างได้ แต่บาง** | แต่ละ class channel แยกกันอยู่แล้ว จึงเป็นผลโดยธรรมชาติ ต้อง ablate เพื่อให้เป็น finding |
| ใช้ใน **blood-smear center-heatmap detector** | **อ้างได้ (to our knowledge)** | ไม่พบงาน blood smear ที่ปรับ target ตามเพื่อนบ้าน (ค้นตื้น ~3 queries) |
| ผลเชิงประจักษ์: count MAE −22%, recall(≥2 neighbours) 0.873→0.895, ไม่เสีย mAP | **อ้างได้ในขอบเขตข้อมูลจริงของเรา** | synthetic discs, 3 seeds เท่านั้น |

### ประโยค claim ที่แนะนำ

- **Contribution (ฉบับสั้น):**
  *"To our knowledge, we are the first to cap the per-instance Gaussian width of a class-specific center heatmap by the distance to the nearest same-class neighbour, and we show that this reduces counting error for overlapping cells in a CenterNet-style detector."*
  ถ้าอยากลดความเสี่ยง ให้เปลี่ยน "first to" เป็น "*we study / propose a simple*"
- **Positioning ต่อ prior work:**
  *"Neighbour-dependent kernels were used for crowd density maps (geometry-adaptive kernels) and for crowd localization blobs (LDC-Net, TopoCount). We adapt this idea to Gaussian peak targets in a center-heatmap detector, combining it with an object-size term, and evaluate it for overlapping blood cells."*
- **ผลการทดลอง:** รายงานตัวเลขพร้อมขอบเขต: *"On a synthetic overlapping-disc benchmark (3 seeds), RBC count MAE decreased from 2.95 to 2.30 (paired bootstrap 95% CI of the difference [−0.755, −0.525]) with no mAP loss."*

### สิ่งที่ห้ามเขียน

- "novel adaptive Gaussian" / "first neighbour-aware kernel" โดยไม่มีเงื่อนไข
- "solves overlapping cells" ข้อมูลเป็น synthetic เท่านั้น
- อ้างว่าต่างจาก LDC-Net เพราะ "ใช้ Gaussian" โดยไม่มีการเทียบจริง
- อ้างผลของ Höfener 2018 หรือ PAC-MAP เพราะยังไม่ได้อ่านเนื้อหา

---

## 3. Paper ที่ต้องอ้างอิง

- **ต้องอ้าง (reviewer จะถามแน่):**
  - Repel coding: Liang et al., arXiv 1904.08864
  - LDC-Net, arXiv 2110.04727
  - MCNN (CVPR 2016) และ CSRNet (CVPR 2018) สำหรับ geometry-adaptive kernel
  - TopoCount, AAAI 2021 (arXiv 2012.12482)
  - CenterNet (arXiv 1904.07850) และ CircleNet (arXiv 2006.02474)
  - SC-CNN (Sirinukunwattana, IEEE TMI 2016) และ Xie et al. (MedIA 2018)
- **ควรอ้าง (ทำให้ related work สมบูรณ์):**
  - IIM, FIDT, LSC-CNN
  - TTFNet, Solar Corona, GGHL (หลักฐานว่า kernel ใหญ่เกินกด peak ใกล้เคียง)
  - SWAHR (adaptive sigma ฝั่ง learned)
  - PAC-MAP (ในฐานะ proximity-aware nuclei detection อ่านก่อนอ้าง)
  - StarDist, Cellpose (ทางเลือกอื่นสำหรับเซลล์ที่แตะกัน)
  - Chula RBC-12, arXiv 2012.01321 (งาน overlapping RBC ของไทย)
- **อ้างเพื่อประกอบ:** Wan & Chan (learned density generator), Beyond Counting (arXiv 1705.10118; kernel กว้างลดคุณภาพ detection)
- **ข้อควรระวัง:** venue ของ LDC-Net, IIM, GGHL, CircleNet ไม่ได้ยืนยันใน notes ให้ตรวจเองก่อนใส่ BibTeX; Liang et al. เจอเฉพาะ arXiv

---

## 4. Experiment / ablation ที่เสริม novelty

เรียงตามความสำคัญ (ตอบ reviewer ได้ตรงที่สุดก่อน)

1. **เทียบ target แข่งโดยตรง** (backbone, stride, seeds เท่ากัน)
   - Fixed-σ CenterNet/CircleNet baseline
   - **Repel-code target** (เป็นงานที่ใกล้ที่สุดที่ใช้ dist1/dist2) ต้องดัดให้เข้ากับ focal loss หรือเทียบเป็นแบบ regression
   - **LDC-Net-style cap:** `min(c, 0.25·d_nn)` ในรูป σ
   - **TopoCount-style:** radius ≤ ½·d_nn
2. **ablation ของตัวสูตร**
   - k-NN mean (k = 1, 2, 3) เทียบ nearest ตามแบบ MCNN/CSRNet
   - **Same-class vs class-agnostic** (วัดตอน RBC ชิด WBC/PLT)
   - **Divisor sweep:** d_nn/(k·s) สำหรับ k = 2, 3, 4, 6, 8 เพื่อแสดงว่า 4 ไม่ใช่ค่าที่เลือกมาเอง (มีเหตุผล saddle exp(−2) ด้านบนแต่ยังต้องวัด)
   - ตัด own-size term / ตัด NN term ทีละตัว ดูว่าแต่ละส่วนช่วยเท่าไร
   - Floor 0.5: ลองที่ 0.25, 0.5, 1.0
3. **Stride และ decoder**
   - stride 2/4/8 และ max-pool kernel (3×3 vs 5×5) ที่ผลของ stride-2 (−16%) น้อยกว่า stride-4 (−22%) ตรงกับ prediction ว่า quantization เป็นปัจจัย
   - ตรวจกรณี center quantization กับ annotation jitter (narrow target ไวต่อทั้งสอง)
4. **Stratified metrics**
   - recall/F1 ตามจำนวนเพื่อนบ้านที่ซ้อน (0, 1, ≥2) และ count MAE ต่อคลาส
   - ดู false positive/duplicate เพราะ narrow target อาจทำให้ peak แตกเป็นสองในเซลล์เดียว
5. **ข้อมูลจริง**
   - BCCD เต็ม 364 ภาพ (มีภาพ RBC ซ้อนที่ TXL-PBC ตัดทิ้ง 172 ภาพ), Chula RBC-12 (706 ภาพ, 20,875 cells, บาง release มีเฉพาะ point labels ซึ่งเข้ากับ center-heatmap)
   - จุดนี้สำคัญที่สุดสำหรับการ publish เพราะ synthetic discs ไม่ใช่ RBC จริง
6. **สถิติ:** ≥ 5 seeds, paired bootstrap หรือ paired test, รายงาน CI ทุกตาราง (3 seeds ยังน้อย แม้ CI ปัจจุบันไม่ข้าม 0)

---

## 5. สิ่งที่ยังไม่ได้ยืนยัน

- **Search coverage ตื้น:** แต่ละหัวข้อใช้ ~17–25 tool calls; ไม่ได้ค้น workshop papers, thesis, GitHub forks และวารสารภาษาจีน ซึ่งอาจมี neighbour-capped radius ที่ซ่อนอยู่
- **Paper ที่ยังไม่ได้อ่านเนื้อหา:**
  - Höfener 2018 (อาจศึกษา kernel width ต่อ crowding), PAC-MAP (สูตร target), Wan & Chan (generator ใช้ระยะเพื่อนบ้านหรือไม่)
  - งาน crowd localization ปี 2022–2025 ที่อ้าง FIDT/IIM
  - P2PNet, TopoCount ในรายละเอียดสูตรนอกเหนือจากที่ดึงมา
- **สูตรทางการของ CenterNet `gaussian_radius`** อ้างจาก mirror ของ mmdet; โค้ด official และคำว่า "σ = diameter/6" มาจากความจำ ไม่ได้ดึงยืนยัน
- **CircleNet σ_p:** paper ไม่ระบุ ต้องอ่านโค้ด hrlblab/CircleNet ว่าใช้ `gaussian_radius` แบบ CenterNet หรือไม่
- **Venue/ปีของ LDC-Net, IIM, GGHL, Liang et al.** ยังไม่ยืนยัน (Liang et al. ไม่พบ journal)
- **ตัวเลขผลของเราเอง:** ยังไม่มีข้อมูลบน blood smear จริง, ไม่มีการเทียบกับ repel/LDC-Net/k-NN mean, 3 seeds; ตัวเลข −16% ของ stride-2 ใน notes ไม่ได้ระบุ baseline ที่ใช้เทียบ ต้องระบุให้ชัดก่อนเขียน
- **กลไกที่ว่า "narrow target ลด merged peak":** เป็นข้อสันนิษฐานที่สอดคล้องกับ Solar Corona และ FIDT แต่ notes ไม่พบ controlled experiment ในงานอื่น ผลของเราจึงเป็นหลักฐานแรกในจุดนี้เท่าที่ตรวจ ซึ่งก็หมายความว่ายังไม่มีใครทำซ้ำ

---

## สรุป

ความใหม่ของงานนี้ควรวางเป็น **"adaptation + evidence"** คือการนำกฎ NN-cap ที่มีอยู่ในโลก crowd มาประกอบกับ own-size term ใน Gaussian peak target ของ CenterNet-style detector สำหรับเซลล์เม็ดเลือด แล้วมีหลักฐานเชิงปริมาณว่าลด count error โดยไม่เสีย mAP ซึ่งขอบเขตนี้ปกป้องได้ ข้อเสียคือ moat บางมาก (LDC-Net ต่างกันแค่ blob vs Gaussian) ดังนั้นสิ่งที่ทำให้ claim แข็งขึ้นจริงคือการเทียบตรงกับ repel-code และ LDC-Net-style cap กับ k-NN mean และ divisor sweep บนข้อมูลจริง (BCCD เต็ม, Chula RBC-12) มากกว่าการเลือกคำพูด ถ้าผลเทียบออกมาว่า LDC-Net-style cap ให้ผลเท่ากัน ให้ปรับ claim เป็น "empirical study of neighbour-capped Gaussian targets for blood cells" แทน
