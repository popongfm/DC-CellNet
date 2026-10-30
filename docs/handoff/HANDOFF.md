# DC-CellNet — Handoff (2026-10-30)

เอกสารนี้ใช้ส่งต่องานให้ Claude session ใหม่ (เช่น Claude Code บน cloud) ทำต่อได้ทันที อ่านไฟล์นี้ก่อน แล้วตามลิงก์ไปไฟล์อื่น

---

## 1. สถานะโดยย่อ

| ช่วง | สถานะ | ผลหลัก |
|---|---|---|
| M0–M1 | ✅ | evaluator กลาง + splits; YOLO26m บน TXL-PBC pooled test: mAP50 0.976, mAP50-95 0.864 |
| M2 (L_cc + density, shared) | 🔴 NO-GO | density head ทำ detector แย่ลง |
| M2b (density detached) | 🔴 NO-GO | L_cc ไม่มีผล |
| M2c (L_cc oracle) | 🔴 NO-GO | แม้เป้าถูกต้องสมบูรณ์ L_cc ก็ไม่ช่วย → **ปิดแนวทาง L_cc** |
| M2d `d_sigma` (adaptive σ) | 🟢 GO | RBC MAE 2.95→2.30 (−22%) |
| M2d `d_s2` (stride 2) | 🟢 GO | 2.95→2.49 (−16%) |
| M2d `d_both` (σ + stride 2) | 🟢 GO | **2.95→1.48 (−50%)**, recall bin≥3 0.839→0.906, mAP50-95 0.852 |
| Novelty check adaptive σ | ✅ | ไม่พบงานที่ทำตรงกัน; ใกล้สุด LDC-Net, TopoCount, MCNN/CSRNet, repel coding → ต้อง ablation |
| Phase 3 spec | ✅ อนุมัติแล้ว | `docs/specs/2026-10-30-phase3-design.md` |
| P3a plan | ✅ | `docs/plans/2026-10-30-phase3-p3a.md` |
| P3a Task A1 (target variants + configs) | ✅ commit 50843ea, review ผ่าน (branch `claude/funny-bell-91wy2w`) | รอ Mickie smoke + real run (README "P3a") |
| P3b plan (TXL-PBC LOSO 75 runs) | ✅ เขียนแล้ว รอ review | `docs/plans/2026-10-30-phase3-p3b.md` (B1–B5) |
| P3c (BCCD relabel 204 ภาพ) | ✅ แผนเขียนแล้ว รอ review | `docs/plans/2026-10-30-phase3-p3c.md` (C1–C3; C3 ต้องรอ P3b B4/B5); audit: label RBC ขาดทุกภาพ (12/12) |

ผลแต่ละขั้นฉบับเต็ม: `docs/handoff/results/`
ledger (ประวัติ task, review, rulings R1–R24): `docs/handoff/ledgers/`

---

## 2. ที่อยู่ของสิ่งต่าง ๆ

- **Repo:** GitHub private `popongfm/DC-CellNet` branch `dccnet` (ต้นฉบับมาจาก `mickieppf/Blood-Cell-Detection`) — โค้ดอยู่ใน `dccnet/`
- **เครื่อง GPU ของ Mickie:** Windows, `C:\Research` (git clone เดียวกัน), conda env `dccnet` (torch 2.11 cu128), RTX 5060 Ti 8 GB
  - ข้อมูล: `C:\Research\dccnet\data\TXL-PBC_Dataset\TXL-PBC`, `C:\Research\dccnet\data\BCCD_Dataset\BCCD`, `data\splits\*.json`, `data\synthetic\` (VERSION 2)
  - ผลรัน: `C:\Research\dccnet\runs\` (ไม่อยู่ใน git) — รายงาน: `C:\Research\dccnet\reports\`
- **Progress log:** GitHub issue #1 ใน `popongfm/DC-CellNet` (ต่อ webhook เข้า Discord)
- **Claude Project "Blood Smear Research":** มี spec/รายงานทั้งหมดใน Project docs

---

## 3. งานที่ค้างอยู่ตอนส่งต่อ (ทำต่อทันที)

### 3.1 P3a Task A1 — ตรวจว่า commit แล้วหรือยัง
- ถ้าใน `git log` มี commit ของ heatmap variants (`heatmap_sigma` agnostic/knn3/mcnn, `heatmap_divisor`, `heatmap_shape: repel`, configs `exp_m2_p3a_*`) → ไปขั้น review (ข้อ 3.2)
- ถ้ายังไม่มี (working tree มีไฟล์แก้ค้างและไม่มี commit) → ทิ้งงานค้าง (`git stash`) แล้ว dispatch implementer ใหม่ด้วย brief: `docs/plans/2026-10-30-phase3-p3a.md` (Task A1) + `docs/handoff/implementer-context.md`

### 3.2 Review → sync → ให้ Mickie รัน
1. Task reviewer (Opus) ตาม `docs/handoff/reviewer-instructions.md`; named risks ตาม Review Focus ในแผน P3a
2. fix loop ถ้ามี finding → re-review
3. push `dccnet` ขึ้น GitHub → Mickie `git pull` บน Windows (ดูข้อ 5)
4. Mickie: smoke ใน `runs_smoke_p3a` → real run (คำสั่งอยู่ใน `dccnet/README.md` หัวข้อ "P3a")
5. ผลกลับมา: `compare_runs.py` ต่อแขน (`--candidate d_both --baseline <arm> --ablation <arm>`), ตีความตามข้อ 4

### 3.3 ระหว่าง P3a รัน (~24–28 ชม.)
- เขียนแผน **P3b** (spec §3) ด้วย `superpowers:writing-plans`: tasks โดยประมาณ
  1. `train.py --split` override + `p3b_*.yaml` configs (d_both, d_sigma, circlenet; epochs 150, patience 30, select_from 0)
  2. `scripts/baselines/yolo_train.py` (data YAML ต่อ split, ภาพเดิม, label ลำดับคลาสเรา)
  3. `scripts/baselines/frcnn_train.py` + `frcnn_predict.py` (torchvision fasterrcnn_resnet50_fpn, 50 ep, patience 10, batch 4)
  4. `scripts/run_exp.bat`/`.sh` runner ทั่วไป (ENV `EXP_PREFIX`, `EXP_SPLITS`, `EXP_ARMS`, `EXP_SEEDS`; resumable; rename-aside)
  5. `scripts/summarize_loso.py` (ตาราง pooled/LOSO/Δ + paired bootstrap)
- เขียนแผน **P3c** (spec §4): `prelabel_bccd.py` (YOLO26m conf≥0.5, dedupe IoU>0.7, เติม IoU<0.3, export X-AnyLabeling YOLO txt + `review_list.csv`), `import_bccd_relabel.py` → `splits/bccd_overlap_v2.json`
  - pre-label ต้องรันบนเครื่อง Mickie (ต้องใช้ YOLO weights: `C:\Research\dccnet\runs\detect\runs\yolo26m_pooled\weights\best.pt`)

---

## 4. เกณฑ์ตัดสินที่ใช้ (pre-registered)
- GO ต่อการเทียบ = (paired bootstrap 95% CI ของ ผลต่าง RBC |count error| ต่อภาพ < 0) AND (pooled recall bin 2+3 สูงกว่า) AND (ชนะทุก seed); ข้อมูลไม่ครบ → INCONCLUSIVE
- P3a: d_both ต้องชนะ agnostic / knn3 / mcnn / repel ถึงจะอ้างว่าส่วนประกอบของกฎมีความหมาย; div2/div8 = ประกอบ
- P3b: หยุดคุยหลัง pooled ถ้า d_both ไม่ดีกว่า circlenet
- ถ้าแขนอื่นชนะ/เสมอ → รายงานตามจริง ปรับ claim

---

## 5. วิธีทำงานกับเครื่อง Windows จาก cloud session

Cloud session ไม่มี bridge เข้าเครื่อง Mickie → ใช้ GitHub เป็นตัวกลาง:
- Claude: commit + `git push origin dccnet`
- Mickie (Anaconda Prompt): `cd /d C:\Research` → `git pull <remote ที่ชี้ popongfm/DC-CellNet> dccnet`
  - ครั้งแรกอาจต้อง `git remote add dccnet https://github.com/popongfm/DC-CellNet.git`
- ผลรันกลับมา: Mickie zip ไฟล์ที่ต้องใช้แล้วแนบในแชต หรือ push เฉพาะไฟล์สรุปไปที่ branch แยก (`results`); ไฟล์ที่ compare_runs ต้องการต่อ run: `summary.json`, `summary_test.json`, `sweep_val.json`, `preds_test.json`, `metrics.jsonl` + `data/splits/<split>.json`
- คำสั่ง tar ที่ใช้มาตลอด (รันใน Git Bash หรือ WSL ได้):
  `tar czf runs_<name>.tgz data/splits/synthetic.json runs/exp_m2_<prefix>_*/{summary.json,summary_test.json,sweep_val.json,preds_test.json,metrics.jsonl}`

ถ้า session ใหม่ **ผูกกับเครื่อง Mickie ได้** (Claude desktop app, remote-devices tools) ใช้วิธีเดิม: bundle → `C:\Research\.sync\` → `git pull --ff-only` และ **ขอสิทธิ์ลบไฟล์ใน `C:\Research` ตั้งแต่ต้น session** (ruling R24) ไม่งั้น git ค้าง lock

---

## 6. การตั้ง environment บน cloud
```bash
git clone -b dccnet https://github.com/popongfm/DC-CellNet repo
uv venv -p 3.11 venv && VIRTUAL_ENV=venv uv pip install torch torchvision numpy opencv-python-headless pillow pyyaml scipy pandas matplotlib pycocotools imagehash pytest openpyxl
mkdir -p data_src && cd data_src
git clone --depth 1 https://github.com/lugan113/TXL-PBC_Dataset txl_pbc
git clone --depth 1 https://github.com/Shenggan/BCCD_Dataset bccd_full
cd ../repo/dccnet && mkdir -p data && ln -s ../../../data_src/txl_pbc/TXL-PBC data/txl_pbc && ln -s ../../../data_src/bccd_full data/bccd_full
python scripts/prepare_data.py --txl-root data/txl_pbc --bccd-root data/bccd_full/BCCD --data-root data
python scripts/make_synthetic.py --out data/synthetic --data-root data --seed 0
python -m pytest -q   # ควรผ่านทั้งหมด (211+)
```
ไม่มี GPU บน cloud → เทรนจริงทำบนเครื่อง Mickie เท่านั้น; cloud ใช้เขียนโค้ด/เทสต์ CPU/วิเคราะห์ผล

---

## 7. Skills และ subagent ที่ใช้ (ทำซ้ำแบบเดิม)

### Skills
| ขั้น | Skill | ใช้ทำอะไร |
|---|---|---|
| ออกแบบงานใหม่ | `superpowers:brainstorming` | จัดประเภท (spike/bounded/architectural), ถามทีละข้อ, เสนอ 2–3 ทาง, design ทีละส่วน, เขียน spec |
| แผนงาน | `superpowers:writing-plans` | แตก spec เป็น task (interfaces, tests, global constraints, review focus) |
| ลงมือ | `superpowers:subagent-driven-development` | implementer → task reviewer → fix loop → re-review ต่อ task; ledger; final whole-branch review |
| ตรวจความใหม่ / literature | `anthropic-skills:deep-research` | researcher subagents 3–4 ตัวขนานกัน + report writer |
| ภาพประกอบ | `image-generation:image-generation` | prompt สำหรับ GPT Image |

### Subagents (ทั้งหมดเป็น `general-purpose`, ระบุ model ทุกครั้ง)
| บทบาท | Model | Prompt/อินพุต |
|---|---|---|
| Implementer (task ทั่วไป) | sonnet | brief ของ task + `docs/handoff/implementer-context.md` + report path |
| Implementer (โค้ดวิกฤต: loss, evaluator, training loop, targets, stride) | opus | เหมือนบน |
| Task reviewer | sonnet (ทั่วไป) / opus (โค้ดวิกฤต, GPU run ยาว) | `docs/handoff/reviewer-instructions.md` + brief + global constraints + report + diff package + named risks |
| Re-reviewer (fix round) | sonnet/opus | รายการ finding + fix diff เท่านั้น |
| Final whole-branch reviewer | opus | rubric `requesting-code-review/code-reviewer.md` + branch diff + ledger deferred minors |
| Researcher (deep-research) | sonnet | หัวข้อแคบ + คำถาม + แหล่ง + path ของ notes |
| Report writer | sonnet | notes folder + คำถามเต็ม + path ผลลัพธ์ |

หลักที่ใช้มาตลอด: implementer ไม่ spawn subagent; controller ไม่แก้โค้ดเอง (ยกเว้น config-only, ruling R23); ทุก ruling ลง ledger พร้อม "cost if wrong"; รัน GPU ยาวต้องผ่าน review + smoke test บน Windows ก่อน

---

## 8. Rulings สำคัญที่ยังมีผล (ฉบับเต็มใน ledger)
- R8 batch 8 + AMP (8 GB); R13 peak_thr 0.01; R16 decode 0.001 สำหรับ mAP; R17/R18 density softplus bias −7; R20 select_from_epoch 15 (M2) / 0 (P3b); R21 L_cc sigmoid count; R22 ยังไม่ merge `dccnet` เข้า main; R23 config-only commit โดย controller ได้; R24 ขอสิทธิ์ลบไฟล์ก่อน git pull บน device
- ห้ามลบไฟล์ของ Mickie โดยไม่ได้รับอนุญาตรายการชัดเจน; ห้าม push branch อื่นนอกจาก `dccnet` ใน `popongfm/DC-CellNet`

## 9. เรื่องที่ต้องคุยกับ Mickie ต่อ
- อ่าน PDF ของ PAC-MAP (bioRxiv 2024) และ Höfener 2018 ก่อนเขียนเปเปอร์ (novelty)
- กำหนดเวลาแก้ label BCCD ใน X-AnyLabeling
- การ merge branch `dccnet` และการทำความสะอาดโฟลเดอร์ `runs_smoke*`, `*_aborted_*`
