# SDD ledger — plan: docs/plans/2026-10-26-dccellnet-m0-m2.md

Spec: docs/specs/2026-10-26-dccellnet-design.md

## Environment
- Canonical repo: user's Windows C:\Research (device; reachable via device_bash at $HOME/mnt/Research; branch dccnet).
- Dev clone: /home/claude/repo (container) — remote `device` = bundle. Tests run with /home/claude/venv (Python 3.11, torch CPU).
- Real data in container: /home/claude/data_src/txl_pbc/TXL-PBC, /home/claude/data_src/bccd_full (git clones of public repos).
- Sync container→device: git bundle → /mnt/user-data/outputs → device_commit_files → device `.sync/` → `git pull --ff-only`.

## Pre-flight scan
| Rows | Produces vs consumes / self-consistency | Finding |
|---|---|---|
| T0 self | Step 5 installs torch in device VM | VM has 3 GB RAM/no GPU/180 s cap; container has 7 GB + venv → ruling R1 |
| T0 ↔ T1+ | T0 commits on device; T1+ developed in container | ordering → ruling R2 |
| T1 → all | `load_config`, `Sample(image_path, source, width, height, boxes, labels)` | consistent |
| T2 → T5,T8,T10,T12 | `boxes_to_circles`, `circles_to_boxes`, `overlap_levels`, `circle_iou` | consistent names/types |
| T3 → T4,T5,T7,T13 | `load_txl -> (samples, official)`, `pooled_split`, `loso_split`, `write_split/read_split` | consistent |
| T4 self | old Roboflow dataset path exists only on device | ruling R3 |
| T5 → T6,T13,T14 | prediction record keys; `coco_map`, `count_errors`, `overlap_metrics`, `bootstrap_ci`, `paired_bootstrap_diff` | consistent |
| T6 self / T14 self | include Windows GPU steps by Mickie | ruling R4 |
| T7 → T13,T14 | `make_synthetic -> list[Sample]`, `synthetic.json` via write_split | consistent |
| T8 → T9,T10,T12 | target keys heatmap,density,ind,mask,log_r,offset,r_valid,cls,circle | consistent with T10 consumers |
| T8 self | offset test (100.5,200.25)/4 → cell (25,50), offset (0.125,0.0625) | arithmetic checks |
| T9 ↔ T11 | decode takes `log_r`; model outputs `log_r` | consistent |
| T11 self | repr box / convnext raise NotImplementedError | consistent with plan scope |
| T12 self | 640×480 letterbox: scale 1, pad_y 80 → center (120,200) | arithmetic checks |
| T13 ← all | uses decode + coco_map + unletterbox | consistent |

## Rulings
- R1: Ruling: run all pytest in container venv (Python 3.11), skip installing torch in device VM — VM too small and no GPU; Windows env (Mickie) is the GPU runtime — cost if wrong: none for correctness; tests not run on Windows itself (Mickie runs check_env + one smoke run).
- R2: Ruling: Task 0 executes on the device (gitignore + snapshot commit + disk report there), then device branch is fetched into container before Task 1 — WIP files exist only on device — cost if wrong: an extra fetch.
- R3: Ruling: prepare_data.py takes optional `--extra-dedupe-dir` (repeatable); container run omits the old Roboflow set, Mickie/device run includes it — cost if wrong: dedupe report incomplete until device run.
- R4: Ruling: Tasks 6 and 14 are marked complete when their code is reviewed; the Windows GPU steps (T6 steps 5–7, T14 steps 5–7) are handed to Mickie and the M1/M2 gates are evaluated when results arrive — a session can't wait on GPU hours — cost if wrong: gate results arrive later than task completion.
- R5: Ruling: data/ in container is a symlink set: dccnet/data/txl_pbc → /home/claude/data_src/txl_pbc/TXL-PBC, dccnet/data/bccd_full → /home/claude/data_src/bccd_full — cost if wrong: none (gitignored).

## Progress
- R6: Ruling: GPU is RTX 5060 Ti (Blackwell sm_120), not RTX 4070 as spec says; requirements note torch must be a cu128 build (>=2.7) on Windows; batch sizes in configs pending VRAM (8/16 GB) — cost if wrong: OOM → lower batch.
- Task 0: dispatched (device), base 1c4352e
- Task 0: ⚠️ resolved by controller: .sync/ in .git/info/exclude ✓, disk_report.md exists ✓, no .pt tracked ✓
- Task 0: minor (deferred): disk_report.py dead `sizes` dict + double walk; getsize on zip pairs not try/except; README lacks .sync note; duplicate __pycache__/ in .gitignore; .vscode/settings.json committed
- Task 0: complete (commits 1c4352e..0d4a822, review clean)
- R7: Ruling: on device the user cloned TXL to dccnet/data/TXL-PBC_Dataset (data root = dccnet/data/TXL-PBC_Dataset/TXL-PBC); all loaders/scripts take data roots as CLI args/config, no hardcoded folder name — cost if wrong: none.
- Note: C:\Research is only 3.9 GB; the 97%-full drive is elsewhere on C: — cleaning this folder frees ≤ ~2.5 GB.
- R8: Ruling: VRAM 8 GB → base.yaml train.batch = 8 with AMP; YOLO baseline batch 8 (not 16 as old runs); if OOM drop to 4 — cost if wrong: slower training or one OOM retry.
- Task 1: dispatched, base 0d4a822
- Task 1: ⚠️ resolved: pyyaml listed in dccnet/requirements.txt (Task 0) ✓
- Task 1: minor (deferred): cycle test lacks match=; base.yaml test leaves some values unasserted; non-mapping YAML gives AttributeError; Sample no validation
- Task 1: complete (commits 0d4a822..6ea99ba, review clean)
- Task 2: dispatched, base 6ea99ba
- Task 2: minor (deferred): grad tests only isfinite (no gradcheck); non-concentric containment value untested; fp16 acos risk → callers must cast .float() (carry to Task 10); no shape assertion on reshape
- Ruling R9: commit trailer stays "Claude Opus 5.5" per controller attribution reminder (subagent saw a different one) — cost if wrong: cosmetic.
- Task 2: complete (commits 6ea99ba..93ec311, review clean)
- Task 3: dispatched, base 93ec311
- Task 3: real data: TXL 1260 (Raabin 500, PBC 500, BCCD 160, BCDD 100; train 882/val 252/test 126; RBC 16302, WBC 1298, PLT 543); BCCD full 364 (RBC 4153, WBC 372, PLT 361; 2 invalid boxes skipped). TXL classes.txt order WBC,RBC,Platelet → mapped by name.
- Task 3: minor (deferred): substring class match; malformed label lines / missing label file silently dropped; duplicate metadata stems overwrite; pooled_split bare KeyError; dead line in test; some untested branches; double logging in bccd loader
- Task 3: complete (commits 93ec311..0be30d2, review clean)
- Task 4: dispatched, base 0be30d2
- FINDING (for Mickie, affects M3): overlap_bccd = 204 images (364-160). BCCD_selection.xlsx lists 172 rejected as 'RBC_noboundary' (likely RBC annotations missing) — overlap set labels likely incomplete; spec §3.6 audit becomes mandatory before any M3 overlap claim.
- Task 4: ⚠️ extra-dedupe-dir untested — controller: run on device later with old Roboflow dir (R3); code read by reviewer looks correct
- Task 4: minor (deferred): no hamming_matrix popcount test; prepare_data not unit-tested (write_yolo in script); roots outside data-root error late; unreadable extra file crashes
- Task 4: data: N_overlap 204; TXL 6 near-dup pairs (2 cross train/val) — reported only
- Task 4: complete (commits 0be30d2..1432dfe, review clean)
- Task 5: dispatched, base 1432dfe
- R10: Ruling: evaluator functions take data_root kwarg; records keyed by image path relative to data_root (same as split JSON) — plan signature lacked it — cost if wrong: signature churn in Tasks 6/13/14 (carried in dispatches).
- R11: Ruling: coco per_class = AP50 plus per_class_50_95 added; eval/common.py helper accepted — cost if wrong: key rename later.
- Task 5: minor (deferred): summary lacks per_image count errors (needed for paired CIs → Task 14 compare_runs recomputes from preds); no paired mAP diff; NaN bootstrap draws silently dropped; mAP CI class set varies per resample (LOSO risk); duplicated CI code; shared seed undocumented
- Task 5: complete (commits 1432dfe..70b7d4b, review clean; opus reviewer cross-checked AP vs COCOeval.summarize within 1e-12)
- Task 6: dispatched, base 70b7d4b
- Task 6: ⚠️ ultralytics path untested here — handed to Mickie's Windows run (R4)
- Task 6: minor (deferred): README hardcodes --score-thr 0.35 placeholder; sweep recomputes overlap_counts 19×; yolo_predict main untested; empty subset no warning
- Task 6: complete (commits 70b7d4b..4b9ab5f, review clean) — M1 code done; M1 gate pending Mickie's YOLO run
- Sync: container→device at 4b9ab5f (ff). Device BCCD path = dccnet/data/BCCD_Dataset/BCCD (README says bccd_full — told Mickie).
- Task 7: dispatched, base 4b9ab5f
- Task 7: data: ~45% RBC in bin 3+, ~22% bin 0 (dense clumps) — accepted as stress test; --clump-prob available
- Task 7: minor (deferred): no test for GT-vs-drawn; round(r) vs exact r ≤0.5px; margin r+1 untested; out not under data-root unguarded
- Task 7: complete (commits 4b9ab5f..a187d38, review clean)
- Task 8: dispatched, base a187d38
- Task 8: note for Task 12: centers outside map clamp to edge cell (offset out of [0,1)) → dataset aug must drop cells whose center leaves the image
- Task 8: minor (deferred): far-outside centers silently get zero density mass; NaN centers cryptic error; border-peak / subpixel density center / slot order untested
- Task 8: complete (commits a187d38..d7471fe, review clean)
- Task 9: dispatched, base d7471fe
- Task 9: minor (deferred): multi-tile fill untested; no density shape check; round half-to-even; IoU boundary untested; NxN NMS memory; redundant astype/detach; negative max_dets
- Task 9: complete (commits d7471fe..f19f63b, review clean)
- Task 10: dispatched, base f19f63b
- R12: Ruling: compute_losses() aggregator added to losses/__init__.py in Task 10 (plan left aggregation implicit for Task 13) — cost if wrong: small refactor.
- Task 10: concern accepted: clamp-floor plateau counts as peaks → ~0.005 constant loss per empty window, no gradient; documented — cost if wrong: logged L_cc floor offset
- R13: Ruling (Task 10 Important #1, plan-mandated peak mask): gate peaks P = local-max ∧ H > τ with τ = cfg loss.cc.peak_thr = 0.01 (not 0.1: split-candidate peaks below 0.1 must still receive gradient to emerge); add peak_thr to base.yaml; also log loss_cc_bg (sum of H over local maxima ≤ τ, diagnostic only). Spec §2.3 intent = count detections, not background maxima — cost if wrong: τ is an extra hyper-parameter; ablate τ ∈ {0, 0.01, 0.1} in M5.
- R14: Ruling (Task 10 Important #2): focal_loss re-clamps after float cast; model must do sigmoid+clamp in fp32 (carry to Task 11); radius exp clamp log_r ≤ 10 in circle construction.
- Task 10: fix round 1 dispatched
- Task 10: fix round 1/5 (2 addressed, 0 open — peak_thr gate + loss_cc_bg; fp16 focal re-clamp + log_r clamp; commits 269274c..78747bb)
- Task 10: minor (deferred): exp(log_r) clamp zero-grad >10; zero tensor reused for l_d/l_cc; 0*nonfinite l_cc → NaN during warmup; fp16 peak ties (moot if model fp32 heatmap)
- Task 10: complete (commits f19f63b..78747bb, review clean after round 1)
- Task 11: dispatched, base 78747bb
- Task 11: minor (deferred): heatmap-mean test single seed; clamp zero-grad saturation; match message weak
- Task 11: complete (commits 78747bb..8341226, review clean)
- Task 12: dispatched, base 8341226
- R15: Ruling: Task 12 RNG ruling was flawed (num_workers=0 → identical aug every epoch); add CellDataset.set_epoch(epoch) and seed = (torch.initial_seed(), epoch, index) — enters fix loop with review findings; Task 13 calls set_epoch each epoch.
- Task 12: fix round 1/5 (3 addressed, 0 open — set_epoch seed, drop r'<0.5, centroid test; commits e7a230f..7ae97de)
- Task 12: minor (deferred): letterbox round vs s ≤0.5px; stain jitter on pad; edge crossing ignores 2px margin
- Task 12: complete (commits 8341226..7ae97de, review clean after round 1)
- Task 13: dispatched, base 7ae97de
- R16: Ruling: for mAP, decode with score_thr 0.001 & max_dets 1000 (parity with YOLO conf 0.001); cfg decode.score_thr used only as counting/overlap operating threshold (val-tuned later) — cost if wrong: mAP under-estimated if left at 0.3.
- R17: Ruling (Task 13 review Critical, cross-task): density head dies (ReLU + unnormalized count-L1). Fix: density output = softplus(d.float()) with final bias init −9.0 (≈8e-5/cell, ~2 counts per 160² map); density_loss count term normalized: mean_(B,C) |Σpred−Σgt| / max(Σgt, 1); MSE term unchanged. Add val per-class predicted vs GT density count to metrics.jsonl (val_den_count_{RBC,WBC,Platelets}, val_gt_count_*). Spec §2.2 says ReLU — softplus keeps ≥0 semantics — cost if wrong: retune λ_d.
- Task 13: fix round 1 dispatched (R17 + atomic best.pt + summary config_stem + warn if density_fill with low decode thr + count GradScaler-skipped steps)
- GitHub: pushed dccnet to private popongfm/DC-CellNet (remote 'github') at 5a5cb53; push after each task completes (Mickie approved).
- Discord: via GitHub webhook (Mickie configures). Progress log = issue #1 popongfm/DC-CellNet; post comment at each task completion / user hand-off / gate result.
- R18: Ruling: density bias init −7.0 instead of −9.0 (probe: RBC log-growth ≈0.011/step → from 0.5 to ~48 needs ~420 steps ≈ warm-up length; −7 gives ~23 counts/class/map at init, shrinking WBC/PLT overshoot is multiplicative and equally fast) — cost if wrong: early density overshoot for sparse classes during warm-up (λ_cc=0 then).
- Task 13: fix round 1/5 (5 addressed + R18 amendment, 0 open; commits 5a5cb53..b605951)
- Task 13: minor (deferred): density_fill no-op at 0.001 decode (warned); boxes not clipped; large preds JSON; selection by mAP vs gate metric (compare_runs flags best_epoch<15); num_workers 0 throughput
- Task 13: complete (commits 7ae97de..b605951, review clean after round 1)
- Task 14: dispatched, base b605951
- R19: Ruling: M2 synthetic configs use train.epochs_max 60, train.patience 20 (spec 300/30 targets real data; M2 is a toy mechanism check; 9 runs must finish overnight on 8 GB GPU) — cost if wrong: under-trained arms; both arms share the budget so the comparison stays fair.
- M1 GATE: PASS — YOLO26m pooled test (126 img, thr 0.70 from val sweep): mAP50 0.976 [0.960,0.991], mAP50-95 0.864 [0.841,0.888]; RBC count MAE 1.02; recall bins 0/1/2/3 = .975/.939/.950/.951. Trained 150 ep, best ep 129, batch 8. Weights at dccnet/runs/detect/runs/yolo26m_pooled (ultralytics nested path).
- Task 14: fix round 1 dispatched (bat for/f quote stripping → call; rename-aside aborted runs; JSONDecodeError; n_boot text; README Windows smoke)
- Task 14: fix round 1/5 (4 addressed, 0 open; commits 3e84df7..e289dc3)
- Task 14: minor (deferred): %RANDOM% collision; load_run KeyError on missing field; pre-first-save kill reuses folder; bat header lacks M2_PYTHON; bat untested on cmd (Windows smoke)
- Task 14: complete (commits b605951..e289dc3, review clean after round 1)
- FINAL REVIEW: dispatched (opus) over 609a693..e289dc3; Mickie running Windows smoke test (runs_smoke) in parallel; real M2 run held until final review clean.
- FINAL REVIEW: Fix before M2 run. Findings: #1 Critical checkpoint selection window; #2 ImageNet normalization; #3 --map-ci-n 0 in runner; #4 gate adds dccell vs density; #5 per-seed 3/3 condition; #6 INCONCLUSIVE verdict; #7 synthetic 0.5px draw offset; design note: L_cc soft count Σ H·P < n with H<1 → peak-sharpener.
- R20: Ruling: train.select_from_epoch = warmup+ramp (15) in base.yaml for all arms; patience counts from it; smoke-safe clamp — cost if wrong: none for fairness.
- R21: Ruling (design note): L_cc soft count uses indicator s(H)=sigmoid((H−c)/T) at peaks with c = loss.cc.count_center 0.3 (= decode.score_thr), T = loss.cc.count_temp 0.05; config loss.cc.count_fn: sigmoid (default) | linear (ablation, old behavior). Rationale: count detections, not peak heights; gradient concentrates at decision boundary where split candidates live — cost if wrong: one more ablation in M5; reverting = config flag.
- FINAL REVIEW fix wave: dispatched (one fixer, opus)
- FINAL fix wave: commits e289dc3..f39c15a, 173 tests
- FINAL REVIEW: re-review Ready for M2 run; spec amendments committed 01eb8bd; workspace kept until M2 gate evaluated (ruling R22).
- R22: Ruling: keep SDD workspace (ledger) until the M2 gate is evaluated; finishing-a-development-branch (merge decision) deferred until after M2 verdict — no merge into user's main without asking.
- Device synced to 01eb8bd; Mickie told to stop running smoke (bat replaced under it) and re-run smoke into runs_smoke2, then real run.
- M2 GATE: NO-GO (pre-registered). Gate1 dccell vs circlenet FAIL (MAE 3.240 vs 3.227, CI [-0.057,0.083]; bin2+3 recall .857 vs .860). Gate2 dccell vs density PASS (MAE diff -0.088, CI [-0.150,-0.037], all seeds <0, recall .857 vs .854). Diagnostic: density head hurts vs circlenet (mAP50-95 .743 vs .790; MAE +0.10). All 9 runs hit epochs_max 60 (best_epoch 46–54; mAP still rising) → under-trained. Density counts ≈ GT (RBC 4089 vs 3956). Per plan: stop and discuss before M3.
- M2b: Mickie chose option A; short design approved in chat (detached density branch, 120 epochs, same gate). base 01eb8bd. dispatched.
- M2b: complete (01eb8bd..e1a96a0, review clean; minors: BN stats on detached drift, no init-parity test, deeper head confound vs M2 density)
- Waiting: Mickie running M2b smoke + real run (M2_CONFIGS=b_circlenet b_density b_dccell). Next: stage runs/exp_m2_b_* + compare_runs --baseline b_circlenet --candidate b_dccell --ablation b_density; compare to M2.
- M2b GATE: NO-GO. Detach removed density harm (mAP50-95 b_circlenet .837, b_density .839, b_dccell .839). L_cc effect vanished: b_dccell vs b_density MAE diff -0.003 CI[-0.050,0.040]; vs b_circlenet +0.017 CI[-0.042,0.072]. Longer budget improved all (RBC MAE 3.23→2.95, bin3 recall .82→.84). Interpretation: M2 gate-2 win was L_cc compensating shared-head interference, not separating overlaps. dccell_s1 was interrupted once and auto-renamed/retrained (aborted folder ignored). Next: discuss pivot (option C) with Mickie.
- M2c: Mickie chose A′ (oracle L_cc); short design approved. base e1a96a0. dispatched.
- M2c: complete (e1a96a0..1fbcf90, review clean, opus probes 0 mismatches; minors: clamp hides bad ind/cls; target_counts shape unchecked; note: L_cc can only raise existing local maxima — a NO-GO rules out L_cc as formulated, not count supervision in general)
- Device synced to 1fbcf90 (delete permission re-granted this session; cleared git lock + partial-checkout file exp_m2_b_oracle.yaml that git itself had created).
- M2c ORACLE: NO-GO. b_oracle vs b_circlenet RBC MAE 2.940 vs 2.945, diff -0.005 CI[-0.048,0.042]; bin2+3 recall .871 vs .873; mAP .839 vs .837. L_cc loss only 0.062→0.035 over training (detector peaks already match counts; residual errors = merged peaks w/o local max, which L_cc can't split). Per pre-registered rule → close L_cc direction; discuss pivot C (and D = finer heatmap for merged clumps).
- BCCD AUDIT (option 3): inspected 12/30 sampled overlap_bccd images — 12/12 have unlabeled RBCs (often ~30–60% of visible RBCs missing, e.g. BloodImage_00143 has 5 labeled of ~15), plus duplicate boxes (00010, 00083). Far above 10% threshold → spec §3.6: relabel required before using as overlap test set.
- M2d (option D, arms d_sigma + d_s2) short design approved; BCCD relabel chosen (option 1, pre-label for CVAT pending). base 1fbcf90. dispatched.
- M2d: complete (1fbcf90..975c253, review clean; e2e stride-2 probe mAP 0.9967 = stride-4). Minors: config test ignores train.batch for both arms; README runtime understated (d_s2 up to ~2×/epoch → 6 runs ~9–10.5 h); batch-4 fallback = stride+batch confound; adaptive σ shrinks σ for ~63% of synthetic cells (not just clumps); 5.6% cells have same-class neighbour <1.5 cells at stride 4 (1.5% at s2) → maxpool decode ceiling.
- Device synced to 975c253 (delete permission re-granted; cleared lock + partial-checkout d_* configs created by git).
- M2d GATE: GO for both. d_sigma: RBC MAE 2.303 vs 2.945 (−0.642, CI[−0.755,−0.525]), bin2+3 recall .895 vs .873, all seeds; mAP .841. d_s2: MAE 2.487 (−0.458, CI[−0.548,−0.373]), recall .888, all seeds; mAP .842. d_sigma better and zero inference cost. d_s2_s2 had one aborted+retrain. Next: novelty check for adaptive heatmap σ; combined arm?; carry into C on TXL-PBC.
- R23: Ruling: d_both config (config-only, plus config-diff test) written by controller without subagent dispatch — no logic change; cost if wrong: one config typo caught by test.
- Device synced to 42b0203. R24: Ruling: request delete permission at the start of every new session BEFORE any device git pull (permission expires per session; skipping it leaves lock files).
- NOVELTY (adaptive σ): neighbour-adaptive widths exist in crowd work (LDC-Net min(c,0.25·d_nn) dilated blobs class-agnostic; MCNN/CSRNet kNN kernels; TopoCount ½d_nn binary; repel coding Liang). Not found: per-instance Gaussian σ capped by same-class d_nn in CenterNet-family / blood smear. Claim 'to our knowledge'; must ablate vs repel-code, LDC-Net cap, TopoCount cap, class-agnostic, kNN mean, divisor sweep. Report in Project docs.
- M2d d_both: GO. RBC MAE 1.48 vs b_circlenet 2.945 (−50%, CI[−1.652,−1.275]) and vs d_sigma 2.303 (−0.823, CI[−0.953,−0.693]), all seeds; bin2+3 recall .927; bin3 .906; mAP50-95 .852. Super-additive (σ −0.64, s2 −0.46, both −1.47). Next: Phase 3 spec (C on real data with d_both main + ablations for novelty).
