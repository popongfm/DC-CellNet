# SDD ledger — plan: docs/plans/2026-10-30-phase3-p3b.md
Spec: docs/specs/2026-10-30-phase3-design.md §3. Base branch dccnet @ df3c571 (P3a = eeb5f06). Baseline suite 323 passed.
Pre-flight scan:
| Pair / task | Produces → consumes | Finding |
|---|---|---|
| B1 → B4 | train.py `--split`, `--run-name`, `--epochs-max` | consistent |
| B2 → B4 | yolo_train CLI (`--model <arm>.pt`, `--epochs-max`), weights at <run>/weights/best.pt → existing yolo_predict | consistent; EXP_DEVICE not passed to yolo (plan) |
| B3 → B4 | frcnn_train/predict CLI, `--device`, `--epochs-max` | consistent |
| B1/B2/B3 → B5 | summary.json best_epoch, select_from_epoch 0, train_seconds, n_params | consistent |
| B4 → B5 | run dir `<prefix>_<split>_<arm>_s<seed>`; B5 `--prefix p3b` | consistent (B5 builds `p3b_<split>_` for discover_runs) |
| B3 test_image_key_parity ↔ B1 predict.py | needs a DC-CellNet checkpoint | implementer builds an untrained ckpt in tmp; fine |
| B1 self | Step 2 names test_train_cli/test_config but accum test lives in test_train_smoke | Ruling: Step 2 RED run includes tests/test_train_smoke.py too — trivial — cost if wrong: none |
| B2 self | `import scripts.baselines.yolo_train` but scripts/ has no __init__.py | Ruling: test may import via importlib from file path (no new packages) — keeps repo layout — cost if wrong: none |
| B3 self | consistent | — |
| B4 self | yolo aborted marker = weights/ w/o summary.json; stub must create weights/ for yolo | consistent |
| B5 self | consistent | — |
- Task B1: dispatched, base df3c571
- Task B1: implementer DONE, commit ee7ecb7 (340 passed)
- Task B1: task review (opus) Approved, spec ✅, 0 Critical/Important
- Task B1: minor (deferred): accum not exactly gradient-equivalent for focal loss (num_pos normalised per micro-batch); docstring should say approximation
- Task B1: minor (deferred): short final micro-batch overweighted in last accum group (no drop_last)
- Task B1: minor (carried to B4): train.py --split is relative to data-root; predict/evaluate --split take full path
- Task B1: minor (deferred): train_seconds includes per-epoch validation
- Task B1: minor (carried to B4 README): AMP+accum path untested on CUDA → smoke one epoch with --batch 4 --accum-steps 2
Task B1: complete (commits df3c571..ee7ecb7, review clean)
- Task B2: dispatched, base ee7ecb7
- Task B2: implementer DONE, commit c76a86c (346 passed)
- Ruling: B2 best_epoch — Ultralytics 8.4.x strip_optimizer (final_eval) sets best.pt["epoch"] = -1 and stopper = None, so the plan's ckpt-first rule yields -1 for every finished run (verified in ultralytics-8.4.194 torch_utils.py:950, trainer.py:842,953). Detection fitness = mAP50-95 only, best.pt saved on first max (strict <). Decision: ckpt epoch used only if >= 0, otherwise results.csv first-max of metrics/mAP50-95(B) (no warning for -1, warning only if CSV also unusable → error). — plan's intent ("epoch actually saved by its fitness rule") preserved — cost if wrong: best_epoch off for a YOLO version with different fitness; only affects reported epoch, not metrics.
- Task B2: fix round 1/5 (1 addressed, 0 open — ckpt epoch -1 → results.csv; commits c76a86c..a77f8f8) [controller-found before review]
- Task B2: task review (sonnet) Approved, 0 Critical/Important
- Ruling: escalate reviewer Minor "pooled export rmtrees data/yolo_pooled (deletes data_pooled.yaml)" to a fix — handoff §8 forbids deleting Mickie's files without explicit permission; export dir becomes <data-root>/yolo_p3b_<split stem>/ (never collides with prepare_data's yolo_pooled) — deviates from plan text yolo_<stem> — cost if wrong: one extra 1260-image copy on disk.
- Task B2: minor (deferred): stale export reuse if a split is regenerated (splits fixed this phase)
- Task B2: minor (deferred): non-int ckpt epoch falls through silently
- Task B2: fix round 2/5 (2 addressed, 0 open — export dir yolo_p3b_<stem>; stub-ultralytics main() test; commits a77f8f8..15ff105)
Task B2: complete (commits ee7ecb7..15ff105, review clean)
- Task B3: dispatched, base 15ff105
- Task B3: implementer DONE_WITH_CONCERNS, commit 82d127a (358 passed); warmup spans min(1000, steps-1) iterations across epochs per Global Constraints text (kept)
- Ruling (carried to B5): records carry no entry for an image with zero detections, so "preds cover the same keys as the split" cannot be a hard requirement. B5 key check = every pred key must be in the split subset keys (catches path-convention mismatch, the real risk) → else INCONCLUSIVE listing keys; images with no records in an arm → warning with count, not INCONCLUSIVE — cost if wrong: a missing-image bug that drops whole images silently would show only as a warning (count visible in report).
- Task B3: task review (opus) Needs fixes — Important: all-non-finite accumulation window → scaler.step AssertionError on CUDA (default accum 1)
- Ruling: B3 warmup = torchvision reference (epoch 0 only, warmup_iters = min(1000, optimizer steps in epoch 0 − 1)) — spec §3.3 says "SGD ตาม torchvision reference" (binding over plan wording "first 1000 iterations") — cost if wrong: ~3 extra warmup epochs on pooled if Mickie wanted the long warmup; one-line change.
- Task B3: minor (deferred): no FrozenBN pretrained ckpt round-trip test (reviewer verified by hand)
- Task B3: fix round 1/5 (4 addressed, 0 open — empty-window step guard, true-size last window, epoch-0 warmup, <1px-high test; commits 82d127a..38e0e3e); controller re-ran tests/test_frcnn.py
Task B3: complete (commits 15ff105..38e0e3e, review clean)
- Task B4: dispatched, base 38e0e3e
- Task B4: implementer DONE_WITH_CONCERNS, commit e311aab (368 passed); .bat untested (no cmd)
- Task B4: task review (opus) Needs fixes — Important: README OOM recipe doubles FRCNN effective batch
- Task B4: minor (deferred): no test of EXP_PREDICT_ARGS / empty-arg absence / summary.json-present skip-train path
- Task B4: minor (deferred): bash skip may hit WSL bash on Windows; .bat frcnn match case-insensitive
- Task B4: minor (deferred, B3 test): test_frcnn_train_smoke has a 60 s wall-clock assertion — flaky under CPU contention (138 s observed); controller re-ran full suite clean 369
- Task B4: fix round 1/5 (4 addressed, 0 open — OOM recipe per family, tar excludes/ignore-failed-read, smoke duration, call-sequence test; commits e311aab..1620a09)
- Task B4: minor (deferred): duplicate test_runner_call_sequence_and_no_empty_args definition in tests/test_run_exp.py:100/112 (first shadowed)
Task B4: complete (commits 38e0e3e..1620a09, review clean)
- Task B5: dispatched, base 1620a09
- Task B5: implementer DONE_WITH_CONCERNS, commit c020344 (381 passed); map CI ~21 min/arm over 5 splits on CPU at n=1000 (default kept, README notes --map-ci-n 0)
- Task B5: task review (opus) Needs fixes — Important: no best_epoch/select_from validity check → invalid runs gated as GO/NO-GO
- Task B5: minor (deferred): count_errors recomputed twice per run
- Task B5: fix round 1/5 (8 addressed, 0 open — best_epoch validity, null-mAP blanks, MAE cross-check, fold-alignment test, empty seeds, rename, per-split train time, tar json; commits c020344..c66d3f4); controller re-ran full suite
- Task B5: minor (deferred): latency.json schema {fps} assumed (no task defines it); map CI ~1.8 h CPU at n=1000
Task B5: complete (commits 1620a09..c66d3f4, review clean)
- FINAL REVIEW (opus, df3c571..c66d3f4): with fixes. Important: README smoke must exercise .bat resume branches (re-run → "done already"; delete summary_test.json → skip train). Minors: duplicate test name; 60 s wall-clock assert in test_frcnn; frcnn metrics.jsonl append mode; preds_test.json size check note; YOLO best_epoch tie → Ultralytics keeps LAST tie (fallback uses first); summarize warns on non-protocol batch/accum; stage-2 README OOM reminder; Windows sleep/update README line. Verified e2e with real scripts on CPU (bash runner, resume, summarize).
- Final fix wave: queued after C2 (no parallel implementers); then one scoped re-review.
- P3b final fix wave: dispatched, base 47d37b7
- P3b final fix wave: commit 37c6ea1 (414 passed); re-review dispatched
- P3b final fix wave re-review: all 8 addressed, no new breakage (commit 37c6ea1)
P3b FINAL REVIEW: clean after one fix wave (df3c571..37c6ea1). Push held until C2 fix (47d37b7 precedes 37c6ea1).
