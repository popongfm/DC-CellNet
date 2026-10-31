# SDD ledger — plan: docs/plans/2026-10-30-phase3-p3c.md
Spec: docs/specs/2026-10-30-phase3-design.md §4. Base branch dccnet. Depends on P3b B4/B5 for C3.
Pre-flight scan:
| Pair / task | Produces → consumes | Finding |
|---|---|---|
| C1 → C2 | relabel.py from/to_yolo_lines, manifest (hash, image, origin) | consistent |
| P3b B4/B5 → C3 | run_exp per-run block, summarize_loso loaders | consistent |
| C3 self | Step 1 test_recheck_sample_and_agreement + Step 5 `--sample-recheck/--recheck` in import_bccd_relabel.py, but Files list omits import_bccd_relabel.py / relabel.py / test_relabel.py | Ruling: C3 Files include those three — plan text mandates the feature — cost if wrong: none |
- Task C1: dispatched, base c66d3f4
- Task C1: implementer DONE_WITH_CONCERNS, commit 33ebdd5 (test_relabel 11 passed; full suite 396 + flaky FRCNN 60 s timing assert under load)
- Task C1: task review (sonnet) Approved, 0 Critical/Important
- Task C1: minor (carried to C2): label text written via write_text → CRLF on Windows; manifest hash is of LF text. Ruling: C2 --progress must hash newline-normalised text (read_text + normalise) — and C2 makes C1 write labels with newline="\n" + utf-8 explicitly — cost if wrong: progress count wrong on Windows.
- Task C1: minor (carried to C2): from_yolo_lines accepts nan/inf, w/h ≤ 0, values outside [0,1] → C2 rejects non-finite and w/h ≤ 0 (values slightly >1 allowed? → reject outside [0,1] with tolerance 1e-6)
- Task C1: minor (deferred): no clipping of boxes to image bounds; boundary/chained-dedupe/CLI-error tests missing; --force has no warning; partial export on mid-loop failure
- Task C1: README.txt with relabel rule written into export dir (accepted; C2/C3 must not duplicate it)
Task C1: complete (commits c66d3f4..33ebdd5, review clean)
- Task C2: dispatched, base 33ebdd5
- Task C2: implementer DONE, commit 47d37b7 (412 passed)
- Task C2: task review (sonnet) Needs fixes — Important: --progress hash keeps blank lines/trailing whitespace the parser ignores → false 'edited'
- Task C2: fix round 1 queued until P3b final fix wave commits (both touch README.md)
- Task C2: fix round 1/5 (6 addressed, 0 open — progress hash normalisation, README full predict cmd + --force warning, missing-manifest warning, totals, dup stems; commits 37c6ea1..a39033e); controller re-ran full suite
- Task C2: minor (deferred): internal whitespace runs not collapsed in progress hash (touched file only)
Task C2: complete (commits 33ebdd5..47d37b7 + 37c6ea1..a39033e, review clean)
- Task C3: dispatched, base a39033e
- Task C3: implementer DONE_WITH_CONCERNS, commit f135940 (428 passed); compare_runs.load_run gained tag='test' (backward compatible); .bat untested
- Task C3: task review (opus) Needs fixes — Important: run_exp.sh do_predict loses ERR trap (no set -E) → predict failures not logged
- Task C3: minor (deferred): docstring section numbering; test_runner_extra_splits lacks EXP_DEVICE/frcnn arm
- Task C3: fix round 1/5 (5 addressed, 0 open — set -E, recheck carry-over, sample.json, n/a render, README note; commits f135940..1a40088); controller re-ran full suite
- Task C3: minor (deferred): thr subshell failure may log FAILED twice (cosmetic)
Task C3: complete (commits a39033e..1a40088, review clean)
- FINAL REVIEW (opus, c66d3f4..1a40088): with fixes. Verified real CPU e2e C1→C2→C3 on 204-image split. Important: (1) --progress text hash breaks after X-AnyLabeling re-export (unformatted floats) → 204/204; (2) README step 3 import-once/export-to-labels gap can lose edits. Minors: --force no backup; stale summary_bccd_overlap_v2 after re-import; --recheck on empty files gives F1 0; bias table reading guide/Δ-vs-candidate/recheck F1 render; utf-8-sig BOM; stale recheck block after re-import; duplicate BCCD predict (skip).
- Ruling: --progress compares parsed values (manifest stores exported boxes+labels; unedited iff same count and every box matched same-class IoU ≥ 0.98, order-independent), replacing the text hash — plan said hash; spec only says "progress helper" — cost if wrong: tiny edits (<2% IoU change) not counted as edits.
- Final fix wave: dispatched, base 1a40088
- P3c final fix wave: commit 0c2e086 (444 passed); re-review (opus): findings 1–8 addressed; residual New Important: X-AnyLabeling 4.2.3 export clamps boxes to [0,W-1]×[0,H-1] → border boxes <50 px drop below IoU 0.98 → --progress miscounts (~19/204 images on real GT); Minor: README 3(d) wrong — X-AnyLabeling silently SKIPS shapes whose label is not in classes.txt (not rejected at import); Nit: same-second --force backup name collision.
- Ruling: parked (no second fix wave per process) — progress miscount — real; affects only the --progress counter, never the import/v2 split — cost if wrong: Mickie sees ~19 extra "edited" files; fix = clamp manifest boxes to W-1/H-1 before labels_unedited (surfaced to user as follow-up).
- Ruling: parked — README 3(d) misspelled-class wording — real and can silently drop boxes from Mickie's export — surfaced to user as a must-do-before-relabel follow-up — cost if wrong: misspelled boxes lost without error.
- Ruling: parked — same-second --force backup collision — nit, requires two --force runs within 1 s — cost if wrong: an older backup nested/overwritten.
P3c FINAL REVIEW: one fix wave done (c66d3f4..0c2e086); 2 residuals parked + surfaced.
