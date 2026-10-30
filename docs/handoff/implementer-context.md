# Implementer context (read after your brief)

Project: DC-CellNet — a CenterNet/CircleNet-style blood-smear cell detector (RBC / WBC / Platelets) with a density head and a local count-consistency loss. Package lives in `dccnet/` of the repo.

## Where you work
- Repo: `/home/claude/repo` (git, branch `dccnet`). Work and commit here with normal Bash/Read/Write/Edit tools.
- Python: ALWAYS use `/home/claude/venv/bin/python` (Python 3.11, torch CPU-capable, numpy, opencv, pycocotools, imagehash, pyyaml, pytest installed). Run tests from `dccnet/`: `cd /home/claude/repo/dccnet && /home/claude/venv/bin/python -m pytest tests/<file> -v`.
- Do not `pip install` anything without reporting it; if a dependency is missing, report NEEDS_CONTEXT.
- Real data (read-only, gitignored symlinks): `dccnet/data/txl_pbc` → TXL-PBC (`images/`, `labels/`, `classes.txt`, `data.yaml`, `metadata_file.xlsx`, `BCCD_selection.xlsx`), `dccnet/data/bccd_full` → Shenggan BCCD_Dataset (`BCCD/Annotations`, `BCCD/JPEGImages`, `BCCD/ImageSets`).
- Code must also run on Windows (paths via pathlib, no shell-specific calls). Data roots are always arguments/config values — never hardcode folder names like `txl_pbc` (the user's Windows copy lives at `dccnet/data/TXL-PBC_Dataset/TXL-PBC`).

## Binding project constraints
Read the "Global Constraints" section of the plan: `/home/claude/repo/docs/plans/2026-10-26-dccellnet-m0-m2.md` (read ONLY that section and the File Structure section — not other tasks). Additional rulings:
- GPU target is RTX 5060 Ti, 8 GB VRAM: default `train.batch: 8`, AMP on CUDA.

## Commits
Commit with: `git commit -m "<subject>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JRJctCoB6ZnbRnRuP6gUwu"`
Never push. Never commit anything under `dccnet/data/`, `dccnet/runs/` or `.superpowers/`.

## Rules
- Do all work yourself; never spawn subagents or reviewers.
- TDD: write failing tests first, show RED, then GREEN.
- Ask (NEEDS_CONTEXT) rather than guess on anything that changes interfaces other tasks rely on.
- Test output must be pristine (no warnings); filter third-party DeprecationWarnings in pytest.ini only if unavoidable and say so.

## Report contract
Write the full report (what you built, TDD evidence RED/GREEN with commands and output, files changed, self-review, concerns) to the report path given in your dispatch. Then reply with ONLY (<15 lines): Status (DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits (short SHA + subject), one-line test summary, concerns, report path.
