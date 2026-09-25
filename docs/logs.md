# Logs

## 2026-09-25: Phase 0 scaffold
- Read inputs: the PS RTF, breakdown md and video transcript. The transcript file is named
  `6ab509c5b7036_ml_challenge_2026_video_eng.txt`, which doesn't match the `*transcript*` glob.
- `../student_resource/` is **missing**. Created placeholders: `utils/validate_submission.py`,
  `Documentation_template.md` (one-line PLACEHOLDER comments), and a dangling symlink
  `dataset -> ../student_resource/dataset`. Replace them with the provided files once available.
- `uv` was not installed. `brew install uv` failed (Xcode license not accepted; needs sudo, not
  run). Installed uv 0.12.19 via `pipx install uv` → `~/.local/bin/uv`.
- Created `.venv` (Python 3.11.13). Installed pandas, numpy, pyarrow, pyyaml, scikit-learn,
  lightgbm, rapidfuzz, unidecode. Pinned with `uv pip freeze` into
  `code/business_entity_resolution/requirements.txt`. Resolved: pandas 3.0.6, numpy 2.4.6,
  pyarrow 25.0.1, scikit-learn 1.9.1, lightgbm 4.7.0, rapidfuzz 3.14.6, unidecode 1.4.0,
  pyyaml 6.0.3.
- **Risk:** `lib_lightgbm.dylib` links `@rpath/libomp.dylib` with rpaths only to Homebrew/MacPorts
  libomp, and neither is installed, so `import lightgbm` will likely fail. Fix: `sudo xcodebuild
  -license accept && brew install libomp`. Not verified by import (the user runs verification).
- **Risk (M1):** pandas 3.x uses the new default string dtype. `dtype=str` + `keep_default_na=False`
  must still yield `""` for empty ID lists. Assert this in the M1 loader.
- Wrote CLAUDE.md, docs/{context,plan,architecture,phases,logs,decisions_mistakes}.md, and
  docs/breakdown.md (source copy + Revision 1). Also config.yaml, README skeleton, and .gitignore.
- `git init` + commit was **not done**. The user declined: they will create a GitHub repo first
  and say when to init/commit/push.
- User requirement: personal md files (CLAUDE.md, docs/) must NOT go into the submission zip.
  M8 packaging copies only the whitelisted paths.
- LightGBM import failed as predicted (`Library not loaded: @rpath/libomp.dylib`). The user ran
  `sudo xcodebuild -license accept && brew install libomp` and reports the phase-0 import/version
  check passes. This is user-reported; the output wasn't seen in this session. Blocker 2 is resolved.
- student_resource found at `~/Downloads/student_resource` (the zip `6ab10eb3b23ba_student_resource.zip`
  sits alongside it). Not moved (2.4 GB). Relinked `dataset -> /Users/arnav/Downloads/student_resource/dataset`
  (absolute). Copied `utils/validate_submission.py` and `Documentation_template.md`; `cmp` says
  IDENTICAL. Its README.md is the same PS text as the RTF, with no new rules.
- Scale (`wc -l`, including header): train S1 2,206,822 / S2 5,034,617 / S3 5,285,604 / GT 2,206,822;
  test S1 1,732,545 / S2 4,887,274 / S3 5,082,317. Far bigger than the breakdown assumed, which
  affects loader memory, blocking chunking and encoder feasibility. Quantify at M2.
- Sample rows: S1 train is US-style ("1795 Westchester Drive, High Point, NC"); GT lists carry up
  to 5+ IDs mixing S2 and S3.
- Remote: https://github.com/Arnav-3012/amazon-ml-challenge. Standing rule (CLAUDE.md updated): the
  agent never commits or pushes; it gives the commands and the user runs them.
