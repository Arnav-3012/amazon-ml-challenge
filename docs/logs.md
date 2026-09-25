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

## 2026-09-25: M1 loader + metric + all-empty baseline (code written, not yet run)
- Pre-checks (tool-verified): every line in all 7 TSVs has exactly 4 tab-separated fields (the GT has 2);
  quote chars appear in 4/6/0 train S1/S2/S3 lines and 134/349/330 test lines, as CSV-style `""` escapes
  → read with `quoting=csv.QUOTE_NONE`. The validator reads line by line (`split("\t")`), consistent with this.
- Finding for M2: India records mix native scripts (Devanagari, Gujarati, Malayalam) with Latin text,
  so multilingual handling matters beyond France.
- Added `src/io.py` (config paths resolved from the repo root; `load_source`, `load_gt` with id-set
  asserts, `write_id_lists`), `src/metric.py` (f05, macro_f05 + singleton breakdown, `__main__`
  self-test), and `src/baseline_empty.py` (train all-empty score == singleton rate assert, cardinality
  histogram, S2/S3 share, rows per source per country for train AND test, all-empty test outputs).
- Awaiting the user's run: `python -m src.metric`, `python -m src.baseline_empty`, validator.
- The user flagged multilingual data. Sampled the first 200k rows/file with a perl script-counter
  (numbers in breakdown.md Revision 1a). S1 is ASCII (except France); India S2/S3 has ~25% native script;
  US S2/S3 names have ~6.8% injected accents. Added breakdown Revision 1a.
- **License flag:** `unidecode` (phase-0 dep) is GPL-2.0+. The challenge rule constrains the *model*
  (MIT/Apache), not libraries, but GPL code in the reproducible package is avoidable risk. The
  alternative is `anyascii` (ISC, permissive, broader script coverage). Decision pending with the user
  before the M3 normaliser.

## 2026-09-25: M1 run results
- User ran metric self-test (PASS) and baseline_empty (assert held): singleton rate = 0.055848
  (123,247 / 2,206,821). Full cardinality histogram and country row counts logged in phases.md.
- Validator invocation failed on the first try: run from `code/business_entity_resolution/`, but
  `utils/` lives at the repo root. Reissued the command with `cd /Users/arnav/amlc2026` first.
