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

## 2026-09-25: M1 validator PASS — M1 closed
- Validator PASS from repo root: 1,732,544/1,732,544 rows in both output files, all empty, no blocking
  issues. `--check-ids` was not used (off by default; irrelevant for an all-empty submission).
- M1 milestone is complete. Next: M2 EDA.

## 2026-09-25: M2 EDA script written; unidecode -> anyascii swap done
- Swapped `unidecode` (GPL-2.0) -> `anyascii` (ISC) in the venv (`uv pip uninstall unidecode`,
  `uv pip install anyascii`) and re-pinned `requirements.txt` to the phase-0 dep set (+ anyascii).
  Noticed the venv also has ipykernel/jupyter/psutil/etc. installed outside this scaffold's scope
  (not from any command run in this session) — left untouched, excluded from requirements.txt since
  they're not phase-0 deps; flagging in case they were installed for unrelated exploratory work.
- Wrote `src/eda.py` covering all 12 requested checks (A1-4 structure, B5-6 scripts, C7-10 difficulty
  incl. 200k-pair sampling, D11-12 vocabulary + France sample), writing to `docs/eda.md`.
- Smoke-tested end-to-end in an isolated /tmp copy against a ~400-entity sampled dataset (not the
  real data) to catch logic/crash bugs before the full 24M-row run: all 12 sections executed and
  produced a well-formed docs/eda.md. Full-scale correctness (e.g. do the numbers make sense) is NOT
  verified — only "does it run and produce all sections" is.
- **Runtime flag:** C8's postcode-presence check uses `df.apply(axis=1)` over full source files (up
  to 5.3M rows) — correct but slow (row-wise Python), likely several minutes. Not a metric-path loop
  (that rule is about metric.py), just a one-time EDA cost. Not optimized further since eda.py is
  run once, not in the pipeline hot path.
- Revision 2 of breakdown.md is NOT yet written — it needs the actual docs/eda.md numbers from a
  full-data run, which only the user can produce (no self-claimed execution).

## 2026-09-25: eda.py extended to match the eda-fe-entity-resolution-hackathon skill contract
- The skill (loaded via /anthropic-skills:eda-fe-entity-resolution-hackathon) has its own numbered
  steps and a mandatory decisions-summary output. Mapped its 7 steps against the already-written
  eda.py: Step2->A2, Step3->A1, Step4->B5/B6/D11/D12, Step5(partial)/Step7->C7 (random negatives
  only). Missing: Step1's null/dup check, Step5's 3-way channel split, Step6's naive-blocking-scale
  estimate, Step7's blocked (not random) negative hardness gap.
- Added section E to `src/eda.py`: E1 (null name/addr + exact dup name+address per source),
  E5 (name-only/address-only/both-strong/both-weak split on true pairs via existing C7 similarity
  arrays, thresholds 80/50 on token_set_ratio), E6/E7 (cheap real top-20 blocking on a 500-S1 sample,
  in-country, name-similarity only — gives a naive candidate-pool-size estimate and a blocked-vs-
  random negative name_sim p50 gap). Plus a closing decisions-summary block with placeholders to
  fill from the real numbers.
- Re-smoke-tested the full script (all sections A-E) on the same ~400-entity sample as before;
  runs clean, section E and the decisions summary render. Cleaned up the /tmp smoke copies again.
- Still not run on real data. Revision 2 of breakdown.md and the decisions summary both wait on that.

## 2026-09-25: eda.py performance rewrite (killed a 27+ min hung run)
- First real run of `src/eda.py` was killed by the user after 27+ minutes at ~100% CPU with
  `docs/eda.md` still empty. Audited every hot path against the ~5-6M row source files:
  - B5 (non-ASCII per source x country x field): was `groupby().apply()` with a Python
    `sum(any(ord(c)>127...))` per group -> now vectorized `str.fullmatch` regex + boolean groupby.sum.
    This was the biggest single culprit (runs over all 6 source files, both fields).
  - C8 (postcode presence, full files) and D12 (France postcode): were `df.apply(axis=1, ...)` ->
    now vectorized `str.contains(regex)` per country.
  - A3 (distractor rate) and A4 (country consistency on true pairs): were pure-Python loops over
    dict items (up to 7.6M iterations) -> now vectorized via `.isin()`/`.map()`/`.groupby()` on
    small DataFrames built from the GT explosion.
  - B6 (non-ASCII true-pair name similarity) and D11 (top-60 tokens per country): were unsampled
    over the full population (~2M+ non-ASCII matches for B6; full name/address columns for D11,
    up to ~1M+ rows/country) calling anyascii/rapidfuzz per item, which is pure-Python and doesn't
    get faster from vectorizing the surrounding loop -> capped with samples (B6: 20k pairs,
    D11: 100k rows/group), consistent with the file's existing sampling policy for C7/C9.
  - E1 (`df.duplicated`) was already vectorized; left as is.
  - Smoke-tested the full rewritten script end-to-end on the same ~400-entity sample: runs in
    ~2 seconds (was previously part of a run still going after 27 min on the small sample's
    real-scale counterpart). Spot-checked A3/A4/B5 output values for correctness, not just
    "did it crash" — numbers are consistent with the earlier perl-based script-mix sample.
  - Real-scale timing is NOT verified (never ran to completion on the full 24M-row dataset,
    by design of the kill) — expected to be low minutes now, but the user's next run is the
    first real measurement.
- **Mistake logged:** shipped an EDA script whose row-wise `.apply`/Python-loop paths were only
  smoke-tested at ~400-row scale, not estimated against the real ~5M-row scale before handing it
  to the user to run. Should have done a back-of-envelope op-count check (rows x per-row overhead)
  before the first real run, not after 27 minutes of wasted wall-clock time.

## 2026-09-25: bumped B6/D11 sample sizes after checking convergence
- Quantified whether sampling B6 (name-sim percentiles) and D11 (top-60 tokens) actually loses
  signal: B6's median estimate has SE ~0.18 points (0-100 scale) at n=20k — negligible, no design
  decision would change. D11's top-60 token overlap vs full population was ~58/60 at n=100k in a
  Zipfian simulation — the top 10-20 tokens (the ones that matter for the normaliser) are stable at
  any sample size; only the rank-60 tail could differ.
- Per the user's request, bumped both anyway for extra margin: B6 20k -> 80k, D11 100k -> 400k.
  Measured per-call cost of anyascii (~0.05us) and rapidfuzz token_set_ratio (~0.66us): the bump
  adds well under 1 second of total runtime. Re-smoke-tested end to end: still ~2s on the small
  sample. Real-scale timing still not measured (design intent: low minutes, unverified).

## 2026-09-25: found and fixed the REAL bottleneck — killed second hung run at 11:48
- Second real run (after the vectorization rewrite) was still running at 11m48s, 100% CPU, no
  output. Re-audited the file instead of waiting past 15 min on a guess, and found the actual bug:
  C7's negative-sampling `while` loop called `rng.choice(list(s1_train))` — rebuilding a list from
  the 2.2M-entry `s1_train` dict on EVERY iteration, up to 200,000 times. Measured cost directly:
  list(dict) with 2.2M keys takes ~10ms; x 200k iterations = ~2000s (~33 min) on this line alone,
  on top of everything else. This loop existed in the ORIGINAL (pre-rewrite) code too — it was
  never the thing I fixed in the first pass, so the first "optimization" round fixed real problems
  (B5/C8/D12/A3/A4) but missed this one, which turned out to dominate.
  Killed the process (was PID 51845). Fixed: hoisted `list(s1_train)` out of the loop into
  `s1_id_list`, built once. Verified in isolation at realistic scale (2.2M S1 entries, 200k
  negative samples): 0.31s vs the ~2000s projected for the buggy version.
- Re-audited the ENTIRE file for any other "rebuild a large list/dict inside a loop" pattern
  (grepped every loop/list()/sample() call) — found none else. The other `list(s1_train)` call
  (E6/E7's sampling) is outside any loop, called once, ~10ms, not a problem.
- Re-smoke-tested end to end on the same ~400-entity sample: ~2s, all 21 sections render.
- **Mistake logged:** the first "audit every hot path" pass was not actually exhaustive — it found
  the `.apply`/dict-loop patterns but missed a rebuild-inside-a-while-loop pattern in existing code
  that wasn't touched by the rewrite. A real audit should have grepped for `list(` / `dict(` calls
  inside every loop body, not just looked for `.apply(axis=1)` and obvious per-row iteration.

## 2026-09-25: eda.py completed successfully; coverage-checked docs/eda.md
- Third run completed: ~37KB, 296 lines, all 12 original checks (A1-D12) + all 3 skill-required
  additions (E1, E5, E6/E7) present and populated. No section missing.
- Found one cosmetic bug while reading closely: D11's header hardcoded "100k rows/group" text but
  the actual sample size (bumped earlier to 400k) was correctly used in the data (`n_sampled=400000`
  shown in every row). Fixed the header to read the D11_SAMPLE variable via f-string so it can't
  drift from the real value again. Does not require a rerun — docs/eda.md's real numbers were
  already correct, only the label text was stale.
- Key numbers now on record (feed Revision 2): assignment constraint holds EXACTLY (0/7,638,365
  matched ids under >1 S1) -> Hungarian/per-component matching is justified, not just "heavy".
  Country consistency on true pairs = 100%, missing-country = 0% -> country is a clean, safe
  blocking pre-filter signal. Naive top-20 blocking pool estimate ~44M pairs -> OVER the ~2M
  brute-force threshold, overturning the original hardware-plan assumption; FAISS or a cheaper
  channel needed at blocking time, not just for special cases. Two-channel split on true pairs:
  66% both-strong, 15.56% name-only, 15.43% address-only, 0.04% both-weak -> both channels
  independently necessary (15%+15% would be lost by either alone), confirms Leaf 2's two-channel
  pick. Blocked-vs-random negative name_sim p50 gap = +27.3 -> blocking negatives are meaningfully
  harder, confirms training on blocking-produced hard negatives over random ones (Leaf 3).

## 2026-09-25: M2 closed — Revision 2 written, decisions summary filled in
- Verified the France "(France)"/"(Frànce)" name marker against the FULL test data (not just the
  15-row D12 sample) before writing it into a permanent doc: S1 8.01%, S2 6.71%, S3 6.61% of France
  rows carry it in business_name; 0.00% in business_address across all three sources. Confirmed
  name-only, resolving the "still open" question from the first Revision 2 draft.
- Appended "## Revision 2 (from EDA)" to docs/breakdown.md: resolved open questions (scale,
  assignment=exact one-to-one, country consistency=100%, distractor rate ~25-27%, script mix exact
  percentages, cross-script similarity after transliteration, postcode near-useless as a blocking
  key, near-zero dupes/empties), leaf confidence updates, and one overturned decision (Leaf 2
  hardware plan: naive blocking pool ~44M pairs is OVER the brute-force ceiling the original plan
  assumed — country pre-filter proposed as the first fix, FAISS only if that's insufficient).
- Filled in docs/eda.md's "Decisions summary" block with the real verdicts (was placeholders).
- M2 is DONE. Next: M3 (blocking v1) per docs/plan.md, informed by Revision 2's country-pre-filter
  decision and confirmed two-channel design.

## 2026-09-25: patched the stale D11 label directly in docs/eda.md (no rerun needed)
- The only post-run change was a header-text fix in src/eda.py (label said "100k", actual
  sampling and every printed n_sampled value was already 400000 — a display-only bug). Patched
  docs/eda.md's D11 header text to match ("100k" -> "400,000") by hand instead of rerunning the
  full script, since no computed value in the file was affected. Data unchanged.
