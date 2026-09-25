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

## 2026-09-25: Revision 2a — correction pass on Revision 2, docs-only
- Three errors in Revision 2 caught and fixed before M3 build (all logged in
  `docs/decisions_mistakes.md`):
  1. The ~1-2M brute-force ceiling is about search-index vectors, not candidate pairs; 44M
     candidate pairs is fine to score in chunks. The real constraint is the search itself
     (2.2M S1 × ~10M S2+S3) — infeasible brute-force regardless of k, so blocking must be an
     inverted-index/ANN structure by design, not a fallback reached for after brute-force proves
     too slow.
  2. Country pre-filter reduction recomputed properly as `1/Σ(share²)`: ≈1.9x on train
     (US 60/India 40), ≈2.5x on test — not the "~3-4x" Revision 2 claimed. Still applied (100%
     country consistency on true pairs makes it free and safe), but it's a companion to ANN
     blocking, not a substitute for it.
  3. Assignment is many-to-one (each S2/S3 → ≤1 S1, each S1 → many), so Hungarian (a one-to-one
     matcher) was the wrong name for the decision-layer assignment step. Correct rule: each
     S2/S3 record goes to its argmax-scoring S1 entity if above threshold. Replaced every
     "Hungarian" reference in `docs/breakdown.md` (Revision 2's two Leaf 4/hardware-plan entries)
     and `docs/phases.md` (M2 closure entry) with this per-record argmax rule.
- Also recorded three new M3-facing findings surfaced while re-reading the data: noise looks
  generator-produced (mine operators from train positive pairs, invert in the normaliser rather
  than hand-writing rules); 3.4% of S2/S3 records have empty address (need explicit `has_address`
  handling, not an implicit both-fields-populated assumption); France S1 uses région vs S2/S3's
  département (different hierarchy levels, not noisy formatting of the same field) and city is a
  weak identity signal there, so house number + street name should carry most of the address-match
  weight for France.
- Files touched: `docs/breakdown.md` (Revision 2a appended; 3 in-place corrections to Revision 2's
  Hungarian/ceiling/3-4x text), `docs/decisions_mistakes.md` (3 mistakes logged), `docs/phases.md`
  (Hungarian reference corrected in the M2 closure entry + this entry), `docs/logs.md` (this entry).
  No code changed.

## 2026-09-25: M3a — noise-operator miner + normaliser v1 + evaluation (code written, NOT run)
- Pre-checks on raw data (grep/pandas peeks, no pipeline code run): alias names are always
  `<generated brand> <kw> <real S1 name>` (kw ∈ formerly [known as]/dba/doing business as/trading as/
  t/a/a/k/a/aka/née, S3 only, ~0.1-0.4% each) → core = right side. `(India)` sits in 2.0% of S1 names
  (organic, like France's `(France)` at 8% of test S1) → one `country_marker` rule, India is the train
  proxy for the France-only strip. Digits inside words in S2 names: 0/1/5/8/6 (2/3/9 ≈ 0); `1` = `l`.
  `n°` becomes `ndeg` after anyascii; `praivet`/`piraivet`/`elelpi` are anyascii of Devanagari legal
  words, not typos.
- `src/noise_ops.py`: 200k seeded true pairs, name + address operators per country, lexicon-gap tables
  (unmapped legal-form look-alikes, state variants, honorific add vs S1 base rate, alias side).
- `src/normalise.py`: 17 ablatable rules in polars expressions (Rust regex: no lookaround, so token
  rules run on lists; digit_fix moved after punct strip for token boundaries). Lexicons per country,
  seeded from EDA D11 + the pre-checks; must be re-checked against `docs/noise_ops.md` before the cache
  run. `--smoke N` prints rate + projection first (eda.py lesson).
- `src/normalise_eval.py`: precision guard uses the EXACT same-country non-pair collision rate from
  group counts (Σ c1·c2 − equal true pairs) / (N1·N2 − T), not 200k random non-pairs: C7 already
  showed exact-name rate 0.00% on 200k random pairs, so a sampled guard could not move.
- `src/io.py`: `load_source(nrows=)`, `load_gt_pairs()` (polars). Config `interim_dir` →
  `artifacts/interim`. Requirements: `polars==1.44.2`, `polars-runtime-32==1.44.2` (MIT; the user installs).
- Only `py_compile` was run on the new modules. No results yet; `docs/noise_ops.md` and
  `docs/normalise.md` don't exist until the user runs the commands.
- 2026-09-25 (M3a, after `docs/noise_ops.md`): lexicon pass from the gap tables + raw greps. Added:
  alias kw `fka|f/k/a` (22.7k S3 names, was missing); rule `id_tag` strips "(ID: 22383)"; domain
  suffix " | www.x.com"; `M/s` prefix (29.3k S3 names) under `honorifics`; `lnc`→inc, `limtid`→ltd;
  US `trail`→trl; India native-script state translits (mharastr, dilli, krnatk, tmilnatu, pscimbng, …);
  junk address components (PO BOX/PMB, "X Region") under `null_token`; `hn` house-number prefix.
  HONORIFICS kept: all six are injected (India smt/shri/sri/mr/dr 1.3-1.7% add vs <=5/80k S1 base;
  US the 1.33% vs 0.13%); `shree` stays (organic, 612 S1). Left unmapped: one-off `private` typos
  (~0.4% of India pairs), city suffixes (city/cdp/township), bombay↔mumbai, generic-word swaps
  (→center/services). Only py_compile run.
- 2026-09-25 (M3a, smoke 20k rows/file): output correct on 36 sample rows, but ~110us/row (see
  decisions_mistakes). Rewrote the normaliser hot path to elementwise-only `list.eval`. Same pass fixed
  3 smoke bugs: "& Co" left a dangling `and` (now stripped when a legal form was removed); India
  "2 Floor" parsed as house number 2 + street "floor" (floor/unit/shop/plot/... components skipped);
  "gabriela's" → "gabriela s" (apostrophes now deleted, not spaced). `M/s` moved into HONORIFICS as `m s`.
  Only py_compile run on the rewrite.
- 2026-09-25 (M3a eval, `docs/normalise.md`): full run 246s for 6 files (peak RSS 7.6GB); eval 434s
  (10.9GB). Recall proxy either-key: US 43.0→86.5%, India 14.4→60.4%. Ablation flagged 3 rules →
  reverted (country_marker, amp_and, landmark). Examples showed 2 defects → fixed: "#98825" name tags
  (1 in S1 vs ~16k per S2/S3) added to id_tag; glued prefixes "n°68"→"ndeg68" (ADDR_PREFIX space made
  optional). Breakdown Revision 3 written. Needs one rerun of normalise + eval to confirm.
- 2026-09-25 (M3a confirm rerun): no ⚠ rules. Either-key recall US 86.61%, India 60.71%; pooled S1
  name collision 50.11% (core+legal 40.38%). "#NNN" and "n°68" fixes confirmed in examples. New bug seen:
  with amp_and reverted, "W & W Minerals" → "w w" → dedupe → "w minerals"; single-letter tokens now
  exempt from dedupe_adjacent. Peak RSS rose to 10.6GB normalise / 11.8GB eval (fits in 24GB).
- 2026-09-25 (M3a final): the single-letter dedupe exemption cost 0.01pp name recall (dedupe gain
  +1.239 → +1.229pp) and fixed "w w minerals". Still no ⚠ rules. M3a closed.
