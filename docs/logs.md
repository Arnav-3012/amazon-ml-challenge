# Logs

## 2026-09-27: CE stack adopted
- CE stack (fine-tuned cross-encoder + LightGBM 2-feature stacker) adopted: OOF(b) 0.9746 -> 0.9775. Rank features and confident-FN override rejected; batch-3 features cut for time. See docs/decisions_mistakes.md.

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
- 2026-09-25 (M3b, code written, not run): `src/phonetic.py` (skeleton + `add_skeletons` over distinct
  tokens), `src/block.py` (channels A/B/C, per-country IDF, df cap, record-centric top-m ∪ S1-centric
  top-k per source, sparse products chunked by an exact nnz upper bound, S1-range merge → parquet parts →
  sink), `src/block_eval.py` → `docs/blocking.md`, `io.write_candidates`, config `blocking:` block.
  Train keeps ranks up to m=10/k=60 (`blockgrid_train_cap{cap}.parquet`); `candidates_train` = grid cut to
  config m/k via `finalize()` (nulls out-of-cut ranks/scores so train matches a direct test run).
  Spec deviations in skeleton: leading vowel → "a" (spec rules fail istrn~eastern), non-initial y = vowel
  (sistms~systems, keyr~care), digit tokens unchanged. Verified here: py_compile; phonetic asserts; top_n
  vs brute force on random sparse matrices at 3 chunk budgets; merge has no duplicate pairs; finalize cut.
  Not run on real data. Judgement: `problem-breakdowns/m3b-blocking.txt`.
- 2026-09-25 (M3b rev, before any real run): A/C bigrams + rarest-2 fallback (config flags, default on).
  Refactor in `block.py`: `token_table` (all linkable tokens, `rare` = df ≤ cap) → `survivors()` (rare tokens,
  or the 2 lowest-df when none, tie by token) → `matrix`; posting lengths from the actual survivors.
  `--dry` now also runs the survivor pass: % records with zero surviving tokens per country × source × channel
  (with and without fallback), product nnz upper bound with bigrams+fallback and rare-only, top-3 cost tokens.
  `add_skeletons` keeps token order (bigrams need it); dedupe moved to `channel_tokens`.
  `python -m src.block --selftest`: synthetic S1/S2/S3 through the real pipeline vs an independent pure-Python
  brute force (own bigrams, skeletons, df, fallback, float32 sums), all channels, 3 chunk budgets, both
  directions. It passes: 184 fallback records and 115 surviving bigram tokens exercised. Still not run on real data.
- 2026-09-25 (M3b dry, train cap 1000, bigrams+fallback on): 85s, peak 8.1GB. Product nnz upper bound per
  direction India 0.96B, US 1.93B (~5.8B over both directions). Fallback cost is almost all US A (446M vs
  125M rare-only on S2; S3 similar), and small elsewhere. Zero-survivor records with fallback: A S2/S3 India
  17.6/12.5%, US 6.2/6.1% (no token shared with any S1); C 0% everywhere; S1 A/C 0%. B has no fallback:
  zero-token India S2/S3 22.1/30.6%, US 18.3/17.1%. Go to smoke.
- 2026-09-25 (M3b smoke 300k rows/file, train): runs end to end in 29s, peak 7.1GB. Sparse products about
  52M nnz/s (794M nnz in 15s); merge about 9M grid pairs/s (66M in 7.5s). Actual nnz is 90-97% of the upper
  bound. Smoke df is about 14x smaller than full, so the cap is looser here and smoke zero-token % is lower
  than dry (India S2 B 6.9 vs 22.1). Full-run projection: products about 2 min, grid about 0.5B rows, peak
  memory estimated 12-14GB (US S1-centric frames about 240M rows, about 3.3GB).
- 2026-09-25 (M3b eval run 1): `python -m src.block_eval` ran about 1h25m and exited without writing
  `docs/blocking.md`. The cause is unknown: it ran in the terminal with no log and the scrollback was not recovered.
  Before the rerun: `shared_flags` no longer uses explode+join (equivalence checked on synthetic data);
  a progress log was added (`[elapsed] i/8 step`, sub-steps per country × source group); a crash now prints
  the traceback + peak RSS. The rerun writes `artifacts/interim/block_eval.log` via tee.
- 2026-09-25 (M3b first PC): candidates_train = 278,134,881 pairs (~126/S1), grid 499.7M; PC = 90.30%
  (6,897,769/7,638,365). Far below the ~99.5% the target needs. Added `src/block_diag.py` for a quick
  per-segment + budget-curve read before the slow eval.
- 2026-09-25 (M3b diag): PC 90.30% at 126 cand/S1; India 85.4% (native 79-80%), US 93.6%. m=10/k=60 grid
  ceiling only 92.6% at 226/S1. True S1 is record-centric rank 1 for 78.7% of pairs, top-10 88.5% (slow tail
  = same-score ties). 565,699 of 740,596 misses are outside every channel's top-10/60. Added channel X (A|t, B|t,
  C tokens in one index; score = A+B+C) so address breaks name ties in one ranking. Selftest passes with X
  (4 channels, 3 budgets). Diag now prints X-only vs all-channel budget curves.
- 2026-09-25 (M3b diag with X): PC 90.92% (+0.6pp); X alone 88.06% at 32 cand/S1 vs all channels 89.04% at 65.
  The tie hypothesis was mostly wrong: X rank-1 is 81.4% vs best-single 82.2%, and the ceiling is still ~92-93%.
  545k of 693k misses lie outside every top-10/60 (no shared token, only over-cap tokens, or deep rank:
  undistinguished). Added `block_eval --sample N` to get the miss-cause split in minutes.
- 2026-09-25 (M3b sampled eval, 200k pairs, 833s; base_pairs 821s of it: the grid/candidates scans are still
  full-size): 18,090 misses. Causes: all shared tokens over the cap 50.0%, rank cut beyond grid 29.4%, rank cut
  inside grid 20.5%, no shared token 3 pairs (0.02%). So char n-grams are not the fix; the lost evidence is the
  name-AND-address conjunction. Added to X (config flags): composite K|name_skel|addr_skel tokens; A joined name
  ("blairhawaii"); sorted bigrams (word swaps). Selftest passes with all three. Not run on real data yet.
- 2026-09-25 (M3b composite run): PC 90.92% → 97.20% (config, all channels, 153.9 cand/S1); X only m=5/k=10
  88.06% → 96.31% at 31.3/S1. India native 80-82% → 94.4-95.3%. X rank-1 81.4% → 94.3%. Misses 693k → 214k.
  Composite K|name|addr + joined name + sorted bigrams KEPT. Diag gains a perfect-matcher macro F0.5 ceiling
  per budget row (formula checked against the exact metric on a toy case).
- 2026-09-25 (M3b budget decision): perfect-matcher macro F0.5 ceiling per budget row. X m5/k10 = 0.9873 at
  31.3 cand/S1 (PC 96.31%); config all-channel m5/k30 = 0.9904 at 153.9; best grid 0.9924 at 277. Knee = X m5/k10
  (after it, +0.001 ceiling costs 15-60 cand/S1). Config: k=10, select=[X] (A/B/C kept as features on selected
  pairs). New: finalize(select=...), `block --refinalize` re-cuts the train grid in seconds. Selftest + a
  finalize unit check pass. Widen later (X m10/k10 0.9887 @ 51) only once M4 runs end to end.
- 2026-09-25 (M3b confirm): refinalize (select=[X], m5/k10) + diag: config PC 96.31% (India 94.71, US 97.39),
  equal to the X column in every segment; 281,532 misses; F0.5 ceiling 0.9873 @ 31.3 cand/S1. M3b closed.
  Test block run deferred to M4 inference so test candidates use the final blocking config.
- 2026-09-25 (M3b test run): `block --split test` 469s, peak 13.7GB; output/candidate_pairs.tsv written, validator
  PASS (1,732,544 S1 rows, 0 empty candidate lists; matching_results still the M1 all-empty file, not submitted).
  Pending: one `--check-ids` validator run to confirm every candidate ID exists in test S2/S3.
- 2026-09-25 (M3b close): validator `--check-ids` PASS: every candidate ID exists in test S2/S3 (9,969,589 valid IDs).
- 2026-09-25 (M4 code, NOT RUN on real data): `src/features.py` (pass 1 per 2M-row chunk: rapidfuzz cpdist name
  ratio/token_sort/token_set/partial/JW, alias best, IDF Jaccard over name tokens/bigrams/skeletons and address
  tokens/skeletons via per-country idf_{split}, legal/marker/number/street/has-address codes, blocking scores+ranks,
  is_s3; pass 2 over the whole split: v − max, rank, margin to best other, per record and per S1, for X/A/B/C
  scores + name/addr token_set; n_cand per record/S1). `src/train.py` (GroupKFold(5) OOF on the 20% S1 subset,
  3:1 negatives 50% hardest, early stop on held-out log-loss; `--loco`; `--final` at 50%, mean best iter ×1.1),
  `src/decide.py` (argmax + global t; vectorised F0.5 asserted equal to metric.macro_f05; splits + LOCO →
  docs/matcher.md), `src/predict.py` (test → matching_results.tsv with ⊆-candidates asserts). io.py gains `StepLog`,
  `peak_rss_mb` (moved from normalise.py, re-exported), `write_candidates(list_col=)`. Config: `features`, `matcher`,
  `decide` blocks; `features_dir` → artifacts/features. Verified on synthetic data only: relative() incl. ties/NaN,
  idf_jaccard, best_alias, code3, id_key injectivity, f05_vec vs metric.f05 (2,000 random cases), with_top tie-break,
  sample_rows, prep()+pair_features() on a toy frame. ml-code-sensei fixes: fill_nan in relative(); idf_jaccard join
  keeps left order (fixed float summation order).
- 2026-09-25 (M4 loss decomposition, OOF 20% subset, t=0.80, read-only scratch scripts): macro F0.5 0.96544, loss 0.03456
  = blocking-unreachable 0.01284 (ceiling 0.98716) + matcher 0.02173. Matcher: removing all FP +0.00773, recovering all
  reachable FN +0.01403. 11.2% of S1 have a reachable FN (57,560 FN pairs, p median 0.46, 94% are their record's argmax);
  2.8% have an FP; 3.8% of singletons get a prediction (-0.00211). In only 14.6% of error S1s does a negative outrank a
  positive: mostly a count/threshold problem, not ranking. Decision rules on p alone: relative-to-S1-max rules +0.0000;
  oracle top-k with k = reachable true count 0.98146 (the count is worth up to ~+0.016, but p does not carry it).
  Leader LB 0.986955 ≈ our blocking ceiling: both blocking and matcher must improve to pass it.
- 2026-09-25 (candidate-size measurement, 20% OOF, read-only): all M4 31.3/S1 PC 96.28 ceiling 0.98716 F0.5 0.96544;
  top-N by X_score: 10 → PC 90.98, F0.5 0.94989; 20 → 92.90, 0.95464; top-N by OOF p: 5 → 0.96138, 8 → 0.96541;
  OOF p ≥ 0.001 → 4.54/S1, PC 96.275, F0.5 0.96544 (no loss). Organiser rule added to context.md; plan reordered.
- 2026-09-25 (M4 predict): test 60,901,445 candidates scored; t=0.80 + argmax → 5,677,069 matches, 1,629,468 of
  1,732,544 test S1 with ≥1 match (5.95% empty; train singleton rate 5.58%). Peak 10.7GB. Pending: validator, LB score.
- 2026-09-25 21:35 IST (M4 LB): public LB 0.957 vs OOF 0.9654 (−0.008). Test predictions per country: France 259k S1,
  4.9% empty, 3.46 matches/S1; India 810k, 6.4%, 3.19; US 663k, 5.8%, 3.31 (train truth 5.6%, 3.46). No France collapse.
  Own inference (unverified): test mix is 47% India vs 40% train (India OOF 0.9515 vs US 0.9747) → mix-weighted OOF
  ~0.961 if France is average; the remainder implies France ~0.925. India under-predicts (3.19 vs 3.46): FN-driven.
- 2026-09-25 (M5-D0 code, NOT RUN on real data): `src/diagnose.py` for D0-1..7 of m5-strategy-l3 §3. Found: train.py
  saves no fold models, and predict.py keeps no test pair p. D0-5 therefore uses models/lgb_final.txt on held-out S1s
  (outside the final 50%); D0-4 max-p re-scores a 60k-record sample per split with lgb_final (train side = records
  whose candidate S1s are all held out, compared by n_cand stratum). D0-1 recomputes exact block ranks (all channels,
  both directions) for every missed pair from idf_train with block.py's functions; asserted against stored X ranks on
  2k hit pairs/country. D0-5 asserts that the no-drop rebuild reproduces every stored feature. Verified on synthetic data
  only: true_rank == block.retrieve ranks (20 random cases x 3 chunk budgets, with ties); per_s1 denominators.
- 2026-09-25 | M5-D0 run 1 aborted in D0-4 (OOM from is_in-in-agg, see decisions_mistakes); D0-1..3 done (D0-1 control asserts 0 mismatches). Fixed; resume with `--d 4 5 6 7`
- 2026-09-25 | M5-D0 complete (docs/diagnose_m5.md). OOF loss 0.0346 splits: blocking miss 0.0140, FN below t (argmax right) 0.0133, FP 0.0077 (69% on distractor records), argmax conflicts 0.0011. Orphan sim (19% S1 drop): -0.0038, FP 334->634, singletons -0.0194. Test rec/S1 5.5-5.8 vs train 4.68, max p>0.5 test 59.9% vs train reweighted 71.9% -> test has more distractor records (inferred ~40% vs 26%). No id/row-order leak (|rho| <= 0.002)
- 2026-09-25 | Distractor shift CONFIRMED, breakdown Revision 4 written, m5-strategy-l3 §5 rewritten to M5-1/M5-2/M5-3
  (docs-only turn, per the session brief). Corrected the density claim: 19% S1 drop -> 4.68/0.81 = 5.78 rec/S1 =
  test's 5.75 (decisions_mistakes.md MISTAKE entry), not "test is denser than the simulation."
- 2026-09-25 | M5-1/M5-2 code written, NOT RUN on real data:
  - `src/gate.py`: pool = block.finalize(grid, m, gate.pool_k, all channels) for m in gate.sweep_m, then a per-S1
    pre-score gate (max over channels of norm score + n_channels_hit; norm = max(score/S1-best, score/record-best)
    in that channel) capped at gate.n. `--split` writes the pool (16 S1-hash buckets, one finalize+gate_bucket pass
    per bucket so all sweep_m share one grid scan); `--eval` -> docs/blocking_v2.md (PC, perfect-matcher F0.5
    ceiling, train/test cand/S1 per m/variant/n, v1-hits-kept%); `--apply` cuts to config gate.m/n/variant ->
    candidates_{train,test}.parquet + candidate_pairs.tsv. `block.py` test now writes blockgrid_test (was:
    candidates_test + candidate_pairs.tsv directly at fixed m/k); its `--refinalize` flag removed (gate.py replaces
    it). diagnose.py D0-8: the D0-1 largest miss cell (US, no address, ASCII, name_sim>=80) — how many US S1s share
    the record's/true S1's core_name, run BEFORE `gate --apply` overwrites candidates_train (D0-1 depends on it).
  - `src/cv_full.py`: GroupKFold(5) on 100% of train S1s (fold = permutation position mod 5, same seed as M4's
    subset permutation but a different derivation). Per fold, an S1-dropout "world": drop 19% of ALL train S1s
    (seeded per fold), recompute only the columns that can change when rows disappear (record-centric ranks,
    features.relative's *_drec/*_rkrec/*_gaprec, n_cand_rec) as memory-mapped .npy overlays on the stored
    feature parts, read via `Data.gather`/`Data.predict` (part-by-part, no full materialisation). Training rows =
    other folds minus the dropped S1s, 10% of those held out for early stopping (all their rows), rest sampled by
    the M4 rule (train.sample_rows, reused). OOF (a) standard = every row scored by its fold's model on stored
    features; OOF (b) test-density = one more dropout world (different seed) scored the same way, PRIMARY metric.
    `score()` reimplements decide.py's argmax+threshold rule vectorised over NaN-able p (rows outside a world are
    NaN) and asserts against `metric.macro_f05` on request. `--check` asserts the no-drop world reproduces every
    stored record-side feature exactly (all rows, not sampled). `--curve` retrains fold 0 at cv_full.curve_fracs of
    its training S1s with fixed rounds (fold 0's best iteration), scored on (b) with the other 4 folds' OOF (b) p
    supplying their competitors. `predict.py --folds`: mean of the 5 fold models, t = OOF (b)'s best t, also saves
    test p to oof/test_p.parquet (stage-2 input per m5-strategy §5 step 4).
  - Verified on synthetic data only (scratchpad/t_m5.py, t_data.py, deleted after use): world() rrank-shift/relative/
    n_cand_rec vs an independent brute force (30 random trials, incl. the no-drop-reproduces-stored case); score()
    vs a brute-force argmax + metric.macro_f05 (20 trials); gate_bucket()'s pre-score/gr_pre/gr_v1 vs brute force
    (20 trials, ties and multi-channel nulls included); Data.gather/Data.predict index alignment across parquet
    part boundaries incl. an empty part (synthetic parquet fixtures). finalize(finalize(g,10,k),5,k) ==
    finalize(g,5,k) checked on a synthetic grid (justifies building the pool once at max(sweep_m) and re-cutting
    per m instead of rescanning). None of this touches real data; the D0-8 numbers, blocking_v2.md sweep, and
    cv_full OOF numbers do not exist until the user runs them.

2026-09-25 | M5-1b/M5-2 memory-guard pass: sibling channel vectorized (rapidfuzz process.cdist), sibling per-pair features (sib_hit/sib_anchor_margin/n_sib_anchors/sib_name_tset) now emitted for ALL v2 pairs not just added ones, gate.py --apply union+dry-estimate+v1-backup-assert+row-set-assert wired (behind gate.sibling.enabled, default false), gate.n 45->60, train.fit() gained an optional weight param (default None, no behavior change for existing callers), cv_full.py gained in_v1 flag / weighted_sample_rows (Bernoulli(ext_neg_rate) on extension-only negatives, composed weight 1/(p_keep_existing*p_keep_ext), valid set unsampled+unweighted) / check_weighted_sampling synthetic 1% test / --preflight (1 fold, 5% S1, RSS+time extrapolation). Nothing run -- files only.

2026-09-26 | Pre-run audit of the M5-1b -> M5-2 command chain. Fixed sibling.py `.get(1)` crash, gate.apply all_features call, features.py two assert bugs (see decisions_mistakes). Made candidates_{train,test}_v1.parquet (cp of current candidates; train 69,043,101 rows / 2,206,821 S1 = 31.29 cand/S1, test 35.15). Checked: pool_{train,test}_m10, blockgrid_{train,test}, predict --folds <-> cv_full.json keys/fold file names. Nothing else run.

2026-09-26 | src/mine_dict.py written (standalone probe, no pipeline change): native-India transliteration map + abbreviation map (first letter + subsequence), keep co>=20 & share>=0.8, mined on 80% S1s, KEY = % of held-out 20% v1 misses (vocab proxy ∪ India-native) sharing a surviving token after the map. D0-1's miss list is not on disk, so misses = true pairs not in candidates_train_v1 and vocab = no shared A/B token with df<=1000 (proxy count printed vs 106,630). Helpers checked on synthetic data only (caught a .drop("k") bug); not run on real data.

2026-09-26 | src/eyeball.py fixes after the first real run (killed after ~1h stuck in pred_contrib, no output). pred_contrib moved from all 13,806,329 OOF rows to section 3's 60 sampled rows (keyed by _oof; stream-filtered feature read); per-step timing prints added. Pre-run schema check found 3 more bugs that would have broken the report: country value is "US" not "United States" (US half of every bucket + all of section 5 empty), `how="outer_coalesce"` (removed in polars 1.44 -> section 6 crash), section 6 select mixed a list.explode with scalar columns (length error) and tagged vocab from S1-side tokens only -- now calls mine_dict.shared (tokens in both sides, df<=cap) directly. Also: balanced_sample tops up either side without duplicates; section 2 GT lookup is one reverse dict. Checked: balanced_sample + shared on toy frames; top3_contrib_for on 5 real FN rows (2.4s, order-independent). Full report not yet re-run.

2026-09-26 | ab_test.py scorer fix (not run). Reported macro_f05 ~0.236 vs OOF 0.965. Cause: Data.predict indexes boosters by fold, so the single fold-0 model only scores fold-0 rows, but score() was scoped to ~dropE over all 5 folds -> ~80% of scored S1s had no p (non-singletons 0, singletons 1): 0.2*0.965 + 0.8*~0.056 ~= 0.238. Fix: scope = fold 0 & ~dropE; prints n scored / n with >=1 candidate / singleton share; SystemExit if baseline outside [0.94, 0.98]. Noise floor: baseline refit at LightGBM seeds 43, 44 on the same rows, std (ddof=1) over 3 -> keep bar max(0.002, 2*std); pairwise unions gated on that bar. Worlds + X/Xv/X_eval built once (all features) and column-sliced per variant (was: 2 worlds + full-part predict per variant). Per-variant wall clock in the table. train.fit gains optional seed= (None = config seed, existing callers unchanged). py_compile OK only.

2026-09-26 | M5 feature A/B done (fixed scorer, seed std ≈0.00007): +a +0.0013, +b +0.0028, +c +0.0050, +d +0.0015, +b+c +0.0071, +all +0.0097 → all 4 groups kept. Gate n=60 deferred for Submit #1 (~2× rows, OOM risk, ~+0.001). Channel D, domain segmentation, generator-inversion dropped. Next: features test → cv_full --preflight → cv_full → predict --folds → validator → Submit #1.

2026-09-26 | cv_full memory-safe full run (not run; py_compile only). Sampler: all positives + sample_rows' hardest-first negatives (weight 1) + other negatives Bernoulli(easy_neg_rate 0.2 x ext_neg_rate if not in_v1), weight 1/keep-prob (replaces the 3:1 uniform draw; ab_test shares it). train.py: dataset() builds + constructs with training params (free_raw_data), fit_ds() trains on Datasets, fit() = both (callers unchanged); params(seed). cv_full.fold_datasets: gather X -> construct -> del, then Xv the same, so one float32 matrix + bins alive at a time. OOF/test-density predict already part-by-part (unchanged). Preflight v2: --preflight runs --preflight-point per preflight_fracs [0.05, 0.2] in fresh subprocesses (clean peak RSS), linear fit a + b*frac for RSS, time, s/round, train rows -> frac 1 x n_folds. Old x20 single-point extrapolation removed (fixed overhead dominates at 5%).

2026-09-26 | check_weighted_sampling: synthetic training pool 55k -> 500k (same generator/seed, 5k valid), rel_diff averaged over sampling seeds 42/43/44 (full fit once), max weight printed; 1% bar kept on the mean; failure message says bias -> do not run cv_full. py_compile only.

2026-09-26 | M5-3 stage2.py written (import check only, not run). Rows = OOF (b) (p_td not null); features: p, record rank / p − other max / top1−top2 / #S1s, S1 rank / Σp / #>0.5 / #cands / p÷max, coref_name/addr (# other records of the S1 with p>0.8, top 5, token_set ≥ 90 to this record), + top-10 stage-1 features by fold gain excluding REC_COLS (stored values are the no-drop world, not (b)). Same S1 folds, inner 10% early stopping, weighted_sample_rows with hardness = stage-1 p. Variants full + rank_only (drops p, rec_dmax, rec_gap, s1_psum, s1_n05). Asserts stage-1 (b) rescored = cv_full.json. --predict refuses unless gain ≥ 0.002; reports p-shift OOF (b) vs test. Deviation from spec: user's coref counts replace the spec's p·sim sibling max / best-sibling-address.

2026-09-26 | stage2 --predict: before writing matching_results.tsv, backup_stage1() copies matching_results.tsv and candidate_pairs.tsv to output/*_stage1.tsv once (an existing backup is never replaced; asserts the source exists). Rule: never overwrite a submission file without a backup. Import check only.

2026-09-26 | block_autopsy.py written (py_compile only, not run). 10% train S1s (hash seed 42 % 10). Q1: missed true pairs vs v1 (candidates_train_v1.parquet) with per-channel rrank/srank buckets from blockgrid_train, record has zero v1 rows, addr_tset (token_set_ratio of joined addr_tokens) ≥ 90, exact core-name equality, 15 examples. Q2 from code: v1 = X rrank ≤ 5 OR X srank ≤ 10 (gate.py:35) = union of both directions, not AND; script also reports the share of v1 rows matching X-only vs any-channel rule. Q3: PC / gate.ceiling / cand/S1 for (a) v1, (b) ∪ any-channel rrank ≤ m (1,2,3,5), (c) ∪ exact name key within country (keys with > 20 train S1s skipped), (d) ∪ B rrank ≤ 3 with addr_tset ≥ 90; (c)/(d) reported for every m. Names/addrs are joined normalised tokens, not features.py strings.

2026-09-26 | M5-2 RUN + Submit #1 (numbers from oof/cv_full.json, oof/predict_timing_folds.json). cv_full: 5 folds on 100% train S1s (2,206,821 S1, 69.0M rows on v1 candidates, gate n=60 deferred), ~16.6M train rows/fold, best_iter 3530-4136, valid logloss ~0.0075. OOF (a) standard 0.97515 @ t=0.73 (India 0.9660, US 0.9813); OOF (b) test-density 0.97459 @ t=0.75 (India 0.9653, US 0.9808, 28,249 FP pairs). vs M4 20% subset OOF 0.9654: +0.0092. predict --folds: 60.9M test candidates, 5-fold mean p, t=0.75 -> 5,688,845 matches, 3.28/S1, 5.96% empty (M4: 5.95%); 10,222 s, peak 7.6GB. Backups: output/{matching_results,candidate_pairs}_sub1.tsv. **Public LB 0.967** (M4 0.957, +0.010). OOF(b)->LB gap -0.0076, the same as M4's -0.008: the S1-dropout world did NOT close it, so the gap is not distractor density alone (candidates: India share 47% vs 40%, France unseen).

2026-09-26 | block_autopsy RUN (docs/block_autopsy.md, 10% train S1s, 764,165 true pairs, peak 9.0GB). v1 misses 28,260 (3.70%); 61.4% of misses are absent from the whole grid (no channel, no direction), and only 5.7% sit at any-channel record rank <= 5. Unions without the GBM: best (c)/(d) m=5 -> ceiling 0.9881 (+0.0010) for +28 cand/S1; m=1 -> +0.0005 for +3.7/S1. 63.6% of misses have equal core name OR addr_tset >= 90, yet the lexical channels do not rank them -> the next recall source is a new signal (encoder E0 / cross-script), not a deeper lexical grid.

2026-09-26 | adv_val.py written (py_compile only, not run). Adversarial validation on test candidates: 300k France vs 300k US+India rows (seeded), same feature_cols as the matcher (asserted train == test), train.fit 300 rounds, GroupKFold(5) by s1_id, target is_france. Reports OOF AUC, top 20 gain features with median/IQR/null per side, AUC with top 5 dropped -> oof/adv_val.json. Two-pass read (row index + flag, then sampled rows) for peak RSS <= 6GB (asserted). Country is the target only. Doc: docs/adv_val.md.

2026-09-26 | test_diag.py written + RUN (docs/test_diag.md, 6 s, peak RSS 4,075 MB after switching aggregations to the polars streaming engine; the first run peaked at 5,107 MB). Test mix: India 46.8% / US 38.3% / France 15.0% of 1,732,544 S1 (train (b): US 60.0% / India 40.0%). Pred/S1 test India 3.22, US 3.33, France 3.37 vs train (b) US 3.30, India 3.21. Empty %: test India 6.26, US 5.83, France 5.33 vs train 5.85 / 6.35. Cand/S1: test ~33-36 vs train (b) ~31. Best p in [0.3, 0.75): test India 1.05%, US 0.51%, France 0.94% vs train US 0.49%, India 0.67%. Records whose best p > 0.75: test 55.7 / 58.0 / 61.1% vs train 57.7 / 56.1%. The test India mid-band share is 1.6x the train India share; France looks like the train countries. The France mid-band examples are mostly sibling businesses at the same or next address (different descriptor word, house number off by 1-4).

2026-09-26 | block_autopsy3.py written (toy brute-force check of ranks/top_m only, not run on data). Same sample + missed pairs as block_autopsy (expect 28,260). Part 0 (v1 from code) in docs/block_autopsy3.md: A/B/C/X all df ≤ 1000, no query truncation except the 2-rarest fallback (A/C/X, not B), no min score, IDF = ln(N/df) over S1+S2+S3 of the country per split, S1 idf-weighted x record binary. Part 1: each missed record vs ALL train S1s of its country, cosine TF-IDF (vocab/IDF fit on that country's S1s), V0 addr words, V1 name char3 no spaces, V2 addr char3 + word bigrams, V3 name+addr char3, V4 = V3 + token_dict, V5 = min rank V1/V2/V4; recall@1/3/5/10/20 by country, (has_address, name_nonascii_raw), and on mine_dict's 20% holdout S1s (the dict was mined on the other 80%, so V4 is leak-free only there). Part 2: top-m (3/5/10) on 200k seeded records: new pairs/rec vs v1, est rows + cand/S1, gate.ceiling from Part-1 ranks, minutes/split with sec/query ∝ n_S1 (covers France). Part 3: best v1 X_score per record, percentile per country, share of V5 gain kept when querying the bottom q%. Chunked S1 csr @ dense query block, 6 threads, 256MB/thread, MemoryError past 10GB RSS.

2026-09-26 | adv_val.py RUN (600k rows, 300k France/300k US+India, peak RSS 2.20GB). AUC 0.99997 (folds 0.99997-0.99998), 0.99969 without top 5 (num_code, num_rel_absdiff, s1_nonascii, unmatched_max_idf_a, unmatched_max_idf_b) -> broad shift, not a few features. Matcher gain (mean fold_0..4.txt) joined onto adv-val top 20: 6 features are top-20 in BOTH lists -- num_code, num_rel_absdiff, unmatched_max_idf_b, addr_tset_gaprec, addr_tset, X_score_gaprec (the matcher's #1 feature, 56% gain). France num_code median 3 vs 1 for US+India -- likely postal/building-code convention difference, feeding num_rel_* too. High-drift/zero-matcher-gain features (s1_nonascii, marker_code, rec_nonascii, B_score*, X_score_ds1) are not concerning. docs/adv_val.md Results filled.

2026-09-26 | eyeball2.py written (py_compile only, not run). OOF (b) = p_td non-null; decision = cv_full.score rule (per-record argmax, ties lowest s1k, t=0.75); FP count asserted == cv_full.json fp_pairs (28,249). Lists: top-60 FPs by p (p > 0.9) and top-60 FNs (true in-world pair, p < 0.5, lowest p), each with raw S1 | record | true S1 (DISTRACTOR / [S1 dropped in (b)]) | best competing in-world S1 + p. Stats on ALL FPs vs TPs (+ by record country): sibling = highest-p other candidate S1 with |A∩A'|/max(|A|,|A'|) >= 0.8 on unique addr_tokens vs the predicted S1; name_in_sib = >= 1 record name token absent from the predicted S1's name is in the sibling's name; num_closer = first digit run of addr_number, |rec-sib| < |rec-pred|. 8 reck-hash partitions, RSS guard 6GB -> docs/eyeball2.md, artifacts/logs/eyeball2_timing.json.

2026-09-26 | loss_breakdown.py written (toy check of shapley/siblings/NONLATIN only, not run on data). World = cv_full (b): drop_mask(seed+2000) S1s, p_td, cv_full.score rule (argmax per record, ties lowest code, t from cv_full.json). Asserts: n_S1 = cv_full n_s1, oof_full rows = candidates_train_v1 rows, GT pairs in v1 = oof positives, macro F0.5 reproduces cv_full.json to 1e-9. Per-S1 loss 1 - F0.5 split exactly by Shapley over 3 fixes (blocking fn_b -> tp, matcher fn_m -> tp, fp -> 0); matcher-FN includes lost-argmax pairs, counted separately in a pair table. Segments (S1-side): country, singleton, cardinality 0/1/2-3/4-6/7+, has_address, non-Latin name (letter outside \p{Latin}), sibling density 0/1/2+ (other train S1s of the country with 5|A∩B| >= 4 max(|A|,|B|) on unique addr_tokens, prefix-filtered set-similarity join on distinct sets, chunked at 10M pairs), name duplicated in country. Top 15 bucket x segment cells. RSS guard 6GB -> docs/loss_breakdown.md.

2026-09-26 | loss_breakdown.py fix: first run crashed at the per-S1 counts (shape 1,787,525 vs 2,206,821). The world S1s kept s1_table's full-train codes, so bincount(minlength=n) returned 2.2M-length arrays. Fixed by re-indexing codes after the drop filter (order kept, so argmax ties are unchanged). The candidate checks had already passed: 69,043,101 rows, 7,356,833 true pairs in v1 = oof positives.

2026-09-26 | selftrain_loco.py written (not run). M6 self-training probe: India stands in for France (LOCO proxy with real labels to grade against). 20% S1 sample (s1_table(0.2), src.train's permutation), fold 0 (split_fold), fast LightGBM (lr 0.1, <=1500 rounds, early stopping unchanged, custom Dataset/params override in fast_fit rather than touching config.yaml). Round 1: US-only fold-0 training rows -> score India at test-density (cv_full.score/world, same seed+2000 eval world as cv_full/ab_test) -> expect ~0.86 ballpark (between M4-era pooled OOF 0.9654 and M4-era India LOCO 0.8221; different sample/params so not a like-for-like target). Pseudo-label round 1's p on India's eval-scope candidates: positive = p>=0.98 AND record (reck) argmax AND no other candidate of the SAME S1 (s1k) has name_tset>=95 AND addr_tset>=90 (ambiguity guard against a second near-duplicate S1); negative = p<=0.02 sampled 3:1 (MC.neg_ratio, seeded). Graded (precision/recall) against d.y (true India labels) -- diagnostic only, never fed into which pseudo-labels get used. Round 2: retrain US rows (full weight) + pseudo-India (weight 0.5) -> score India; gate = adopt for France if round1->round2 delta >= +0.02. Round 3: relabel India with round-2 model (same rule), retrain, rescore, report delta only (no second gate). Writes oof/selftrain_loco.json + docs/selftrain_loco.md. RSS not guarded by a Log subclass here (20% sample, fast rounds; cv_full's own guard already proved the full-data world fits under max_rss_mb, so a 20% slice is expected well inside 10GB, but this has not been run to confirm). Command to run: `python -m src.selftrain_loco`.

2026-09-26 | block_r.py written (code only, not run) + gate.py channel R OR-branch (also code only). block_r productionises block_autopsy3's V4: per-country joint name+addr char-3gram TF-IDF (token_dict applied both sides, reusing block_autopsy3.strings/vectorize/products unchanged), fitted on that country's S1s, queried only for records with best v1 X_score in the bottom blocking.r_query_pct% (default 12) of their country or zero v1 score, top-3 S1s kept (top_m_scored: block_autopsy3.top_m's logic + the score column it doesn't return). THREADS/MAX_RSS_MB patched on the imported block_autopsy3 module (os.cpu_count(), 14GB) since products() resolves them in its own module globals. Output artifacts/interim/blockr_{split}.parquet (s1_id, rec_id, R_score, R_rrank). gate.py: grid() now outer-joins blockr_{split} in (falls back to null R columns if absent); new finalize_r()/r_hit() union R into the row filter and n_channels_hit alongside block.finalize()'s A/B/C/X cut (R has no S1-centric direction, so R_srank never exists); rec_best()/gate_bucket() extended to include an R_rbest norm term; in_pool() ORs r_hit() in; apply() carries R_score/R_rrank into candidates_{split}.parquet (and the sibling null-filler). sibling.py's finalize() call switched to gate.finalize_r() (same bug would've hit it once gate.sibling.enabled flips). New config.yaml keys: blocking.r_query_pct (12), gate.ceiling (0.994). gate.py --eval now asserts the shipped (gate.m, gate.n, gate.variant) row's F0.5 ceiling >= gate.ceiling.

2026-09-26 | World context fix (py_compile + toy checks only, not run on data). Feature classification: PAIR (two records, given fixed IDF/dicts), S1_SIDE (n_cand_s1, *_ds1, *_rks1, *_srank: invariant under whole-S1 removal), CONTEXT = cv_full.REC_COLS (n_cand_rec, *_drec/_rkrec/_gaprec, *_rrank, + CTX_COLS rec_name_hits95, is_noaddr_ambiguous, twin_hits, s1_name_dup, rec_name_novel, rec_name_all_novel). New cv_full.context(split, dead, out, rows): recomputes all of them over ALL rows of the split minus dead S1s (was: only the caller's meta rows, and CTX_COLS never). is_noaddr_ambiguous = stored flag & new hits95 >= 2 (exact: removal only lowers hits95; rec_no_addr isn't stored). rec_novelty = features.rec_name_novel vectorised per record over alive S1 vocab (toy-checked equal, incl. duplicate tokens, empty names). dead_s1k(seed) = drop over ALL train S1s, so worlds have test density at any sample fraction (identical to drop_mask at frac 1). Data: meta built part by part without REL/RR columns (lower RSS), asserts every feature is classified, wcols = columns taken from the context dir. --check [--split test] runs context() alone. features.relative gains s1_side=False (column order unchanged for features.py). world() kept as a compat wrapper for ab_test/invariance/selftrain_loco. cv_full.ab() switched to context(dead_s1k). src/world_skew.py: 2x2 stale/recomputed context (train x eval) on OOF (b), 20% training sample, fast params, every alive S1 scored; FP by distractor/orphan/live record; RSS guard 10GB.

2026-09-26 | selftrain_loco.py fixes after run 2 (round 1 India 0.8601 @ t=0.65, then crash). (1) IndexError in Data.gather: pseudo_rows = concat(pos, neg) was unsorted (neg = rng.choice) but gather searchsorts per part and needs sorted positions -> pseudo rows now index the in-memory Xe_india (aligned to sorted india_rows); pseudo_label returns sorted arrays. (2) pseudo_label re-gathered all 4.52M India rows (~1.4 GB, the 6.4 -> 9.2 GB RSS jump) for two pair columns -> reads name_tset/addr_tset from Xe_india. (3) grade() recall denominator was every positive in the 20% sample (1,471,770, incl. US and dropped S1s) -> now the India eval rows' true labels; run 2's recall 1.28% is invalid, precision 0.9555 pos / 0.9980 neg stands. (4) cv_full changed under the script: world() now drops only the SAMPLE's S1s (~3.8% of all at 20%), not test density -> both worlds switched to context("train", dead_s1k(seed), ..., d.i) + dropped(s1, SEED+2000), the world_skew pattern. Round 1 will move again; runs 1-2 (0.848, 0.860) were in non-test-density worlds and are not the baseline. (5) Local 10 GB RSS guard (cv_full.Log uses 20 GB). Argmax still sees sampled S1s' competitors only (ab_test's limitation), identical across rounds so deltas are paired.

2026-09-26 | Audit of uncommitted cv_full/features/train/predict vs 817979f (read + dry calc + --check; no code changed). Full-run drop: dropped(s1_table(1.0), seed) = 419,296 / 2,206,821 = 0.1900 for seeds 1042-1046 and 2042, India/US 0.190 each, bit-identical to drop_mask at frac 1; the full run uses context(dead_s1k), not world(). `cv_full --check` (train) PASS: 26/26 stored REC_COLS reproduced exactly (69,043,101 rows, 67.5 s, peak 8.1 GB); weighted sampling mean rel_diff 0.00956 < 0.01 (seed 43 alone 0.019: thin margin). Not verified: twin_hits, rec_name_novel, rec_name_all_novel (batch-2 not in the feature parts: train parts 12:07 and test parts 13:17 predate the 21:39 features.py edit; 78 feats on disk, 0 batch-2). Flags: (1) batch-2 half-applied (code only); (2) the unsuffixed --fast full run overwrites models/fold_{k}.txt, oof/oof_full.parquet, oof/cv_full.json = Submit #1 artifacts; (3) ab() resets FC.macro_weight to False, not to its prior value; (4) ab_test/invariance still call world() -> 3.8% of all S1s dropped at 20%. Log: artifacts/logs/cv_full_check_train.log.

2026-09-27 | twin_hits NaN fix: addr_tset NaN (no address) passed fill_null(-1) >= 90 (polars NaN > all), so no-address pairs counted as twins. Now fill_nan(None).fill_null(-1) in features.py pass 2 and cv_full.context(). Stored twin_hits in features/{train,test} are stale until features are regenerated. blocking.r_query_pct 12 -> 10. gate.n is 60 in config (user expected 45; left unchanged pending decision). cv_full OOF scoring already runs per feature part (<=2M rows), keeping only s1k/reck/label/p: no change. Compile check only.

2026-09-27 | ce_probe crash: CrossEncoder.fit (sentence-transformers 6.1) wraps HF Trainer, which needs accelerate>=1.1.0; accelerate was never installed or pinned, and transformers 5.17.0 was only transitive. requirements.txt now pins accelerate==1.15.0 (uv dry-run resolves only that one new package) + transformers==5.17.0. ce_probe.main asserts accelerate is importable before pair building. Not installed or run by me.

2026-09-27 | Dep cross-check: mechanisms.py (lightgbm, numpy, polars, sklearn) and france_deep.py (polars) use no torch/transformers/Trainer; all four scripts (ce_probe, e0, mechanisms, france_deep) import cleanly in .venv, incl. local names (cv_full.drop_mask, loss_breakdown.shapley, normalise.LEX). Only ce_probe needs accelerate.

2026-09-27 | ce_probe deep fix. (1) Crash: CrossEncoder.fit() is broken in sentence-transformers 6.1 (FitMixinLoss calls model(**tokens); v6 BaseModel.forward takes `input`) -> fit_ce now uses CrossEncoderTrainer + BinaryCrossEntropyLoss on a datasets.Dataset (same lr/warmup/linear schedule, seed+data_seed, pin_memory off for MPS); model_kwargs torch_dtype -> dtype. API verified on a 64-pair toy run (train, predict with Sigmoid, save). (2) Silent bug, worse than the crash: build_pairs joined oof.s1k (id_key int, e.g. 1755362802) to s1_table.code (row index) -> only S1s with small numeric ids matched and every pair carried the WRONG S1 text (smoke gave 847/397 pairs instead of 1000/2000). Now works in (s1k, reck) key space and maps back via key_to_id (inverse of id_key); no more feature-part scan. oof.fold == s1_table fold for all 2,206,821 S1s. (3) make_texts joins now maintain_order="left" (labels and scores are aligned by position). Verified build_pairs(smoke): 1000 train (200/800), 2000 eval, labels 100% agree with GT, 0 fold-0 S1s in train. e0.py checked: inference only, encode args valid in v6. Full smoke not run by me.

2026-09-27 | ce_probe auc_report IndexError: root cause 326/2000 smoke eval rows with p_td null (oof_full has 13,127,650 null-p_td rows stage-1 never scored); true0 pulled them in, polars bool mask with nulls -> numpy object array. build_pairs now restricts eval to p_td-scored fold-0 rows + asserts no null p_td. Verified post-training path on smoke data with dummy ce: auc_report, stack (0 NaN), macro_f05_sweep all run (random ce AUC 0.49 as expected).

2026-09-27 | Batch run (mechanisms, france_deep, ce_probe). (1) mechanisms + france_deep stopped on the 10 GB RSS guard (peaks 10.19 / 10.35 GB); machine has 24 GB -> MAX_RSS_MB 18000 in both (france_deep now a named constant; mechanisms message no longer claims "6 GB"/"Nothing written" -- docs/mechanisms.md WAS written at 07:47, guard fires after the report). (2) ce_probe full run died silently at fine-tune start: crash report Python-2026-09-27-074759.ips = SIGSEGV in libomp __kmp_fork_barrier; 3 libomp copies loaded (homebrew via LightGBM, torch, sklearn; cv_full/train import lightgbm at module level). Fix: OMP_NUM_THREADS=1 set at top of ce_probe before any import (verified torch.get_num_threads()==1). Smoke had completed (docs/ce_probe.md 07:41 is the smoke report). (3) The runner's [ -f docs/X.md ] skip checks look in code/business_entity_resolution/docs, but scripts write to repo-root docs -> never skip; use ../../docs.

2026-09-27 | New src/structure.py (raw-data structure probe, 6 questions: ID tags, cardinality, pseudo-brands, distractor transforms, domains, within-source dups). Only dataset/train TSVs via io.load_source/load_gt_pairs + normalise legal lexicons/DIGIT_MAP; no pipeline artifacts. Nearest-S1 search = hashed char_wb 3-gram TF-IDF (2^20, grams in >0.2% of the pool idf 0), stored gram×S1 so query cost ~ sum of df; addr and name+addr indexes built one at a time for the 4 GB RSS cap; queries capped at 20k/group (1k smoke). Compile OK + synthetic in-memory run of all six sections OK; not run on real data by me.

2026-09-27 | france_deep crashed at section A: DataFrame.to_markdown needs optional pkg tabulate (not in venv/requirements). Replaced the 3 to_pandas().to_markdown calls with a local polars md_table(df, fmt) helper (no new dep). Helper checked on a toy frame; full script not re-run by me. structure/ce_probe in that batch were Ctrl-C by user, not failures.

2026-09-27 | Full batch: mechanisms (peak 9.8 GB), france_deep (10.5 GB), ce_probe (6.6 GB; AUC ce 0.984 vs p_td 0.996, band 0.816 vs 0.723; macro F0.5 p_td 0.9764 -> p_new 0.9794) all wrote reports. structure hit its own 4 GB guard after loading S3 (4339 MB), no report. Raised structure MAX_RSS_MB 4096 -> 18000 to match the other probes (24 GB machine); train S1+S2+S3 load alone exceeds 4 GB.
2026-09-27 | src/r_recovery.py: channel R recovers v1's 45,703 missed true pairs (of 1,238,302 in a 20% sample); exact-key channel K also probed here (killed, see decisions_mistakes.md). Report: docs/r_recovery.md.
2026-09-27 | gate variant v1r shipped (v1, then r_hit, then pre-score): fixes R-only rows starved by v1-first ranking, n=45 ceiling 0.9885 (v1+R) -> 0.9946 (v1r). config.yaml gate.variant v1 -> v1r, gate.n 60 -> 45, gate.ceiling 0.994 -> 0.993.
2026-09-27 | src/lb_probe.py: LB split via France-blank probe (LB 0.832) — France F≈0.943, US+India ≈0.969. No docs/lb_probe.md report generated yet.
2026-09-27 | src/floor_tests.py: S1-side tie-breakers + empty-address ties on test, metric definition sanity (world/populations from src.tiebreak). Report: docs/floor_tests.md.
2026-09-27 | src/tiebreak.py: tie-breaking core-name ties with copy signal (populations A=addr_empty, B=core_identical). Report: docs/tiebreak.md.
2026-09-27 | src/addr_empty.py: grammar/ambiguity + lexical rule for the top loss tag (empty record address). Report: docs/addr_empty.md.
2026-09-27 | src/normalise.py: rules token_map (France name map: cb/clb/cub/culb/cbu->club, center->centre, farmacie->pharmacie, fs->fils, maisondesant->maison de sante) and num_zeros (2-4 digit zero-padded addr_tokens, all countries; addr_number already stripped); France street map + crs/q/pass/avenue typos/alee; llp=elelpi in US/FR lexicons. FR legal forms NOT added to US/India (sa/sas/sci/ei are initials there). --split flag; before/after assert vs existing norm parquet (smoke 50k/file: all 6 identical on untouched rows, 3-8% rows touched); norm_timing.json now merged by file.
2026-09-27 | src/normalise.py: ets->etablissements accepted into France token_map (only 2 France S1 names are bare 'ets'); rejected list now b, t, res. Smoke 50k test: untouched rows identical.
2026-09-27 | src/normalise.py: fixed assert_untouched false failure on full test_s1 (TOUCHED missing 'llp' from the new _LEGAL_FR/_LEGAL_US entry). Full test_s1 now passes: 57,872/1,732,544 rows touched, rest byte-identical.
2026-09-27 | Session closed by user: target score judged out of reach; all work (M5 probes, gate v1r, normaliser France map) pushed as-is. Best confirmed LB stays Submit #1 0.967. Normaliser full train re-run + downstream rebuild NOT done.
2026-09-27 | Final public LB 0.971 (user-reported). Root README.md written for GitHub: problem, pipeline, tech stack, results table, worked/dropped, reproduce commands.
