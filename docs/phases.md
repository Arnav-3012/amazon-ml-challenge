# Phases

## Phase 0 / M0: Scaffold (2026-09-25)
**Status:** done. The user reports the env verification import passes (after `brew install libomp`).
- Repo layout mirrors the submission zip; docs written; uv venv (Python 3.11) with phase-0 deps pinned.
- Git deferred: the user is creating the GitHub repo first, then init + commit "phase0: scaffold" on their go.
- **Blockers carried forward:**
  1. ~~`student_resource/` missing~~ RESOLVED 2026-09-25: found at `~/Downloads/student_resource`. `dataset` now links there; validator and template copied byte-identical.
  2. ~~LightGBM needs `libomp`~~ RESOLVED 2026-09-25: the user ran `sudo xcodebuild -license accept && brew install libomp` (keg-only is fine; the rpath hits `/opt/homebrew/opt/libomp/lib`).
- **Next:** M1 (loader + scorer + all-empty baseline + validator PASS). Unblocked. Git init is still waiting on the user's GitHub repo.
- **Scale flag for M1/M3/M7:** train S1 ≈ 2.2M, S2 ≈ 5.0M, S3 ≈ 5.3M rows; test S1 ≈ 1.7M, S2 ≈ 4.9M, S3 ≈ 5.1M (`wc -l`, including header). The loader needs memory-conscious dtypes; blocking must be chunked; encoder throughput over ~10M texts per split must be measured before M7.

## M1: Loader + metric + all-empty baseline (2026-09-25) — DONE
**Status:** metric self-test PASS, baseline assert held, validator PASS on all-empty test outputs.
- Train: 2,206,821 S1 entities. Singleton rate = all-empty macro F0.5 = **0.055848** (123,247 singletons).
  Cardinality: 1:119157 2:375212 3:530841 4:484115 5+:574249 — most S1 entities have 3+ matches; singletons
  are a small (~5.6%) but full-credit-or-zero slice. 7,638,365 matched ids total, S2/S3 share ~48/52.
- Row counts: train S1 2,206,821 (US 1,323,633 / India 883,188); train S2 5,034,616; train S3 5,285,603.
  test S1 1,732,544 (India 809,986 / US 663,106 / **France 259,452**, ~15% of test S1); test S2 4,887,273;
  test S3 5,082,316.
- **Implication for M4/M5:** singleton rate of 5.6% means an all-empty submission scores ~0.056 — any real
  model must clear that trivially, but it sets the floor. With 94% of entities having >=1 true match and a
  heavy tail at 5+, recall/blocking budget per entity matters more than singleton precision for the bulk
  of the score; still never skip singleton detection (F0.5 cliff).
- Wrote all-empty `output/matching_results.tsv` and `output/candidate_pairs.tsv` (1,732,544 rows each).
- Validator: PASS, 1,732,544/1,732,544 rows in both files, all empty (as expected for the all-empty baseline).
  ID-existence check was off (default; needs `--check-ids`, moot for an all-empty submission). Not run yet.
- **Next milestone: M2 (EDA).**

## M2: EDA (2026-09-25) — IN PROGRESS
**Status:** `src/eda.py` written and smoke-tested on a small sample; unidecode->anyascii swap done.
Not yet run on real data. Revision 2 of breakdown.md is written after the real docs/eda.md exists.
- Next: user runs `python -m src.eda` from `code/business_entity_resolution/`, shares docs/eda.md
  (or its key numbers), then Revision 2 gets written from those.

## M2: EDA (2026-09-25) — DONE
**Status:** eda.py ran successfully (after two perf fixes: vectorized hot paths, then a
list-rebuilt-in-loop bug). docs/eda.md complete (21 sections, all A-E checks + decisions summary).
Coverage-checked against both the original 12-point spec and the skill's 8-step checklist: nothing
missing. Revision 2 of docs/breakdown.md written from the real numbers.
- Headline results: assignment constraint holds exactly (0 violations) -> per-record argmax
  assignment justified (many-to-one: each S2/S3 -> its best S1 if above threshold; see Revision 2a
  for why Hungarian, a one-to-one tool, was the wrong name for this); country consistency 100% ->
  safe pre-filter; two-channel blocking confirmed (66% both-strong, ~15%/15% single-channel-only);
  naive blocking pool ~44M pairs overturns the brute-force-suffices assumption -> country
  pre-filter first, FAISS if still needed; blocked negatives are harder than random (gap +27.3) ->
  train on blocking negatives.
- Next milestone: M3 (blocking v1: char-TF-IDF name + address channels + cheap keys, PC/RR per
  country), now informed by the country-pre-filter decision.

## Revision 2a correction (2026-09-25) — docs-only, before M3 build
**Status:** done. No code touched.
- Fixed three errors in Revision 2 (full detail: `docs/breakdown.md` Revision 2a,
  `docs/decisions_mistakes.md`): the brute-force ceiling is a search-index constraint (2.2M S1 ×
  ~10M S2+S3), not a candidate-pairs one — 44M pairs score fine in chunks, but search needs
  ANN/inverted-index by design; country pre-filter is ~1.9x (train) / ~2.5x (test) via
  `1/Σshare²`, not "3-4x"; assignment is many-to-one so it's per-record argmax-above-threshold,
  not Hungarian (corrected everywhere "Hungarian" appeared in docs/).
- Added M3-facing findings: noise looks generator-produced (mine + invert operators from train
  positives); 3.4% of S2/S3 have empty address (need `has_address` handling); France address
  matching should weight house number + street over city, and région (S1) vs département (S2/S3)
  are different hierarchy levels, not noisy duplicates of the same field.
- M3 is still next; this correction changes *how* M3's blocking and decision-layer should be
  built, not the milestone order.

## M3a: Noise operators + normaliser v1 (2026-09-25) — CODE WRITTEN, NOT RUN
**Status:** waiting on the user's runs. No blocking yet (M3b).
- Written: `src/noise_ops.py` → `docs/noise_ops.md`; `src/normalise.py` → `artifacts/interim/norm_*.parquet`;
  `src/normalise_eval.py` → `docs/normalise.md`.
- Gate before the cache run: read the lexicon-gap tables in `docs/noise_ops.md`; trim `HONORIFICS` to
  tokens whose add rate clearly beats their S1 base rate; add any unmapped legal-form/state variants.
- Gate after the eval: any ⚠ rule in the ablation gets reverted or narrowed, and logged in
  `decisions_mistakes.md`.
- **Next:** M3b blocking v1 on the normalised caches.
- 2026-09-25 update: noise_ops + normalise + eval ran. 3 rules reverted, 2 fixes, Revision 3 written.
  Remaining for M3a: one confirm rerun (normalise + eval), then commit. Next: M3b blocking.
- 2026-09-25 M3a DONE: final rerun has no ⚠ rules; "W & W Minerals" → "w w minerals" confirmed. Final
  either-key recall US 86.61%, India 60.71% (non-ASCII 47.85%); pooled S1 name collision 50.11%
  (core+legal 40.38%). 6 files normalise in 230s, peak 10.6GB. Next: M3b blocking on artifacts/interim.

## M3b: Blocking v1 + PC report (2026-09-25) — CODE WRITTEN, NOT RUN
**Status:** waiting on the user's runs.
- Written: `src/phonetic.py`, `src/block.py` → `artifacts/interim/{idf,candidates,blockgrid}_*.parquet` +
  `output/candidate_pairs.tsv` (test); `src/block_eval.py` → `docs/blocking.md`.
- Gate before the full run: `--dry` upper-bound product nnz per channel/source (cap 5000 may be far slower
  than 1000: cost ∝ Σ df_S1·df_R); `--smoke` for nnz/s.
- Gate after the eval: the C marginal on India native-script, and the miss-cause table, pick the next lever
  (m/k, cap, token bigrams, char n-grams).
- 2026-09-25 update: bigrams + rarest-2 fallback added to A/C before any real run; `--dry` reports
  zero-survivor % and the product nnz bound incl. both. Trimmed run order: phonetic → dry → smoke 300000 →
  full train cap 1000 → block_eval. Cap sweeps and the test run are deferred.
- 2026-09-25 update: composite name×address tokens → PC 90.9 → 97.2% (all channels) / 96.3% X-only @31/S1.
  Budget fixed at X m5/k10 (F0.5 ceiling 0.9873). Remaining for M3b: refinalize + diag confirm, test run
  (writes output/candidate_pairs.tsv) + validator. Then M4.
- 2026-09-25 M3b DONE: train candidates = X m5/k10, PC 96.31%, perfect-matcher F0.5 ceiling 0.9873, ~69M pairs.
  Test block run + candidate_pairs.tsv happen at M4 inference. Next: M4 GBM matcher (first real submission).
