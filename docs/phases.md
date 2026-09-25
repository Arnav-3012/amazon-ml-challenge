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
- Headline results: assignment constraint holds exactly (0 violations) -> Hungarian matching
  justified; country consistency 100% -> safe pre-filter; two-channel blocking confirmed (66%
  both-strong, ~15%/15% single-channel-only); naive blocking pool ~44M pairs overturns the
  brute-force-suffices assumption -> country pre-filter first, FAISS if still needed; blocked
  negatives are harder than random (gap +27.3) -> train on blocking negatives.
- Next milestone: M3 (blocking v1: char-TF-IDF name + address channels + cheap keys, PC/RR per
  country), now informed by the country-pre-filter decision.
