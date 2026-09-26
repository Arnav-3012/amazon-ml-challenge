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

## M4: Features + LightGBM matcher + first submission (2026-09-25) — CODE WRITTEN, NOT RUN
**Status:** waiting on the user's runs.
- Written: `src/features.py`, `src/train.py`, `src/decide.py`, `src/predict.py`; report → `docs/matcher.md`.
- Features cover ALL candidates of each split (relative features need every competitor); the 20% subset is a row
  filter in train.py.
- Gate after decide: OOF macro F0.5 vs the 0.9873 blocking ceiling; LOCO gap; argmax on/off delta.
- 2026-09-25 update: all M4 runs done. OOF macro F0.5 0.9654 @ t=0.80 (20% subset; blocking ceiling 0.9872);
  LOCO India −0.13, US −0.02. Final fit 50% subset, 3,433 rounds. Test: 5.68M matches, 5.95% S1 empty.
  Remaining: validator + first LB submission. Next: cheap candidate pruner (plan.md, organiser size rule).
- 2026-09-25 M4 DONE: first real submission, public LB 0.957 (OOF 0.9654). Next: cheap candidate pruner (M5a).

## M5-D0: diagnostics (2026-09-25) — DONE
**Status:** all 7 checks run; results in docs/diagnose_m5.md, summary in logs.md.
- Next session: breakdown Revision 3 (D0), update the m5-strategy §5 gates, confirm or kill §2, pick the next build.
- 2026-09-25 update: done. §2 confirmed (breakdown Revision 4), §5 rewritten: M5-1 → M5-2 → M5-3.

## M5-1: blocking v2 (2026-09-25) — CODE WRITTEN, NOT RUN
**Status:** waiting on the user's runs.
- Written: `src/gate.py` (pool + pre-score gate, `--split/--eval/--apply`), `block.py` test now writes a grid,
  `diagnose.py` D0-8. Config `gate:`.
- Gate: pick m/n/variant from docs/blocking_v2.md: test ≤ ~45 cand/S1, F0.5 ceiling > v1 0.9873.

## M5-2: full-data retrain with S1-dropout (2026-09-25) — CODE WRITTEN, NOT RUN
**Status:** queued overnight after M5-1 + features + `cv_full --check/--smoke`.
- Written: `src/cv_full.py`, `predict.py --folds`. Config `cv_full:`.
- Gate: OOF (b) recovers most of the D0-5 −0.0038; curve slope decides data vs features. M5-3 spec in m5-strategy §5.

**M5-1b/M5-2 status:** files written, nothing run. gate.n=60 decided; sibling.enabled=false pending sweep+eval read. cv_full weighted-CV + --preflight written; --check extended with the synthetic weight-math test. Features-stage sib_* passthrough NOT verified (features.py not read this turn -- confirm ID_COLS/drop-list does not exclude sib_* before relying on it).

**2026-09-26 pre-run audit:** 4 bugs fixed (sibling .get(1), gate all_features call, features.py 2 asserts); v1 backups written. Chain ready: sibling sweep/eval -> gate --apply -> features -> cv_full --check/--preflight/full -> predict --folds -> validator. Reaches M5-2 only; 0.988 needs M5-3 (not coded).

**2026-09-26 mine_dict probe:** src/mine_dict.py written, not run. Decides whether a token dict (cheap) can stand in for / precede the E0 encoder channel on vocab misses.

**2026-09-26 eyeball.py:** first run killed (pred_contrib over 13.8M OOF rows). Fixed + 4 latent bugs (country "US", polars outer_coalesce, section 6 explode/vocab definition). Re-run pending: `python -m src.eyeball`.

**2026-09-26 ab_test scorer fix:** scope was all 5 folds with a fold-0-only model (0.236). Now fold-0 scope + sanity assert + 3-seed noise floor + per-variant timing. Re-run pending: `python -m src.ab_test`.

**2026-09-26 A/B close:** all 4 M5-3 feature groups kept (+all +0.0097). Gate n=60 deferred; channel D, segmentation and generator-inversion dropped. Next: Submit #1 chain, then stage-2 (M5-3).

**2026-09-26 cv_full memory pass:** easy-negative sampling at 0.2 (weighted), Dataset constructed then matrix freed, preflight v2 (5% + 20%, linear fit). Run: `cv_full --check` then `cv_full --preflight`.

**2026-09-26 --check hardening:** weighted-sampling check at 500k rows x 3 sampling seeds (mean rel_diff < 1%).

**2026-09-26 M5-3 stage 2 coded:** `src/stage2.py` (+ `stage2:` config). Run after cv_full + predict --folds: `stage2 --smoke`, `stage2`, then `stage2 --predict` only if the gate passes.

**2026-09-26 stage2 backup:** `--predict` backs up the stage-1 output files to `output/*_stage1.tsv` before overwriting.

**2026-09-26 blocking autopsy coded:** `src/block_autopsy.py` → `docs/block_autopsy.md`. Run: `python -m src.block_autopsy`.

**2026-09-26 M5-2 DONE + Submit #1:** cv_full OOF (b) 0.9746 @ t=0.75 (a: 0.9751); predict --folds -> **LB 0.967** (M4 0.957). OOF->LB gap still -0.008. Backups `output/*_sub1.tsv`.

**2026-09-26 block autopsy DONE:** lexical unions buy <= +0.001 ceiling at +28 cand/S1; 61% of v1 misses are in no channel. Next: stage-2 (M5-3) `stage2 --smoke` -> `stage2`; find the OOF->LB gap (India/France mix); encoder retrieval for the recall ceiling.
