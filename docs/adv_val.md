# Adversarial validation: France vs US+India (test candidates)

**Question.** The OOF(b) → LB gap is -0.008 on both submissions and is not explained by distractor density
(logs 2026-09-26). France is in test only. If the matcher's 78 pair features separate France candidates
from US+India candidates, the gap is feature shift and the drift features are the ones to fix or drop.

**Method** (`code/business_entity_resolution/src/adv_val.py`)
- Rows: 300k test candidate pairs whose S1 country is France, 300k from US+India (seeded, `config.yaml` seed).
  Country comes from `norm_test_s1.parquet`; it is the target only, never a feature.
- Features: exactly `train.feature_cols` (asserted equal for train and test).
- Model: `train.fit` (matcher LightGBM params), 300 rounds, no early stopping, GroupKFold(5) by `s1_id`
  (candidates of one S1 share relative features; a row split would inflate AUC).
- Report: OOF AUC + per-fold; top 20 features by summed gain with median / IQR / null share per side;
  AUC again with the top 5 dropped.
- Memory: pass 1 reads only the row index + country flag, pass 2 reads the sampled 600k × 78 float32
  (~180 MB). Script asserts peak RSS ≤ 6 GB.

**Run** (from `code/business_entity_resolution/`): `python -m src.adv_val` → `oof/adv_val.json` + tables on stdout.

**Reading it**
| AUC | Meaning | Action |
|---|---|---|
| ≤ 0.60 | no usable shift | gap is elsewhere (label/threshold/density) |
| 0.60–0.80 | moderate shift | check whether top drift features are also top matcher-gain features |
| > 0.80 | strong shift | if AUC drops a lot without the top 5, the shift is concentrated: fix/normalise those; if it stays high, the shift is broad (script/tokenisation), not a few features |

Some shift is expected and harmless (e.g. address-length features for a different format). It matters only
where the feature is also high-gain in the matcher.

## Results

`oof/adv_val.json`: 600k rows (300k France / 300k US+India), peak RSS 2.20 GB.

**AUC 0.99997** (folds 0.99997–0.99998), **0.99969 without the top 5** dropped
(`num_code`, `num_rel_absdiff`, `s1_nonascii`, `unmatched_max_idf_a`, `unmatched_max_idf_b`).
Per the table above, >0.80 both before and after dropping the top 5 = **broad shift**, not a few bad
features — the France candidate rows are near-perfectly separable on tokenisation/scale grounds
alone, and dropping the top 5 barely moves it (-0.0003).

**Top 20 by adv-val gain, with matcher gain (mean over `models/fold_0..4.txt`) and matcher rank added:**

| # | feature | adv gain % | FR median | FR IQR | US+IN median | US+IN IQR | matcher gain % | matcher rank | in both top-20 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | num_code | 25.1 | 3.0 | 1.0 | 1.0 | 3.0 | 0.60 | 12 | **yes** |
| 2 | num_rel_absdiff | 14.5 | 13.0 | 42.0 | 20.0 | 483.0 | 0.73 | 11 | **yes** |
| 3 | s1_nonascii | 11.9 | 0 | 1 | 0 | 0 | 0.00 | 78 | no |
| 4 | unmatched_max_idf_a | 8.3 | 10.63 | 1.93 | 12.12 | 2.22 | 0.25 | 25 | no |
| 5 | unmatched_max_idf_b | 6.9 | 10.61 | 2.55 | 11.97 | 2.67 | 1.15 | 10 | **yes** |
| 6 | X_score_ds1 | 4.3 | -110.4 | 71.9 | -188.8 | 146.4 | 0.08 | 41 | no |
| 7 | B_score_ds1 | 3.6 | -20.4 | 9.8 | -20.0 | 19.7 | 0.05 | 47 | no |
| 8 | num_rel_reldiff | 3.6 | 0.455 | 0.778 | 0.451 | 0.900 | 0.14 | 35 | no |
| 9 | addr_tset_gaprec | 2.0 | -27.3 | 44.7 | -44.4 | 42.4 | 1.55 | 7 | **yes** |
| 10 | addr_tset | 2.0 | 68.2 | 45.1 | 48.0 | 28.9 | 7.55 | 3 | **yes** |
| 11 | B_score | 1.6 | 18.4 | 11.4 | 13.0 | 12.3 | 0.06 | 45 | no |
| 12 | A_score_ds1 | 1.5 | -23.4 | 24.3 | -26.8 | 25.3 | 0.16 | 33 | no |
| 13 | street_eq | 1.1 | 0 | 1 | 0 | 1 | 0.12 | 36 | no |
| 14 | X_score_gaprec | 1.1 | -66.8 | 99.6 | -104.3 | 166.8 | 56.18 | 1 | **yes** |
| 15 | s1_name_dup | 1.1 | 0 | 6 | 0 | 8 | 0.16 | 32 | no |
| 16 | rec_nonascii | 1.0 | 0 | 1 | 0 | 0 | 0.01 | 64 | no |
| 17 | marker_code | 0.8 | 0 | 0 | 0 | 0 | 0.01 | 69 | no |
| 18 | nskel_idfj | 0.8 | 0.469 | 0.748 | 0.368 | 0.544 | 0.32 | 22 | no |
| 19 | A_score | 0.7 | 19.6 | 12.3 | 20.8 | 14.2 | 0.18 | 28 | no |
| 20 | askel_idfj | 0.7 | 0.209 | 0.521 | 0.113 | 0.196 | 0.33 | 21 | no |

**Flagged (drift × matcher importance — top-20 in both lists):**
`num_code`, `num_rel_absdiff`, `unmatched_max_idf_b`, `addr_tset_gaprec`, `addr_tset`, `X_score_gaprec`.

`X_score_gaprec` is the matcher's #1 feature by gain (56%) and is also adv-val top-20 (rank 14, 1.1%
gain share) — a small drift share on the single most-relied-on feature is worth checking even though
its adv-val rank is low. `addr_tset` (matcher rank 3, 7.55% gain) is the other one that matters at scale.
`num_code` is the single largest drift source (25% of adv-val gain) and is also matcher-relevant
(rank 12); France addresses carry ~3 numeric codes (postal + building + arrondissement-like patterns)
vs 1 for US/India, so `num_code` and `num_rel_absdiff`/`unmatched_max_idf_b` (num_code feeds the
`num_rel_*` family) look like the same underlying cause, not three independent problems.

**Not concerning:** `s1_nonascii` (adv-val rank 3, 11.9% gain) has zero matcher gain (rank 78, unused
by the matcher) — it separates France perfectly (accented characters) but the model never leans on it.
Same story for `marker_code`, `rec_nonascii`, `B_score`/`B_score_ds1`, `X_score_ds1` — high adv-val
gain, negligible-to-zero matcher gain.

**Read:** the France/US+India split is broad (whole-feature-set separable, dropping top 5 barely
moves AUC), but the France-relevant *and* matcher-relevant overlap is narrow — 6 features, dominated
by `num_code` and the `X_score_gaprec`/`addr_tset` pair. Those 6 are the candidates for a
France-specific look (does `num_code` normalisation assume a US/India address-code convention?)
before touching the rest of the 78.
