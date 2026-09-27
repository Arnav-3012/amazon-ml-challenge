# krish-v2 results (overnight 26→27 Sep, 16 GB MacBook Air M4)

Primary metric = OOF (b), test-density world (19% of S1s dropped), 5-fold GroupKFold by S1, 100% of train S1s.
The ship rule is ≥ +0.002 OOF (b) for model changes, ≥ +0.001 for the seed ensemble, and clear diagnostic evidence for gap fixes.
Each candidate changes one thing. Nothing was tuned against the LB.

## Recommendation
**Submit A** (`submissions/A_krishv2_stage1_exp1_submission.zip`). It is the M5-2 stage-1 pipeline retrained locally, plus two
country-agnostic address-ambiguity features (exp1).
- Against Submit #1 (LB 0.967) it changes exactly one thing: the exp1 features.
- In-distribution it is +0.0004 on OOF (b).
- Label-free, it moves France toward the US/India operating point.
- Expected LB: ~0.967–0.970. **0.98 is out of reach tonight.** The largest measured gain (stage 2, +0.0011) plus every
  other lever sums to well under the +0.013 needed.

If you are willing to override the +0.002 bar, B (A + stage 2) is the next candidate: +0.0011 OOF (b), consistent in every
segment. That is your call; by the rule it does not ship.

## Experiments

| # | change | OOF (b) | Δ vs base | India | US | singleton | runtime | peak footprint | shipped |
|---|---|---|---|---|---|---|---|---|---|
| ref | M4 (20% subset) | 0.9654 (plain OOF) | — | 0.9515 | 0.9747 | — | — | — | LB 0.957 |
| ref | Arnav M5-2 (Submit #1) | 0.97459 @ t 0.75 | — | 0.9653 | 0.9808 | — | — | — | LB 0.967 |
| A | local M5-2 retrain + exp1 (`s1_addr_dup`, `rec_addr_hits95`), t 0.73 | **0.97497** | +0.0004 vs Arnav | 0.9659 | 0.9810 | 0.9793 | features 24+20 min, cv_full 4.05 h, predict 84 min | 20.4 GB (swap) | **yes → A** |
| — | decision layer: `ef_empty` (expected-F0.5 empty-set option) | 0.97497 | +0.00000 | | | | 20 min | 19.1 GB | no |
| — | decision layer: `ef_prefix` (E[F0.5] prefix, 5 E[T] scales) | 0.97452–0.97471 | −0.0003…−0.0005 | | | | | | no |
| — | exp2: separate t for address-shared S1s (t2 0.76) | 0.97498 | +0.00001 | | | | | | no |
| B | stage 2 stacked GBM, carry 10 | 0.97595 @ t 0.68 | +0.00098 | 0.9671 | 0.9819 | 0.9875 | 12 min train | 11.3 GB | no (< +0.002) |
| B | stage 2 stacked GBM, carry 20 | 0.97609 @ t 0.69 | +0.00112 | 0.9673 | 0.9820 | 0.9880 | 15 min train, 7 min test | 17.3 GB | no (< +0.002) → B zip |
| C | per-country t (label-free empty-rate target) | n/a (France unscorable) | t=0.98 costs −0.0128 in-dist | | | | 2 min | 3 GB | **no** (CLAUDE.md rule; not recommended) |
| — | Phase 3 encoder feasibility | skipped | | | | | | | torch install fails twice on exFAT `.venv` |
| D | seed ensemble: 2 × 5 folds (CV_SEED_SHIFT=1 alone 0.97499), same folds + (b) world, averaged | 0.97518 @ t 0.75 | +0.0002 | 0.9661 | 0.9813 | 0.9817 | 4.0 h extra cv_full | ~20 GB | no (< +0.001) |

## Phase 1: the OOF→LB gap (label-free, test)
**Arithmetic.** Submit #1's LB 0.967 with US 0.9808 / India 0.9653 at the test mix (38/47/15% US/India/France) implies
France ≈ 0.937 [inference].

**Profile, train vs test.**
- France S1s share an exact normalised address with another S1 **13.4%** of the time. The other countries are at 4.6–6.3% (US/India, train and test).
- M4 predicted France empty only 4.97% of the time, below the 5.58% generator singleton rate, which is identical in US and India. France also got 3.45 predicted matches/S1 vs 3.18–3.31.
- The normaliser already handles French legal forms (SAS/SARL/SA/SCI), accents, `R.`/`RTE` and regions. So the problem is address ambiguity, not vocabulary.

**Fix (exp1).** Two country-agnostic features, learned from US/India's ~5% shared-address S1s:
- `s1_addr_dup`: other S1s in the same country with an identical normalised address; −1 when there is no address.
- `rec_addr_hits95`: the record's candidate S1s with `addr_tset` ≥ 95.

**Effect, label-free (M4 → A).**

| country | empty % | predicted / S1 |
|---|---|---|
| France | 4.97 → 5.29 | 3.45 → 3.38 |
| US | 5.86 → 5.82 | 3.31 → 3.33 |
| India | 6.39 → 6.21 | 3.18 → 3.23 |

France moved toward the in-distribution operating point. The M4 → A change also includes the M5 features and full data, so the move
cannot be attributed to exp1 alone.

**Per-country threshold (C).** Reaching the US/India empty rate (6.01%) needs t_France = 0.98. On US/India, t = 0.98 costs
−0.0128, which says France's scores are confident rather than miscalibrated. Not recommended, and it breaks the no-country-logic rule.

## Candidate files (all `utils/validate_submission.py --check-ids` PASS)
| zip in `../submissions/` | matches | empty S1 | note |
|---|---|---|---|
| `A_krishv2_stage1_exp1_submission.zip` | 5,704,271 | 102,573 (5.92%) | **recommended** |
| `B_stage2_BELOWBAR_submission.zip` | 5,731,243 | 104,676 (6.04%) | below the +0.002 bar; only on your override |
| `C_countryt_NOT_RECOMMENDED_needs_approval_submission.zip` | — | — | per-country t; do not submit without an explicit decision |

`candidate_pairs.tsv` is the v1 set, exactly the scored set: 60,901,445 pairs, checked with zero difference.
`Documentation_template.md` in the zips is the **unfilled** template; fill it before the final upload.

## Engineering notes
- Everything was written to `models_krish/`, `oof_krish/`, `output_krish/` via `io.path()` `BER_PATH_*` overrides. Arnav's `models/`,
  `oof/`, `output/` and the M4 `output/` were never touched. M4 features moved to `artifacts/features_m4/`.
- `predict --cascade 0.02`: fold 0 scores every row, and the 5-model mean is computed only where fold-0 p ≥ 0.02 (~10% of rows).
  - 0 of 540k sampled skipped rows reached t (max exact 0.575 vs t 0.73).
  - Worst case is a few hundred pairs out of 5.7M, well under 0.0001 F0.5.
- `stage2 p_floor 0.01`: stage 2 models the 12% of rows with stage-1 p ≥ 0.01, and loses 2,459 of 5.96M positives. This is what made it fit in 16 GB.
- Seeds: base run = config seed 42 everywhere; `CV_SEED_SHIFT=1` moves only the training-side seeds.
