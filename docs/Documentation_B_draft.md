# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Prompt & Pray  
**Team Members:** Krish Viramgama, Arnav Chaturvedi, Tanvi Khanvilkar, Muskaan Jaggi  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We generate candidates with within-country inverted-index retrieval over IDF-weighted name, address and phonetic tokens,
plus composite name×address keys: 35 candidates per Source-1 entity on test. A LightGBM matcher scores every candidate
pair with 80 string, IDF, blocking and competitor-relative features. A second, stacked LightGBM re-scores each pair
using its neighbourhood (the record's competing S1s and the S1's other likely records). Each S2/S3 record is assigned to at
most one S1 by argmax plus a single global threshold.
Key ideas:
- **S1-dropout training**, which matches test's distractor density.
- **Address-ambiguity features** for an unseen country where many businesses share one address.
- **Validation at test density** (OOF macro F0.5 0.97609 for this submission).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure.**
  - Train: 2,206,821 S1 (US 60%, India 40%) and 10.3M S2/S3 records. Each S1 has 3.46 true matches on average, 5.58% have none (singletons), and 26% of records match nothing (distractors).
  - The US and India generator constants are identical to within 0.01%.
  - Each S2/S3 record matches at most one S1 (many-to-one).
- **Noise.** Legal-form swaps and drops (Pvt Ltd/LLC/SARL/SAS), aliases ("X fka Y", "M/s", domains), typos and OCR/phonetic
  variants, word swaps, and address component drops/reorders, abbreviated street types and prefixed admin regions.
  About 40% of India and France records contain non-ASCII text: native script or accents. About 3% of records have no address.
- **Shift.**
  - Test has 5.75 S2/S3 records per S1 vs 4.68 in train (+23%). This matches dropping about 19% of S1s while keeping their records as orphans, which roughly doubles the rate of FP-inducing orphan records.
  - Test also contains France (15% of test S1s), which is absent from train.
  - French S1s share an exact normalised address with another S1 13.4% of the time, vs 4.6–6.3% for US/India.
- **Names collide.** 40–54% of S1 core names are shared with another S1 of the same country, so names alone cannot rank
  candidates.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + per-record assignment with a global threshold  
**Core Innovation:**
1. **S1-dropout worlds.** Training folds and the primary validation drop 19% of S1s to reproduce test's distractor density. Every record-relative feature is recomputed in that world.
2. **Address-ambiguity features**, computed identically for every country (country is never a feature).
3. **A stacked stage 2** that reads each pair's neighbourhood: competing S1s and co-referent records.

---

## 3. Candidate Generation (Blocking)
- **Normalisation (pure polars rules, one code path for train and test):**
  - `anyascii` transliteration.
  - Alias split, keeping the right side.
  - Stripping ID tags, domains, trailing phones and honorifics.
  - Legal form moved to its own field.
  - Address rules: house-number parse, street-type canonicalisation, ordinals, null tokens (PO Box), admin-region removal.
  - Country selects lexicons only.
- **Blocking keys (within country; IDF = ln(N/df), tokens with df > 1,000 dropped, rarest-2 fallback):**
  - **A**: core-name tokens + sorted adjacent bigrams + the joined name.
  - **B**: address tokens + a "number street" key.
  - **C**: phonetic consonant skeletons of name and address tokens.
  - **X**: all of the above in one ranking, plus composite "name skeleton | address skeleton" pair tokens.
  - Scores are sparse products chunked by an exact nnz bound; no all-pairs matrix is ever built.
- **Selection:** channel X, record-centric top-5 S1s ∪ S1-centric top-10 records per source.
- **Candidate pairs generated:** 60,901,445 on test (35.15 per S1). Train has 69,043,101 (31.3 per S1).
- **How we protected recall:**
  - The budget was chosen on a perfect-matcher macro-F0.5 ceiling rather than pair recall, because the metric is per S1: 0.9873 at 96.31% pair recall on train.
  - The composite X channel halved candidates at equal recall.
  - An autopsy of the 3.7% of true pairs missed showed that 61% appear in no channel at any rank. Deeper lexical unions added ≤ +0.001 ceiling for +28 candidates/S1, so the set was kept small (smaller candidate sets rank higher).

---

## 4. Matching Model

**Features used (80, stage 1):**
- **Name:** rapidfuzz ratio, token_sort, token_set and partial ratio; Jaro-Winkler; IDF-weighted Jaccard over tokens, bigrams and phonetic skeletons; best alias token_set; length difference; non-ASCII flags; legal-form and "(France)"-marker codes; token_set after a mined transliteration dictionary (807 mappings, e.g. tek→tech, imtrnesnl→international); unmatched-token counts, max IDF and char similarity; `s1_name_dup` (other S1s with the same core name).
- **Address:** token_set; IDF Jaccard over tokens and skeletons; house-number code; street equality; has-address code; number-token relation (equal / zero-pad / truncation / transposition / off-by-small, plus absolute and relative difference).
- **Address-ambiguity (new):**
  - `s1_addr_dup`: other S1s in the same country with an identical normalised address.
  - `rec_addr_hits95`: the record's candidate S1s with address token_set ≥ 95.
  - `is_noaddr_ambiguous`: the record has no address and ≥ 2 near-identical name hits.
- **Blocking and competitor-relative:** per-channel scores and ranks in both directions; channels hit. For X/A/B/C scores and name/address token_set: value − best over the record's other S1s and over the S1's other records, the rank within each, and the margin to the record's runner-up. Candidate counts per record and per S1. Source flag.

**Model type:**
- **Stage 1:** LightGBM binary. Settings: lr 0.05, 127 leaves, min 200 per leaf, feature/bagging fraction 0.8, L2 1.
  - 5-fold GroupKFold by S1 on 100% of train.
  - Each training fold runs in its own 19% S1-dropout world.
  - Early stopping on an inner 10% S1 split, stopping at 3,230–4,260 rounds.
  - All positives are kept, plus hardest-first negatives up to 1.5× the positives; other negatives are kept with probability 0.2 and weighted 1/0.2.
  - Test p is the mean of the 5 fold models. The mean is taken only where fold-0 p ≥ 0.02, which covers about 10% of rows (sampled check: no skipped row reached t).
- **Stage 2:** LightGBM on the stage-1 OOF p of the test-density world, restricted to rows with stage-1 p ≥ 0.01. That keeps 12% of rows and loses 0.04% of positives; other rows keep their stage-1 p.
  - **Record context:** p, rank, p minus the record's best other S1, the record's top-1 minus top-2 gap, and its number of S1s.
  - **S1 context:** rank, Σp, count with p > 0.5, candidate count, and p / max p.
  - **Coreference:** among the S1's top-5 other records with p > 0.8, how many have name (or address) token_set ≥ 90 with this record.
  - **Carried:** the 20 strongest stage-1 features.
  - Same folds and early stopping (1,136–1,466 rounds). Test uses the mean of the 5 stage-2 fold models over the 5-fold stage-1 mean.

**Threshold selection method:**
- Each record goes to its argmax S1 (ties go to the lowest id). The pair is kept if p ≥ t.
- One global t maximises macro F0.5 on OOF at test density, over a grid of 0.02–0.98: t = 0.69 for this submission (A: 0.73).
- No per-country thresholds.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro), 5-fold OOF, test-density world (19% of S1s dropped, 1,787,525 S1s scored):**

| submission | OOF F0.5 | India | US | singletons | FP pairs |
|---|---|---|---|---|---|
| **B: stage 1 + stage 2 (this doc)** | **0.97609** | 0.9673 | 0.9820 | 0.9880 | 25,506 |
| A: stage 1 only (variant) | 0.97497 | 0.9659 | 0.9810 | 0.9793 | 28,310 |

  - Standard OOF (no dropout) for A: 0.97518.
  - Public LB reference: the same stage-1 pipeline without the address-ambiguity features scored 0.967 (OOF 0.97459).
  - France cannot be scored locally. Label-free, the address features moved France's predicted empty rate from 4.97% to 5.29% and its matches/S1 from 3.45 to 3.38, toward the US/India operating point (5.8–6.2%, 3.2–3.3).
- **Common false positives (wrong merges):**
  - 69% fall on distractor records, whose true S1 is absent: the record is pulled to the best remaining look-alike.
  - The highest FP rates are for records with no address and a generic name (8–30% of kept pairs in those cells).
  - Next come same-address/different-name pairs: several businesses in one building.
  - Stage 2 removes about 10% of FPs, mostly singletons (singleton F0.5 0.979 → 0.988).
- **Common false negatives (missed matches):**
  - About half are blocking misses (3.7% of true pairs). They concentrate in US records without an address whose name is shared by many S1s (77% of such misses have ≥ 2 S1s with the true name), and in India transliterations with no shared surviving token.
  - The other half are pairs whose record argmax is correct but whose p is below t.
  - These figures come from diagnostics on the earlier stage-1 model.

**What did not work** (measured, not shipped):
- **Wider/pre-scored blocking.** A gated pool performed worse than v1 at equal size. Rank/name/address unions gave ≤ +0.001 ceiling.
- **Blocking and normaliser ideas:**
  - An exact-address channel.
  - Domain-name segmentation.
  - Generator inversion: 81% of names cannot be reached by rewrite operations.
  - Three normaliser rules reverted by ablation: country marker, "&"→"and", landmark stripping.
- **Decision layer.**
  - Calibrated expected-F0.5 set selection: −0.0003.
  - An empty-set option: ±0.
  - A separate threshold for address-shared S1s: +0.00001.
- **Seed ensemble** (2 × 5 folds): +0.0002.
- **A label-free per-country threshold** would need t = 0.98 for France, which costs −0.013 on US/India. It was not used, and country is never used in decisions.

---

## 6. Conclusion
A lean lexical blocking stage (35 candidates per S1) plus a two-stage LightGBM matcher reaches 0.976 macro F0.5 at test density.
The gains came from:
- validating at test density rather than on the easier train distribution;
- competitor-relative and address-ambiguity features;
- stacking the pair neighbourhood.

The remaining loss is mostly blocking recall on records without an address and on cross-script names. That is the next lever, for example a multilingual encoder retrieval channel.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/` (package, run as modules), `configs/config.yaml` (seed 42, every path and
knob), `requirements.txt` (pinned; LightGBM needs `libomp` on macOS). Data expected at `dataset/{train,test}/` under the
submission root. Run from `code/business_entity_resolution/`:
```
python -m src.normalise                          # normalised caches (train + test)
python -m src.block --split train                # rank grid + IDF (train)
python -m src.block --split test                 # rank grid + IDF (test)
python -m src.gate --apply-v1                    # v1 candidates (X, m=5, k=10) -> output/candidate_pairs.tsv
python -m src.mine_dict                          # transliteration/abbreviation token dictionary (train only)
python -m src.features --split train
python -m src.features --split test
python -m src.cv_full                            # stage 1: 5-fold models + OOF, best t
python -m src.predict --folds --cascade 0.02     # variant A -> output/matching_results.tsv (+ test p)
python -m src.stage2 --only full                 # stage 2: 5-fold models + OOF
python -m src.stage2 --predict --ignore-gate     # submission B -> output/matching_results.tsv
```
Modules:
- `normalise`, `phonetic`: text normalisation.
- `block`, `gate`: candidate generation.
- `features`: pair features.
- `train`, `cv_full`: stage 1.
- `stage2`: stacked stage 2.
- `decide`, `predict`: assignment and output.
- `metric`: exact macro-F0.5 replica.

Validation: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`.

Hardware: 16 GB Apple M4. Memory knobs are in the config (`features.chunk_rows`, `cv_full.max_rss_mb`, `stage2.p_floor`).
Approximate wall time: features 45 min, stage 1 4 h, test scoring 1.5 h, stage 2 30 min.

### B. Additional Results
| step | effect on OOF F0.5 (test density) |
|---|---|
| earlier baseline (20% of train, standard OOF) | 0.9654 |
| + full data, S1-dropout, second feature set | 0.97459 |
| + address-ambiguity features (A) | 0.97497 (+0.0004) |
| + stage 2 (B) | 0.97609 (+0.0011) |

Blocking: pair recall 96.31%, perfect-matcher F0.5 ceiling 0.9873, 31.3 candidates/S1 on train and 35.2 on test.
