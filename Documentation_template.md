# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME — to fill]  
**Team Members:** [TEAM MEMBERS — to fill]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A three-stage lexical pipeline: a rule-based normaliser (transliteration, legal-form / honorific / street-type
canonicalisation), an IDF-weighted inverted-index blocker with a composite name × address channel that keeps 96.3%
of true pairs at ~31–35 candidates per S1, and a LightGBM pair classifier over 78 name / address / retrieval /
neighbourhood features. The set decision exploits the data's structure (every S2/S3 record belongs to at most one
S1): each record goes to its highest-scoring S1 if p ≥ t, with t chosen on out-of-fold macro F0.5 under a
simulated test density.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Assignment structure:** 7,638,365 train true pairs; no S2/S3 id is matched to more than one S1 → a
  record-level argmax is a valid constraint. Matches per S1: median 3, p90 6, max 11 (S2 ≤ 5, S3 ≤ 6).
- **Distractors:** 25–27% of S2/S3 records match no S1, in every source × country cell.
- **Country:** 100% of true pairs share the country → country is a safe *partition* key for retrieval, but test
  adds an unseen country (France: 259k S1s), so it is never a model feature.
- **Scripts:** S2/S3 India names are 18–28% non-ASCII (Devanagari, Kannada, Telugu, Tamil, Gujarati, Bengali, …),
  India addresses ~23%; S1 is essentially ASCII. After transliteration (anyascii), median token_set similarity of
  non-ASCII true pairs is 86.7 (p10 60.7): transliteration recovers most but not all.
- **Noise operators (mined from true pairs):** legal-form variants (pvt ltd / private limited / llc …), honorifics,
  abbreviations and acronym dots, alias parentheses, trailing phone numbers / domains / ID tags, digit
  corruption, duplicated tokens, street-type variants (st / street, rd / road), ordinals, admin-region suffixes.
- **Density shift:** train S1s have ~4.7 candidate records each vs ~5.75 on test (more distractors per S1 on test).

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (+ structured set decision)  
**Core Innovation:** (1) a composite blocking channel whose tokens are *pairs* of (name skeleton, address skeleton),
which keeps name-AND-address evidence that single tokens lose to the document-frequency cap; (2) an S1-dropout
"test-density" out-of-fold world: 19% of train S1s are removed and every record-side feature is recomputed, so the
threshold is tuned at the candidate density the test set actually has.

---

## 3. Candidate Generation (Blocking)
- **Normalisation first (`src/normalise.py`):** 15 ordered rules (anyascii → alias split → id tag → acronym dots →
  domain strip → trailing phone → digit fix → dedupe → legal form → honorifics → number prefix → street type →
  ordinal → null token → admin region). Exact core-name agreement on India true pairs: 6.5% → 54.5%; US 13.6% → 61.6%.
- **Blocking keys used (`src/block.py`)**, all within country, score = Σ idf(shared tokens), idf = ln(N_country/df),
  tokens with df > 1000 dropped (rarest-2 fallback for records with none):
  - A: core-name tokens + adjacent bigrams
  - B: address tokens + an exact "#number street-core" key
  - C: phonetic skeletons of name and address tokens
  - X: A ∪ B ∪ C plus composite "name-skeleton | address-skeleton" pair tokens (one ranking over all evidence)
  - Retrieval runs as sparse (records × vocab) @ (vocab × S1) products in memory-bounded chunks.
- **Candidate set:** channel X, record-centric top-5 S1 per record ∪ S1-centric top-10 records per S1 per source.
  All four channels' scores and ranks are kept as features.
- **Candidate pairs generated:** test 60,901,445 (35.2 per S1, `output/candidate_pairs.tsv`); train 69,043,101
  (31.3 per S1).
- **How you ensured true matches were not lost:** pair completeness on train 96.31% (India 94.71%, US 97.39%);
  a perfect matcher on this candidate set would score macro F0.5 0.9873. Both directions (record → S1 and S1 →
  record) are kept, so an S1 with many look-alikes still reaches its records and vice versa. Of the 3.69% missed
  true pairs, 62% rank just outside the cut and 38% share no surviving token at all (mostly India native-script
  names).

---

## 4. Matching Model

**Features used (78, `src/features.py`):**
- Name features: rapidfuzz ratio / token_sort / token_set / partial_ratio, Jaro-Winkler, best alias-variant
  token_set, IDF-weighted Jaccard over name tokens, name bigrams and phonetic skeletons, length difference,
  non-ASCII flags, legal-form agreement code, "(country)" marker code, count of other S1s with an identical core
  name, a dictionary-remapped name token_set (inactive in this build: its dictionary file was not generated).
- Address features: token_set, IDF-weighted Jaccard over address tokens and skeletons, house-number agreement
  code and a number-relation class (equal / zero-pad / prefix / suffix / transposed / off-by-small / different,
  plus absolute and relative difference), street-core equality, has-address code.
- Other: every blocking channel's score and both ranks, number of channels hitting, S2 vs S3, unmatched-token
  counts / max IDF / character similarity of the unmatched tokens, and neighbourhood features: for 6 key scores,
  value minus the best over the record's candidate S1s, minus the best over the S1's candidates, rank within the
  record and within the S1, and margin over the record's second-best S1; candidate counts per record and per S1;
  "no-address record with ≥ 2 near-identical-name S1s".

**Model type:** LightGBM binary classifier (127 leaves, lr 0.05, feature/bagging fraction 0.8, early stopping 100
rounds on an inner 10% S1 hold-out; 3,748–4,606 trees per fold). Five folds, GroupKFold by S1 over 100% of train
S1s; each fold trains in its own S1-dropout world on all positives, the hardest negatives and importance-weighted
easy negatives (~16.6M rows per fold).  
**Threshold selection method:** per record keep only its highest-p S1 (ties → lowest S1 id), then one global t
that maximises macro F0.5 on the test-density out-of-fold predictions (grid search) → t = 0.75. Test is scored
with fold model 0 (see §6).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), out-of-fold on 2.2M train S1s:** 0.9739 at test density (19% S1 dropout; India 0.9638,
  US 0.9807); 0.9746 on the standard OOF. An earlier build of the same pipeline (5-model average) scored 0.967 on
  the public leaderboard (OOF 0.9746), i.e. OOF overstates the leaderboard by ~0.008 (India's larger test share
  and the unseen France explain part of it).
- **Common false positives (wrong merges):** mostly an *extra* record attached to an S1 that already has true
  matches (≈ 92% of false-positive pairs in our diagnostics), typically a same-chain / same-name business at a
  different address or a same-address different business (shared buildings); singleton S1s with a wrong match
  are the minority but cost the most per pair, since precision is weighted double.
- **Common false negatives (missed matches):** (1) blocking misses (3.7% of true pairs): native-script or heavily
  abbreviated names that share no rare token after transliteration, and pairs just below the rank cut; (2)
  records whose best-scoring S1 is a near-duplicate sibling S1, so the argmax sends them elsewhere; (3) true
  pairs with no address on one side and a generic name, where p stays below t.

---

## 6. Conclusion
Careful normalisation, a composite name × address blocking channel and a gradient-boosted matcher with
neighbourhood features reach 0.974 out-of-fold macro F0.5 against a 0.987 perfect-matcher ceiling on our
candidates. The biggest lessons: tune the decision at the density the test set has, never let country become a
feature, and budget compute for the target machine (the full 5-model average needed ~15 h of prediction on our
16 GB machine, so the submission uses one fold model at the same out-of-fold threshold).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `README.md` (environment, data layout, the exact 10-step reproduction with
measured time / memory per step), `requirements.txt` (pinned), `configs/config.yaml` (every seed, threshold grid
and knob), `src/`. Entry points, run from `code/business_entity_resolution/`:
1. `python -m src.normalise`
2. `python -m src.block --split train` / `--split test`, then `--split train --finalize` / `--split test --finalize`
   → `output/candidate_pairs.tsv`
3. `python -m src.features --split train` / `--split test`
4. `python -m src.cv_full` → `models/fold_{0..4}.txt`, `oof/cv_full.json` (OOF scores and threshold)
5. `python -m src.predict --folds --models 1` → `output/matching_results.tsv`

### B. Additional Results

| stage | metric | value |
|---|---|---|
| normalisation | India / US exact core-name agreement on true pairs | 54.5% / 61.6% (from 6.5% / 13.6%) |
| blocking | pair completeness (train) | 96.31% (India 94.71%, US 97.39%) |
| blocking | candidates per S1 (train / test) | 31.3 / 35.2 |
| blocking | perfect-matcher macro F0.5 ceiling | 0.9873 |
| matcher | OOF macro F0.5, standard / test density | 0.9746 @ t=0.74 / 0.9739 @ t=0.75 |
| matcher | per fold best iteration | 3748 / 4485 / 4606 / 3937 / 3980 |
| matcher | per fold held-out log-loss | 0.0078 / 0.0077 / 0.0077 / 0.0079 / 0.0078 |
