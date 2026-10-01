# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Prompt&Pray

**Team Members:** Arnav Chaturvedi , Krish Viramgama , Muskaan Jagg , Tanvi Khanvilkar

**Submission Date:** 1-oct-2026

---

## 1. Executive Summary
Our pipeline has four stages: lexical blocking on inverted indexes, a capped candidate gate, a LightGBM pair
classifier with 93 features, and a many-to-one decision rule (each S2/S3 record goes to at most one S1). Three
choices drove most of the score:
- a composite **name × address** blocking channel, which breaks ties between S1s that share a name;
- a **per-S1 gate** that keeps at most 45 candidates and was tuned against the metric's own ceiling, not pair recall;
- **S1-dropout cross-validation**, which reproduces the test set's density of orphan records.

The final submission scores **0.971 macro F0.5 on the public leaderboard** (0.9806 local out-of-fold at test
density). It uses no pretrained model, no external data and no country feature.

---

## 2. Methodology

### 2.1 Problem Analysis
Key findings from EDA (`src/eda.py`) and later probes, all on train only unless stated:

- **Assignment is many-to-one.** Every S2/S3 id appears under at most one S1 (0 of 7.64M ids are shared), so a
  record can be assigned to its single best S1.
- **Cardinality:** p50 = 3 matched records per S1, p99 = 8, max = 11. Some S1s have no match at all. Those score
  1.0 only when the prediction is empty, so F0.5 punishes any false positive on them heavily.
- **Distractors:** about 26% of S2/S3 records match no S1, in both countries and both sources.
- **Country never differs** between the two sides of a true pair (100% agreement), so we block within each
  country. This needs no country-specific rules and works for France, which appears only in test.
- **Scripts:** S1 is almost all ASCII, but 18–28% of Indian S2/S3 names and addresses are in native scripts. US
  S2/S3 names are about 6.5% non-ASCII. In test, France S1 is 16% non-ASCII in names and 28% in addresses
  (accents).
- **Name noise:** in 65.5% of true pairs the core name is identical after normalisation. The rest are prefixes
  (S1 truncated by 1–3 tokens, with or without an extra word), subsets, supersets, reorderings, abbreviations and
  typos. Legal-form drift (`Pvt Ltd` / `Private Limited`) is common.
- **Shared names:** 38–50% of S1 names occur at more than one S1 in the same country (chains, common names). For
  these, only the address can tell the S1s apart.
- **Density shift:** train has 4.68 S2/S3 records per S1 and test has 5.75. Test therefore contains relatively
  more records whose S1 is absent, and these become false-positive pressure.
- **Unseen country:** test is 46.8% India, 38.3% US and 15.0% France (by S1).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (inverted-index blocking → candidate gate → LightGBM → many-to-one decision)  
**Core Innovation:**
1. A composite `name|address` skeleton channel (X), plus a TF-IDF fallback channel (R) for records the lexical
   channels serve weakly.
2. A per-S1 candidate gate whose budget was chosen with the **perfect-matcher F0.5 ceiling**. This is the macro
   F0.5 that an oracle matcher would score on the candidate set, so the budget is judged by the real metric.
3. **S1-dropout CV.** We drop 19% of train S1s per fold, so local validation sees the test density (4.68 / 0.81
   = 5.78 ≈ 5.75). Record-context features are recomputed inside that world.

Pipeline:
```
normalise → block (A, B, C, X) → block_r (R) → gate (≤45 per S1) → features (93) → cv_full (5-fold LightGBM) → predict/decide
```

---

## 3. Candidate Generation (Blocking)

**Normalisation (`src/normalise.py`).** Text is transliterated to ASCII with `anyascii` and lower-cased. Legal
forms (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, …) are split into their own column, not deleted, so `Acme Inc` and
`Acme LLC` remain distinguishable. We also apply per-country abbreviation lexicons, house-number extraction,
removal of leading zeros from 2–4 digit address numbers, and phonetic/consonant skeletons. Every rule passed a
*precision guard*: it must not create too many new exact collisions between non-matching pairs. Three intuitive
rules failed it and were reverted: `&` → `and`, stripping landmarks, and stripping `(India)`.

- **Blocking keys used:**
  - **A:** name tokens, adjacent bigrams in sorted order, and the space-free joined name.
  - **B:** address tokens.
  - **C:** phonetic/skeleton name tokens and their bigrams.
  - **X (composite):** `name-skeleton | address-skeleton` pair tokens. A pair shares an X token only when it
    agrees on name and address together.
  - Mechanics for A/B/C/X: inverted indexes built per country and per split. IDF is computed over S1+S2+S3 and
    tokens with document frequency above 1000 are dropped. A record with no indexable token keeps its 2 rarest
    tokens. Scoring is a chunked sparse product (S1 IDF-weighted × record binary) and keeps the union of the
    record-centric top-m and the S1-centric top-k.
  - **R:** character 3-gram TF-IDF over name + address, fitted per country on S1. It is queried only for the
    bottom 10% of records by best X score (plus records with no X hit) and keeps the top 3 S1s.
- **Gate (`src/gate.py`).** We pool all channels (record rank ≤ 10 or S1 rank ≤ 30) and rank each S1's pool.
  Variant `v1r` puts the X-selected set first, then R hits, then a pre-score. We keep at most **45 candidates
  per S1**.
- **Candidate pairs generated:** **77,949,788** on test, an average of 45.0 per S1 over 1,732,544 S1s (cap 45).
  On train the same rule gives 99,279,505 pairs for 2,206,821 S1s.
- **How you ensured true matches were not lost:**
  - The budget (m, n, variant) was chosen on train by the *perfect-matcher macro F0.5 ceiling*, the score an
    oracle matcher would reach on the candidate set. We did not optimise pair recall alone, because the metric
    is per entity.
  - The shipped configuration has a ceiling of **0.9946**. Channel X alone at 31 candidates per S1 gave 0.9873.
    Adding R and the `v1r` gate ordering gave +0.007.
  - `gate --eval` asserts that the ceiling is ≥ 0.993 before candidates are written.
  - Probes that did not pay off were dropped: deeper lexical blocking (≤ +0.001 ceiling for +28 candidates per
    S1), an exact-name/address key channel (0.37% precision) and domain-name segmentation.
  - `output/candidate_pairs.tsv` is exactly the set the matcher scores. `predict.py` asserts that the number of
    scored pairs equals the number of candidates and that matches ⊆ candidates.

---

## 4. Matching Model

**Features used (93, `src/features.py`):**
- **Name features:**
  - RapidFuzz `ratio`, `token_sort`, `token_set`, `partial` and Jaro-Winkler.
  - IDF-weighted Jaccard on name tokens, bigrams and skeletons.
  - Alias/abbreviation best match, and `token_set` after a dictionary mined from train pairs.
  - Unmatched-token count, max IDF and character similarity, in each direction.
  - Name coverage in each direction, length difference, legal form on each side and their relation
    (same / compatible / conflicting), and non-ASCII flags.
- **Address features:**
  - Address `token_set` (raw and alias-expanded) and IDF-weighted Jaccard on tokens and skeletons.
  - House-number relation: equal, contained, in range, absolute and relative difference.
  - Street equality, locality relation, address coverage in each direction, length ratio, and has-address flags.
- **Other:**
  - Blocking signals: score, record rank and S1 rank per channel A/B/C/X, plus the number of channels hit.
  - **Record-relative features** for the X/A/B/C scores and the name/address `token_set` scores: the difference
    from the best competing S1 for the same record, rank within the record, gap to the runner-up, and the same
    from the S1's side.
  - Context counts: candidates per record and per S1, near-duplicate names among a record's candidates,
    same-name/same-address "twin" S1s, duplicated S1 name, name novelty, and an ambiguity flag for records with
    no address.
  - A source flag (S2 vs S3).
  - **Country is never a feature**, because France is unseen in train.

**Model type:**
- LightGBM binary classifier: `num_leaves` 127, learning rate 0.1 with at most 1500 rounds (`cv_full --fast`),
  `min_data_in_leaf` 200, feature and bagging fraction 0.8, L2 1.0, `deterministic: true`, seed 42, and early
  stopping (100 rounds) on held-out S1s (best iterations 769–979).
- 5-fold `GroupKFold` by S1, so no S1 appears in both training and validation.
- Training rows: all positives, hard negatives taken hardest-first, and easy and extension-only negatives
  subsampled with inverse-probability weights.
- Each fold is trained in an **S1-dropout world**: 19% of S1s are removed, their records become orphans, and
  record-context features are recomputed without them.
- Test probability = the mean of the 5 fold models.

**Threshold selection method:**
- Decision rule: each S2/S3 record is assigned to its argmax-p S1 (ties go to the lowest id) and kept only if
  p ≥ t. This enforces the many-to-one structure.
- The global t is chosen by maximising macro F0.5 on out-of-fold predictions in the test-density world,
  searching 0.02–0.99 in steps of 0.01. Result: **t = 0.76**.
- No per-country thresholds are used.

---

## 5. Results & Error Analysis

| Submission | Change | Local OOF macro F0.5 | Public LB |
|---|---|---|---|
| M4 | Blocking v1 + LightGBM on 20% of train | 0.9654 | 0.957 |
| Submit #1 | Full-data 5-fold, S1-dropout CV, +4 feature groups | 0.9746 | 0.967 |
| **Final** | Gate v1r (≤ 45 per S1) + channel R + normaliser updates, retrained | **0.9806** | **0.971** |

- **F_0.5 Score (macro):**
  - **0.9806** out-of-fold at test density (t = 0.76; India 0.9757, US 0.9839).
  - 0.9817 in the standard OOF world.
  - Public leaderboard: **0.971**.
  - Local OOF ran about 0.008 above the leaderboard throughout, so we judged changes on paired local deltas.
  - A probe submission with France predictions left empty split the leaderboard by country: France ≈ 0.943,
    US + India ≈ 0.969. Most of the remaining gap comes from the unseen country.
- **Common false positives (wrong merges):**
  - The final model makes 34,016 false-positive pairs at test density, which accounts for about 17% of the
    remaining loss.
  - **Sibling businesses:** the same chain or a near-identical name at the same or an adjacent address
    (house number off by 1–4, different descriptor word).
  - **Records with no address**, whose name matches several S1s.
  - **Orphan records:** records whose true S1 is absent, attached to the nearest look-alike S1. The S1-dropout
    training world exists to make the model hold these back.
  - On the France slice of test, mid-confidence examples are mostly siblings.
- **Common false negatives (missed matches):**
  - In a loss decomposition of the Submit #1 model, **51% of the loss came from blocking misses**, 32% from
    matcher false negatives and 17% from false positives.
  - 61% of blocking misses were absent from every lexical channel. Typical cases are cross-script records where
    transliteration diverges, heavy name rewrites (DBA or brand names in place of legal names, domain-style
    names) and partial or landmark-only addresses. Channel R and the `v1r` gate were built for these and lifted
    the blocking ceiling from 0.9873 to 0.9946.
  - The main matcher false negatives are S1 names truncated by one to three tokens plus an extra word, and true
    pairs that lose the record argmax to a sibling S1.

---

## 6. Conclusion
- A transparent lexical pipeline went from 0.957 to 0.971 on the public leaderboard: composite name × address
  blocking, a candidate gate tuned against the per-entity ceiling, and a LightGBM matcher trained at test density.
- The most reusable lessons are three. Evaluate blocking with the metric's own ceiling rather than pair recall.
  Make local validation match the test's distractor density. Hold every normalisation rule to a precision guard.
- The largest remaining loss is the unseen country (France ≈ 0.943 vs 0.969 elsewhere) and records that no
  lexical signal reaches.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
  README.md            end-to-end reproduction (exact commands)
  requirements.txt     pinned dependencies (Python 3.11)
  configs/config.yaml  every hyper-parameter, path and the seed (42)
  src/
    io.py              TSV loading (sep="\t", dtype=str, keep_default_na=False), paths, writers
    normalise.py       step 1: transliteration, legal forms, lexicons, skeletons
    block.py           step 2: channels A/B/C/X (inverted index + sparse products) + v1 candidates
    mine_dict.py       step 3: token / locality maps mined from 80% of train S1s
    block_r.py         step 4: channel R (char-3gram TF-IDF fallback; retrieval core in tfidf.py)
    gate.py            step 5: per-S1 gate (≤45) → output/candidate_pairs.tsv
    features.py        step 6: 93 pair features (phonetic.py: skeleton keys)
    cv_full.py         step 7: 5-fold S1-dropout LightGBM + OOF threshold
    predict.py         step 8: score test, decide → output/matching_results.tsv
    train.py, decide.py, metric.py, sibling.py   helpers (metric.py = exact local copy of the LB metric)
```

The entry points are run in this order from `code/business_entity_resolution/`: `src.normalise`, `src.block`,
`src.mine_dict`, `src.block_r`, `src.gate` (`--split` for each split, then `--eval` and `--apply`),
`src.features`, `src.cv_full --fast` and `src.predict --folds`. The README gives the exact commands.

### B. Additional Results

**Blocking ceiling by configuration (train, perfect-matcher macro F0.5):**

| Configuration | Candidates per S1 | Ceiling |
|---|---|---|
| Channel X only (m = 5, k = 10) | 31.3 | 0.9873 |
| X + R, X-first ordering, n = 45 | ≤ 45 | 0.9885 |
| **X + R, v1r ordering, n = 45 (shipped)** | **≤ 45** | **0.9946** |

**Compliance notes:**
- No external data, APIs, lookups or geocoding.
- The final pipeline uses **no pretrained model**: LightGBM is trained from scratch.
- Two pretrained models were used only in experiments that did not ship, both ≤ 8B parameters:
  `intfloat/multilingual-e5-small` (MIT), as a retrieval probe, and `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (Apache-2.0), as a reranker probe.
- Country is used only to partition blocking (true pairs never cross countries). It is not a model feature, is
  not one-hot encoded and has no hard-coded per-country decision.
- **Disclosure:** France has no training rows. The France abbreviation map in `normalise.py` (for example
  `clb` → `club`, `ets` → `etablissements`, street-type typos) was mined from **unlabelled test input text**
  only, with no labels and no external sources.
- Runs are deterministic: seed 42 and LightGBM `deterministic: true`.
