# Amazon ML Challenge 2026 — Business Entity Resolution: Problem Breakdown

**Problem:** For every clean Source-1 (S1) business, output the full set of noisy S2/S3 record IDs that describe the same real-world business (possibly empty), using only name + address; scored by per-entity macro F0.5.

**Domains:** ML (supervised pairwise clf, tabular features on text) · IR (candidate retrieval / blocking) · decision theory (set selection under F-beta) · DE (reproducible pipeline)

| | |
|---|---|
| **In** | S1 (clean, de-duplicated reference), S2, S3 (noisy vendor records): ID, name, address. Train has labels; test does not. All TSV |
| **Out** | `matching_results.tsv`: one row per S1 ID → comma-separated S2/S3 IDs (empty if none). Final archive: + `candidate_pairs.tsv`, runnable pipeline, methodology doc |
| **Success** | Macro F0.5 over S1 entities (precision weighted 2x recall). Singleton: empty pred = 1, anything else = 0 |
| **Givens** | No shared key; name+address only; multi-region data; no external DBs/APIs/lookups; blocking audited; top packages reproduced |

**Hidden requirements:**
- Metric is per-entity then averaged → decisions are per *set*, not per *pair*. Pair-level F1 threshold tuning optimises the wrong thing.
- Empty-vs-nonempty is a 0/1 cliff per entity → singleton detection is a first-class problem.
- Candidate file is audited → blocking must be a genuine reduction (not all-pairs) with high recall; must be deterministic and reproducible.
- Methodology doc + runnable pipeline → every step scripted, seeded, no manual edits.
- "Region-specific patterns" → normalisation cannot be one global English ruleset (abbreviations, legal suffixes, landmark phrasing, postcode formats, transliteration).

**Assumptions:**
- S1 de-duplicated ⇒ each S2/S3 record belongs to ≤1 S1 entity (assignment constraint). Verify on train labels before relying on it.
- S2/S3 records that match no S1 entity exist (distractors) — expected from "lookalikes" description.
- Train/test are disjoint universes of businesses → model must generalise from similarity features, not memorise names.

**Open questions (check rules/data first):**
- Are pretrained models (sentence-transformers, multilingual encoders, local LLMs) allowed, or does "no external" cover them too? Changes L3 heavy option.
- Data scale (rows per source) → sets blocking compute budget and whether cross-encoders are feasible.
- Singleton rate and match-cardinality distribution in train (mostly 0/1 matches or many?).
- Does any S2/S3 ID appear under two S1 entities in train labels? (tests assignment assumption)
- Do S2↔S3 records also co-refer (useful transitive signal)?
- Which regions/countries and are there region/country/postcode fields hidden inside address strings?
- Submission count limits → how much you can probe the leaderboard vs trust local CV.

---

## Big picture
Not "a classifier" — it's record linkage against a master list: retrieve → score pairs → choose a set per entity. Key insight: the scorer rewards per-entity set quality with a hard singleton cliff and a precision bias, so the final decision layer (what set, including empty, to emit per S1 entity) matters as much as the model. Hard parts: noisy multi-region text, recall ceiling from blocking, metric mismatch between pair training and set scoring.

**Cut:** pipeline cut (normalise → block → match → decide) + prediction-vs-decision cut (pair probability ≠ output set) + constraint cut (validation/reproducibility as own leaves because audits and leakage can sink the entry).

---

## Leaves

### 1. Normalisation & representation
- **Type:** Data prep › text canonicalisation › region-aware string normalisation (rules + data-driven token weighting)
- **Why:** only two noisy free-text fields; same business written as "Acme Robotics Incorporated" vs "ACME Robotics Inc", abbreviated addresses, landmark references
- **Why not:** ML-learned normaliser (seq2seq) → no aligned clean/noisy pairs at token level, overkill · raw strings → similarity scores dominated by boilerplate
- **Approaches:** lowercase/strip punct/unicode fold (always) · abbreviation + legal-suffix maps per region, split address into number/street/locality/postcode/landmark (standard) · IDF/frequency-based token weighting learned from all sources, transliteration folding (heavy, if multilingual)
- **Pick:** standard + IDF weights; mine abbreviation pairs from train positive matches rather than hand-writing everything
- **Concept:** rare tokens are identity-bearing ("Acme"), common ones are noise ("Pvt", "Ltd", "Enterprises", "Road"). `idf(t) = log(N / df(t))`. Keep generic tokens as separate features, don't delete them blindly (they help reject "Acme Robotics" vs "Acme Foods")
- **Trap:** stripping legal suffixes/numbers too aggressively → merges distinct businesses (fatal under F0.5); tempting because it lifts recall
- **Eval:** positive-pair similarity distribution shifts up, negative stays low; blocking recall rises at fixed candidate budget
- **Confidence:** medium — depends on how many regions/languages exist

### 2. Blocking / candidate generation
- **Type:** IR › candidate retrieval › recall-first multi-pass blocking (union of keys + top-k nearest neighbours)
- **Why:** all-pairs S1×(S2+S3) infeasible; video states blocking sets the recall ceiling; output (`candidate_pairs.tsv`) is audited
- **Why not:** ranking/LTR as final task → output is an unordered set, but ranking view is right *inside* this leaf · clustering → S1 is a clean reference, so it's linkage to a master, not unsupervised dedup
- **Approaches:** exact keys (postcode, name prefix, phonetic code) (baseline, cheap, brittle) · TF-IDF char n-gram on name and address separately, top-k cosine via sparse matmul/ANN, union with key passes (standard) · zero-shot small sentence-encoder (bge-small/gte-small, 33M params) embeddings + brute-force cosine top-k, fine-tune only if zero-shot underperforms (heavy, if pretrained allowed)
- **Pick:** union of name-channel top-k + address-channel top-k + cheap keys + embedding-channel top-k; both channels because matches can share name *or* address (video explicitly)
- **Hardware note (M4 Pro 24GB unified):** bge-small/gte-small (384-dim, fp16 ~70MB) via MLX or `sentence-transformers` on MPS. Brute-force numpy cosine (`emb @ query.T`, `argpartition` top-k) handles candidate pools up to ~1-2M rows in seconds — skip FAISS/vector DB until past that. `# lean: move to FAISS-IVF only past ~2-5M vectors`
- **Concept:** measure `pair completeness PC = true pairs retained / all true pairs` vs `reduction ratio RR = 1 - candidates / all pairs`. Tune k per channel on train to reach PC ~ plateau, then stop
- **Trap:** tuning blocking by final leaderboard only → you can't see which misses are blocking vs model; also one-channel blocking (name only) misses "abbreviated name, same address"
- **Eval:** PC and candidates-per-entity on train, per region; list of missed positives inspected by hand
- **Confidence:** high on type, medium on k/budget until scale known

### 3. Pair matcher
- **Type:** ML › supervised › binary classification › tabular (engineered similarity features) on candidate pairs, imbalanced
- **Why:** labels exist (train ground truth expands to positive pairs; blocked non-matches = hard negatives); output per pair is match/non-match
- **Why not:** unsupervised thresholds on one similarity → can't combine name vs address evidence (same address/different business, similar name/different address) · end-to-end DL on raw strings from scratch → data likely too small, GBM on similarity features is the proven ER baseline · LLM API → prohibited
- **Approaches:** logistic regression on a few similarities (baseline) · GBM (LightGBM/XGBoost) on ~30-80 features: Jaro-Winkler, token-set ratio, IDF-weighted Jaccard, char n-gram cosine, number/postcode equality, rare-token overlap, length diffs, region flags, source (S2 vs S3) flag, candidate rank + rank-gap features (standard) · + embedding cosine similarity (from Leaf 2's encoder) as an extra GBM feature; fine-tuned cross-encoder only if GBM+embedding-feature plateaus (heavy, if pretrained allowed)
- **Pick:** GBM + embedding-similarity feature; add context features (this pair's score vs best/second-best candidate for the same S1 entity and same S2/S3 record) — they carry the "is this the best explanation" signal
- **Hardware note:** embedding feature reuses Leaf 2's encoder output, no extra fine-tune cost. Skip the cross-encoder tier unless GBM plateaus — 24GB unified handles it (batch 64-128, ~15-30 min for 2-3 epochs) but it's the most time-expensive option on the list
- **Concept:** train on exactly the candidate distribution blocking produces (train/serve consistency). Name and address are two partially independent witnesses; the model learns when one overrides the other
- **Trap:** random negatives → easy task, great offline AUC, fails on lookalikes; random pair split → leakage (same entity on both sides)
- **Eval:** PR-AUC + log-loss on out-of-fold pairs; calibration curve (needed by Leaf 4)
- **Confidence:** high

### 4. Set-level decision (singleton + subset selection + assignment)
- **Type:** Decision theory › set prediction under non-decomposable metric (expected F-beta maximisation) + bipartite assignment constraint
- **Why:** score is per S1 entity on the whole set; singleton cliff (empty correct = 1, any FP = 0); F0.5 punishes extra wrong IDs more than missing ones
- **Why not:** single global pair threshold → optimises pair-level metric, ignores set size and the cliff · per-entity F1 logic → wrong beta
- **Approaches:** tune one global threshold for macro F0.5 on OOF (baseline) · per-entity: sort candidates by calibrated p, evaluate prefixes k = 0..K, pick k maximising expected F0.5 (standard) · + resolve conflicts so each S2/S3 record goes to ≤1 S1 entity (argmax or Hungarian-style), + separate "is singleton" model using entity-level features (max p, gap, #candidates) (heavy)
- **Pick:** start global threshold to get a number, move to expected-F prefix selection + assignment constraint once calibration is good
- **Concept:** worked intuition — one candidate with prob p: predict it → E[F] = p, predict empty → E[F] = 1-p, so emptiness is a real option. With a true set {a,b}: predicting {a} scores 0.83, predicting {a,b,c-wrong} scores 0.71 → dropping doubtful extras pays
- **Trap:** treating "empty" as failure / always emitting top-1 because it feels like hedging; forgetting that calibrated probabilities are required for expected-F math
- **Eval:** local macro F0.5 on OOF entities, split by singleton vs non-singleton vs multi-match to see where points are lost
- **Confidence:** medium — depends on singleton rate and whether assignment assumption holds

### 5. Validation & local metric
- **Type:** ML eval › grouped CV (group = S1 entity) + exact local replica of the scorer
- **Why:** pairs of one entity are correlated; leaderboard submissions are limited and noisy
- **Why not:** random k-fold on pairs → leakage · leaderboard-only tuning → overfit to public split
- **Approaches:** single holdout of entities (baseline) · GroupKFold by S1 ID producing OOF probabilities for Leaves 3-4 (standard) · stratify groups by region and singleton/multi-match (heavy)
- **Pick:** GroupKFold, stratified by region + cardinality if cheap
- **Concept:** OOF predictions give honest probabilities to tune thresholds/decision layer without touching test
- **Trap:** tuning the decision layer on in-fold predictions (overconfident) → too many merges on test
- **Eval:** local score tracks leaderboard in direction across submissions
- **Confidence:** high

### 6. Deliverable & reproducibility
- **Type:** DE › batch pipeline › deterministic end-to-end script + validation script check
- **Why:** final archive must run, candidate file audited, top teams reviewed
- **Why not:** notebooks with manual cells → fails review
- **Approaches:** one script per stage + config (standard) · Makefile/CLI with seeds, versioned artefacts (heavy)
- **Pick:** staged scripts: normalise → block (writes candidate_pairs) → features → model → decide (writes matching_results) → validate
- **Concept:** candidate_pairs must be the actual input to the matcher (audit consistency)
- **Trap:** read TSVs without explicit `sep='\t'` (and without keeping IDs as strings / empty lists intact)
- **Eval:** clean run from raw files reproduces submitted score; official validator passes
- **Confidence:** high

---

## Dependencies
`1 → 2 → 3 → 4`; `5` wraps 3-4 (OOF probs); `6` wraps all. Label/EDA checks (open questions) gate the Leaf 4 design.

## Riskiest leaf: 4 (set-level decision)
Blocking is the most *work* and sets the ceiling, but it's measurable and fixable. The decision layer is where a strong pair model silently loses points: global pair thresholds, ignoring singletons, uncalibrated probs, double-assigned records. It's the leaf most teams misjudge.

## Key formulas
```
F_beta = (1+beta^2) P R / (beta^2 P + R);  beta=0.5 → 1.25 P R / (0.25 P + R)
Macro score = mean over S1 entities of F0.5(pred_set, true_set); empty/empty = 1
Pair completeness PC = |true pairs in candidates| / |true pairs|
Reduction ratio RR = 1 - |candidates| / (|S1| * (|S2|+|S3|))
idf(t) = log(N / df(t))
Expected-F set: S* = argmax_S E_T[F0.5(S, T)], search S over top-k prefixes by p, k=0..K
```

## Glossary
| Term | Meaning |
|---|---|
| Entity resolution / record linkage | deciding which records refer to same real-world thing |
| Singleton | S1 entity with no S2/S3 match; correct output = empty |
| Blocking | cheap candidate filter; recall-first |
| Hard negative | non-match that looks similar (same address or similar name) |
| OOF | out-of-fold predictions from CV |
| Calibration | predicted p ≈ observed match frequency |
| Assignment constraint | each S2/S3 record linked to ≤1 S1 entity |

## Hardware plan (M4 Pro, 24GB unified)
- Encoder: `bge-small-en-v1.5` or `gte-small` (33M params, 384-dim) — small encoders match/beat large ones on ER benchmarks; no quantization needed at this size
- Zero-shot first (20-min separation check on train pairs) → fine-tune only if it underperforms GBM's text features
- Fine-tune if needed: MultipleNegativesRankingLoss, batch 64-128, MPS backend, ~15-30 min for 2-3 epochs — no gradient accumulation needed
- ANN: brute-force numpy cosine top-k, not FAISS, until candidate pool exceeds ~1-2M rows
- Everything else (GBM, TF-IDF blocking) is CPU-light and not memory-bound on this machine

## Brainstorm seeds
- Two-channel view: model "name match strength" and "address match strength" separately, then a combiner; explicitly handle same-address-different-business (malls, office towers) and chains (same name, many addresses).
- Competition features: per-record rank features ("is this S1 the best candidate for this S2 record?") often beat raw similarity.
- Use S2↔S3 agreement as evidence: if an S2 and S3 record strongly match each other and one matches entity E, boost the other.
- Region detection first, then region-specific normalisers/models or region as a feature.
- Mine abbreviation/synonym maps automatically from aligned tokens in train positive pairs.
- Error-bucket dashboard: blocking-miss vs model-FN vs model-FP vs singleton-FP, per region — drives iteration order.
- Methodology doc as a differentiator: report PC/RR, per-region scores, ablations.

---

## Revision 1 (from PS)
Source: `docs/source/Business Entity Resolution Challenge.rtf` (authoritative) + video transcript.
The original leaves above are unchanged. These deltas override them where they conflict.

- **Domain shift:** test contains **France, unseen in train**, so this is zero-shot cross-country
  generalisation. **Riskiest leaf is now cross-country generalisation** (was Leaf 4).
- **Country:** use it for a blocking pre-filter (only if EDA confirms country is consistent across
  sources for true matches) and to route normalisers. **NEVER a GBM feature.** Never hard-code,
  filter or one-hot it.
- **Encoder (Leaf 2/3):** `bge-small-en` is English-only. Candidates are `multilingual-e5-small`
  (MIT) or `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0). Verify the license on the model card
  before use.
- **Normalisation (Leaf 1):** add a small set of hand-written French rules: legal forms
  (SARL/SAS/SASU/SA/EURL/SCI); street types (rue/av/bd/pl/chemin/allée/quai); accent folding;
  bis/ter. Also strip landmark phrases ("near/opp/behind X"). Compute IDF **per country** over
  train+test sources (unlabeled, so allowed).
- **Matcher (Leaf 3):** favour relative features (score vs best/2nd-best per S1 record and per S2/S3
  record, ranks) over absolute similarities, since absolutes shift across countries. Add token-sort
  and token-set features for word transpositions. DBA names mean the address channel must carry some
  matches on its own.
- **Validation (Leaf 5):** add leave-one-country-out CV (train US → eval India, and the reverse) as
  the France proxy. Report the in-domain GroupKFold vs LOCO gap.
- **Decision (Leaf 4):** calibration fitted on US/India may be overconfident on France. Size any
  conservative adjustment from the measured LOCO gap; don't guess it.
- **Resolved open Qs:** pretrained models are allowed (MIT/Apache-2.0, <=8B params). Countries are
  US and India, plus France in test only.
- **Still open (for M2 EDA):** scale; singleton rate; match cardinality; whether any S2/S3 ID sits
  under >1 S1 entity; S2↔S3 co-reference; country consistency across sources for true matches;
  whether country/postcode appears inside address strings.

## Revision 1a (script-mix sample, 2026-09-25)
Evidence: the first 200k rows of each source file, counting script per field × country (full-file numbers come at M2).

| | S1 | S2/S3 |
|---|---|---|
| US | pure ASCII | ~6.8% of names carry Latin accents; addresses ASCII |
| India | ~ASCII (0.05% accents) | ~18–27% of names and ~23% of addresses non-ASCII: Devanagari ~7–14%, other Indic (Gujarati, Malayalam, …) ~6–10% |
| France (test) | ~16% of names, ~28% of addresses accented | ~24% of names accented, ~19% of addresses |

- **Cross-script matching is a first-class India problem, and it's in train (so learnable and measurable).**
  Native script is mostly phonetic transliteration of English/Latin words, not translation.
- **Normalisation (Leaf 1):** the canonical space is ASCII Latin (S1 already lives there). Transliterate
  every field to ASCII before char n-grams and string similarities; keep the raw string for the encoder.
  Accent folding covers US noise and France.
- **Encoder (Leaf 2/3):** a multilingual encoder alone may miss transliterated English ("यूनिवर्सल" =
  "universal"). The transliterated char-TF-IDF channel is the primary cross-script channel; the encoder
  is complementary. Measure both separately on India cross-script positives.
- **Blocking (Leaf 2):** report PC for India split by "S2/S3 record has native script: yes/no".
- **Test mix differs from train:** in the sample, India is ~47% of test S1 vs ~40% of train S1, and US
  is lower. Macro F0.5 is therefore weighted more toward India on test.

## Revision 2 (from EDA, `docs/eda.md`, full train+test scan + sampled checks)

**Resolved open questions:**
- **Scale:** train S1 2.21M / S2 5.03M / S3 5.29M; test S1 1.73M / S2 4.89M / S3 5.08M rows.
- **Singleton rate:** 5.58% (from M1). **Cardinality:** median 3 matches/S1, p90 6, max 11.
- **S2/S3 under >1 S1:** **0 of 7,638,365 matched ids** — the one-to-one-per-S2/S3-record
  assumption holds EXACTLY on train, not just approximately (assignment is many-to-one: each
  S2/S3 record → ≤1 S1, each S1 → many). Per-record argmax-with-threshold assignment in the
  decision layer is now the standard pick, not a "heavy" option (strict-old-man correction).
- **Country consistency:** 100.00% of true pairs have S1.country == match.country; 0% missing
  country either side. Country is a clean, safe blocking pre-filter and normaliser router — the
  EDA-gate from Revision 1a is satisfied.
- **Distractor rate:** ~25-27% of S2/S3 records match no S1 entity, consistent across S2/S3 and
  US/India — distractors are a real, evenly-distributed presence, not a corner case.
- **Non-ASCII / script mix (confirms Revision 1a, now exact):** India S2/S3 name+address ~18-28%
  non-ASCII (Devanagari, Kannada, Telugu, Tamil, Gujarati, Bengali, Malayalam all present); US S2/S3
  names ~6-7% accented; **France (test) ~16-28% non-ASCII** across name/address, both S2 and S3.
  Top scripts by volume: LATIN (accents) > DEVANAGARI > KANNADA/TELUGU/TAMIL/GUJARATI/BENGALI.
- **Cross-script similarity is high after transliteration:** for true pairs with a non-ASCII match,
  `token_set_ratio(anyascii(match), S1 name)` has p10=60.7, p50=86.7, p90=100 (n=80,000 sampled).
  Transliteration recovers most of the signal; the hard tail (p10~60) is real but a minority.
- **S2/S3 postcode presence is near zero for India (0.00%) and low for US (~10%) and France
  test (~0.4-0.5%)** — postcode cannot be a primary blocking key for any country; it's a
  confirming signal at best, not a filter.
- **Exact duplicate name+address within a source:** S2 ~0.5%, S3 ~0.3-0.4% (train and test) —
  low enough to ignore for M3, not worth a dedup pre-pass.
- **Empty name/address:** 0% everywhere — no null-handling branch needed in the normaliser.

**Leaf confidence updates / decisions this overturns:**
- **Leaf 2 (blocking), hardware plan — OVERTURNED, see Revision 2a for a correction to this
  entry.** Naive top-20-per-channel blocking on the full S1 gives an estimated **~44M candidate
  pairs** (`BLOCK_K=20 * |S1|=2,206,821`) — this number itself is fine to score in chunks; the
  real cost is the *search* (2.2M S1 queries × ~10M S2+S3 candidates), which is infeasible
  brute-force regardless of k. **Decision: M3's blocking must use an inverted-index/ANN structure
  by design (not brute-force cosine as a fallback), plus (b) restrict the candidate pool per S1 to
  its country before scoring (country consistency = 100% on train supports this).** Country
  pre-filtering is applied first for its 100%-safe recall (zero cost given the country-consistency
  finding) — it reduces the search space by ~1.9x on train / ~2.5x on test (1/Σshare², not 3-4x;
  see Revision 2a), not enough on its own to make brute force viable, so it's a companion to ANN
  blocking, not a replacement for it. Confidence: high (data-verified).
- **Leaf 2 (blocking), two-channel design — CONFIRMED, not just plausible.** On 200k sampled true
  pairs: 66.01% both-strong, 15.56% name-only-strong, 15.43% address-only-strong, 0.04% both-weak.
  Either channel alone would miss ~15% of true pairs the other channel alone would catch — the
  two-channel union is necessary, not redundant. Confidence: high.
- **Leaf 3 (matcher), hard negatives — CONFIRMED.** Blocked negatives (top-20 by name similarity,
  same country, sampled) have name_sim p50 = 60.6 vs random negatives' p50 = 33.3 (gap +27.3
  points on a 0-100 scale). Training on blocking-produced negatives is materially harder and
  therefore necessary — random negatives would give a falsely easy offline AUC, exactly the trap
  the original breakdown named. Confidence: high.
- **Leaf 4 (decision layer), assignment constraint — UPGRADED from "verify on train" to
  "confirmed, build it as standard."** 0 violations in 7.6M matched ids means the per-record
  argmax-above-threshold assignment step (many-to-one: each S2/S3 → its best S1 if above
  threshold) is justified outright, not conditional. See Revision 2a: Hungarian (one-to-one) is
  the wrong tool for a many-to-one assignment and should not be used here.
- **Leaf 1 (normalisation) — CONFIRMED region-specific, scope now precise.** D11's top-60 tokens
  per country/source give the actual normaliser vocabulary (not guessed): India needs
  limited/private/ltd/pvt/llp + common OCR-style typos already visible in S2 (`praivet`,
  `limitet`, `piraivet` — phonetic/OCR noise on top of the legal-suffix set, not just legal-suffix
  variation); France needs sarl/sas/eurl/sasu/sci/sa + rue/avenue/bis/allee/chemin/bd, matching
  Revision 1a's rule list exactly, plus heavy landmark/prefix noise ("Amicale du", "Comite",
  "Établissements"). US needs llc/inc/corp/pllc/lp + street-type abbreviations (st/rd/dr/ave/ln)
  AND full state names appearing in S3 addresses but abbreviated in S1/S2 (`texas` vs `tx`) — this
  is a new, previously-unflagged US normalisation need: state name/abbreviation must be folded to
  one form, source-dependent.
- **New finding, not in Revision 1/1a:** France business names carry an embedded "(France)" or
  "(Frànce)" parenthetical marker on **6.6-8.0% of records, in all three sources (S1 8.01%, S2
  6.71%, S3 6.61% of France test rows)** — checked against the full France test slice, not just
  the D12 sample. **Confirmed name-only: 0.00% of France addresses carry it in any source.** This
  looks like injected/synthetic noise specific to the France test slice, not an organic naming
  pattern — strip it as a normalisation step (name field only) so it doesn't inflate or deflate
  name similarity for reasons unrelated to real business identity.
- **C9 (within-entity clustering) result is informative but modest:** among-matched-record name
  similarity p50 = 88.0 vs matched-to-S1 name similarity p50 = 98.7 (n=2000 sampled S1 entities
  with >=3 matches). Matched records resemble S1 slightly more than they resemble each other —
  weak evidence for clustering S2∪S3 as a support signal, not a primary channel. Leaf 3's
  "S2↔S3 agreement" brainstorm idea is worth a small feature, not a redesign.

**Still open (push to M3 EDA-as-you-build, not blocking M3 start):**
- Exact per-country reduction-ratio/PC tradeoff once a real blocking k and country pre-filter are
  chosen (this EDA gives the *before* picture, not the *after*).
- FAISS need is conditional on the country-pre-filter decision above; re-measure candidate-pool
  size after applying it before committing to FAISS in M3.

## Revision 2a (correction)

Corrections to Revision 2, caught before M3 build:

1. **The ~1-2M brute-force ceiling referred to vectors in the search index, not candidate pairs.**
   44M candidate pairs is fine to score in chunks — that's a downstream, embarrassingly-parallel
   step. The real cost is the *search* itself: 2.2M S1 queries against ~10M S2+S3 index vectors,
   which is infeasible brute-force regardless of what k is chosen. **Blocking must be built on an
   inverted-index / ANN structure by design from the start, not reached for as a fallback once
   brute-force is shown too slow.**
2. **Country pre-filter reduction is not 3-4x.** The correct reduction ratio for a filter that
   partitions into groups by share `s_i` is `1/Σ(s_i²)`. On train (US 60% / India 40%, two
   groups): `1/(0.6² + 0.4²) = 1/0.52 ≈ 1.9x`. On test (three groups, India/US/France roughly
   ~47/38/15 per the M1/M2 splits): **≈2.5x**. Still worth applying — it's a free, 100%-safe
   reduction given 100% country consistency on true pairs — but it does not make brute-force
   search viable on its own; ANN/inverted-index blocking is required regardless.
3. **Assignment is many-to-one, not one-to-one — Hungarian is the wrong tool.** Each S2/S3 record
   maps to ≤1 S1 entity, but each S1 entity maps to many S2/S3 records (median 3, p90 6). Hungarian
   algorithm solves one-to-one bipartite assignment and is the wrong fit here. **Correct rule: each
   S2/S3 record is assigned to its argmax-scoring S1 entity if that score is above threshold**
   (a per-record decision, not a global one-to-one matching). Every "Hungarian" reference elsewhere
   in `docs/` (Revision 2's Leaf 4 assignment-constraint entries, `docs/phases.md`'s M2 closure
   entry) has been corrected in place to this per-record argmax rule.

**New findings for M3 (from continued data reading, not yet in Revision 2):**
- Data noise looks synthetic/enumerable — it reads like it was produced by a small set of
  generator operators (fixed abbreviation swaps, fixed typo patterns, fixed marker insertions like
  the France "(France)" suffix), not organic free-text variation. **Action: mine the actual
  operators from train positive pairs (aligned token-level diffs between S1 and its matched
  S2/S3 records) and invert them explicitly in the normaliser, rather than hand-writing a generic
  ruleset and hoping it covers what the generator does.**
- **3.4% of S2/S3 records have an empty address field** (name-only matches) — the normaliser and
  matcher need an explicit `has_address` handling path (e.g. a feature flag + fallback to
  name-only similarity), not an implicit assumption that both fields are always populated.
- **France: S1 uses région while S2/S3 use département** — these are different administrative
  levels of the same hierarchy, not the same field with noisy formatting; a normaliser or matcher
  that string-compares them directly will systematically fail on France. City also carries little
  identity signal for France matches (weak field) — **house number + street name carries most of
  the real address-match signal for France** and should be weighted accordingly.

## Revision 3 (M3a normaliser v1, `docs/noise_ops.md` + `docs/normalise.md`)
- **Name alone is not an identity key.** Raw lowercased names already collide within S1 for 38.3% of
  records (India 43.9%, US 34.6%), and S1 is de-duplicated on name+address. Name-channel blocking will
  return many same-name/different-address S1 candidates; the matcher's address evidence decides.
- **Legal form must stay a feature, not be deleted.** Removing it from core_name gives +25.3pp name
  recall but S1 collision 38.3→50.2% and non-pair collisions 1.0→9.1 ppm; `core_name|legal_form` holds
  collision at 40.4% (3.0 ppm). Keep `legal_form` as its own matcher feature, not a hard key: 13-18% of
  true pairs drop it and 8% (US) add one.
- **India address key is weak by design:** exact (number, street) equality 13.4% of India pairs vs 64.8%
  US; India addresses rarely start with a house number. India address matching must come from
  addr_tokens similarity (TF-IDF), not a number+street key. US ceiling: 18.6% of US matches drop the
  house number entirely.
- **Recall proxy after v1** (name OR addr exact): US 86.5%, India 60.4% (non-ASCII India 47.6%: the
  transliteration gap is the main India recall problem for blocking; char n-grams, not exact keys).
- Reverted rules: country_marker, amp_and, landmark (decisions_mistakes.md).
