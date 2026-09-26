# M5+ Strategy: from LB 0.957 to 0.986 (L3 research + cross-check)

Date: 2026-09-25 21:45 IST. Deadline: 2026-09-27 23:59 IST (~50 h, of which ~8 h go to sleep and ~5 h to M8 packaging).
Mode: lean-senior + ai-is-smart-for-me + ml-code-sensei. Tags: **[S]** = sourced, **[D]** = derived from our own eda.md numbers, **[I]** = own inference (untested).

---

## 0. Verdict

**Solution with fixes.** The plan in plan.md has the right levers but the wrong order and one wrong premise.

1. **Don't write M5a (pruner) first.** Its stated goal ("size is ranked") contradicts the PS. The PS says `candidate_pairs.tsv` is *not scored*; it is audited for recall ceiling and reduction ratio. At 35 candidates per S1 against ~10M S2/S3 records, RR is already ~0.9999965. A fixed 5-10 prune also conflicts with our own cardinality numbers: p99 = 8 and max = 11, and 26% of entities have 5+ matches. A pruner is only worth building as a **compute gate** in front of expensive encoder features, and then it must be a probability threshold, not a fixed k.
2. **The biggest unexplained lever is the OOF→LB gap, and eda.md already contains a probable cause** (§2). Test has **~23% more S2/S3 records per S1 than train, in every country** [D]. If matches per S1 stay constant, the test distractor rate is ~40% vs 26% in train. That is a precision shift the OOF never saw, and it would hit US and India too, not just France.
3. **Encoders are worth it, but as a targeted channel/feature gated by a 1-hour measurement, not as the main model.** The literature says fine-tuned cross-encoders are volatile and slow, pre-trained models are already strong on semi-synthetic perturbation data, and generic multilingual sentence encoders can be near-random on character-level cross-script matching (§4).

Answer to "write M5a now?": **no. First run D0 diagnostics (~2 h, §3)**, then build in the order in §5.

---

## 1. Loss budget (what 0.986 needs)

| Item | Now | Needed for LB 0.986 |
|---|---|---|
| Blocking ceiling loss | 0.0128 | ≤ 0.004 |
| Matcher FN loss | 0.0140 | ≤ 0.004 |
| Matcher FP loss | 0.0077 | ≤ 0.003 |
| OOF→LB gap | ~0.008 | ≤ 0.003 |
| **Total** | **0.043** | **≤ 0.014** |

Every row must roughly **third**. There is no single fix. The gap row is the cheapest per point if §2 is right, because it needs no new model, only a training-distribution fix.

---

## 2. New finding: test has a distractor-density shift [D]

Computed from eda.md row counts (E1 and B5):

| Split / country | S1 | S2+S3 | Records per S1 |
|---|---|---|---|
| train US | 1,323,633 | 6,186,873 | **4.67** |
| train India | 883,188 | 4,133,346 | **4.68** |
| test US | 663,106 | 3,817,031 | **5.76** |
| test India | 809,986 | 4,717,565 | **5.82** |
| test France | 259,452 | 1,434,993 | **5.53** |
| train all | 2,206,821 | 10,320,219 | 4.68 |
| test all | 1,732,544 | 9,969,589 | **5.75 (+23%)** |

Train has exactly 3.46 matched records per S1 (7,638,365 / 2,206,821), and the distractor rate is ~26% **identically** across S2/S3 and US/India (A3). That identity looks like a generator constant [I]. If test keeps 3.46 matches per S1, then:

- test distractors ≈ 1 − 3.46 / 5.75 = **~40%**, vs 26% in train [I]
- the likely mechanism: test S1 was subsampled (about 19% of entities removed) and their S2/S3 records were left behind as orphans [I]. The other possibility is higher cardinality in test. D0-4 below tells the two apart without labels.

**Why it matters:** orphans whose true S1 is missing will argmax to the *next-best* S1. Our argmax-per-record assignment turns them into false positives that the OOF never contained. That is exactly an FP-shaped gap, and precision counts double in F0.5.

**Fix if confirmed (cheap, no new model):** simulate test density on train by **dropping ~19% of S1 entities** before feature construction. Their records become orphans. Train and evaluate on that. It's S1-dropout augmentation: the model learns "best S1 for this record, but still not a match". Relative features (score − max over the record's candidates) stay valid because they're recomputed on the dropped S1 set.

---

## 3. D0 diagnostics: run these before any new model (~2 h total)

All read existing artifacts (`candidates_train`, OOF predictions, `norm_*`). None retrain anything.

| # | Diagnostic | Decides |
|---|---|---|
| D0-1 | **Blocking-miss anatomy:** the 3.7% of true pairs missed, split by country × native-script(y/n) × has_address × name_sim bucket × addr_sim bucket. Plus: does the true S1 appear at rank k+1..200 in any channel (a depth problem) or never (a vocabulary problem)? | Depth → raise k behind a p-gate. Vocabulary + native script → encoder retrieval channel. Empty address + generic name → accept the loss. |
| D0-2 | **FN anatomy (0.014):** for each missed-but-candidate true pair, was it (a) below threshold with the record's argmax = this S1, or (b) the record's argmax went to *another* S1? Plus a p histogram. | (a) → stage-2 model / calibration. (b) → assignment is costing recall; use a soft assignment margin. |
| D0-3 | **FP anatomy (0.0077):** singleton-FP vs extra-FP in non-singletons; for each FP, name_sim/addr_sim and whether it's same-address-different-name or same-name-different-address. | Which features stage 2 needs. |
| D0-4 | **Test density check (label-free):** on test, compute (i) mean predicted matches per S1 per country, (ii) fraction of S2/S3 records whose max p > 0.5. Compare to train OOF: 3.46/S1 and ~74%. | Record fraction ≈ 60% → distractor shift confirmed, do §2. Mean cardinality ≫ 3.46 → cardinality shift instead. |
| D0-5 | **Orphan simulation:** drop 19% of train OOF S1s, recompute relative features, re-predict with the *existing* model, re-decide, and score the remaining S1s. | If OOF drops by ~0.005-0.008, you have found the gap. |
| D0-6 | **Per-country OOF + generator constants:** singleton rate and mean cardinality for US vs India separately. | If equal, use them as label-free priors for France (§5 step 5). |
| D0-7 | **Leak sanity (1 min):** correlation between S1 and matched S2/S3 ID numbers, and file row order. | Should be ~0. If not, **don't use it.** The top packages are reviewed, and exploiting a leak is a disqualification risk. Just note it. |

### Claude Code prompt for D0 (paste as-is)
```
Apply lean-senior. Write src/diagnose.py (no training) that reads artifacts/interim/candidates_train.parquet,
the M4 OOF predictions, norm_train_s*.parquet and GT, and prints ONE markdown report to docs/diagnose_m5.md with
the tables D0-1..D0-7 from docs/m5-strategy-l3.md §3. Constraints: polars/duckdb, vectorised, no iterrows;
sample ≤ 500k rows where a full pass would exceed 2 min; seed 42. D0-4 runs on test predictions if they exist,
else it is skipped with a note. D0-5: drop 19% of OOF S1 ids (seeded), recompute only the relative/rank
features, re-predict with the saved fold models, re-run decide.py logic, and report OOF macro F0.5 before/after.
Output numbers only; no conclusions. Do not modify any existing module.
```

---

## 4. Evidence ledger (encoders, cross-encoders, decision theory)

| Claim | Source type | Verdict | What it means here |
|---|---|---|---|
| A PLM cross-encoder (Ditto) matched two company datasets (789K × 412K) at 96.5% F1 | primary (paper) [S1] | Verified-primary | Company name+address ER is the Ditto sweet spot. That was F1, pair-level, and on a GPU. |
| Fine-tuned PLM matchers are not robust to out-of-distribution entities | primary [S2] | Verified-primary | Risk for France (unseen). Keep the GBM on similarity features as the backbone. |
| Fine-tuned bi-encoder = best accuracy/speed balance; it repairs blocking recall by reshaping the space | primary (KBS 2026) [S3] | Verified-primary | If D0-1 shows vocabulary misses, a fine-tuned small bi-encoder retrieval channel is the textbook fix. |
| Fine-tuned cross-encoders are volatile (negative average gain for MiniLM/BGE), and on semi-synthetic perturbation data pre-trained models are already strong, so fine-tuning can hurt | primary [S3] | Verified-primary | Our data is perturbation-generated. Expect small encoder gains; measure before investing. |
| Cross-encoder resolution took >2 h on a 1M-record dataset, on a GPU | primary [S3] | Verified-primary | 60M test pairs on an M4 is infeasible. Only the uncertain band, and only if it's small. |
| A classifier over [u; v; |u−v|; engineered features] beats cosine thresholding by 8–18 F1 points | primary [S3] | Verified-primary | Use embeddings as **GBM features**, not as a standalone decision. Matches our plan. |
| multilingual-e5-small: MIT, 12 layers, 384-dim, ~118M params, needs a `query:` prefix for symmetric tasks | primary (model card) [S4] | Verified-primary | License passes (MIT, ≤ 8B). Use `query: ` on both sides. |
| potion-multilingual-128M: static (Model2Vec) distillation of bge-m3, orders of magnitude faster than transformers | primary [S5] | Verified-primary (library MIT); **model-card license: verify before use** | Fast fallback if MPS throughput is too slow. |
| Generic multilingual sentence transformers can be near-random on character-level cross-script lexical retrieval | primary, different script pair [S6] | Verified-but-misused risk: Tajik↔Persian, not Hindi↔English | Don't assume e5 fixes Devanagari transliterations. Our anyascii + char n-gram path may already beat it. **Measure on India native-script slices.** |
| Pairwise matching ignores global consistency; select/compare strategies help | primary [S7] | Verified-primary | Supports the stage-2 per-S1 context model and set-level decision. |
| Decision-theoretic expected-F beats a tuned threshold with a good calibrated model and under domain adaptation; threshold tuning is more robust to misspecification | primary (ICML 2012) [S8] | Verified-primary | Implement expected-F0.5 but **A/B it against the tuned threshold on OOF**; keep the winner per country. |
| Test distractor rate ~40% vs 26% | own inference from row counts [D][I] | Unverified | D0-4 / D0-5 confirm or kill it. |
| France LB ≈ 0.91 | own inference: 0.85·0.9654 + 0.15·x = 0.957 → x ≈ 0.91 [I] | Unverified, and confounded by §2 and the public-subset mix | Don't size France work until D0-5 separates the two causes. |

**Kill-search findings.** Nothing here is novel in ER: two-stage blocking+matching, GBM on similarities, and embeddings as features are all table stakes [S3]. The genuinely new parts for this entry are: (1) **S1-dropout augmentation to match test distractor density**, (2) **label-free generator-constant priors for France**, and (3) expected-F0.5 set decisions with a record-level assignment. Pitch those three in the methodology doc.

---

## 5. Build order (value per hour, gated)

Rewritten after D0 (breakdown.md Revision 4). §2 is confirmed: the 19% S1 drop reproduces test density.

| Step | What | Gate / target | Est. time |
|---|---|---|---|
| 1 | **D0 diagnostics** (§3) | DONE: docs/diagnose_m5.md | — |
| 2 = M5-1 | **Blocking v2:** union A/B/C into the output (`select=[X]` drops 31.1k = 11% of misses) + depth (S1-centric k = 30, record-centric m ≤ 10 from the grid), gated per S1 by a **cheap pre-score** (max over channels of normalised channel score + n_channels_hit; not the GBM). Diagnostic only: in the US no-address name_sim ≥ 80 cell, how many S1s share the record's normalised name | report PC, cand/S1 (train + test) and the F0.5 ceiling at the chosen gate; **≤ ~45 cand/S1 on test**; ceiling must beat v1 (0.9873). Shared name typically > 1 S1 → retrieval can't fix that cell → stage-2 sibling features. Vocabulary misses (38%): no new channel, E0 decides | 1 h + runs |
| 3 = M5-2 | **Features on v2 + ONE full-data retrain with S1-dropout:** 100% of train S1s, GroupKFold-5 by s1_id, fold models + OOF saved. Training folds only: drop 19% of S1s (seeded per fold), their records' remaining candidates are all negatives, record-side relative/rank features recomputed. OOF two ways: (a) standard, (b) **test-density OOF** (held-out fold with 19% of S1s dropped; **PRIMARY metric from now on**). Learning curve 20/50/100% on (b) at fixed rounds | (b) recovers most of the D0-5 −0.0038; curve still rising at 100% → data-limited (more rounds/leaves), flat → feature-limited. Peak memory ≤ 20 GB | overnight |
| 4 = M5-3 | **Stage 2 = stacked GBM on stage-1 OOF probabilities:** p, rank in S1, p − max over other S1s for the record, Σp per S1, count(p > 0.5) per S1, and **sibling features**: for record r and S1 s, max over the other candidates r′ of s of p(r′, s)·sim(r, r′) for name and for address, plus "best sibling has an address" | targets D0-2 (a) 0.0133 and D0-3 FP 0.0077. **Silent killer:** stage-2 training features come from stage-1 OOF only; test stage-1 = the mean of the fold models | 3-4 h |
| 5 | **Encoder E0 (1 h go/no-go):** measure multilingual-e5-small MPS throughput (texts/s) on 50k real names; add cosine(name), cosine(addr), cosine(name+addr) as GBM features on a 5% subset | **Go** if OOF (b) gains ≥ 0.002, or if it recovers ≥ 30% of the D0-1 vocabulary misses at k = 10. Else stop and use potion-multilingual as a cheap feature only | 1 h |
| 6 | **Encoder E1 (if go):** embed all records once (~12M test texts, ~1-2.5 h [I]), ANN retrieval channel (country-filtered, top-10) → union into the pool **before** the pre-score gate | blocking ceiling ≥ 0.995 | overnight |
| 7 | **Fine-tuned bi-encoder** (MNRL, hard:easy negatives 1:1 [S3], 1 epoch) **only if** E1 shows pre-trained retrieval misses a lot of D0-1 | re-embed cost again | 3-4 h |
| 8 | **Cross-encoder on the uncertain band only** (p ∈ [0.05, 0.95]) **only if** that band is < ~2M pairs and steps 2-4 plateau. Needs cross-fitted scores to stack | highest risk, lowest priority [S3] | 6 h+ |
| 9 | **Decision layer (last):** isotonic calibration on OOF (b) → expected-F0.5 **only for the singleton empty-set option**; A/B vs the tuned threshold. Threshold tuning alone is dead (+0.0002 in D0-5); assignment constraint deprioritised (0.0011 in D0-2). France prior only if D0-6 holds (it does: US = India within 0.01%) | +≤ 0.002 [I] | 1-2 h |
| 10 | M8 packaging, clean-run repro, validator | hard stop: start by Sun 17:00 IST | 5 h |

**M5-3 spec (not coded yet):**
- **Rows and labels:** the (b) test-density world of M5-2 (rows of the remaining S1s, p = `p_td` from `oof/oof_full.parquet`),
  so stage 2 learns at test density. Folds = the stage-1 S1 folds; early stopping on an inner 10% S1 split, as in M5-2.
- **Features per (record r, S1 s):**
  - p, and the rank of p within s.
  - p − max p over r's other S1s (0 when r has none).
  - Σp over s, count(p > 0.5) in s, and n_cand of s.
  - Siblings: over the other candidates r′ of s, restricted to the top 5 by p (bounds the cost at ≤ 5 record×record
    comparisons per row), max of p(r′, s)·token_set(core_name r, r′) and of p(r′, s)·token_set(addr r, r′).
  - "Best sibling has an address": the address flag of that argmax r′.
  - Plus name_tset, addr_tset and has_addr_code from stage 1.
- **Silent killers:**
  - Stage-2 training features come only from stage-1 OOF p.
  - On test, stage-1 p = the mean of the 5 fold models (`oof/test_p.parquet`). An average is smoother than one model's
    OOF p; watch the p-histogram shift between OOF (b) and test.
- **Decision:** argmax per record + one t on stage-2 OOF (b).
- **Gate:** stage-2 OOF (b) ≥ stage-1 OOF (b) + 0.002, else don't ship.

**Pruner (old M5a):** replaced by the M5-1 pre-score gate (a per-S1 cap, chosen from a train sweep and checked on test). Never a fixed top-5/10 on X alone.

---

## 6. Independent review (cold re-derivation, then checked against the plan)

No subagent was spawned. The review was written without looking at §5 and then applied to it.

| Attack (sceptical Amazon reviewer) | Holds? | Fix or limit |
|---|---|---|
| "Your OOF is on a 20% subset and the model is trained on it; test uses a 20%-trained model" | Yes | Step 2: full-data retrain. |
| "Stage-2 stacking leaks if stage-1 probabilities are in-fold" | Yes, a classic silent killer | Stage 2 trains only on stage-1 OOF; test stage-1 = average of the fold models. |
| "Threshold/calibration tuned on 26% distractors will over-merge at 40%" | Probable | §2 + D0-5 + step 3. |
| "France priors from a US/India generator constant are an assumption" | Yes | Only applied if D0-6 shows US = India within ±1%; otherwise the France threshold = the global one. |
| "Encoder features change the candidate set → candidate_pairs.tsv no longer equals the model input" | Real audit risk | Write candidate_pairs **after** the final gate, from exactly the rows stage 2 scores. The validator subset check enforces it. |
| "Can a coached/adversarial input bypass it?" | N/A (offline competition) | — |
| "Encoder license/size rule" | Passes | e5-small MIT, ~118M params [S4]. Record the model card URL + license in requirements/README. |

**Pre-mortem (how this fails by Sunday):**
1. Time runs out mid-encoder with no packaged submission → **kill criterion:** if step 6 isn't a clear go by Sat 12:00 IST, drop steps 7-9.
2. The stage-2 leak inflates OOF, and LB drops → check: the stage-2 OOF gain must show up on LB within 1 submission, else revert.
3. The §2 hypothesis is wrong (the extra records are higher cardinality, not distractors) → D0-4 separates these in 10 minutes; only build step 3 if confirmed.

---

## 7. Submission discipline
Submit only after steps 2, 3+5, and 4 (≤ 4 submissions), each changing one thing, so the LB tells you which lever moved. Log OOF vs LB per submission in logs.md; the gap trend is itself the check on §2.

---

## Unverified (needs a number before it's used)
- The test distractor rate and orphan mechanism (§2) → D0-4, D0-5.
- The France LB score → only via the D0-5 decomposition.
- e5-small throughput on M4 MPS → E0.
- potion-multilingual-128M model-card license → check the HF card before use.
- The submission limit → not published anywhere I could find; check the portal.

## References
- [S1] Li et al., *Deep Entity Matching with Pre-Trained Language Models* (Ditto), PVLDB 14(1), 2020. https://arxiv.org/abs/2004.00584 (primary)
- [S2] Peeters, Steiner, Bizer, *Entity Matching using Large Language Models*, arXiv 2310.11244. https://arxiv.org/abs/2310.11244 (primary)
- [S3] Karapiperis, Akritidis, Bozanis, *The impact of fine-tuning on entity resolution: An experimental evaluation*, Knowledge-Based Systems 338, 2026. https://www.sciencedirect.com/science/article/pii/S095070512600170X (primary, open access)
- [S4] intfloat/multilingual-e5-small model card. https://huggingface.co/intfloat/multilingual-e5-small (primary)
- [S5] MinishLab, potion-multilingual-128M / Model2Vec. https://huggingface.co/minishlab/potion-multilingual-128M ; https://github.com/MinishLab/model2vec (primary)
- [S6] *TajPersLexon: cross-script low-resource NLP*, arXiv 2605.06886. https://arxiv.org/pdf/2605.06886 (primary; different script pair)
- [S7] *Match, Compare, or Select? An Investigation of LLMs for Entity Matching*, COLING 2025. https://aclanthology.org/2025.coling-main.8.pdf (primary)
- [S8] Ye, Chai, Lee, Chieu, *Optimizing F-measures: A Tale of Two Approaches*, ICML 2012. https://arxiv.org/abs/1206.4625 (primary)
- Zeakis et al., *Pre-trained Embeddings for Entity Resolution*, PVLDB 16(9), 2023. https://arxiv.org/abs/2304.12329 (background)
- Own data: docs/eda.md (A3, B5, E1 row counts), M3b/M4 numbers from this project.
