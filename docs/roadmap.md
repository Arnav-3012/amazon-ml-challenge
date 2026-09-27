# Roadmap after the challenge (written 27 Sep, from tonight's measurements)

Starting point: submission B, OOF (b) 0.97609. Its losses come from docs/error_diagnosis.md.
- **Blocking misses** cost 0.0129 LB-weighted. India alone costs 0.0206: 62% of its "address shared, name changed" misses are native-script records.
- **Matcher false negatives** cost 0.0066.
- **False positives** cost 0.0033.
- **France** has no labels; its score is estimated at ~0.937.

Effort estimates assume one person.

## 1. Neural retrieval with a model fine-tuned on our matched pairs (RTX 3060 laptop, 6 GB) — highest value
Pre-trained retrieval (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 118M) on missed pairs, measured tonight:
missed-pair recall@25 at a 500k same-country pool is India 27.7% / US 39.4%; even blocking's hits only reach 81% / 92% (docs/phase3_retrieval.md). It is too weak and too slow to add as a channel as-is: ~1.2–1.7k texts/s on M4 MPS makes ~5.7 h for 24M texts.
- **Data.**
  - 7.6M train true pairs, as (S1 text, record text).
  - Hard negatives from the v1 candidates the matcher scored below 0.05.
  - Text = normalised name | address, which measured better than raw text. Add a raw-script copy of India records as an extra positive view, so the model learns हाईटेक ↔ hitech.
- **Training.**
  - MultipleNegativesRankingLoss, 1 epoch on ~2M pairs.
  - fp16, batch 128, max_len 48.
  - ~1.5–2 h on a 3060 [estimate].
  - Keep a 10% S1 holdout. Gate: recall@25 of v1 misses at a full-country pool ≥ 40%.
- **Index.** Embed all 24M texts (~1 h on a 3060 at ~6–8k/s [estimate]). FAISS IVF-Flat per country (nlist ≈ 4√N), top-25 in both directions.
- **Integration.**
  - Union the new pairs with v1, capped at +10 per S1 to keep candidate sets small.
  - Rebuild features over the **union**, not the new pairs alone: the competitor-relative features need every candidate of a record.
  - Retrain stage 1 and stage 2 (4 h on the M4, or ~1 h on a cloud CPU box).
- **Effort:** 1.5–2 days.
- **Upside:** if half the blocking cost is recovered, about +0.006 LB [inference]. This is the only lever sized like the gap.

## 2. Cross-encoder re-ranker on the uncertain band
- **Scope.** Rows with stage-2 p in [0.05, 0.95]: about 3% of candidate rows [to measure]. A small multilingual cross-encoder (MiniLM-L12, Apache-2.0) is fine-tuned on OOF rows of that band.
- **How it is used.** Cross-fitted by the same S1 folds, its score becomes a stage-3 feature, never a replacement decision.
- **Effort:** 2 days. Test scoring of ~2M pairs at ~300 pairs/s on a 3060 takes ~2 h.
- **Upside.** Mostly the "argmax right, p below t" FNs, which are 41k in India and 48k in US on OOF. Moderate, and the literature says fine-tuned cross-encoders are volatile, so gate it on OOF.

## 3. India script and transliteration features
- **Problem.** `anyascii` turns हाईटेक फाइनेंस into "haaiittek phaainens", which shares no token with "hitech finance".
  - That is 62% of India's biggest miss bucket.
  - It is also 25% of India's below-threshold FNs.
- **Plan:**
  - Map the anyascii output through an ITRANS-style vowel/aspirate collapse (aa→a, ii→i, ph→f, kh→k, tt→t) before the phonetic skeleton.
  - Mine a larger native→Latin dictionary from matched pairs (mine_dict already found 754 exact mappings; extend it to character n-gram rules).
  - Add a character-3-gram TF-IDF cosine on the collapsed forms as a blocking channel and as a feature.
  - Common-token down-weighting is already covered by IDF. Add a "name made only of top-100 tokens" flag for shree/om/global names.
- **Effort:** 1 day. Blocking + features + retrain = one overnight run.
- **Upside:** India blocking cost 0.0206 → maybe 0.014 [inference].

## 4. France data strategy (no labels)
- **Known.**
  - The normaliser covers French legal forms (SARL/SAS/SASU/EURL/SNC/SA/SCI), street types (rue, avenue, bd, place, chemin, route) and accents.
  - Gaps: `&` vs `et`, `Ets` vs `Établissements`, elisions (l'/d'), and BP/Cedex tokens.
  - 13.4% of French S1s share an exact address, vs ~5%.
- **Plan:**
  1. A French-style noise proxy on a labelled US/India holdout (B1) as the only measurable France signal. Use it to choose rules and features.
  2. Close the lexical gaps with country-agnostic rules. Check that they change no train rows, so train features stay valid.
  3. Pseudo-labelling on confident France test pairs. Continue each fold model with `init_model` at low weight, and replay labelled rows.
     - Needs an explicit rules check first, since it trains on test data.
     - Guard: US/India OOF (b) may drop by at most 0.001.
  4. Label-free monitors per country: empty rate vs the 5.58% generator singleton rate, and matches/S1 vs 3.46.
- **Effort:** 1–1.5 days.

## 5. Larger ensembles
- **Measured tonight:** 2 seeds × 5 folds gave +0.0002 OOF (b), with seed agreement ±0.00002. Seed averaging is saturated.
- **Diversity has to come from different views instead:**
  - a stage-1 model without the relative features;
  - a CatBoost model on the same features;
  - the encoder-based model from §1.
- Stack them in stage 2, which already takes stage-1 p as input.
- **Effort:** 0.5 day per model plus retrains. **Upside:** +0.0005–0.001 [inference].

## What did not work (measured)
| idea | result |
|---|---|
| gated wider blocking pool (pre-score) | worse than v1 at equal size |
| rank / name-key / address unions | ≤ +0.001 ceiling for +28 candidates/S1 |
| exact-address channel, domain segmentation, generator inversion | dropped (18% standalone recall; 25% segmentation accuracy; 81% of names unreachable) |
| normaliser rules country_marker, amp_and, landmark | reverted by ablation |
| expected-F0.5 set selection / empty-set option | −0.0003 / ±0 |
| separate threshold for address-shared S1s | +0.00001 |
| seed ensemble (2 × 5 folds) | +0.0002 |
| per-country threshold (label-free) | needs t = 0.98 for France, which costs −0.013 on US/India; country logic is also against our rules |
| pre-trained multilingual encoder as a retrieval channel | see §1: recall too low at realistic pool sizes, and too slow on the M4 |
| installing torch into an exFAT venv | fails (AppleDouble stubs); use APFS |
