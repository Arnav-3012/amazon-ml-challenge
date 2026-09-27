# Afternoon mission results (27 Sep, 13:48–18:10): attack the three loss sources

**Recommendation: keep B.** No candidate beat B (OOF (b) 0.97609) under rule 6: ≥ +0.002, or ≥ +0.0005 for an ensemble, with no segment dropping by more than 0.001.

| candidate | change vs parent | OOF (b) | Δ vs B | India | US | singletons | PC / ceiling | runtime | shipped |
|---|---|---|---|---|---|---|---|---|---|
| **B** (uploaded) | stage 1 + stage 2 carry20 | **0.97609** | — | 0.9673 | 0.9820 | 0.9880 | 96.31% / 0.9873 | — | **yes** |
| D | ensemble with Arnav's stage 1 | — | — | — | — | — | unchanged | — | **skipped**: Arnav's oof/oof_full.parquet + test_p.parquet absent. Tonight's own 2-seed stage-1 ensemble gave +0.0002 at stage 1 |
| E | pre-trained multilingual retrieval channel | — | — | — | — | — | not built | 25 min probe | **no**: dropped at the gate. India 500k-pool top-25 recall of misses 27.7% (< 30%), US 39.4%; full embed 8.4 h (docs/phase3_retrieval.md) |
| F | France pseudo-labelling (stage-1 fold models continued with init_model), then stage 2 retrained | 0.97591 | −0.00018 | 0.9670 (−0.0003) | 0.9819 (−0.0001) | 0.9876 | unchanged | ~3.5 h | **no**: below the bar; France changed only +0.2% net pairs (empty 5.42% → 5.38%, away from US/India); needs a rules confirmation anyway |

Stage-1-only view of F: OOF (b) 0.97502 vs 0.97497 (+0.00005). The guard held on US/India (+0.0001 / −0.00003).

## What the diagnostics established (docs/error_diagnosis.md, docs/france_proxy.md)
- **Loss budget of B (LB-weighted, US/India):**
  - Blocking misses cost 0.0129, and India alone 0.0206. 62% of India's largest miss bucket is native-script names with matching addresses.
  - Matcher FN costs 0.0066.
  - FP costs 0.0033.
- **France (label-free):**
  - The normaliser already covers French legal forms, street types and accents.
  - Legal-form differences in France follow the sibling-distractor pattern: uniform over forms; in train such swaps are 0% true. So B's rejection of them is most likely correct.
  - French lexical oddities (Cedex, BP, `all.`, `et`) occur in ≤ 1% of records.
  - France's real shift is address ambiguity: 38% of its predicted matches have ≥ 2 near-equal-address candidate S1s, vs 7.4% for US/India. B is robust to it on the proxy (+0.0003).
- **The French-noise proxy** dropped B by 0.052, but its largest component (−0.040, legal forms) contradicts the real data. A synthetic proxy needs calibrating against label-free target statistics before its magnitude means anything.

## Why nothing fits before 22:30
The one lever sized like the loss is blocking recall for cross-script names. It needs a fine-tuned encoder on a GPU and a stage-1/2 retrain: about 1.5–2 days. See docs/roadmap.md §1 and §3.

## Artefacts
- **Docs:** docs/error_diagnosis.md, docs/phase3_retrieval.md, docs/france_proxy.md, docs/roadmap.md.
- **Scripts:** experiments/{error_diag,retrieval_probe,france_proxy,pseudo_label}.py.
- **F outputs, not shipped:** models_pl/, oof_pl/, output_pl/matching_results.tsv.
- **~/p3venv:** 1.1 GB on the internal drive; can be deleted.
