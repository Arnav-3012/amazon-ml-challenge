# B1 France proxy + evidence from the real France test data (27 Sep, 14:50)

**Proxy** (`experiments/france_proxy.py`):
- French-style noise on the raw text of 5,000 labelled fold-0 US/India S1s and their true records, then re-normalised and re-featurised with the real code.
- The noise-off rebuild reproduces the stored features (asserted).
- Candidates and blocking scores are held fixed, so this measures **matcher** robustness only.
- Scored with B (stage-1 fold models + stage 2), OOF-valid.

| noise component (alone) | Δ macro F0.5 of B on the holdout | stage 1 only |
|---|---|---|
| all components together (2k S1 smoke) | −0.0516 | −0.0523 |
| legal form → random French form (SARL/SAS/SA/EURL/SNC/SASU) | **−0.0401** | −0.0365 |
| street words → rue/route/avenue/allée/bd/chemin (+ abbreviations) + Cedex | −0.0042 | −0.0047 |
| accents + `&`/`and` → `et` | −0.0020 | −0.0021 |
| shared address density raised to ~13% | +0.0003 | +0.0002 |

## Is the proxy realistic? Label-free checks on the real France test data
1. **Legal forms.** Among near-identical pairs (name token_set ≥ 95, address ≥ 90) with *different* legal forms and no sibling S1 (a sibling = same name with an equal legal form):
   - Train truth: India's truncation swaps are mostly true (`ltd pvt`→`pvt` 79%, →`ltd` 61%); swaps to a different form are 0% true. US llc↔inc/corp/co/ltd are 0% true (sibling distractors).
   - France's swaps are **spread uniformly over all other forms** (sarl↔sas/eurl/sa/sasu/sci/snc, ~4.5–5k each). That is the US distractor signature, not India's truncation noise.
   - B rejects them (mean p 0.026), most likely correctly. **The proxy's biggest component (−0.040) mis-specifies France.**
2. **Lexical patterns.** In France's S2/S3 records: Cedex 0.01%, BP 0.06%, `all.` 0.77%, ` et ` 1.0% (S1 names use `&`, which the normaliser strips).
   - The normaliser already canonicalises French legal forms and rue/R., av, bd, place/pl., chemin/ch., route/rte, and strips accents.
   - Rules for the remaining gaps would move France by ≪ 0.001.
3. **Address ambiguity** is France's real shift. 38% of France's predicted matches have ≥ 2 candidate S1s with a near-equal address (US/India 7.4%), and 13% of French S1s share an exact address.
   - In the proxy, B is robust to it (+0.0003).
   - Caveat: with fixed candidates, new cross-cluster pairs are not generated, so this understates the effect.

## Conclusion
No evidence-backed matcher fix for France was found. The largest synthetic drop comes from an assumption the real data contradicts, and the realistic components cost B ≤ 0.004 on the proxy.
The France gap (estimated at ~0.937) is therefore not explained by French legal forms or address wording on the matcher side. Remaining candidates: blocking recall for France (unmeasurable without labels) and distribution effects that only labelled French data could show.
Because the proxy is mis-specified, it cannot serve as a ship criterion for France-targeted changes (candidate F).
