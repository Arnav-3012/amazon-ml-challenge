# CE probe (--stack2): rank-aware stacker on the fine-tuned CE, scored in cv_full's (b) world

Reuses `oof/ce_probe_scores.parquet` (`cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned, see the docstring); no re-scoring, no retraining.

## 1. Baseline (fixed)
The previous probe scored 0.7996: it kept the 19% of fold-0 S1s that are dead in world SEED+2000 (no predictions, ntrue > 0 -> F0.5 0) and took the record argmax over fold-0 rows only. Now: cv_full's rule over every oof row with p_td (all folds), scope = alive fold-0 S1s.

| check | macro F0.5 | t | n_s1 |
|---|---|---|---|
| oof/cv_full.json b_test_density (all folds) | 0.9746 | 0.75 | 1787525 |
| recomputed here, all folds (must match to 1e-06) | 0.974585 | 0.75 | 1787525 |
| **fold 0, alive S1s** (within 0.01 of json) | **0.9746** | 0.74 | 357255 |

cv_full.json stores no per-fold F0.5, so fold 0 is checked against the all-fold value.

## 2-3. Stacked ΔF0.5, scope p_td in [0.0005, 0.9995] (fold 0)
Stack rows: 731179 pairs, 306461 S1s, pos rate 0.4788. LightGBM 200 rounds, inner GroupKFold(5) by s1k; p_new inside the scope, p_td elsewhere (other folds too).

| model | AUC (stack rows) | t | F0.5 | ΔF0.5 | Δ India | Δ US |
|---|---|---|---|---|---|---|
| p_td | 0.9917 | 0.74 | 0.9746 | — | 0.9656 | 0.9807 |
| stack1 [logit p_td, ce] | 0.9940 | 0.68 | 0.9775 | +0.0028 | +0.0051 | +0.0013 |
| stack2 (+ranks) | 0.9936 | 0.69 | 0.9775 | +0.0028 | +0.0051 | +0.0013 |

Stack2 gain importance (full fit): logit_p_td 0.0798, ce 0.9198, ce_rank 0.0, ce_margin 0.0, ce_is_argmax 0.0, p_td_rank 0.0004, n_scored_cands 0.0

### Where the stack2 gain comes from: p_new applied inside ONE p_td band (p_td elsewhere), same fitted stacker
| p_td band | pairs | pos rate | t | ΔF0.5 | Δ India | Δ US |
|---|---|---|---|---|---|---|
| [0.0005, 0.02) | 251768 | 0.0040 | 0.74 | -0.0001 | -0.0000 | -0.0001 |
| [0.02, 0.3) | 110845 | 0.0844 | 0.74 | +0.0001 | +0.0003 | +0.0001 |
| [0.3, 0.9) | 58595 | 0.5524 | 0.7 | +0.0026 | +0.0044 | +0.0014 |
| [0.9, 0.9995] | 309971 | 0.9915 | 0.74 | +0.0002 | +0.0005 | -0.0001 |

Band ablation Δs need not sum to the full-scope Δ (one global t each, shared record argmax).

Decision flips p_td (its t) -> stack2 (its t), fold-0 scope S1s, by the predicted pair's p_td band:
| band | gained TP | lost TP | new FP | fixed FP |
|---|---|---|---|---|
| [0.0005, 0.02) | 0 | 0 | 89 | 0 |
| [0.02, 0.3) | 1251 | 0 | 347 | 0 |
| [0.3, 0.9) | 7108 | 1686 | 841 | 2019 |
| [0.9, 0.9995] | 2 | 739 | 2 | 531 |
| outside stack | 0 | 0 | 0 | 0 |

## 4. Confident-FN rescue (true pair, p_td < 0.02, alive fold-0 S1) under stack2 @ t=0.69
- Confident FN: 1039 pairs / 1039 records (old probe: 1039 pairs, dead S1s included); 1014 pairs have p_td >= 0.0005 (the rest cannot change).
- Rescued (true pair now predicted): **0 pairs / 0 records**; F0.5 gain of those rescues +0.00000.
- Of the confident-FN records, assigned to a wrong S1: 1 under p_td -> 0 under stack2.
- New FPs overall (predicted by stack2, not by p_td): **1279** (by band: {'0': 89, '1': 347, '2': 841, '3': 2}; 0 on confident-FN records); F0.5 cost of the new FPs **-0.00096**.
- All flips: gained TP 8361, lost TP 2425, new FP 1279, fixed FP 2550.

## Caveats
- CE scores exist for fold-0 rows only: ce_rank / ce_margin / n_scored_cands see the record's fold-0 in-scope candidates (~1/5 of its competitors). On test every in-scope candidate would be scored, so the rank features would be computed over a harder set. p_td_rank uses all folds.
- Confident-error records' extra CE rows (label-selected) are excluded from every rank.
- Earlier sections of this doc (throughput, test scope hours) are in git history; rerun without --stack2 to regenerate them.

## Steps
```
{"step": "load world", "s": 1.6, "peak_rss_mb": 4037, "rows": 55915451}
{"step": "baseline", "s": 32.0, "peak_rss_mb": 5755, "all_folds": 0.974585182294365, "fold0": 0.9746269174888498, "t": 0.74, "n_s1": 357255, "w_rows": 40508995}
{"step": "stack features", "s": 1.1, "peak_rss_mb": 5755, "rows": 731179, "s1": 306461}
{"step": "fit stack1 [logit p_td, ce]", "s": 14.1, "peak_rss_mb": 5755, "auc": 0.9940145964022715}
{"step": "fit stack2 (+ranks)", "s": 16.8, "peak_rss_mb": 5755, "auc": 0.9935831779691506}
{"step": "score stack1 [logit p_td, ce]", "s": 8.3, "peak_rss_mb": 6955, "f05": 0.9774625995676499, "t": 0.68, "d": 0.002835682078800139}
{"step": "score stack2 (+ranks)", "s": 8.1, "peak_rss_mb": 6955, "f05": 0.9774606943026453, "t": 0.69, "d": 0.0028337768137954855}
{"step": "band [0.0005, 0.02)", "s": 8.2, "peak_rss_mb": 6955, "pairs": 251768, "d": -6.86519518038331e-05, "t": 0.74}
{"step": "band [0.02, 0.3)", "s": 8.1, "peak_rss_mb": 6955, "pairs": 110845, "d": 0.00014353790238053055, "t": 0.74}
{"step": "band [0.3, 0.9)", "s": 8.0, "peak_rss_mb": 6955, "pairs": 58595, "d": 0.002608434950490679, "t": 0.7}
{"step": "band [0.9, 0.9995]", "s": 8.0, "peak_rss_mb": 6955, "pairs": 309971, "d": 0.0001562341963482483, "t": 0.74}
{"step": "rescue", "s": 0.3, "peak_rss_mb": 6955, "cfn_pairs": 1039, "cfn_recs": 1039, "cfn_stackable_pairs": 1014, "cfn_rescued_pairs": 0, "cfn_rescued_recs": 0, "cfn_recs_wrong_s1_new": 0, "cfn_recs_wrong_s1_td": 1, "new_fp": 1279, "new_fp_cfn_recs": 0, "new_fp_by_band": {"0": 89, "1": 347, "2": 841, "3": 2}, "fixed_fp": 2550, "lost_tp": 2425, "gained_tp": 8361, "new_fp_cost": -0.0009625691677196402, "rescue_gain": 0.0, "t_new": 0.69}
```
