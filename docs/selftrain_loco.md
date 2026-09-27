# Self-training LOCO probe (India as France proxy)

20% S1 sample, fold 0, fast params (lr=0.1, <=1500 rounds), test-density scorer.

## Scores (India, macro F0.5)
| round | setup | macro_f05 | best_t | n_s1 |
|---|---|---|---|---|
| 1 | US only | 0.85428 | 0.61 | 143557 |
| 2 | US + pseudo-India (round 1, w=0.5) | 0.86574 | 0.76 | 143557 |
| 3 | US + pseudo-India (round 2 relabel, w=0.5) | 0.87885 | 0.87 | 143557 |

Round 1 -> 2 delta: **+0.01146** (gate +0.02000) -> **DO NOT ADOPT** for France.
Round 2 -> 3 delta: +0.01311.

## Pseudo-label quality vs true India labels
| round | class | n | precision | recall | n_true_total |
|---|---|---|---|---|---|
| 1 | positive (p>=0.98, argmax, no ambiguous rival) | 19352 | 0.9578 | 0.0394 | 471015 |
| 1 | negative (p<=0.02, 3:1 sampled) | 58056 | 0.9973 | 0.0143 | 4062958 |
| 2 | positive (relabel, round-2 model) | 23356 | 0.9419 | 0.0467 | 471015 |
| 2 | negative (relabel, round-2 model) | 70068 | 0.9979 | 0.0172 | 4062958 |

## Rule
- positive: p >= 0.98, argmax of its record, and no other candidate of the same S1 has name_tset >= 95 AND addr_tset >= 90 (ambiguity guard)
- negative: p <= 0.02, sampled 3:1 vs positives (seeded)
- pseudo-labeled rows enter training at weight 0.5, alongside full-weight US rows

## Notes
- India stands in for France (both LOCO-country proxies); grading uses TRUE India labels for diagnosis only -- never fed into training.
- Precision/recall above are read straight off `docs/selftrain_loco.md`'s generating run (`oof/selftrain_loco.json` has the exact floats); rerun to refresh both.
- Gate: adopt self-training for France if round 1 -> round 2 gains >= +0.02 macro F0.5 on India.
