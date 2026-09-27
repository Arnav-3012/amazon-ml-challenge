# Phase 3 / Workstream C: neural retrieval feasibility (27 Sep, 14:23) — DROPPED at the gate

**Setup**
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M), pre-trained with no fine-tuning, on MPS.
- **Environment:** `~/p3venv` on the internal APFS drive (torch 2.14, sentence-transformers 6.1, faiss-cpu 1.15).
- **Text:** normalised core name + " | " + normalised address tokens. A raw-text variant was worse in the smoke run (India miss top-25: 50.5% vs 57.0% at a 5k pool).
- **Queries:** per country, 5,000 sampled train true pairs missed by the v1 candidates (+ 2,000 hit pairs as a control), seed 42. The query is the S1.
- **Pool:** the partners + N random same-country S2/S3 records. Recall@k = the partner is in the S1's top-k by cosine (FAISS inner product).
- **Script:** `experiments/retrieval_probe.py`; raw numbers in `experiments/retrieval_probe.json`.

## Results (recall %, missed pairs = the ones blocking lost; hit = control)
| country | pool | kind | @10 | @25 | @50 |
|---|---|---|---|---|---|
| India | 50k | miss | 32.2 | 37.5 | 41.9 |
| India | 500k | miss | 23.8 | **27.7** | 29.9 |
| India | 500k | hit (control) | 79.6 | 81.2 | 83.0 |
| US | 50k | miss | 48.4 | 55.5 | 61.5 |
| US | 500k | miss | 32.6 | **39.4** | 44.6 |
| US | 500k | hit (control) | 90.5 | 91.8 | 93.0 |

- **Throughput:** 766 (India) / 848 (US) texts/s at batch 128, so ~805/s overall. All train + test texts = 24,229,173, which takes **8.4 h** to embed.
- **Memory:** 7.4 GB RSS, 8.4 GB peak footprint, after switching to lazy loading. The first version peaked at 17.6 GB.

## Gate (16:30 rule: 500k top-25 > 30% AND the full run packaged by 22:00): FAIL
- **Recall:** India 27.7% < 30%; US 39.4% passes. Real per-country pools are 4.1–6.2M records (8–12× the 500k pool), and recall fell 10–16 points from 50k to 500k, so full-scale top-25 recall would be lower still.
- **Time:** 8.4 h of embedding alone, before FAISS search, features for the new pairs, scoring and packaging. It cannot finish by 22:00.
- **Control:** even pairs the lexical blocking already finds are only ~81% (India) / ~92% (US) retrievable at top-25. The pre-trained encoder is weaker than the lexical channels on this data, so its value is only as a complementary channel.

## What would make it work (see docs/roadmap.md §1)
- **Fine-tuning** on our 7.6M matched pairs, with hard negatives from the v1 candidates. Step 0 showed India's misses are mostly cross-script names whose addresses match, which a pre-trained paraphrase model is not trained for.
- **A GPU** (RTX 3060: an estimated ~6–8k texts/s, so ~1 h for 24M texts), with FAISS IVF per country.
- **Rebuilt features over the union** of old and new pairs (competitor-relative features need every candidate of a record), then a stage-1/2 retrain.
