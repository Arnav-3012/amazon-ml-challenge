# Business Entity Resolution — Amazon ML Challenge 2026

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-2E8B57)
![Polars](https://img.shields.io/badge/Polars-1.44-CD792C?logo=polars&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/sentence--transformers-6.1-FFD21E?logo=huggingface&logoColor=black)
![Apple Silicon](https://img.shields.io/badge/M4_Pro-24GB-000000?logo=apple&logoColor=white)

![Public LB](https://img.shields.io/badge/public_LB_(macro_F0.5)-0.971-2F81F7)
![OOF](https://img.shields.io/badge/local_OOF-0.9746-2F81F7)
![Candidates](https://img.shields.io/badge/candidates%2FS1-%E2%89%A445-2F81F7)
![Blocking ceiling](https://img.shields.io/badge/blocking_ceiling-0.9946-2F81F7)
![Scale](https://img.shields.io/badge/train_S1_entities-2.2M-2F81F7)
![External data](https://img.shields.io/badge/external_data-none-555555)

Links business records from three vendors that share **no common ID**. For every clean Source 1 business,
the pipeline finds every Source 2 / Source 3 record that describes the same real-world business
(zero, one or many), using only `business_name`, `business_address` and `country`.

**Final public leaderboard: 0.971 macro F0.5** (first submission 0.957).

| Submission | What changed | Local OOF | Public LB |
|---|---|---|---|
| M4 | Blocking v1 + LightGBM matcher, trained on 20% of train | 0.9654 | 0.957 |
| Submit #1 | Full-data 5-fold matcher, test-density CV, 4 new feature groups | 0.9746 | 0.967 |
| Final | Further blocking / normalisation / matcher work (below) | — | **0.971** |

---

## The problem

- **Scale:** 2.2M S1 businesses in train, 1.7M in test, and several million noisy S2/S3 records per split.
  Comparing every pair is impossible (about 10¹³ pairs).
- **Noise:** abbreviations, legal-suffix drift (`Pvt Ltd` / `Private Limited`), DBA names, typos, word swaps,
  transliteration (about 25% of Indian S2/S3 text is in native script), partial addresses, landmark addresses
  ("Near SBI ATM").
- **Distribution shift:** train covers the US and India. **Test adds France, which appears in no training data.**
- **Metric:** macro F0.5 per S1 entity, which counts precision twice as much as recall. An S1 with no true
  match scores 1.0 only when its prediction is empty, so one wrong match costs that entity its whole score.
- **Second objective:** the organisers also rank the size of the candidate set, so a smaller candidate set per S1
  is better.
- **Rules:** no external data, APIs or geocoding. Models must be MIT or Apache-2.0 licensed with ≤ 8B parameters.

## Pipeline

```
raw TSVs
  │
  ▼
1. normalise   ── ASCII transliteration, legal-form split, per-country lexicons, phonetic skeletons
  │
  ▼
2. block       ── inverted-index channels A/B/C + composite name×address channel X
  │               + channel R (char-3gram TF-IDF, only for records weakly served by X)
  ▼
3. gate        ── pre-score gate, capped at ≤ 45 candidates per S1  →  output/candidate_pairs.tsv
  │
  ▼
4. features    ── ~80 pair features: string similarity, IDF overlap, legal form, blocking ranks,
  │               record-relative features (margin vs the best competing S1)
  ▼
5. matcher     ── LightGBM, 5-fold GroupKFold by S1, trained in an "S1-dropout" world that matches test density
  │
  ▼
6. decide      ── each record goes to its argmax S1, kept only above a threshold tuned on OOF
                  →  output/matching_results.tsv
```

### 1. Normalisation
- Everything is mapped to ASCII-transliterated Latin (`anyascii`); the raw text is kept for the encoder experiments.
- The legal form (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, …) moves into its own column instead of being deleted, so
  `Acme Inc` and `Acme LLC` do not merge.
- Every rule was checked against a **precision guard**: how many exact collisions between non-matching pairs it
  creates. Three intuitive rules (`&`→`and`, stripping landmarks, stripping `(India)`) lost on that test and
  were reverted.

### 2–3. Blocking and the candidate gate
- Retrieval uses inverted indexes: per-country IDF with a document-frequency cap, record-centric top-m ∪
  S1-centric top-k, and chunked sparse products with an exact memory bound.
- **Channel X** scores name and address evidence together with composite `name|addr` skeleton tokens. This halved
  the number of candidates at equal recall, because 38–50% of S1 names are shared with another S1 and only the
  address can break those ties.
- The blocking budget was chosen by the **perfect-matcher F0.5 ceiling**, not by pair recall, because the metric
  is per entity.
- **Channel R** (char-3-gram TF-IDF over name and address together) runs only on the bottom 10% of records by
  their channel X score. It recovers pairs that the lexical channels miss.
- The **gate** ranks the pooled candidates of each S1 and keeps at most 45. This lifted the blocking ceiling from
  0.9873 to 0.9946 at a small candidate set.

### 4–6. Matcher and decision
- **LightGBM** binary classifier (127 leaves, lr 0.05, early stopping), trained with hard-negative sampling.
- **S1-dropout CV:** train has 4.68 records per S1 and test has 5.75. Dropping 19% of train S1s per fold gives
  4.68 / 0.81 = 5.78, so the local CV sees the same "orphan record" pressure as test.
- **Country is never a feature.** France only appears in test, so a country feature would be out of
  distribution there.
- **Decision rule:** each S2/S3 record is assigned to its single best S1 (a many-to-one match), and a global
  threshold tuned on OOF decides whether to keep it.

## Tech stack

| Area | Tools |
|---|---|
| Language / env | Python 3.11, `uv` |
| Data processing | Polars (lazy + streaming), PyArrow / Parquet, NumPy, SciPy sparse |
| String matching | RapidFuzz, `anyascii`, custom phonetic skeletons, TF-IDF (scikit-learn) |
| Model | LightGBM (deterministic mode) |
| Transformers (experiments) | sentence-transformers, `intfloat/multilingual-e5-small` (retrieval probe), `cross-encoder/ms-marco-MiniLM-L-6-v2` (fine-tuned reranker probe), PyTorch MPS |
| Hardware | MacBook M4 Pro, 24 GB unified memory, CPU only for the main pipeline |

All dependencies are pinned in [`code/business_entity_resolution/requirements.txt`](code/business_entity_resolution/requirements.txt).

## What worked, and what didn't

Every idea was measured before it went into the pipeline. The full log is in
[`docs/decisions_mistakes.md`](docs/decisions_mistakes.md).

**Worked**
- The joint name×address channel instead of ranking each channel alone.
- The S1-dropout CV world, which matches test density.
- Four feature groups (A/B test, seed std ≈ 0.00007): +0.0097 OOF together, although two groups were below the
  bar on their own.
- Fine-tuned cross-encoder stacked on the GBM probability for uncertain pairs: +0.0028 OOF on fold 0.

**Measured and dropped**
- Deeper lexical blocking: at most +0.001 ceiling for +28 candidates per S1. 61% of misses are absent from every
  lexical channel.
- Exact-key channel: 0.37% precision.
- Domain-name segmentation and generator inversion: 81% of names cannot be reached by rewrite operations.
- Self-training / pseudo-labelling, using India as a stand-in for France (LOCO).

**Where the remaining score is lost**
- A probe submission with France left blank split the leaderboard by country: **France ≈ 0.943, US + India ≈ 0.969.**
  Most of the remaining gap comes from the unseen country, which fits the challenge design.
- OOF scores ran about 0.008 above the public LB throughout, so leaderboard gains were judged on local CV.

## Repository layout

```
code/business_entity_resolution/
  configs/config.yaml     all hyper-parameters, paths and the seed (42)
  src/                    pipeline modules (run as python -m src.<module>) + diagnostic probes
  requirements.txt
docs/                     design notes, generated evaluation reports, work log, decisions & mistakes
utils/validate_submission.py   official submission validator
output/                   submission files (gitignored)
dataset/                  challenge data (not included)
```

## Reproduce

The challenge data is not redistributed. Place it at `dataset/train/*.tsv` and `dataset/test/*.tsv`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt
brew install libomp          # macOS: required by LightGBM

cd code/business_entity_resolution
python -m src.normalise                                  # both splits
python -m src.block   --split train && python -m src.block   --split test
python -m src.mine_dict                                  # token maps (train only)
python -m src.block_r --split train && python -m src.block_r --split test
python -m src.gate --split train && python -m src.gate --split test
python -m src.gate --eval && python -m src.gate --apply  # writes output/candidate_pairs.tsv
python -m src.features --split train && python -m src.features --split test
python -m src.cv_full --fast                             # 5-fold OOF + threshold (final run used --fast)
python -m src.predict --folds                            # writes output/matching_results.tsv

cd ../.. && python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Peak memory is about 10–18 GB on the full data. Runs are deterministic (seed 42, LightGBM `deterministic: true`).

## Notes
- The France token map in `normalise.py` was mined from unlabelled test text (no labels, no external data).
- Built in about 72 hours for the Amazon ML Challenge 2026.
