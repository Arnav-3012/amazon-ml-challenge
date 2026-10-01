# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 (S1) business, this pipeline finds the Source 2 / Source 3 records that describe the same
real-world business. An S1 can have zero, one or several matches. It reads only the provided train/test TSVs
and uses no external data and no pretrained model.

It regenerates both submission files:
- `output/candidate_pairs.tsv`: the blocking candidate set. The matcher scores exactly these pairs.
- `output/matching_results.tsv`: one row per test S1, with S2/S3 ids.

Final run: public LB **0.971** macro F0.5; local OOF at test density **0.9806** (t = 0.76).

## Layout

Paths are relative to the **zip root**, which is the parent of `code/`:

```
<zip root>/
  dataset/train/*.tsv, dataset/test/*.tsv     ← place the challenge data here (not shipped)
  output/                                     ← both submission TSVs are written here
  artifacts/, oof/, models/                   ← intermediates, created automatically
  code/business_entity_resolution/
    configs/config.yaml   every hyper-parameter, path and the seed (42)
    requirements.txt      pinned dependencies
    src/                  pipeline (run as `python -m src.<module>` from this folder)
```

`src/io.py` resolves the zip root as two levels above this folder, so commands work from any machine. All TSVs
are read with `sep="\t", dtype=str, keep_default_na=False`.

## Environment

- Python 3.11. Tested on macOS (Apple M4 Pro, 24 GB RAM), CPU only.
- Peak memory in the final run's logs was 13.5 GB (`gate --apply`). Logged step times: `block_r --split test`
  41 min, `cv_full --fast` about 1 h, `predict --folds` 39 min.

```bash
# from the zip root
uv venv --python 3.11 .venv                      # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt
brew install libomp                              # macOS only: OpenMP runtime needed by LightGBM
source .venv/bin/activate
```

`torch`, `transformers`, `sentence-transformers`, `accelerate` and `datasets` are pinned only for the
experimental probes (`e0.py`, `ce_probe.py`). The reproduction steps below do not import them.

## Reproduce end-to-end

Run from `code/business_entity_resolution/`, in this order:

```bash
cd code/business_entity_resolution

# 1. Normalise all six source files (transliteration, legal forms, lexicons, skeletons)
python -m src.normalise                       # both splits

# 2. Lexical blocking channels A/B/C/X (inverted indexes, per country) -> rank grid, IDF, v1 candidates
python -m src.block --split train
python -m src.block --split test

# 3. Token maps mined from train pairs (abbreviations, transliterations, locality aliases)
python -m src.mine_dict

# 4. Channel R (char-3gram TF-IDF) for records weakly served by X
python -m src.block_r --split train
python -m src.block_r --split test

# 5. Candidate gate: pool channels, rank per S1, cap at 45 (variant v1r)
python -m src.gate --split train
python -m src.gate --split test
python -m src.gate --eval                     # train: asserts blocking F0.5 ceiling >= 0.993 (shipped: 0.9946)
python -m src.gate --apply                    # writes ../../output/candidate_pairs.tsv (77,949,788 pairs)

# 6. Pair features (93 columns)
python -m src.features --split train
python -m src.features --split test

# 7. 5-fold S1-dropout LightGBM + OOF threshold (writes models/fold_{0..4}.txt, oof/cv_full.json)
python -m src.cv_full --fast                  # the final submission used --fast (lr 0.1, <=1500 rounds)

# 8. Score test with the 5 fold models (mean p), decide, write the submission
python -m src.predict --folds                 # writes ../../output/matching_results.tsv
```

Optional checks used in the final run, none of which change any output: `python -m src.cv_full --check` and
`python -m src.cv_full --check --split test` recompute the context features and compare them with the stored
ones. `python -m src.decide --eval` prints the OOF summary per country.

Then validate from the zip root:

```bash
cd ../..
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

`utils/validate_submission.py` is the organisers' validator and is not part of this folder. `predict.py`
asserts the same invariants itself: every test S1 gets one row, the scored pairs equal the candidates, and the
matches are a subset of the candidates.

## Determinism

The seed is 42, set in `configs/config.yaml` and used for every sampling step and fold split. LightGBM runs
with `deterministic: true` and `force_row_wise: true`. Every step is a plain module with a CLI; there is no
notebook code.

## What each step does

| Step | Module | Key settings (`config.yaml`) |
|---|---|---|
| Normalise | `normalise.py` | anyascii transliteration; legal form split into its own column; per-country lexicons (the France map was mined from unlabelled test text only) |
| Block | `block.py` | `blocking.df_cap` 1000, `m` 5, `k` 10; channel X = composite `name\|addr` skeleton tokens |
| Token maps | `mine_dict.py` | mined on 80% of train S1s; the rest is held out to check the lift |
| Channel R | `block_r.py`, `tfidf.py` | bottom `r_query_pct` = 10% of records by X score, top 3 S1s |
| Gate | `gate.py` | `gate.m` 10, `pool_k` 30, `n` 45, `variant` v1r |
| Features | `features.py` | string similarity, IDF overlap, numbers, legal form, blocking ranks, record-relative margins; **no country feature** |
| Matcher | `cv_full.py`, `train.py` | GroupKFold(5) by S1, 19% S1-dropout per fold (test density), hard-negative sampling |
| Decide | `predict.py`, `decide.py` | per record argmax S1, keep if p ≥ t from OOF (b) (t = 0.76) |

`metric.py` is an exact local copy of the leaderboard metric (`python -m src.metric` runs its self-test).
`sibling.py` is an optional gate extension, off in the config. `train.py` also has a single-model mode, which
earlier submissions used. Each module has a self-test or a `--smoke` mode where that was practical:
`python -m src.block --selftest`, `python -m src.phonetic`, `python -m src.normalise --smoke 20000`.
