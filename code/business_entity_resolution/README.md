# Business Entity Resolution — Amazon ML Challenge 2026

Links every Source 1 business to its matching Source 2 / Source 3 records (possibly none).
Scored by macro F0.5 per S1 entity.

## Environment
- Python 3.11, [uv](https://github.com/astral-sh/uv)
- `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt`
- macOS: LightGBM needs `libomp` (`brew install libomp`)

## Data
TBD. Expects `dataset/train/*.tsv` and `dataset/test/*.tsv` at the repo root. All TSVs are read with
`sep="\t", dtype=str, keep_default_na=False`.

## Normalisation (`src/normalize.py`, `src/prep.py`)
One rulebook for every record (country is never used to route rules): anyascii transliteration and
accent folding, legal-form canonicalisation incl. OCR/transliteration variants (praivet -> pvt),
DBA/formerly/nee splitting, US/Indian state and street-type folding, France "(France)" marker
removal, consonant-skeleton keys for cross-script matching. `python -m src.prep --split all`
caches `interim/norm_v*_{split}_s{n}.parquet`.

## Blocking (candidate generation, `src/block.py`)
Two channels (name char-3-grams + skeleton; address words + skeleton) x two directions
(S1 -> top-k records; each record -> top-k S1), searched inside each country group with
cost-capped sparse top-k matrix products. `--mode pc` prints pair completeness per country,
channel, direction and k; `--mode full` writes `interim/cands_{split}.parquet`.
The final `output/candidate_pairs.tsv` is written by the matching stage and is exactly the set
the matcher scores.

## Matching
TBD. Pairwise features → GBM → per-entity set decision.

## Output
TBD. Writes `output/matching_results.tsv`, then run:
```
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Reproduce end-to-end
TBD (single entrypoint command).
