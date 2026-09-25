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

## Blocking (candidate generation)
TBD. Writes `output/candidate_pairs.tsv`, which is exactly the set the matcher scores.

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
