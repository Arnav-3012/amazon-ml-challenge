# Business Entity Resolution — Amazon ML Challenge 2026

Links every Source 1 business to its matching Source 2 / Source 3 records (possibly none).
Scored by macro F0.5 per S1 entity.

## Environment
- Python 3.11, [uv](https://github.com/astral-sh/uv)
- `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt`
- macOS: LightGBM needs `libomp` (`brew install libomp`)
- Tested on a 16 GB Apple M4. Memory knobs live in `configs/config.yaml` (`features.chunk_rows`, `cv_full.max_rss_mb`,
  `stage2.p_floor`); they change memory, not results.

## Data
Expects `dataset/train/*.tsv` and `dataset/test/*.tsv` at the submission root (next to `code/`). All TSVs are read with
`sep="\t", dtype=str, keep_default_na=False` (no quoting). No external data is used. Every path and the seed (42)
are in `configs/config.yaml`; intermediate files go to `artifacts/`, models to `models/`, OOF predictions to `oof/`.

## Blocking (candidate generation)
`src.normalise` (rule-based normaliser), then `src.block` (within-country inverted-index retrieval over IDF-weighted
name, address and phonetic tokens plus composite name×address keys). Then `src.gate --apply-v1` cuts the rank grid
to channel X: record-centric top-5 S1s ∪ S1-centric top-10 records per source. It writes `output/candidate_pairs.tsv`,
which is exactly the set the matcher scores: 60,901,445 test pairs, 35.15 per S1.

## Matching
- `src.features`: 80 pair features per candidate (name, address, address ambiguity, blocking, competitor-relative).
- `src.cv_full`: stage 1, a LightGBM with 5-fold GroupKFold by S1 and S1-dropout worlds that match test density.
- `src.stage2`: a stacked LightGBM on stage-1 out-of-fold p, reading each pair's neighbourhood.
- Decision: each S2/S3 record goes to its argmax S1 if p ≥ one global threshold, chosen on test-density OOF.

## Output
`output/matching_results.tsv` (one row per test S1, empty list allowed) and `output/candidate_pairs.tsv`. Check both with
the official validator (`utils/validate_submission.py` from the challenge's student_resource):
```
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Reproduce end-to-end
Run from `code/business_entity_resolution/`, in order:
```
python -m src.normalise                          # normalised caches (train + test)
python -m src.block --split train                # rank grid + IDF (train)
python -m src.block --split test                 # rank grid + IDF (test)
python -m src.gate --apply-v1                    # v1 candidates (X, m=5, k=10) -> output/candidate_pairs.tsv
python -m src.mine_dict                          # transliteration/abbreviation token dictionary (train only)
python -m src.features --split train
python -m src.features --split test
python -m src.cv_full                            # stage 1: 5-fold models + OOF, best t
python -m src.predict --folds --cascade 0.02     # variant A -> output/matching_results.tsv (+ test p)
python -m src.stage2 --only full                 # stage 2: 5-fold models + OOF
python -m src.stage2 --predict --ignore-gate     # submission B -> output/matching_results.tsv
```
Approximate wall time on a 16 GB M4: features 45 min, stage 1 4 h, test scoring 1.5 h, stage 2 30 min.
Without `--ignore-gate`, `stage2 --predict` refuses unless stage 2 beats stage 1 by `stage2.keep_gain` (0.002) on the
test-density OOF. Submission B (+0.0011) is written with `--ignore-gate`.
