# Business Entity Resolution — Amazon ML Challenge 2026

Links every Source 1 (S1) business to its matching Source 2 / Source 3 records (possibly none). Scored by macro
F0.5 per S1 entity. Pipeline: normalise → lexical blocking (IDF inverted index, 4 channels) → 78 pair features →
LightGBM (5-fold GroupKFold by S1, S1-dropout "test-density" OOF) → per-record argmax + one global threshold.

## Environment
- Python 3.11, [uv](https://github.com/astral-sh/uv); all dependencies pinned in `requirements.txt`
  (MIT / BSD / Apache-2.0 / MPL-2.0; no pretrained models, no external data or APIs).
- `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt`
  (Windows: `.venv\Scripts\python.exe`). macOS: LightGBM needs `libomp` (`brew install libomp`).
- Deterministic: every seed comes from `configs/config.yaml` (`seed: 42`); LightGBM runs with `deterministic: true`.
- Memory: the 15.7 GB Windows machine that produced the submission ran every step below; peaks are listed per step.
  Memory knobs (`blocking.nnz_budget/s1_range/...`, `features.chunk_rows/prep_batch`) change memory only, never output.

## Data
Place the challenge files at the repo root: `dataset/train/{train_source1,train_source2,train_source3,train_gt}.tsv`
and `dataset/test/{test_source1,test_source2,test_source3}.tsv` (paths in `configs/config.yaml` → `paths`).
Every TSV is read with `sep="\t", dtype=str, keep_default_na=False`. Intermediate files go to `artifacts/`,
models to `models/`, OOF predictions to `oof/`, submission files to `output/`.

## Reproduce end-to-end (from `code/business_entity_resolution/`)
Times / peak memory measured on the 12-core, 15.7 GB Windows machine (2026-09-25..27).

| # | command | writes | time / peak |
|---|---|---|---|
| 1 | `python -m src.normalise` | `artifacts/interim/norm_{train,test}_s{1,2,3}.parquet` | 19 min / 9.5 GB |
| 2 | `python -m src.block --split train` | `blockgrid_train_cap1000`, `idf_train` | 68 min / 7.5 GB ¹ |
| 3 | `python -m src.block --split test` | `blockgrid_test_cap1000`, `idf_test` | 29 min / 7.8 GB ¹ |
| 4 | `python -m src.block --split train --finalize` | `candidates_train.parquet` (69,043,101 pairs) | 2 min |
| 5 | `python -m src.block --split test --finalize` | `candidates_test.parquet` (60,901,445 pairs) + **`output/candidate_pairs.tsv`** | ~3 min |
| 6 | `python -m src.features --split train` | `artifacts/features/train/part-*.parquet` (80 cols) | 44 min / 10.9 GB |
| 7 | `python -m src.features --split test` | `artifacts/features/test/part-*.parquet` | 39 min / 10.5 GB |
| 8 | `python -m src.cv_full` | `models/fold_{0..4}.txt`, `oof/oof_full.parquet`, `oof/cv_full.json` | 14.1 h / 11.4 GB ws, 17.2 GB committed |
| 9 | `python -m src.predict --folds --models 1` | **`output/matching_results.tsv`**, `oof/test_p.parquet` | 2.66 h / 9.3 GB |
| 10 | `python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test` (from the repo root: drop the `../../`) | PASS | — |

¹ Measured with a memory-reduced `block.py` (same output, verified identical) that was lost before packaging; the
shipped `block.py` produces the same grid but was not re-run on this machine and needs more RAM (`blocking.nnz_budget`
/ `s1_range` are its memory knobs; lower them on a small machine). Its test grid is built at m ≤ 10 / k ≤ 30 (≥ the
v1 cut 5 / 10, so step 5 is unaffected); step 4 was checked identical to the train candidates actually used.

Notes
- Step 9: `--models 1` scores test with `fold_0` only, at the OOF (b) best threshold from `oof/cv_full.json`
  (0.75). OOF p is itself one fold model per row, so the threshold transfers directly. `--folds` without
  `--models` averages all 5 fold models (~15 h on the machine above; not used for the submission for time).
- Steps 4-5 are the v1 candidate set (channel X, record-centric top-5 S1 per record ∪ S1-centric top-10 records
  per S1 and source). `src.gate --apply` writes a wider v2 set that was explored and NOT submitted.
- `output/candidate_pairs.tsv` is exactly the set the model scores; `src.predict` asserts every match is in it
  and that no record is matched to more than one S1.
- Checks: `python -m src.block --selftest`, `python -m src.features --selftest`, `python -m src.cv_full --check`.
- Country is used only as a partition key for IDF / retrieval (true pairs never cross countries) and is never a
  model feature (`train.feature_cols` asserts it); test contains an unseen country (France).

## Layout
- `src/io.py` paths, TSV I/O, memory/timing log · `src/normalise.py` + `src/phonetic.py` text normalisation
- `src/block.py` blocking (channels A/B/C/X, rank grid, `--finalize` candidates)
- `src/features.py` pair features · `src/train.py` LightGBM helpers · `src/cv_full.py` 5-fold stage-1 + OOF
- `src/decide.py` decision rule (argmax per record + global t) · `src/metric.py` macro F0.5 · `src/predict.py` test scoring
- Analysis / diagnostics only (not needed to reproduce the submission): `eda`, `noise_ops`, `normalise_eval`,
  `block_eval`, `block_diag`, `block_autopsy`, `gate`, `sibling`, `composition_depth`, `diagnose`, `eyeball`,
  `mine_dict`, `ab_test`, `stage2`, `baseline_empty`.
