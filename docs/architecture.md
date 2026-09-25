# Architecture: path map
Format: `path | purpose | what to check when something related breaks`. Update this whenever a path is added.

- `CLAUDE.md` | agent rules (safety, hard challenge rules) | rule conflict → CLAUDE.md wins over docs
- `.gitignore` | keep data/outputs/venv out of git | file unexpectedly committed/ignored → patterns + `.gitkeep` negations
- `docs/context.md` | why: challenge, metric, rules, deliverables | spec question → the source RTF is authoritative
- `docs/breakdown.md` | technical breakdown + revisions | approach doubt → latest Revision section overrides leaves
- `docs/plan.md` | current milestones + compute policy | scope creep → is it in MVP (M0–M5, M8)?
- `docs/architecture.md` | this map | path missing here → add it
- `docs/phases.md` | per-phase status | phase status unclear → last entry
- `docs/logs.md` | chronological work log | "what changed when" → grep by date
- `docs/decisions_mistakes.md` | right calls / mistakes | repeating an error → check Mistakes first
- `docs/source/` | original PS (RTF), our breakdown, video transcript | never edit; spec disputes resolve here
- `code/business_entity_resolution/` | submission code root (copied into zip) | repro fails → run from repo root, paths in config
- `code/business_entity_resolution/src/` | pipeline package (empty at M0) | ImportError → `__init__.py`, run from repo root
- `code/business_entity_resolution/configs/config.yaml` | seed + all paths (relative to repo root) | FileNotFound → paths here vs cwd
- `code/business_entity_resolution/README.md` | reproduction instructions | reviewer can't reproduce → README commands vs actual entrypoint
- `code/business_entity_resolution/requirements.txt` | exact pinned deps (`uv pip freeze`) | version drift → re-freeze; macOS LightGBM needs `libomp`
- `dataset` | symlink → `/Users/arnav/Downloads/student_resource/dataset` (absolute, gitignored, 2.4 GB) | FileNotFound → was Downloads cleaned or the folder moved? Relink with `ln -sfn <new>/dataset dataset`
- `utils/validate_submission.py` | official validator, byte-identical copy of `student_resource/utils/` | never edit; re-`cmp` against the source if in doubt
- `Documentation_template.md` | methodology template (unfilled copy) for the zip | fill in at M8; the source copy stays in student_resource
- `output/` | `matching_results.tsv`, `candidate_pairs.tsv` (gitignored) | validator fails → candidates ⊇ matches, one row per test S1
- `artifacts/` | misc run artifacts, reports (gitignored) | stale results → delete and rerun
- `interim/` | normalised/parsed data caches (gitignored) | normaliser change not reflected → clear cache
- `features/` | pair feature tables (gitignored) | train/serve skew → same feature code for train and test
- `oof/` | out-of-fold predictions (gitignored) | overconfident thresholds → tuned on OOF, not in-fold?
- `models/` | trained models (gitignored) | wrong model loaded → filename/seed/config hash
- `.venv/` | uv venv, Python 3.11 (gitignored) | ImportError → `.venv/bin/python`, reinstall from requirements.txt
