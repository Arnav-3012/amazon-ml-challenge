# CLAUDE.md — Amazon ML Challenge 2026 (Business Entity Resolution)

Repo root = final submission zip root. Details live in `docs/`; this file holds the rules only.

## Safety
- Never run `git commit`, `git push` or any publishing command yourself. Give the user the commands; the user runs them.
  Remote: https://github.com/Arnav-3012/amazon-ml-challenge (branch `main`).
- Never install, delete or modify anything outside what the current phase task requires.
- You write code; the user runs it. Never claim you ran, tested or verified anything unless a
  tool result in this session shows it.

## Working rules
- Every code change states: file(s) touched + the ONE command to verify it.
- Before proposing an approach to any sub-problem, read `docs/architecture.md`, `docs/plan.md`,
  `docs/breakdown.md`. Don't re-derive recorded decisions without new evidence.
- Style: lean-senior (answer first, baseline before complexity, no waste). Use problem-judgement
  for new sub-problems; strict-old-man L2 when evaluating a design choice.
- Before ending any turn with changes: append to `docs/logs.md` and `docs/phases.md`. Log wrong
  turns in `docs/decisions_mistakes.md` the moment they are identified.
- New path → add a line to `docs/architecture.md`.

## Challenge hard rules
- Read TSVs with `sep="\t", dtype=str, keep_default_na=False`. Empty ID lists must stay empty strings.
- No external data, APIs, lookups or geocoding. Only the provided train/test files.
- Pretrained/final models: MIT or Apache-2.0 license and <=8B params. Check the model card.
- `output/candidate_pairs.tsv` = exactly the set the model scores; matches ⊆ candidates.
- Every test S1 entity gets exactly one row; S2/S3 IDs only; no duplicates in a list.
- Never hard-code, filter or one-hot country, and never use it as a model feature (test has unseen France).
- Deterministic: seed from `code/business_entity_resolution/configs/config.yaml`; no notebook-only code.

## Map
- `docs/context.md`: why, metric, rules, deliverables
- `docs/breakdown.md`: technical breakdown + revisions
- `docs/plan.md`: current milestones
- `docs/architecture.md`: every path, its purpose, what to check
- `docs/phases.md`, `docs/logs.md`, `docs/decisions_mistakes.md`: history
- Env: `.venv` (uv, Python 3.11); deps pinned in `code/business_entity_resolution/requirements.txt`
