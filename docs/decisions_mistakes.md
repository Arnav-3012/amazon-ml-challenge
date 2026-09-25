# Decisions & Mistakes
Log wrong turns the moment they are identified. Format: `date | item | why`.

## Right calls
- 2026-09-25 | Repo mirrors the submission zip from day 1 | packaging = copy; no late restructuring risk
- 2026-09-25 | Country is never a model feature | test has unseen France; a country feature would be out-of-distribution
- 2026-09-25 | Zip = whitelist copy (output/, code/business_entity_resolution/, Documentation_template.md) | personal/agent docs (CLAUDE.md, docs/) must never ship
- 2026-09-25 | Read TSVs with `quoting=csv.QUOTE_NONE` | every line has exactly 4 tab fields, but ~800 lines carry CSV-style quotes; default quoting risks merging rows. Quote chars stay as text noise for the normaliser
- 2026-09-25 | Paths resolve from the repo root via `__file__`, not cwd | `python -m src.x` runs from code/business_entity_resolution while data/output sit at the root (same layout in the zip)
- 2026-09-25 | Canonical matching space = ASCII-transliterated Latin; raw kept for the encoder | S1 is ASCII while India S2/S3 is ~25% native script (Revision 1a)

## Mistakes
_(none yet)_
