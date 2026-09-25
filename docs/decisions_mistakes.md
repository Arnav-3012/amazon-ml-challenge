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
- 2026-09-25 | eda.py first version had unvectorized `.apply(axis=1)`/dict-loop paths over 5M+ row files, only ever smoke-tested at ~400-row scale | wasted 27+ min of the user's wall-clock time before being killed; should have sanity-checked op-count (rows x per-row cost) against real file sizes before declaring it ready, not just checked "does it run on a tiny sample"
- 2026-09-25 | First eda.py "audit every hot path" pass missed a `list(2.2M-entry dict)` rebuilt inside a 200k-iteration while loop (C7 negative sampling) — the real bottleneck, ~33 min alone | should have grepped for list()/dict() calls inside loop bodies specifically, not just `.apply(axis=1)` patterns; caught only on the SECOND killed run, wasting another ~12 min of the user's time
