# Plan (current only; history lives in phases.md / logs.md)

## Pipeline stages
normalise → block (writes `output/candidate_pairs.tsv`) → features → GBM → decide (writes
`output/matching_results.tsv`) → validate

## Milestones
| ID | Milestone | Done when |
|---|---|---|
| M0 | Scaffold (repo, env, docs) | tree + venv; git init/commit deferred until the user's GitHub repo exists |
| M1 | Loader + exact local macro-F0.5 scorer + all-empty baseline + official validator PASS on a dummy output | all-empty score = singleton rate; validator PASS |
| M2 | EDA (`eda-fe-entity-resolution-hackathon` skill) → answer open Qs → breakdown Revision 2 | open Qs in breakdown answered with numbers |
| M3 | Blocking v1: char-TF-IDF name + address channels + cheap keys | PC/RR reported per country |
| M4 | GBM v1 + GroupKFold OOF + global threshold | **FIRST REAL SUBMISSION** |
| M5 | Decision layer: expected-F0.5 prefix selection + singleton handling + assignment constraint | OOF macro F0.5 beats M4 |
| M6 | LOCO CV + France hardening (French rules, per-country IDF, relative features) | LOCO gap reported, narrowed |
| M7 | Multilingual encoder channel (zero-shot) in blocking + as GBM feature | PC and OOF F0.5 deltas reported |
| M8 | Package: clean-run repro from raw, validator PASS, methodology doc | zip builds from a clean run and contains ONLY `output/`, `code/business_entity_resolution/`, `Documentation_template.md` (no CLAUDE.md, docs/, utils/, artifacts) |

- **MVP** = M0–M5 + M8. **Stretch** = M6–M7.

## Target (set 2026-09-25, after the M4 loss decomposition)
- **Beat the leader (0.986955), stretch 0.99** test macro F0.5, judged on local CV (the private LB decides).
- Budget arithmetic: 0.99 means total loss ≤ 0.010. The best blocking grid measured so far (ceiling 0.9924) already
  costs 0.0076, which leaves the matcher ≤ 0.0024 (it loses 0.0217 today). So 0.99 needs a new recall source in
  blocking, not only a better matcher.
- **Second objective (organiser update): smallest candidate set per S1.** Output size is ranked, so recall must
  come from a wider INTERNAL retrieval followed by a cheap learned pruner (blocking outputs only, no string
  features), never from a wider output. Measured on 20% OOF: pruning 31.3 → 4.5 cand/S1 with the full model's p
  costs 0 F0.5 (upper bound for the pruner).
- Order: M4 submit → cheap pruner (gate: smallest set with F0.5 loss ≤ 0.001) → stage-2 matcher with S1 context +
  learning curve → wider internal retrieval behind the pruner → encoder (M7) as a retrieval channel AND a
  cross-encoder on uncertain pairs, fed to stage-2. Encoder moves from "cut if short" to in-scope.
- **Cut if short (first cut first):** AWS sweeps, cross-encoder, encoder fine-tune (zero-shot encoder channel stays).

## Compute
- Local M4 Pro is primary.
- AWS ($200 credits, spot CPU) only for parallel CV/param sweeps once the M4 pipeline is stable.
- GPU burst (Kaggle/Colab first, SageMaker spot as fallback) only if the zero-shot encoder loses to
  the GBM text features, or test embedding takes more than ~20 min locally.
- No endpoints, no Bedrock or any hosted LLM (external-API rule).
- The same entrypoint runs everywhere; no notebook-only code.
