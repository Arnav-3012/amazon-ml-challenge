# Context: why this repo exists

Authoritative spec: `docs/source/Business Entity Resolution Challenge.rtf`. Video transcript:
`docs/source/6ab509c5b7036_ml_challenge_2026_video_eng.txt`. Technical breakdown: [breakdown.md](breakdown.md).

## Challenge
Amazon ML Challenge 2026, Business Entity Resolution. Business records arrive from 3 independent
vendors with no shared identifier. Only `business_name`, `business_address` and `country` exist.
For every Source 1 entity, output all Source 2/Source 3 records describing the same real-world
business (zero, one or many).

## Sources
- **S1** is the clean, de-duplicated reference list (Amazon Business sign-ups). One row = one entity.
- **S2 and S3** are noisy vendor fragments: abbreviations, legal-suffix drift, DBA/trade names,
  `&` vs "and", word transpositions, typos, transliteration, partial/reordered addresses, landmark
  references ("Near SBI ATM"), missing PIN/state.
- Source is given by the ID prefix (`S1-`/`S2-`/`S3-`) and the file. There is no source column.
- Country: train = US and India; **test adds France (unseen)**. Treat it as an open set of labels.

## Metric: macro F0.5 per S1 entity
`F0.5 = 1.25·P·R / (0.25·P + R)`, computed per S1 entity, then averaged over all S1 entities.
Precision counts twice as much as recall ("when in doubt, don't merge").

Worked example (from the PS): predict `[S2-00047, S2-00193, S3-00812]`, truth `[S2-00047, S3-00812]`
→ P = 2/3, R = 1 → F0.5 = (1.25·0.667·1)/(0.25·0.667 + 1) = **0.714**.

**Singleton rule:** an S1 entity with no true matches scores 1.0 for an empty prediction and 0.0
for any prediction. Singletons are in the average, so an all-empty submission scores exactly the
singleton rate.

## Outputs (both in `output/`, tab-separated)
- `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) is the only scored file,
  uploaded to the Portal.
- `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`) is not scored. It is
  **audited** for blocking recall and reduction ratio, and to verify the pipeline. It must be the
  final set the model runs inference on, not an earlier blocking pass. Matches must be a subset of
  candidates (the validator warns otherwise).
- Rules for both: one row per test S1 entity, empty list allowed, S2/S3 IDs that exist in test
  only, no duplicate IDs in a list, no duplicate S1 rows. Failing validation means the file isn't
  scored.
- Validator: `python3 utils/validate_submission.py --matching output/matching_results.tsv
  --candidate output/candidate_pairs.tsv --test-dir dataset/test`. It's stdlib only, prints PASS
  (exit 0) or a list of issues (exit 1), and does not compute the score.

## Final submission zip
```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   (filled in; .pdf also fine)
```
The code folder must regenerate both outputs from train/test data alone. Top teams' packages are
reproduced and reviewed before rankings are confirmed. This repo mirrors that layout.

## Methodology document
Must cover: methodology, candidate generation/blocking strategy, model architecture and feature
engineering, and anything else relevant. No page limit; favour clarity and technical depth.

## Rules
- **No external data:** no ER APIs, business registries, geocoding, or internet augmentation.
  Violating this means disqualification.
- **Model license:** the final model must be MIT or Apache-2.0 and <=8B params.
- Hosted LLMs/Bedrock count as external APIs, so they are not used.

## Leaderboard
The public LB uses a subset of test and the private LB uses the rest. Final ranking = private LB.
Always submit predictions for the full test set. Trust local CV over public-LB probing.

## Hardware & time
- M4 Pro, 24GB unified memory, MLX/MPS. Small encoders (~33M–120M params) only. GBM and TF-IDF are
  CPU-light.
- AWS ($200 credits, spot) only for parallel sweeps later. See [plan.md](plan.md).
- 72-hour challenge window: MVP first, stretch after.

## Update (2026-09-25): candidate set size is ranked
`candidate_pairs.tsv` is part of the final submission. Blocking must scale (no all-pairs comparison) and cut the
search space to a small candidate set per S1. Organisers review candidate_pairs.tsv and the code producing it; a
SMALLER candidate set per S1 ranks higher in the final evaluation, beyond the public/private LB score.
