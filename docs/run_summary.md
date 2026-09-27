# Pipeline run summary: Windows machine (15.7 GB RAM)

Status of the production rerun (block → features → train → decide → predict → validate) on this machine, with
the locked config (blocking X, m5/k10, df_cap 1000; matcher subset 0.2 / final 0.5). Only steps with real output
are filled in. Reference numbers come from the 24 GB machine's runs (docs/logs.md, docs/phases.md).

## Environment checks (2026-09-25)
| check | result |
|---|---|
| normalised caches `artifacts/interim/norm_*.parquet` | present; built before the M4 commit, but that commit changed only imports in normalise.py (rules identical), so the caches are valid |
| `src/io.py` on Windows | **was broken**: top-level `import resource` (Unix-only) crashed every module. Fixed with a win32 branch in `peak_rss_mb()` |
| `python -m src.phonetic` | PASS |
| `python -m src.block --selftest` | PASS (4 channels, 3 budgets, 171 fallback records) |
| stale inputs | `artifacts/interim/cand_{train,test}.parquet` are the old blocking.py output (not read by the current pipeline); `output/candidate_pairs.tsv` is stale until `block --split test` reruns |
| available RAM | 6.0 GB at first; **7.6 GB after closing apps** (background services hold ~6.8 GB of working sets). Page file 12 GB on D: → commit limit 29.5 GB, so an over-RAM step pages instead of failing |

## Memory gate (reference peak RSS vs what's available here)
| step | reference peak | fits in 6.0 GB? |
|---|---|---|
| `block --dry` | 8.1 GB | no |
| `block --smoke 300000` | 7.1 GB | no |
| `block` full (test) | 13.7 GB | no; needs ~14 GB free, or a lower `blocking.nnz_budget` (a memory knob only: output unchanged) |
| `features` | 11.5 GB | no |
| `predict` | 10.7 GB | no |

## block --dry (train), 2026-09-25
`python -m src.block --split train --dry` (log: `artifacts/logs/block_dry.log`). Finished all channels; only the
final summary print crashed (cp1252 console vs polars table borders: use `PYTHONIOENCODING=utf-8`).
**10 min 18 s (reference 85 s), peak 11.4 GB working set**. It paged heavily (~89k pages/s, 2.5 GB resident of
9.3 GB committed). The 8.1 GB reference predates channel X/composite tokens, so it isn't comparable.

| country | source | channel | records | zero-survivor % | with fallback % | product nnz upper bound |
|---|---|---|---|---|---|---|
| India | S2 | A | 2,017,799 | 28.26 | 14.85 | 99.4M |
| India | S2 | B | 2,017,799 | 22.10 | – | 175.0M |
| India | S2 | C | 2,017,799 | 9.33 | 0.00 | 222.6M |
| India | S2 | X | 2,017,799 | 0.76 | 0.00 | 1,322.1M |
| India | S3 | A | 2,115,547 | 20.09 | 9.54 | 122.2M |
| India | S3 | B | 2,115,547 | 30.55 | – | 154.6M |
| India | S3 | C | 2,115,547 | 8.48 | 0.00 | 220.1M |
| India | S3 | X | 2,115,547 | 0.82 | 0.00 | 1,313.8M |
| US | S2 | A | 3,016,817 | 12.62 | 3.30 | 189.8M |
| US | S2 | B | 3,016,817 | 18.25 | – | 193.0M |
| US | S2 | C | 3,016,817 | 7.41 | 0.00 | 306.2M |
| US | S2 | X | 3,016,817 | 0.54 | 0.00 | 1,174.2M |
| US | S3 | A | 3,170,056 | 12.79 | 3.30 | 199.6M |
| US | S3 | B | 3,170,056 | 17.11 | – | 209.6M |
| US | S3 | C | 3,170,056 | 7.30 | 0.00 | 326.5M |
| US | S3 | X | 3,170,056 | 0.50 | 0.00 | 1,318.2M |

Total nnz bound 7.55B (India 3.63B, US 3.92B). X (the selected channel) leaves ≤0.82% of records with no token
and 0% with fallback. Step times: India df pass 110 s, US df pass 195 s, survivors 55–75 s per source.

**Memory verdict:** 11.4 GB peak on 7.6 GB available made dry 7x slower. The full run (reference test 13.7 GB,
probably more now) will page harder. Waiting on the user before smoke/full run.

## Steps and outputs
| step | status | key numbers | reference |
|---|---|---|---|
| block --dry | **done** | 10m18s, peak 11.4 GB; X zero-survivor ≤0.82%; nnz bound 7.55B | 85 s, 8.1 GB (before channel X) |
| block --smoke 300000 | not run | | 29 s, 7.1 GB |
| block train + test | not run | | test 469 s, 13.7 GB |
| block_diag, block_eval | not run | | PC 96.31%, ceiling 0.9873, 31.3 cand/S1 |
| features (smoke, train, test) | not run | | peak 11.5 GB |
| train (CV), train --loco | not run | | OOF 0.9654; LOCO India −0.13, US −0.02 |
| train --final, decide | not run | | t = 0.80, 3,433 rounds |
| predict + validator | not run | | 60.9M test candidates, 5.68M matches |

## block.py memory rewrite (2026-09-26): output-identical, NOT yet run at full scale
Available RAM after restart: **7.27 GB**. Direction 2 (A/B/C score only X's pairs) rejected: it changes the A/B/C
feature columns and the eval grid (docs/decisions_mistakes.md). Direction 1 done as "same computation
resequenced": batched tokenise → disk cache, one channel at a time, row-batched survivors, rc/sc top-n frames on
disk, merge one s1 slice at a time, idf parts on disk. Knobs: token_batch 250k, merge_row_group 1M,
nnz_budget 10M (was 40M), s1_range 50k (was 100k), all output-neutral.

**Identity check (old vs new code, smoke 100k rows/file, forced many batches/slices/chunks):**
| output | rows | identical (values, incl. float scores, and row order) |
|---|---|---|
| candidates_train_smoke | 2,342,641 | yes |
| blockgrid_train_cap1000_smoke | 28,305,562 | yes |
| idf_train_smoke | 1,133,830 | yes |
| candidates_test_smoke (India, US, France) | 2,343,967 | yes |
| idf_test_smoke | 1,170,254 | yes |
| survivor stats (dry numbers) | 24 rows | yes |

`--selftest` passes, now also checking batched tokenise, cached counts and batched survivors against one pass.
Smoke peak: train 4.76 → 1.73 GB, test 2.47 → 1.34 GB.

**Full-scale peak estimate (largest group: US train, channel X).** M = measured at full scale; B = measured per
batch (bounded by a knob); U = upper bound from measured token counts; D = derived from row counts × dtype widths.

Highest point: X, source S3, S1-centric retrieval. Held at the same time:
| item | size | basis |
|---|---|---|
| X token table: 9.91M linkable tokens × ~72 B | ~0.7 GB | M (vocab, dry log) + D |
| S1 postings post1/post1_rare: 9.91M × 8 B × 2 | 0.16 GB | D |
| S1 index qw + qw_t: ≤ 45.6M X tokens × 8 B × 2 | ≤ 0.73 GB | U (as if every S1 token survives the cap) |
| S3 record matrix r + r.T: ≤ 104.1M × 8 B × 2 | ≤ 1.67 GB | U |
| one product chunk + top-n sort, nnz_budget 10M | ~0.5 GB | D (~50 B per product nnz) |
| sc frame being built: 1.32M S1 × k 60 × 14 B | 1.11 GB | D |
| ids + process baseline | ~0.5 GB | M (~0.3–0.4 GB baseline) |
| **total** | **≤ ~5.4 GB** | vs 7.27 GB available → ~1.9 GB headroom |

Other stages (all lower):
| stage | estimate | basis |
|---|---|---|
| tokenise one source (raw lists + 250k-row batch) | ≤ ~1.5 GB | B: batch transient 43–348 MB |
| X token counts + combine (3 count frames 0.73 GB + concat + group_by) | ≤ ~3.5 GB | M (count frames) + D |
| X survivors on S3 (table + S1 index + S3 list 0.96 GB + batch 0.53 GB + kept ≤ 1.35 GB) | ≤ ~4.5 GB | M + B + U |
| merge, 50k-S1 slice | ~1–1.5 GB | D |
| test only: write_candidates group_by over ~61M pairs (code unchanged) | ~3–5 GB | D, **not measured**: the largest remaining uncertainty |

Before (for comparison): the old dry peaked at 11.4 GB (measured). The old full run held all 8 rc/sc frames of a
source (~6 GB for US S3 train, D) on top of that.

**Proposed confirmation before the full run:** the new `block --split train --dry` at full scale. Predicted peak
≤ ~4.5 GB with no paging (it runs tokenise, counts and survivors, i.e. the rows above without products). If it
goes over ~5.5 GB, stop and re-examine before the full run.

## New `block --split train --dry`, full scale (2026-09-26): STOPPED, over the 5.5 GB limit
| step | s | peak MB |
|---|---|---|
| India tokenize | 177.3 | 2,936 |
| India A / B / C df + S1 survivors | 3.0 / 13.6 / 11.8 | 2,936 |
| **India X df + S1 survivors** | 48.1 | **5,957** |
| India S2/S3 survivors | 82.2 | 5,957 |
Stopped before US (the larger group). Estimate was ≤ ~4.5 GB. The X count/combine + S1-survivor stage was never
measured on a slice; that is where it went over. The alert failed (sampler CSV bug), so the overrun was found in
the step log, not live. Paging: not measured (sampler broken). Old code at the same point: 8,131 MB after India df.


## Blocking, full scale with the rewritten block.py (2026-09-26): DONE, gate met
The user chose to keep going through high memory (no kill switch) and report the numbers. Keep-awake was held during
runs after the smoke run showed the machine entering standby mid-run.

| step | wall | peak RSS | disk paging | result | reference |
|---|---|---|---|---|---|
| block --smoke 300000 (train) | 768 s (~530 s of it in standby) | 3,427 MB | not measured (sampler path bug) | exit 0 | 29 s, 7.1 GB |
| block --split train | 4,082 s | **7,543 MB** (US X df + S1 survivors) | 33% of samples > 1k page-ins/s, max 61k/s, min avail 242 MB | 69,043,101 candidates (31.3/S1), grid 611.6M | ~69M, 31.3/S1 |
| block --split test | 1,717 s | **7,827 MB** (final write of candidates + TSV) | 41% of samples > 1k page-ins/s, max 47k/s, min avail 2,269 MB | 60,901,445 candidates (35.2/S1); candidate_pairs.tsv 1,732,544 rows, 0 empty | 60,901,445; 469 s, 13.7 GB |
| block_diag | 107 s | – | – | **X m5/k10: PC 96.31%, ceiling 0.9873, 31.3/S1**; India 94.71 / US 97.39; missed 281,532 | identical |

Blocking reproduces the reference exactly. Peak memory is ~7.5-7.8 GB (vs 13.7 GB for the old code on the Mac) and
it pages heavily on this machine but completes.
| block_eval (full, unsampled) | 65 s | **OOM**: 32.8 GB committed > 29.5 GB limit | 100% of samples paging, min avail 4 MB | failed in base_pairs; docs/blocking.md not regenerated | the committed docs/blocking.md is a `--sample 200000` run |
