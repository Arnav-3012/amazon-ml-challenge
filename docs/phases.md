# Phases

## Phase 0 / M0: Scaffold (2026-09-25)
**Status:** done. The user reports the env verification import passes (after `brew install libomp`).
- Repo layout mirrors the submission zip; docs written; uv venv (Python 3.11) with phase-0 deps pinned.
- Git deferred: the user is creating the GitHub repo first, then init + commit "phase0: scaffold" on their go.
- **Blockers carried forward:**
  1. ~~`student_resource/` missing~~ RESOLVED 2026-09-25: found at `~/Downloads/student_resource`. `dataset` now links there; validator and template copied byte-identical.
  2. ~~LightGBM needs `libomp`~~ RESOLVED 2026-09-25: the user ran `sudo xcodebuild -license accept && brew install libomp` (keg-only is fine; the rpath hits `/opt/homebrew/opt/libomp/lib`).
- **Next:** M1 (loader + scorer + all-empty baseline + validator PASS). Unblocked. Git init is still waiting on the user's GitHub repo.
- **Scale flag for M1/M3/M7:** train S1 ≈ 2.2M, S2 ≈ 5.0M, S3 ≈ 5.3M rows; test S1 ≈ 1.7M, S2 ≈ 4.9M, S3 ≈ 5.1M (`wc -l`, including header). The loader needs memory-conscious dtypes; blocking must be chunked; encoder throughput over ~10M texts per split must be measured before M7.
