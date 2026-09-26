"""M4 inference: models/lgb_final.txt scores every test candidate (artifacts/features/test, part by part), then
the decide.py rule: per record keep its argmax-p S1 (ties -> lowest s1_id), keep it if p >= t (oof/decide.json).
Writes output/matching_results.tsv (one row per test S1, "" when none) and asserts matches ⊆ candidates.
--folds (M5-2): p = mean of models/fold_{k}.txt, t = best t of the test-density OOF (oof/cv_full.json, b); the
scored pairs are also kept in oof/test_p.parquet (stage-2 input).

Run from code/business_entity_resolution/:  python -m src.predict [--folds]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from .block import norm_path
from .decide import with_top
from .io import CFG, StepLog, path, write_candidates
from .train import feature_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", action="store_true")
    a = ap.parse_args()
    log, M, O = StepLog(), path("models_dir"), path("oof_dir")
    if a.folds:
        cv = json.loads((O / "cv_full.json").read_text())
        feats, t = cv["features"], cv["b_test_density"]["best_t"]
        bsts = [lgb.Booster(model_file=str(M / f"fold_{k}.txt")) for k in range(CFG["matcher"]["n_folds"])]
    else:
        feats = json.loads((M / "lgb_final.json").read_text())["features"]
        t = json.loads((O / "decide.json").read_text())["t"]
        bsts = [lgb.Booster(model_file=str(M / "lgb_final.txt"))]
    assert feature_cols("test") == feats, "test feature columns != training features"
    scored = []
    for f in sorted((path("features_dir") / "test").glob("part-*.parquet")):
        df = pl.read_parquet(f)
        X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        p = np.mean([b.predict(X) for b in bsts], axis=0).astype(np.float32)
        scored.append(df.select("s1_id", "rec_id").with_columns(p=pl.Series(p)))
        log(f"score {f.name}", rows=df.height, models=len(bsts))
    scored = pl.concat(scored)
    if a.folds:
        scored.write_parquet(O / "test_p.parquet")
    cands = pl.scan_parquet(path("interim_dir") / "candidates_test.parquet").select("s1_id", "rec_id")
    n_cand = cands.select(pl.len()).collect().item()
    assert scored.height == n_cand, f"scored {scored.height:,} pairs != {n_cand:,} candidates"

    matches = with_top(scored).filter(pl.col("top") & (pl.col("p") >= t)).select("s1_id", "rec_id")
    outside = matches.lazy().join(cands, on=["s1_id", "rec_id"], how="anti").select(pl.len()).collect().item()
    assert outside == 0, f"{outside} matches not in candidate set"
    assert matches["rec_id"].is_unique().all(), "a record matched to >1 S1"
    s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"])["entity_id"]
    write_candidates(path("matching_results"), matches.lazy(), s1_ids, list_col="matched_entity_ids")
    n_with = matches["s1_id"].n_unique()
    log("decide + write", t=t, matches=matches.height, s1_with_match=n_with, s1_total=s1_ids.len(),
        empty_pct=round(100 * (1 - n_with / s1_ids.len()), 2))
    log.dump(O / f"predict_timing{'_folds' if a.folds else ''}.json")


if __name__ == "__main__":
    main()
