"""M4 inference: models/lgb_final.txt scores every test candidate (artifacts/features/test, part by part), then
the decide.py rule: per record keep its argmax-p S1 (ties -> lowest s1_id), keep it if p >= t (oof/decide.json).
Writes output/matching_results.tsv (one row per test S1, "" when none) and asserts matches ⊆ candidates.

Run from code/business_entity_resolution/:  python -m src.predict
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from .block import norm_path
from .decide import with_top
from .io import StepLog, path, write_candidates
from .train import feature_cols


def main() -> None:
    log = StepLog()
    meta = json.loads((path("models_dir") / "lgb_final.json").read_text())
    t = json.loads((path("oof_dir") / "decide.json").read_text())["t"]
    feats = meta["features"]
    assert feature_cols("test") == feats, "test feature columns != training features"
    bst = lgb.Booster(model_file=str(path("models_dir") / "lgb_final.txt"))
    scored = []
    for f in sorted((path("features_dir") / "test").glob("part-*.parquet")):
        df = pl.read_parquet(f)
        p = bst.predict(df.select(pl.col(feats).cast(pl.Float32)).to_numpy()).astype(np.float32)
        scored.append(df.select("s1_id", "rec_id").with_columns(p=pl.Series(p)))
        log(f"score {f.name}", rows=df.height)
    scored = pl.concat(scored)
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
    log.dump(path("oof_dir") / "predict_timing.json")


if __name__ == "__main__":
    main()
