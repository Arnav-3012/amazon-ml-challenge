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
    ap.add_argument("--cascade", type=float, default=0.0,
                    help="--folds only: score every row with fold 0; average all folds only where fold-0 p >= this "
                         "(rows below keep fold-0 p; they can't be a record argmax >= t). 0 = off (exact mean)")
    ap.add_argument("--tags", nargs="+", default=[""], help="--folds: fold-model suffixes to average (seed runs)")
    ap.add_argument("--cv-json", default="cv_full.json", help="--folds: json with features + b best t")
    ap.add_argument("--check-frac", type=float, default=0.01, help="cascade: share of skipped rows re-scored exactly")
    a = ap.parse_args()
    log, M, O = StepLog(), path("models_dir"), path("oof_dir")
    if a.folds:
        cv = json.loads((O / a.cv_json).read_text())
        feats, t = cv["features"], cv["b_test_density"]["best_t"]
        bsts = [lgb.Booster(model_file=str(M / f"fold_{k}{tg}.txt"))
                for tg in a.tags for k in range(CFG["matcher"]["n_folds"])]
    else:
        feats = json.loads((M / "lgb_final.json").read_text())["features"]
        t = json.loads((O / "decide.json").read_text())["t"]
        bsts = [lgb.Booster(model_file=str(M / "lgb_final.txt"))]
    assert feature_cols("test") == feats, "test feature columns != training features"
    scored, rng, chk = [], np.random.default_rng(CFG["seed"]), []
    for f in sorted((path("features_dir") / "test").glob("part-*.parquet")):
        df = pl.read_parquet(f)
        X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        if a.cascade > 0 and len(bsts) > 1:
            p = bsts[0].predict(X)
            hi = np.flatnonzero(p >= a.cascade)
            lo = np.flatnonzero(p < a.cascade)
            p[hi] = np.mean([p[hi]] + [b.predict(X[hi]) for b in bsts[1:]], axis=0)
            c = lo[rng.random(len(lo)) < a.check_frac]  # exact mean on a sample of the skipped rows
            if len(c):
                pc = np.mean([b.predict(X[c]) for b in bsts], axis=0)
                chk.append((pc, p[c]))
            p = p.astype(np.float32)
            log(f"score {f.name}", rows=df.height, models=len(bsts), full_rows=len(hi),
                max_exact_on_skipped=round(float(pc.max()), 4) if len(c) else None)
        else:
            p = np.mean([b.predict(X) for b in bsts], axis=0).astype(np.float32)
            log(f"score {f.name}", rows=df.height, models=len(bsts))
        scored.append(df.select("s1_id", "rec_id").with_columns(p=pl.Series(p)))
    if chk:
        ex, ap_ = np.concatenate([c[0] for c in chk]), np.concatenate([c[1] for c in chk])
        log("cascade check", n=len(ex), max_exact=float(ex.max()), max_abs_diff=float(np.abs(ex - ap_).max()),
            n_exact_ge_t=int((ex >= t).sum()))
        assert (ex < t).all(), "cascade: a skipped row's exact mean reaches t -> rerun without --cascade"
    scored = pl.concat(scored)
    if a.folds:
        scored.write_parquet(O / ("test_p.parquet" if a.tags == [""] else "test_p_ens.parquet"))
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
