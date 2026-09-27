"""Candidate F (PREPARED, needs Krish's rules confirmation before shipping): France pseudo-labelling of stage 1.

Pseudo-labels: B's final test p (oof_krish/test_p2.parquet) on France S1 rows: positive = the record's argmax with
p >= 0.97, negative = p <= 0.02; capped per S1 (3 positives, 5 negatives, seeded). Rows joined to the test feature
parts BY KEY (s1_id, rec_id), never by index. Replay: an equal number of labelled train rows, drawn only from the
training S1s of fold k (other folds), so each continued fold model never sees its own OOF S1s.
Fold k model (models_krish/fold_k.txt) is continued with lgb init_model for ROUNDS rounds on pseudo (weight W_PSEUDO)
+ replay (weight 1) -> models_pl/fold_k.txt. Then OOF (b) is rescored with the continued models (same (b) world as
cv_full) -> oof_pl/oof_full.parquet + oof_pl/cv_full.json, so `stage2 --only full` and `predict` run unchanged with
BER_PATH_oof_dir=oof_pl BER_PATH_models_dir=models_pl.
Run from repo root:  .venv/bin/python experiments/pseudo_label.py [--smoke]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("BER_PATH_models_dir", "models_pl")
os.environ.setdefault("BER_PATH_oof_dir", "oof_pl")
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.block import norm_path  # noqa: E402
from src.cv_full import SEED, Data, Log, drop_mask, s1_table, score, truth_keys, world  # noqa: E402
from src.decide import with_top  # noqa: E402
from src.io import path  # noqa: E402
from src.train import params  # noqa: E402

SRC_M, SRC_O = REPO / "models_krish", REPO / "oof_krish"
POS_T, NEG_T, CAP_POS, CAP_NEG, ROUNDS, W_PSEUDO = 0.97, 0.02, 3, 5, 150, 0.5


def pseudo_rows(feats: list[str], smoke: bool, log: Log) -> tuple[np.ndarray, np.ndarray]:
    fr = (pl.read_parquet(norm_path("test", 1), columns=["entity_id", "country"])
          .filter(pl.col("country") == "France").select(s1_id="entity_id"))
    p2 = pl.read_parquet(SRC_O / "test_p2.parquet").join(fr, on="s1_id", how="semi")
    top = with_top(p2)
    rng = np.random.default_rng(SEED)
    pos = top.filter(pl.col("top") & (pl.col("p") >= POS_T)).with_columns(y=pl.lit(1, pl.Int8))
    neg = top.filter(pl.col("p") <= NEG_T).with_columns(y=pl.lit(0, pl.Int8))
    cap = lambda df, k: (df.with_columns(_r=pl.Series(rng.random(df.height)))
                         .filter(pl.col("_r").rank("ordinal").over("s1_id") <= k).drop("_r"))
    lab = pl.concat([cap(pos, CAP_POS), cap(neg, CAP_NEG)]).select("s1_id", "rec_id", "y")
    if smoke:
        lab = lab.sample(min(lab.height, 20000), seed=SEED)
    files = sorted(str(p) for p in (path("features_dir") / "test").glob("part-*.parquet"))
    X = (pl.scan_parquet(files).join(lab.lazy(), on=["s1_id", "rec_id"], how="inner")
         .select(*[pl.col(f).cast(pl.Float32) for f in feats], "y").collect(engine="streaming"))
    assert X.height == lab.height, f"pseudo rows lost in the key join: {X.height} of {lab.height}"
    log("pseudo labels", rows=X.height, pos=int(X["y"].sum()), france_s1=fr.height,
        pos_pool=pos.height, neg_pool=neg.height)
    return X.select(feats).to_numpy(), X["y"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="20k pseudo rows, 10 rounds, fold 0 only, no OOF rescore")
    a = ap.parse_args()
    log, M, O = Log(), path("models_dir"), path("oof_dir")
    M.mkdir(parents=True, exist_ok=True)
    O.mkdir(parents=True, exist_ok=True)
    cv = json.loads((SRC_O / "cv_full.json").read_text())
    feats = cv["features"]
    Xp, yp = pseudo_rows(feats, a.smoke, log)
    s1 = s1_table(1.0)
    d = Data(s1)
    assert d.feats == feats, "train feature columns != cv_full features"
    rounds = 10 if a.smoke else ROUNDS
    folds = [0] if a.smoke else range(5)
    rng = np.random.default_rng(SEED + 7)
    for k in folds:
        train_rows = np.flatnonzero(d.fold != k)
        rep = np.sort(rng.choice(train_rows, len(yp), replace=False))
        Xr = d.gather(rep, None)
        X = np.vstack([Xp, Xr])
        y = np.concatenate([yp, d.y[rep]])
        w = np.concatenate([np.full(len(yp), W_PSEUDO, np.float32), np.ones(len(rep), np.float32)])
        base = lgb.Booster(model_file=str(SRC_M / f"fold_{k}.txt"))
        ds = lgb.Dataset(X, label=y, weight=w, feature_name=feats, free_raw_data=True)
        bst = lgb.train(params(), ds, rounds, init_model=base, keep_training_booster=False)
        bst.save_model(M / f"fold_{k}{'_smoke' if a.smoke else ''}.txt")
        log(f"fold {k} continued", rows=len(y), pseudo=len(yp), replay=len(rep), trees=bst.num_trees())
        del X, Xr, ds
    if a.smoke:
        log.dump(O / "pseudo_label_timing_smoke.json")
        return
    boosters = [lgb.Booster(model_file=str(M / f"fold_{k}.txt")) for k in range(5)]
    drop = drop_mask(s1.height, SEED + 2000)
    keep = ~drop[d.code]
    wdir = path("features_dir") / "_worlds_pl" / "eval"
    world(d.meta, keep, wdir, log)
    p_td = d.predict(boosters, wdir, keep)
    shutil.rmtree(wdir.parent, ignore_errors=True)
    log("OOF (b) rescored")
    old = pl.read_parquet(SRC_O / "oof_full.parquet")
    assert np.array_equal(old["_i"].to_numpy(), d.i)
    old.with_columns(p_td=pl.Series(p_td).fill_nan(None)).write_parquet(O / "oof_full.parquet")
    b = score(s1, d, p_td, ~drop, exact=truth_keys(s1))
    out = {**{k: v for k, v in cv.items() if k != "b_test_density"}, "b_test_density": b,
           "pseudo": {"pos_t": POS_T, "neg_t": NEG_T, "cap_pos": CAP_POS, "cap_neg": CAP_NEG, "rounds": ROUNDS,
                      "w_pseudo": W_PSEUDO, "rows": int(len(yp)), "pos": int(yp.sum())},
           "note": "a_standard/p_std copied from oof_krish (not rescored); b_test_density rescored with continued models"}
    (O / "cv_full.json").write_text(json.dumps(out, indent=1))
    print("b_test_density", {k: v for k, v in b.items() if k != "curve"}, flush=True)
    log("scored", b=round(b["macro_f05"], 5), base_b=round(cv["b_test_density"]["macro_f05"], 5))
    log.dump(O / "pseudo_label_timing.json")


if __name__ == "__main__":
    main()
