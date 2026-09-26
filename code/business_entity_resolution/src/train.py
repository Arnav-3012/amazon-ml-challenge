"""M4 LightGBM pair matcher on artifacts/features/train (see src.features).

Subset = first `fraction` of a seed-42 permutation of the sorted train S1 ids, so the 20% iteration subset
is nested in the final-fit subset. Every candidate of a subset S1 is loaded (its relative features were
computed against the full candidate set).
Training rows = all positives + neg_ratio x as many negatives: hard_frac of them hardest-first by the max
channel score, the rest uniform (seeded). Negative subsampling shifts the base rate, so p is not calibrated;
decide.py tunes the threshold on OOF p (M5 needs a recalibration step before expected-F0.5 decisions).

Modes (run from code/business_entity_resolution/):
  python -m src.train            # GroupKFold(5) by s1_id on the subset; held-out fold scored on ALL its
                                 # candidates (also the early-stopping set: picks the iteration only)
                                 # -> oof/oof_train.parquet, oof/cv_metrics.json
  python -m src.train --loco     # per country c: train on the other countries' subset rows at the mean CV
                                 # best iteration (no early stopping), score c -> oof/loco_train.parquet
  python -m src.train --final    # final_fraction subset, mean CV best iteration x final_round_mult
                                 # -> models/lgb_final.txt + models/lgb_final.json
Country is used only to split rows for --loco; it is never a feature.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import GroupKFold

from .block import norm_path
from .features import ID_COLS
from .io import CFG, StepLog, load_gt_pairs, path

MC, SEED = CFG["matcher"], CFG["seed"]
SCORE_COLS = ["A_score", "B_score", "C_score", "X_score"]


def parts(split: str) -> str:
    return str(path("features_dir") / split / "part-*.parquet")


def feature_cols(split: str) -> list[str]:
    feats = [c for c in pl.scan_parquet(parts(split)).collect_schema().names() if c not in ID_COLS]
    assert not any("country" in c.lower() for c in feats), "country must never be a feature"
    return feats


def subset_ids(fraction: float) -> pl.Series:
    ids = pl.read_parquet(norm_path("train", 1), columns=["entity_id"])["entity_id"].sort()
    perm = np.random.default_rng(SEED).permutation(ids.len())
    return ids.gather(perm[: round(fraction * ids.len())])


def load_meta(ids: pl.Series) -> pl.LazyFrame:
    """Subset rows (global row index _i, ids, label, hardness) of the train feature parts, file order."""
    gt = load_gt_pairs().select("s1_id", rec_id="match_id", label=pl.lit(1, pl.Int8))
    return (pl.scan_parquet(parts("train")).with_row_index("_i").filter(pl.col("s1_id").is_in(ids.implode()))
            .join(gt.lazy(), on=["s1_id", "rec_id"], how="left", maintain_order="left")
            .with_columns(pl.col("label").fill_null(0), hard=pl.max_horizontal(SCORE_COLS).fill_null(0)))


def matrix(lf: pl.LazyFrame, feats: list[str]) -> np.ndarray:
    return lf.select(pl.col(feats).cast(pl.Float32)).collect().to_numpy()  # nulls -> NaN (LightGBM missing)


def sample_rows(idx: np.ndarray, y: np.ndarray, hard: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """All positives of idx + neg_ratio x as many negatives: hard_frac hardest by `hard`, the rest uniform."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_neg = min(len(neg), MC["neg_ratio"] * len(pos))
    n_hard = round(MC["hard_frac"] * n_neg)
    order = neg[np.argsort(-hard[neg], kind="stable")]
    rand = rng.choice(order[n_hard:], n_neg - n_hard, replace=False)
    return np.sort(np.concatenate([pos, order[:n_hard], rand]))


def params(seed: int | None = None) -> dict:
    return {**MC["lgb"], "seed": SEED if seed is None else seed, "metric": "binary_logloss"}


def dataset(X: np.ndarray, y: np.ndarray, feats: list[str], weight: np.ndarray | None = None,
            reference: lgb.Dataset | None = None, seed: int | None = None) -> lgb.Dataset:
    """Constructed (binned) Dataset. free_raw_data drops LightGBM's reference to X, so the caller can `del X`
    right after and only the bins stay in memory. Same params as training (binning params are fixed at construct)."""
    return lgb.Dataset(X, label=y, weight=weight, feature_name=feats, reference=reference, params=params(seed),
                       free_raw_data=True).construct()


def fit_ds(dtr: lgb.Dataset, rounds: int, dva: lgb.Dataset | None = None, seed: int | None = None) -> lgb.Booster:
    kw = {}
    if dva is not None:
        kw = {"valid_sets": [dva], "valid_names": ["heldout"],
              "callbacks": [lgb.early_stopping(MC["early_stopping"], verbose=False), lgb.log_evaluation(100)]}
    return lgb.train(params(seed), dtr, rounds, **kw)


def fit(X: np.ndarray, y: np.ndarray, feats: list[str], rounds: int, valid: tuple | None = None,
        weight: np.ndarray | None = None, seed: int | None = None) -> lgb.Booster:
    """weight: per-row training weight (e.g. inverse sampling probability), None = uniform. The valid set is
    never weighted -- early stopping and the reported heldout logloss must read as an unweighted, unsampled
    metric, or the stopping decision itself would be biased by the sampling scheme.
    seed: LightGBM seed override (bagging/feature_fraction draws); None = config seed."""
    dtr = dataset(X, y, feats, weight, seed=seed)
    dva = dataset(valid[0], valid[1], feats, reference=dtr, seed=seed) if valid is not None else None
    return fit_ds(dtr, rounds, dva, seed)


def calibration(y: np.ndarray, p: np.ndarray) -> list[dict]:
    b = np.minimum((p * 10).astype(int), 9)
    return [{"bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int((b == k).sum()),
             "mean_p": float(p[b == k].mean()) if (b == k).any() else None,
             "pos_rate": float(y[b == k].mean()) if (b == k).any() else None} for k in range(10)]


def cv(log: StepLog) -> None:
    feats, ids = feature_cols("train"), subset_ids(MC["subset_fraction"])
    lf = load_meta(ids)
    meta = lf.select("s1_id", "rec_id", "label", "hard").collect()
    X = matrix(lf, feats)
    y, hard = meta["label"].to_numpy(), meta["hard"].to_numpy()
    log("load subset", rows=len(y), pos=int(y.sum()), s1=ids.len(), n_feats=len(feats))
    groups = meta["s1_id"].rank("dense").to_numpy()
    p, fold = np.zeros(len(y), np.float32), np.zeros(len(y), np.int8)
    best, gain = [], np.zeros(len(feats))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=MC["n_folds"]).split(X, y, groups)):
        tr = sample_rows(tr, y, hard, np.random.default_rng(SEED + k))
        bst = fit(X[tr], y[tr], feats, MC["num_boost_round"], valid=(X[va], y[va]))
        p[va], fold[va] = bst.predict(X[va], num_iteration=bst.best_iteration), k
        best.append(bst.best_iteration)
        gain += bst.feature_importance("gain", iteration=bst.best_iteration)
        log(f"fold {k}", train_rows=len(tr), train_pos=int(y[tr].sum()), heldout_rows=len(va),
            best_iter=bst.best_iteration, heldout_logloss=round(float(log_loss(y[va], p[va], labels=[0, 1])), 5))
    out = path("oof_dir")
    out.mkdir(parents=True, exist_ok=True)
    meta.select("s1_id", "rec_id", "label").with_columns(p=pl.Series(p), fold=pl.Series(fold)).write_parquet(
        out / "oof_train.parquet")
    imp = sorted(zip(feats, gain / MC["n_folds"]), key=lambda t: -t[1])
    m = {"fraction": MC["subset_fraction"], "n_s1": ids.len(), "rows": len(y), "pos": int(y.sum()),
         "best_iters": best, "mean_best_iter": float(np.mean(best)),
         "pr_auc": float(average_precision_score(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1])),
         "calibration": calibration(y, p), "gain_top30": [{"feature": f, "gain": float(g)} for f, g in imp[:30]],
         "features": feats}
    (out / "cv_metrics.json").write_text(json.dumps(m, indent=1))
    log("cv done", pr_auc=round(m["pr_auc"], 5), logloss=round(m["logloss"], 5), mean_best_iter=m["mean_best_iter"])
    for r in m["calibration"]:
        print(r)
    for r in m["gain_top30"]:
        print(r)


def loco(log: StepLog) -> None:
    feats, ids = feature_cols("train"), subset_ids(MC["subset_fraction"])
    rounds = round(json.loads((path("oof_dir") / "cv_metrics.json").read_text())["mean_best_iter"])
    country = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    lf = load_meta(ids).join(country.lazy(), on="s1_id", how="left", maintain_order="left")
    meta = lf.select("s1_id", "rec_id", "label", "hard", "country").collect()
    X = matrix(lf, feats)
    y, hard, ctry = meta["label"].to_numpy(), meta["hard"].to_numpy(), meta["country"].to_numpy()
    log("load subset", rows=len(y), rounds=rounds)
    out = []
    for k, c in enumerate(sorted(set(ctry))):
        tr = sample_rows(np.flatnonzero(ctry != c), y, hard, np.random.default_rng(SEED + 100 + k))
        va = np.flatnonzero(ctry == c)
        bst = fit(X[tr], y[tr], feats, rounds)
        out.append(meta[va].select("s1_id", "rec_id", "label", eval_country="country")
                   .with_columns(p=pl.Series(bst.predict(X[va]).astype(np.float32))))
        log(f"loco eval {c}", train_rows=len(tr), eval_rows=len(va))
    pl.concat(out).write_parquet(path("oof_dir") / "loco_train.parquet")


def final(log: StepLog) -> None:
    feats, ids = feature_cols("train"), subset_ids(MC["final_fraction"])
    cvm = json.loads((path("oof_dir") / "cv_metrics.json").read_text())
    assert cvm["features"] == feats, "feature set changed since CV; rerun src.train"
    rounds = round(cvm["mean_best_iter"] * MC["final_round_mult"])
    lf = load_meta(ids)
    meta = lf.select("_i", "label", "hard").collect()
    y = meta["label"].to_numpy()
    sel = sample_rows(np.arange(len(y)), y, meta["hard"].to_numpy(), np.random.default_rng(SEED))
    keep = meta["_i"].gather(sel)
    del meta
    X = matrix(lf.filter(pl.col("_i").is_in(keep.implode())), feats)  # same file order -> aligned with y[sel]
    log("load final rows", s1=ids.len(), subset_rows=len(y), train_rows=len(sel), pos=int(y[sel].sum()), rounds=rounds)
    bst = fit(X, y[sel], feats, rounds)
    out = path("models_dir")
    out.mkdir(parents=True, exist_ok=True)
    bst.save_model(out / "lgb_final.txt")
    (out / "lgb_final.json").write_text(json.dumps({"features": feats, "rounds": rounds, "fraction": MC["final_fraction"],
                                                    "train_rows": len(sel), "params": params()}, indent=1))
    log("final fit", rounds=rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--loco", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    mode = "loco" if a.loco else "final" if a.final else "cv"
    log = StepLog()
    {"cv": cv, "loco": loco, "final": final}[mode](log)
    log.dump(path("oof_dir") / f"train_timing_{mode}.json")


if __name__ == "__main__":
    main()
