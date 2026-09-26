"""M5-2 full-data stage-1 matcher: GroupKFold(5) by s1_id on 100% of train S1s, with S1-dropout on the training folds.

Folds: S1 fold = (position in a seed-42 permutation of the sorted S1 ids) mod n_folds; an S1's rows never split.
Dropout world: drop_frac of ALL train S1s removed (seeded). Their rows vanish, so their records become orphans whose
remaining candidates are all negatives (labels unchanged). Recomputed over the remaining rows: n_cand_rec, the
record-side relative features (features.relative: *_drec, *_rkrec, *_gaprec) and record-centric ranks {ch}_rrank
(minus the dropped rows ranked ahead in the same record). S1-side and pair features cannot change. Not simulated:
pairs that would enter the candidate set (they have no pair features), a rank crossing the m cut, the gate.
Fold k: its own world (seed+1000+k). Training S1s = other folds minus dropped; inner_valid_frac of them (seed+3000+k)
are the early-stopping set (all their rows); the rest are sampled as in M4 (train.sample_rows). -> models/fold_{k}.txt
OOF: each row scored by its fold's model, then the decide.py rule (argmax per record, one global t), macro F0.5:
  (a) standard: stored features, over all S1s.
  (b) test density (PRIMARY): one world (seed+2000), over the remaining S1s; the argmax sees remaining rows only.
-> oof/oof_full.parquet (_i = row in features/train, s1k, reck, label, fold, p_std, p_td), oof/cv_full.json.
Memory: features are read part by part; only the sampled training rows are held as float32. Stops if peak RSS > max_rss_mb.

Run from code/business_entity_resolution/:
  python -m src.cv_full --check    # the no-drop world must reproduce every stored record-side feature (all rows)
  python -m src.cv_full --smoke    # 2% of S1s, 50 rounds, *_smoke outputs: end-to-end crash test
  python -m src.cv_full            # 5 folds + OOF (a) and (b)
  python -m src.cv_full --curve    # fold 0 at curve_fracs of its training S1s, fixed rounds = fold 0 best iter, on (b)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import log_loss

from .block import BCFG, CHANNELS, norm_path
from .decide import f05_vec
from .features import REL_COLS, id_key, relative
from .io import CFG, StepLog, load_gt_pairs, path
from .metric import macro_f05
from .train import SCORE_COLS, dataset, feature_cols, fit, fit_ds, parts, sample_rows

FC, MC, SEED = CFG["cv_full"], CFG["matcher"], CFG["seed"]
RR_COLS = [f"{ch}_rrank" for ch in CHANNELS]
REC_COLS = ["n_cand_rec", *(f"{c}_{x}" for c in REL_COLS for x in ("drec", "rkrec", "gaprec")), *RR_COLS]
TS = np.round(np.arange(*CFG["decide"]["t_grid"]), 4)


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        if self.rows[-1]["peak_rss_mb"] > FC["max_rss_mb"]:
            raise SystemExit(f"STOP: peak RSS {self.rows[-1]['peak_rss_mb']} MB > {FC['max_rss_mb']} MB after '{step}'")


class Data:
    """Row metadata of features/train (file order) restricted to the S1s of `s1`, plus the part layout."""

    def __init__(self, s1: pl.DataFrame):
        self.files = sorted(str(p) for p in Path(parts("train")).parent.glob("part-*.parquet"))
        n = np.array([pq.ParquetFile(f).metadata.num_rows for f in self.files])
        self.bounds = list(zip(np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n)))
        self.feats = feature_cols("train")
        gt = load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"), label=pl.lit(1, pl.Int8))
        self.meta = (pl.scan_parquet(self.files).with_row_index("_i")
                     .select("_i", *REL_COLS, *RR_COLS, "X_srank", s1k=id_key("s1_id"), reck=id_key("rec_id"),
                             hard=pl.max_horizontal(SCORE_COLS).fill_null(0).cast(pl.Float32))
                     .join(s1.lazy().select("s1k", "code"), on="s1k", how="inner", maintain_order="left")
                     .join(gt.lazy(), on=["s1k", "reck"], how="left", maintain_order="left")
                     .with_columns(pl.col("label").fill_null(0)).collect())
        assert self.meta["_i"].is_sorted() and self.meta["_i"].n_unique() == self.meta.height
        self.i = self.meta["_i"].to_numpy()
        self.code = self.meta["code"].to_numpy()
        self.y = self.meta["label"].to_numpy()
        self.fold = s1["fold"].to_numpy()[self.code]
        # v1 = the M3b candidate set (X_rrank <= blocking.m or X_srank <= blocking.k); everything else is a
        # v2-only "extension" row. Memory guard (M5-3) subsamples extension-only negatives, never v1 or positive rows.
        self.in_v1 = (((self.meta["X_rrank"] <= BCFG["m"]) | (self.meta["X_srank"] <= BCFG["k"]))
                      .fill_null(False).to_numpy())

    def chunks(self):
        """(meta slice a:b, part rows) per feature part."""
        for f, (lo, hi) in zip(self.files, self.bounds):
            a, b = np.searchsorted(self.i, [lo, hi])
            if a < b:
                yield f, a, b, self.i[a:b] - lo

    def gather(self, pos: np.ndarray, wdir: Path | None) -> np.ndarray:
        """Rows `pos` (sorted meta positions) as a float32 matrix; record-side columns from world `wdir`."""
        X = np.empty((len(pos), len(self.feats)), np.float32)
        for f, a, b, rows in self.chunks():
            u, v = np.searchsorted(pos, [a, b])
            if u < v:
                sel = rows[pos[u:v] - a]
                X[u:v] = pl.read_parquet(f, columns=self.feats).select(pl.col(self.feats).cast(pl.Float32)).to_numpy()[sel]
        if wdir is not None:
            for j, c in enumerate(self.feats):
                if c in REC_COLS:
                    X[:, j] = np.load(wdir / f"{c}.npy", mmap_mode="r")[pos]
        return X

    def predict(self, boosters: list[lgb.Booster], wdir: Path | None = None, keep: np.ndarray | None = None) -> np.ndarray:
        """p for every meta row from its fold's booster (NaN where keep is False)."""
        p = np.full(len(self.i), np.nan, np.float32)
        mm = {c: np.load(wdir / f"{c}.npy", mmap_mode="r") for c in REC_COLS} if wdir is not None else {}
        for f, a, b, rows in self.chunks():
            X = pl.read_parquet(f, columns=self.feats).select(pl.col(self.feats).cast(pl.Float32)).to_numpy()[rows]
            for j, c in enumerate(self.feats):
                if c in mm:
                    X[:, j] = mm[c][a:b]
            for k, bst in enumerate(boosters):
                r = np.flatnonzero((self.fold[a:b] == k) & (True if keep is None else keep[a:b]))
                if len(r):
                    p[a + r] = bst.predict(X[r])
        return p


def s1_table(frac: float) -> pl.DataFrame:
    ids = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"]).sort("entity_id")
    perm = np.random.default_rng(SEED).permutation(ids.height)
    fold = np.empty(ids.height, np.int8)
    fold[perm] = np.arange(ids.height) % MC["n_folds"]
    t = (ids.select("country", s1_id="entity_id").with_columns(s1k=id_key("s1_id"), fold=pl.Series(fold))
         .join(load_gt_pairs().group_by("s1_id").len("ntrue"), on="s1_id", how="left", maintain_order="left")
         .with_columns(pl.col("ntrue").fill_null(0).cast(pl.Int32)))
    if frac < 1:
        t = t.filter(pl.Series(np.isin(np.arange(ids.height), perm[: round(frac * ids.height)])))
    return t.with_row_index("code")


def drop_mask(n: int, seed: int) -> np.ndarray:
    d = np.zeros(n, bool)
    d[np.random.default_rng(seed).permutation(n)[: round(FC["drop_frac"] * n)]] = True
    return d


def world(meta: pl.DataFrame, keep: np.ndarray, out: Path, log: Log) -> None:
    """Record-side columns after removing the rows with keep=False -> out/{col}.npy (float32, meta row order,
    NaN on removed rows and where the value is null)."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    idx = np.flatnonzero(keep)

    def save(name: str, v: pl.Series) -> None:
        full = np.full(len(keep), np.nan, np.float32)
        full[idx] = v.cast(pl.Float32).to_numpy()
        np.save(out / f"{name}.npy", full)
    kept = meta.select("s1k", "reck", *REL_COLS).filter(pl.Series(keep))
    keys = kept.select("s1k", "reck")
    save("n_cand_rec", keys.select(pl.len().over("reck")).to_series())
    for c in REL_COLS:
        r = relative(keys, kept[c])
        for x in ("drec", "rkrec", "gaprec"):
            save(f"{c}_{x}", r[f"{c}_{x}"])
    del kept, keys
    for c in RR_COLS:  # record-centric ranks are ordinal (unique per record and channel): subtract removed rows ahead
        x = pl.col(c)
        ahead = (pl.col("_gone") & x.is_not_null()).cast(pl.Int32).cum_sum().over("reck", order_by=c)
        v = meta.select("reck", c).with_columns(_gone=pl.Series(~keep)).select((x - ahead).alias(c))
        save(c, v.to_series().filter(pl.Series(keep)))
    log(f"world {out.name}", kept_rows=len(idx), removed_rows=int(len(keep) - len(idx)))


def truth_keys(s1: pl.DataFrame) -> dict:
    t = (load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))
         .join(s1.select("s1k", "code"), on="s1k").group_by("code").agg("reck"))
    return dict(t.iter_rows())


def score(s1: pl.DataFrame, d: Data, p: np.ndarray, scope: np.ndarray, exact: dict | None = None) -> dict:
    """decide.py rule on p (NaN = row absent): argmax per record (ties -> lowest s1_id), then one global t;
    macro F0.5 over the S1s in `scope` (blocking misses count)."""
    ok = ~np.isnan(p)
    top = (pl.DataFrame({"code": d.code[ok], "reck": d.meta["reck"].to_numpy()[ok], "p": p[ok], "y": d.y[ok] == 1})
           .sort(["reck", "p", "code"], descending=[False, True, False]).filter(pl.col("reck").is_first_distinct()))
    c, pp, yy = top["code"].to_numpy(), top["p"].to_numpy(), top["y"].to_numpy()
    n, ntrue = s1.height, s1["ntrue"].to_numpy()

    def per(t: float) -> np.ndarray:
        k = pp >= t
        return f05_vec(np.bincount(c[k & yy], minlength=n), np.bincount(c[k], minlength=n), ntrue)
    cur = np.array([per(t)[scope].mean() for t in TS])
    bi = int(np.argmax(cur))
    t, f = float(TS[bi]), per(float(TS[bi]))
    k = pp >= t
    npred = np.bincount(c[k], minlength=n)
    if exact is not None:
        pred = pl.DataFrame({"code": c[k], "reck": top["reck"].to_numpy()[k]}).group_by("code").agg("reck")
        pred = {a: set(b) for a, b in pred.iter_rows()}
        ex, _ = macro_f05(pred, {i: frozenset(exact.get(i, ())) for i in np.flatnonzero(scope)})
        assert abs(ex - cur[bi]) < 1e-9, (ex, cur[bi])
    ctry, single = s1["country"].to_numpy(), ntrue == 0
    return {"best_t": t, "macro_f05": float(cur[bi]), "n_s1": int(scope.sum()),
            "f05_singleton": float(f[scope & single].mean()), "f05_nonsingleton": float(f[scope & ~single].mean()),
            **{f"f05_{x}": float(f[scope & (ctry == x)].mean()) for x in sorted(set(ctry[scope]))},
            "pred_matches_per_s1": float(npred[scope].mean()), "fp_pairs": int((k & ~yy).sum()),
            "logloss": float(log_loss(d.y[ok], p[ok], labels=[0, 1])),
            "curve": {f"{x:.2f}": float(v) for x, v in zip(TS, cur)}}


def weighted_sample_rows(idx: np.ndarray, y: np.ndarray, hard: np.ndarray, in_v1: np.ndarray,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """All positives + the negatives sample_rows() takes hardest-first (n_hard = hard_frac x min(#neg,
    neg_ratio x #pos)), weight 1 + every other ("easy") negative kept with prob easy_neg_rate x (ext_neg_rate
    if extension-only, i.e. not in_v1, else 1), weight 1/that prob. Returns (kept row idx sorted, weight).
    Each row is kept with prob 1 or a known prob, so the weighted rows are an unbiased estimate of the
    full-population gradient."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_hard = round(MC["hard_frac"] * min(len(neg), MC["neg_ratio"] * len(pos)))
    order = neg[np.argsort(-hard[neg], kind="stable")]
    hard_sel, easy = order[:n_hard], order[n_hard:]
    pk = FC["easy_neg_rate"] * np.where(in_v1[easy], 1.0, FC["ext_neg_rate"])
    kept = rng.random(len(easy)) < pk
    rows = np.concatenate([pos, hard_sel, easy[kept]])
    w = np.concatenate([np.ones(len(pos) + len(hard_sel)), 1.0 / pk[kept]])
    order = np.argsort(rows, kind="stable")
    return rows[order], w[order].astype(np.float32)


def check_weighted_sampling(log: Log) -> None:
    """Synthetic set: a full-population lgb fit vs. a weighted_sample_rows-sampled + weighted fit must reach
    within 1% logloss of each other on a held-out synthetic valid set. Isolates the weighting math from the
    real pipeline (Data/world) -- this only tests that w = 1/(easy_neg_rate * ext keep prob) is an unbiased
    importance weight, not the real feature pipeline. 500k training rows, rel_diff averaged over 3 sampling
    seeds: at this size the MC noise is well under 1%, so a mean above the bar means bias, not noise."""
    rng = np.random.default_rng(SEED)
    n, n_feats = 505_000, 6
    X = rng.normal(size=(n, n_feats)).astype(np.float32)
    true_w = rng.normal(size=n_feats)
    logit = X @ true_w
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    in_v1 = rng.random(n) < 0.3  # ~30% "v1" rows, matching the real pipeline's v1-is-a-minority-of-v2 shape
    hard = rng.random(n).astype(np.float32)
    feats = [f"f{i}" for i in range(n_feats)]

    Xva, yva = X[:5_000], y[:5_000]
    idx = np.arange(5_000, n)

    bst_full = fit(X[idx], y[idx], feats, 200, valid=(Xva, yva))
    p_full = bst_full.predict(Xva)

    ll_full = log_loss(yva, p_full, labels=[0, 1])

    diffs, max_w = [], 0.0
    for s in (SEED, SEED + 1, SEED + 2):
        tr, w = weighted_sample_rows(idx, y, hard, in_v1, np.random.default_rng(s))
        p_w = fit(X[tr], y[tr], feats, 200, valid=(Xva, yva), weight=w).predict(Xva)
        ll_w = log_loss(yva, p_w, labels=[0, 1])
        diffs.append(abs(ll_w - ll_full) / ll_full)
        max_w = max(max_w, float(w.max()))
        log(f"check weighted sampling seed {s}", logloss_full=ll_full, logloss_weighted=ll_w, rel_diff=diffs[-1],
            train_rows_full=len(idx), train_rows_weighted=len(tr), max_weight=float(w.max()))
    mean_diff = float(np.mean(diffs))
    print(f"weighted-sampling check: rel_diff {[round(x, 5) for x in diffs]} mean {mean_diff:.5f}, "
          f"max weight {max_w:g}", flush=True)
    assert mean_diff < 0.01, (f"weighted-sample logloss vs full-train: mean rel_diff {mean_diff:.4f} > 1% over 3 "
                              f"sampling seeds at {len(idx)} rows -- BIAS, not noise; do not run cv_full")


def fold_datasets(d: Data, tr: np.ndarray, w: np.ndarray, va: np.ndarray, wdir: Path) -> tuple[lgb.Dataset, lgb.Dataset]:
    """Constructed train/valid Datasets; each float32 matrix is dropped right after binning, so at most one
    matrix + the bins are alive at a time."""
    X = d.gather(tr, wdir)
    dtr = dataset(X, d.y[tr], d.feats, w)
    del X
    Xv = d.gather(va, wdir)
    dva = dataset(Xv, d.y[va], d.feats, reference=dtr)
    del Xv
    return dtr, dva


def split_fold(s1: pl.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dropped S1 mask of fold k's world, training S1 mask, inner early-stopping S1 mask)."""
    n = s1.height
    drop = drop_mask(n, SEED + 1000 + k)
    tr = np.flatnonzero((s1["fold"].to_numpy() != k) & ~drop)
    inner = np.zeros(n, bool)
    inner[np.random.default_rng(SEED + 3000 + k).choice(tr, round(FC["inner_valid_frac"] * len(tr)), replace=False)] = True
    train = np.zeros(n, bool)
    train[tr] = True
    return drop, train & ~inner, inner


def run(s1: pl.DataFrame, d: Data, tag: str, smoke: bool, log: Log) -> None:
    work, out_m, out_o = path("features_dir") / f"_worlds{tag}", path("models_dir"), path("oof_dir")
    out_m.mkdir(parents=True, exist_ok=True)
    rounds = 50 if smoke else MC["num_boost_round"]
    hard = d.meta["hard"].to_numpy()
    folds = []
    for k in range(MC["n_folds"]):
        drop, train, inner = split_fold(s1, k)
        wdir = work / f"fold{k}"
        world(d.meta, ~drop[d.code], wdir, log)
        tr, w = weighted_sample_rows(np.flatnonzero(train[d.code]), d.y, hard, d.in_v1, np.random.default_rng(SEED + k))
        va = np.flatnonzero(inner[d.code])  # early-stopping/valid set: unsampled, unweighted (see weighted_sample_rows)
        dtr, dva = fold_datasets(d, tr, w, va, wdir)
        shutil.rmtree(wdir)
        log(f"fold {k} data", train_rows=len(tr), train_pos=int(d.y[tr].sum()), valid_rows=len(va),
            train_s1=int(train.sum()), dropped_s1=int(drop.sum()), weighted_rows=int((w > 1).sum()),
            ext_rows_kept=int((~d.in_v1[tr]).sum()))
        bst = fit_ds(dtr, rounds, dva)
        del dtr, dva
        bst.save_model(out_m / f"fold_{k}{tag}.txt")
        folds.append({"fold": k, "best_iter": bst.best_iteration, "train_rows": len(tr), "valid_rows": len(va),
                      "valid_logloss": float(bst.best_score["heldout"]["binary_logloss"])})
        log(f"fold {k} fit", **folds[-1])
    boosters = [lgb.Booster(model_file=str(out_m / f"fold_{k}{tag}.txt")) for k in range(MC["n_folds"])]
    p_std = d.predict(boosters)
    log("OOF (a) predicted")
    drop = drop_mask(s1.height, SEED + 2000)
    keep = ~drop[d.code]
    world(d.meta, keep, work / "eval", log)
    p_td = d.predict(boosters, work / "eval", keep)
    shutil.rmtree(work)
    log("OOF (b) predicted")
    d.meta.select("_i", "s1k", "reck", "label").with_columns(
        fold=pl.Series(d.fold), p_std=pl.Series(p_std), p_td=pl.Series(p_td).fill_nan(None)).write_parquet(
        out_o / f"oof_full{tag}.parquet")
    res = {"a_standard": score(s1, d, p_std, np.ones(s1.height, bool)),
           "b_test_density": score(s1, d, p_td, ~drop, exact=truth_keys(s1))}
    res["b_test_density"]["f05_at_a_t"] = res["b_test_density"]["curve"][f"{res['a_standard']['best_t']:.2f}"]
    (out_o / f"cv_full{tag}.json").write_text(json.dumps(
        {"n_s1": s1.height, "rows": len(d.i), "pos": int(d.y.sum()), "drop_frac": FC["drop_frac"],
         "best_iters": [f["best_iter"] for f in folds], "folds": folds, **res, "features": d.feats}, indent=1))
    for name, r in res.items():
        print(name, {k: v for k, v in r.items() if k != "curve"}, flush=True)
    log("scored", a=round(res["a_standard"]["macro_f05"], 5), b=round(res["b_test_density"]["macro_f05"], 5))


def curve(s1: pl.DataFrame, d: Data, tag: str, log: Log) -> None:
    """Learning curve on fold 0, evaluated on (b): curve_fracs of the fold's training S1s (nested seeded prefixes),
    fixed rounds = fold 0's best iteration; 100% = the saved fold-0 model. Fold-0 rows get the curve model's p,
    the other folds keep their p_td from oof_full, so the argmax sees every competitor."""
    o = path("oof_dir")
    cv = json.loads((o / f"cv_full{tag}.json").read_text())
    assert cv["features"] == d.feats, "features changed since the main run"
    rounds = cv["best_iters"][0]
    oof = pl.read_parquet(o / f"oof_full{tag}.parquet", columns=["_i", "p_td"])
    assert np.array_equal(oof["_i"].to_numpy(), d.i)
    p_main = oof["p_td"].fill_null(np.nan).to_numpy().astype(np.float32)
    work = path("features_dir") / f"_worlds{tag}"
    drop, train, _ = split_fold(s1, 0)
    order = np.random.default_rng(SEED + 4000).permutation(np.flatnonzero(train))
    hard = d.meta["hard"].to_numpy()
    world(d.meta, ~drop[d.code], work / "fold0", log)
    sets = {}
    for fr in FC["curve_fracs"]:
        sub = np.zeros(s1.height, bool)
        sub[order[: round(fr * len(order))]] = True
        tr = sample_rows(np.flatnonzero(sub[d.code]), d.y, hard, np.random.default_rng(SEED))
        sets[fr] = (tr, int(sub.sum()), d.gather(tr, work / "fold0"))
        log(f"curve data {fr}", train_rows=len(tr))
    dropE = drop_mask(s1.height, SEED + 2000)
    keepE = ~dropE[d.code]
    world(d.meta, keepE, work / "eval", log)
    rows0 = np.flatnonzero((d.fold == 0) & keepE)
    X0 = d.gather(rows0, work / "eval")
    shutil.rmtree(work)
    scope = (s1["fold"].to_numpy() == 0) & ~dropE
    out = []

    def point(fr: float, n_s1: int, n_rows: int, p: np.ndarray) -> None:
        r = score(s1, d, p, scope)
        out.append({"frac": fr, "train_s1": n_s1, "train_rows": n_rows, "rounds": rounds,
                    "b_macro_f05": r["macro_f05"], "best_t": r["best_t"],
                    "b_f05_at_main_t": r["curve"][f"{cv['b_test_density']['best_t']:.2f}"],
                    "fold0_logloss": float(log_loss(d.y[rows0], p[rows0], labels=[0, 1]))})
        log(f"curve {fr}", **out[-1])
    for fr, (tr, n_s1, X) in sets.items():
        bst = fit(X, d.y[tr], d.feats, rounds)
        p = p_main.copy()
        p[rows0] = bst.predict(X0)
        point(fr, n_s1, len(tr), p)
    point(1.0, len(order), cv["folds"][0]["train_rows"], p_main)
    (o / f"curve{tag}.json").write_text(json.dumps(out, indent=1))


def preflight_point(frac: float, log: Log) -> dict:
    """One fold-0 run at `frac` of S1s, the same steps as run() (meta, world, sample, datasets, fit) with
    preflight_rounds fixed rounds. Called in a fresh process so peak RSS belongs to this point only."""
    t0 = time.monotonic()
    s1 = s1_table(frac)
    d = Data(s1)
    log("preflight meta", frac=frac, rows=len(d.i), s1=s1.height)
    drop, train, inner = split_fold(s1, 0)
    wdir = path("features_dir") / f"_worlds_preflight_{frac}" / "fold0"
    world(d.meta, ~drop[d.code], wdir, log)
    tr, w = weighted_sample_rows(np.flatnonzero(train[d.code]), d.y, d.meta["hard"].to_numpy(), d.in_v1,
                                 np.random.default_rng(SEED))
    dtr, dva = fold_datasets(d, tr, w, np.flatnonzero(inner[d.code]), wdir)
    shutil.rmtree(wdir.parent)
    t1 = time.monotonic()
    fit_ds(dtr, FC["preflight_rounds"], dva)
    log("preflight fit", frac=frac)
    return {"frac": frac, "s1": s1.height, "meta_rows": len(d.i), "train_rows": len(tr),
            "secs": time.monotonic() - t0, "fit_s_per_round": (time.monotonic() - t1) / FC["preflight_rounds"],
            "peak_rss_mb": log.rows[-1]["peak_rss_mb"]}


def preflight(log: Log) -> None:
    """preflight_point at each preflight_fracs (one subprocess each), then per-fold RSS = a + b*frac and
    time = c + d*frac fitted through the points and read at frac = 1. A fixed-overhead term (a, c) is why a
    single 5% point x20 is not a valid extrapolation. Folds run sequentially, so full-run peak RSS ~ one fold
    at frac 1; full time ~ n_folds x fold time (upper bound: meta load is paid once) + OOF predict (not measured).
    Fit time is at preflight_rounds; the real fold runs to early stopping, so fit_s_per_round is printed too."""
    pts = []
    for fr in FC["preflight_fracs"]:
        r = subprocess.run([sys.executable, "-m", "src.cv_full", "--preflight-point", str(fr)],
                           cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, text=True, check=True)
        print(r.stdout, end="", flush=True)
        pts.append(json.loads(next(x for x in r.stdout.splitlines() if x.startswith("PREFLIGHT_POINT "))[16:]))
    fr = np.array([p["frac"] for p in pts])
    print(f"\n{'frac':>5s} {'S1':>9s} {'train_rows':>11s} {'secs':>8s} {'fit_s/round':>11s} {'peak_rss_mb':>11s}")
    for p in pts:
        print(f"{p['frac']:5.2f} {p['s1']:9d} {p['train_rows']:11d} {p['secs']:8.1f} {p['fit_s_per_round']:11.3f} "
              f"{p['peak_rss_mb']:11.0f}")
    fits = {k: np.polyfit(fr, [p[k] for p in pts], 1) for k in ("peak_rss_mb", "secs", "fit_s_per_round", "train_rows")}
    at1 = {k: float(np.polyval(v, 1.0)) for k, v in fits.items()}
    for k, (b, a) in fits.items():
        print(f"fit {k:15s} = {a:.4g} + {b:.4g} * frac  -> at frac 1: {at1[k]:.4g}")
    print(f"\nextrapolated full run ({MC['n_folds']} folds, 100% S1): peak RSS ~{at1['peak_rss_mb']:.0f} MB "
          f"(guard {FC['max_rss_mb']}), ~{at1['train_rows'] / 1e6:.1f}M train rows/fold, "
          f"time ~{MC['n_folds'] * at1['secs'] / 60:.0f} min at {FC['preflight_rounds']} rounds/fold "
          f"+ {at1['fit_s_per_round']:.2f} s per extra round per fold + OOF predict", flush=True)
    log("preflight done", **{f"{k}_at1": v for k, v in at1.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--curve", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--preflight-point", type=float, help=argparse.SUPPRESS)  # internal: one preflight subprocess
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    log, tag = Log(), "_smoke" if a.smoke else ""
    if a.preflight_point is not None:
        print("PREFLIGHT_POINT " + json.dumps(preflight_point(a.preflight_point, log)), flush=True)
        return
    if a.preflight:
        preflight(log)
        log.dump(path("oof_dir") / "cv_full_timing_preflight.json")
        return
    s1 = s1_table(0.02 if a.smoke else 1.0)
    d = Data(s1)
    log("meta", rows=len(d.i), s1=s1.height, pos=int(d.y.sum()), n_feats=len(d.feats))
    mode = "check" if a.check else "curve" if a.curve else "cv"
    if a.check:
        assert not a.smoke, "--check runs on all rows"
        wdir = path("features_dir") / "_worlds" / "check"
        world(d.meta, np.ones(len(d.i), bool), wdir, log)
        bad = {}
        for c in REC_COLS:
            stored = pl.scan_parquet(d.files).select(pl.col(c).cast(pl.Float32)).collect().to_series().to_numpy()
            got = np.load(wdir / f"{c}.npy")
            ne = int((~((stored == got) | (np.isnan(stored) & np.isnan(got)))).sum())
            if ne:
                bad[c] = ne
        shutil.rmtree(wdir)
        assert not bad, f"no-drop world differs from stored features (rows per column): {bad}"
        log("check OK", columns=len(REC_COLS))
        check_weighted_sampling(log)
    elif a.curve:
        curve(s1, d, tag, log)
    else:
        run(s1, d, tag, a.smoke, log)
    log.dump(path("oof_dir") / f"cv_full_timing_{mode}{tag}.json")


if __name__ == "__main__":
    main()
