"""World skew: does leaving CONTEXT features at their stored (all-S1) value in the S1-dropout worlds move OOF (b)?

Arms (differ ONLY in cv_full.CTX_COLS; every other REC_COLS feature is recomputed by cv_full.context in both):
  stale       CTX_COLS keep the stored all-S1 value in every world (= cv_full before 2026-09-26).
  recomputed  CTX_COLS from context() over the surviving S1s / candidates (= test, where features see test as-is).
Train: 5 folds, fold k's world (seed+1000+k), training S1s = world_skew.sample_frac of s1_table's permutation
(other folds, alive, minus the inner early-stopping set), cv_full.weighted_sample_rows, --fast params
(learning_rate_fast, max_rounds_fast, early stopping on the inner set). Same rows and seeds for both arms.
Models cache to models/world_skew/{arm}_fold{k}.txt; a rerun loads them and skips the refit.
Eval: the (b) world (seed+2000), scored on a fixed 20% S1 sample (world_skew.sample_frac of s1_table's
GroupKFold permutation, same sample as training and identical for both arms -- NOT every alive S1) by its fold's
model. 2x2: {stale, recomputed} model x {stale, recomputed} eval features. The honest cell for a model is its
recomputed-eval column (test features are computed on the test world as-is); recomputed/recomputed is the fixed
pipeline. Arms run sequentially end-to-end (fit both folds' models -> eval stale fully -> eval recomputed fully)
so at most one arm's boosters + eval predictions are ever live.
Per cell: cv_full.score (argmax per record, ties lowest code, best global t) over the sampled S1s; FP pairs;
per record class at the cell's t: distractor (no true S1 in train), orphan (every true S1 dropped in (b)), live
(>= 1 true S1 alive): records, predicted, FP, FP rate = FP / records.
Also the context shift itself: per CTX column, share of (b) rows (sampled S1s only) whose recomputed value
differs from stored.
docs/world_skew.md additionally reports, per arm on its own honest (matching) eval features: macro F0.5 at a
fixed t=0.75 and at each arm's best t, their delta (recomputed - stale), per-country F0.5 @ t=0.75, and peak
RSS per Log step.
-> oof/world_skew.json, docs/world_skew.md, oof/world_skew_timing.json. Stops if peak RSS > world_skew.max_rss_mb.

Run from code/business_entity_resolution:  python -m src.world_skew
"""
import gc
import json
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl

from .cv_full import CTX_COLS, SEED, TS, Data, Log, context, dead_s1k, dropped, fold_datasets, s1_table, split_fold, \
    weighted_sample_rows
from .decide import f05_vec
from .features import id_key
from .io import CFG, ROOT, load_gt_pairs, path
from .train import MC, fit_ds, perf_cores

WS = CFG["world_skew"]
CHUNK = 2_000_000
MODELS = path("models_dir") / "world_skew"


def rec_class(s1: pl.DataFrame, alive: np.ndarray) -> pl.DataFrame:
    """reck -> live (>= 1 true S1 alive) for every record with a true S1; records absent here are distractors."""
    live = s1.filter(pl.Series(alive))["s1k"]
    return (load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))
            .group_by("reck").agg(live=pl.col("s1k").is_in(live.implode()).any()))


def fp_by_class(meta: pl.DataFrame, p: np.ndarray, t: float, cls: pl.DataFrame) -> dict:
    """cv_full.score's decision (argmax per record, ties -> lowest code, then p >= t) broken down by record class."""
    ok = ~np.isnan(p)
    top = (meta.filter(pl.Series(ok)).with_columns(p=p[ok]).sort(["reck", "p", "code"], descending=[False, True, False])
           .filter(pl.col("reck").is_first_distinct()))
    live = pl.col("live")
    x = (top.join(cls, on="reck", how="left", maintain_order="left")
         .with_columns(cls=pl.when(live.is_null()).then(pl.lit("distractor")).when(live).then(pl.lit("live"))
                       .otherwise(pl.lit("orphan")), pred=pl.col("p") >= t)
         .group_by("cls").agg(records=pl.len(), pred=pl.col("pred").sum(), fp=(pl.col("pred") & ~pl.col("label")).sum()))
    return {r["cls"]: {"records": r["records"], "pred": r["pred"], "fp": r["fp"], "fp_rate": r["fp"] / r["records"]}
            for r in x.iter_rows(named=True)}


def predict_scoped(d: Data, boosters: list[lgb.Booster], wdir, scope: np.ndarray) -> tuple[pl.DataFrame, np.ndarray]:
    """Rows where scope is True, predicted by their fold's booster, in <=CHUNK-row slices per feature part.
    Only scope rows are ever read off disk (row-index filter pushdown into scan_parquet, not read-then-index),
    float32, d.feats only. Keeps only (code, reck, label, p) meta + p -- never a full-length p or full gather."""
    mm = {c: np.load(wdir / f"{c}.npy", mmap_mode="r") for c in d.wcols}
    meta_out, p_out = [], []
    for f, a, b, rows in d.chunks():
        sel = np.flatnonzero(scope[a:b])
        for lo in range(0, len(sel), CHUNK):
            s = sel[lo:lo + CHUNK]
            r = rows[s]  # part-local row positions to read, ascending (rows is ascending per Data.chunks)
            X = (pl.scan_parquet(f).select(pl.col(d.feats).cast(pl.Float32)).with_row_index("_r")
                 .filter(pl.col("_r").is_in(pl.Series(r))).drop("_r").collect().to_numpy())
            for j, c in enumerate(d.feats):
                if c in mm:
                    X[:, j] = mm[c][a + s]
            p = np.full(len(s), np.nan, np.float32)
            for k, bst in enumerate(boosters):
                m = np.flatnonzero(d.fold[a:b][s] == k)
                if len(m):
                    p[m] = bst.predict(X[m])
            meta_out.append(d.meta[a:b][s].select("code", "reck", "label"))
            p_out.append(p)
            del X
    gc.collect()
    return pl.concat(meta_out), np.concatenate(p_out)


def score_scoped(s1: pl.DataFrame, meta: pl.DataFrame, p: np.ndarray, ref_t: float = 0.75) -> dict:
    """cv_full.score, off (meta: code/reck/label, p) instead of a full Data -- same argmax/threshold/metric logic,
    scope = every S1 in s1 (caller already restricted meta/p to the eval sample). logloss over every scored row
    (pre-argmax), matching cv_full.score's d.y[ok]/p[ok]. Adds per-country F0.5 and the metric at a fixed ref_t
    (both best-t and ref_t use the same per-S1 f05_vec, just a different threshold)."""
    from sklearn.metrics import log_loss
    ok = ~np.isnan(p)
    ll = float(log_loss(meta["label"].to_numpy()[ok], p[ok], labels=[0, 1]))
    top = (meta.filter(pl.Series(ok)).with_columns(p=p[ok])
           .sort(["reck", "p", "code"], descending=[False, True, False]).filter(pl.col("reck").is_first_distinct()))
    c, pp, yy = top["code"].to_numpy(), top["p"].to_numpy(), (top["label"] == 1).to_numpy()
    n, ntrue, ctry = s1.height, s1["ntrue"].to_numpy(), s1["country"].to_numpy()

    def per(t: float) -> np.ndarray:
        k = pp >= t
        return f05_vec(np.bincount(c[k & yy], minlength=n), np.bincount(c[k], minlength=n), ntrue)
    cur = np.array([per(t).mean() for t in TS])
    bi = int(np.argmax(cur))
    t = float(TS[bi])
    f_best, f_ref = per(t), per(ref_t)
    k = pp >= t
    return {"best_t": t, "macro_f05": float(cur[bi]), "ref_t": ref_t, "macro_f05_ref_t": float(f_ref.mean()),
            "n_s1": n, "fp_pairs": int((k & ~yy).sum()), "logloss": ll,
            "by_country": {x: {"best_t": float(f_best[ctry == x].mean()), "ref_t": float(f_ref[ctry == x].mean())}
                           for x in sorted(set(ctry))},
            "curve": {f"{x:.2f}": float(v) for x, v in zip(TS, cur)}}


def shift(d: Data, wdir, scope: np.ndarray, cols: list[str]) -> dict:
    """Per CTX column over the eval sample's rows: share changed vs stored, mean stored, mean recomputed.
    Row-index filter pushdown (scope selects << the full world), same trick as predict_scoped."""
    pos = np.flatnonzero(scope)
    rows = d.i[pos]
    out = {}
    for c in cols:
        stored = (pl.scan_parquet(d.files).select(pl.col(c).cast(pl.Float32)).with_row_index("_r")
                  .filter(pl.col("_r").is_in(pl.Series(rows))).drop("_r").collect().to_series().to_numpy())
        new = np.load(wdir / f"{c}.npy", mmap_mode="r")[pos]
        same = (stored == new) | (np.isnan(stored) & np.isnan(new))
        out[c] = {"changed_share": float(1 - same.mean()), "mean_stored": float(np.nanmean(stored)),
                  "mean_recomputed": float(np.nanmean(new))}
    return out


def report(res: dict) -> str:
    cells, L = res["cells"], []
    L += ["# World skew: stale vs recomputed context features in the S1-dropout worlds", "",
          "Generated by `python -m src.world_skew` (see the module docstring for the protocol). Rows = model trained "
          "with that context, columns = eval features. **Honest cell = recomputed eval** (test features see test "
          "as-is). recomputed/recomputed = the fixed cv_full pipeline. Eval is a fixed 20% S1 sample, not every "
          "alive S1 (memory guard).", "",
          f"Training S1s: {res['train_s1_per_fold']} per fold (sample_frac {WS['sample_frac']}); eval: "
          f"{res['n_s1_sampled']} sampled S1s ({res['n_s1_sampled_alive']} alive in (b)). "
          f"CTX features in the model: {', '.join(res['ctx_cols'])}.", "",
          "## Macro F0.5 (b), eval sample", "",
          "| train \\ eval | stale | recomputed |", "|---|---|---|"]
    for tr in ("stale", "recomputed"):
        L.append(f"| {tr} | " + " | ".join(f"{cells[f'{tr}/{ev}']['macro_f05']:.5f} "
                                           f"(t {cells[f'{tr}/{ev}']['best_t']:.2f})"
                                           for ev in ("stale", "recomputed")) + " |")
    L += ["", "## Per cell", "",
          "| train/eval | F0.5 | F0.5 @ stale/stale t | FP pairs | distractor FP / records (rate) | orphan FP / "
          "records (rate) | live FP | logloss |", "|---|---|---|---|---|---|---|---|"]
    for k, c in cells.items():
        fc = c["fp_by_class"]

        def cr(n: str) -> str:
            v = fc.get(n, {"fp": 0, "records": 0, "fp_rate": float("nan")})
            return f"{v['fp']:,} / {v['records']:,} ({v['fp_rate']:.4f})"
        L.append(f"| {k} | {c['macro_f05']:.5f} | {c['at_ref_t']:.5f} | {c['fp_pairs']:,} | {cr('distractor')} | "
                 f"{cr('orphan')} | {fc.get('live', {}).get('fp', 0):,} | {c['logloss']:.5f} |")
    L += ["", "## Context shift in the (b) world (eval sample)", "",
          "| feature | changed share | mean stored | mean recomputed |", "|---|---|---|---|"]
    L += [f"| {c} | {v['changed_share']:.4f} | {v['mean_stored']:.4f} | {v['mean_recomputed']:.4f} |"
          for c, v in res["shift"].items()]
    L += ["", "## Fits", "", "| arm | fold | best iter | valid logloss |", "|---|---|---|---|"]
    L += [f"| {f['arm']} | {f['fold']} | {f['best_iter']} | {f['valid_logloss']:.5f} |" for f in res["fits"]]

    diag = {a: cells[f"{a}/{a}"] for a in ("stale", "recomputed")}  # each arm on its own honest (matching) features
    L += ["", "## Per-arm macro F0.5 (honest cell: arm evaluated on its own features)", "",
          "| arm | F0.5 @ t=0.75 | F0.5 @ best t | best t |", "|---|---|---|---|"]
    L += [f"| {a} | {c['macro_f05_ref_t']:.5f} | {c['macro_f05']:.5f} | {c['best_t']:.2f} |" for a, c in diag.items()]
    d075 = diag["recomputed"]["macro_f05_ref_t"] - diag["stale"]["macro_f05_ref_t"]
    dbest = diag["recomputed"]["macro_f05"] - diag["stale"]["macro_f05"]
    L += ["", f"Delta (recomputed - stale): {d075:+.5f} @ t=0.75, {dbest:+.5f} @ each arm's best t.", "",
          "## Per-country F0.5 @ t=0.75 (honest cell)", "",
          "| country | stale | recomputed | delta |", "|---|---|---|---|"]
    for x in sorted(diag["stale"]["by_country"]):
        s_v = diag["stale"]["by_country"][x]["ref_t"]
        r_v = diag["recomputed"]["by_country"][x]["ref_t"]
        L.append(f"| {x} | {s_v:.5f} | {r_v:.5f} | {r_v - s_v:+.5f} |")
    L += ["", "## Peak RSS per step", "", "| step | s | peak_rss_mb |", "|---|---|---|"]
    L += [f"| {r['step']} | {r['s']} | {r['peak_rss_mb']} |" for r in res["log_rows"]]
    return "\n".join(L) + "\n"


def fit_arm(d: Data, s1: pl.DataFrame, samp: np.ndarray, arm: str, wcols: list[str], work, log: Log) -> tuple[
        list[lgb.Booster], list[dict], list[int]]:
    """Fold models for one arm: load models/world_skew/{arm}_fold{k}.txt if present, else fit and save.
    Refits use work/fold{k} (recomputed context), same rows/seeds/--fast params for every arm."""
    overrides = {"learning_rate": MC["learning_rate_fast"], "num_threads": perf_cores()}
    rounds = min(MC["num_boost_round"], MC["max_rounds_fast"])
    hard, ntrue = d.meta["hard"].to_numpy(), s1["ntrue"].to_numpy()[d.code]
    d.wcols = wcols
    boosters, fits, n_train = [], [], []
    for k in range(MC["n_folds"]):
        mpath = MODELS / f"{arm}_fold{k}.txt"
        _, train, inner = split_fold(s1, k)
        train, inner = train & samp, inner & samp
        n_train.append(int(train.sum()))
        if mpath.exists():
            boosters.append(lgb.Booster(model_file=str(mpath)))
            log(f"fold {k} {arm} cached", train_rows=n_train[-1])
            continue
        context("train", dead_s1k(SEED + 1000 + k), work / f"fold{k}", d.i, log)
        tr, w = weighted_sample_rows(np.flatnonzero(train[d.code]), d.y, hard, d.in_v1, np.random.default_rng(SEED + k),
                                     ntrue)
        va = np.flatnonzero(inner[d.code])
        dtr, dva = fold_datasets(d, tr, w, va, work / f"fold{k}", overrides)
        bst = fit_ds(dtr, rounds, dva, overrides=overrides)
        del dtr, dva
        MODELS.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(mpath))
        boosters.append(bst)
        fits.append({"arm": arm, "fold": k, "best_iter": bst.best_iteration, "train_rows": len(tr),
                     "valid_logloss": float(bst.best_score["heldout"]["binary_logloss"])})
        log(f"fold {k} {arm} fit", **fits[-1])
        shutil.rmtree(work / f"fold{k}", ignore_errors=True)
    return boosters, fits, n_train


def main() -> None:
    log = Log()
    log.max_mb = WS["max_rss_mb"]
    s1 = s1_table(1.0)
    d = Data(s1)
    samp = s1.select(pl.col("s1k").is_in(s1_table(WS["sample_frac"])["s1k"].implode())).to_series().to_numpy()
    ctx = [c for c in CTX_COLS if c in d.feats]
    assert ctx, "no CTX_COLS in the model features: nothing to compare"
    arms = {"stale": [c for c in d.wcols if c not in ctx], "recomputed": list(d.wcols)}
    log("meta", rows=len(d.i), s1=s1.height, sampled_s1=int(samp.sum()), ctx=ctx)
    work = path("features_dir") / "_worlds_skew"

    alive = ~dropped(s1, SEED + 2000)
    scope = alive[d.code] & samp[d.code]  # eval rows: the fixed sample, restricted to alive S1s in (b)
    cls = rec_class(s1, alive)

    context("train", dead_s1k(SEED + 2000), work / "eval", d.i, log)  # built once, reused by every arm + shift()

    fits, n_train, cells = [], None, {}
    for arm, wcols in arms.items():  # arms sequential: fit both folds, eval both eval-cols, then discard
        bst, fit_rows, n_train_arm = fit_arm(d, s1, samp, arm, wcols, work, log)
        fits += fit_rows
        n_train = n_train_arm
        for ev_arm, ev_wcols in arms.items():
            d.wcols = ev_wcols
            meta, p = predict_scoped(d, bst, work / "eval", scope)
            c = score_scoped(s1, meta, p, ref_t=0.75)
            c["fp_by_class"] = fp_by_class(meta, p, c["best_t"], cls)
            cells[f"{arm}/{ev_arm}"] = c
            del meta, p
            gc.collect()
            log(f"cell {arm}/{ev_arm}", f05=round(c["macro_f05"], 5), t=c["best_t"], fp=c["fp_pairs"],
                distractor_fp_rate=c["fp_by_class"].get("distractor", {}).get("fp_rate"))
        del bst
        gc.collect()

    ref_t = f"{cells['stale/stale']['best_t']:.2f}"
    for c in cells.values():
        c["at_ref_t"] = c["curve"][ref_t]
    d.wcols = arms["recomputed"]
    res = {"sample_frac": WS["sample_frac"], "train_s1_per_fold": n_train, "n_s1_sampled": int(samp.sum()),
           "n_s1_sampled_alive": int((samp & alive).sum()), "ctx_cols": ctx, "ref_t": float(ref_t), "fits": fits,
           "shift": shift(d, work / "eval", scope, ctx), "cells": cells, "log_rows": log.rows}
    shutil.rmtree(work, ignore_errors=True)
    (path("oof_dir") / "world_skew.json").write_text(json.dumps(res, indent=1))
    (ROOT / "docs" / "world_skew.md").write_text(report(res))
    for k, c in cells.items():
        print(k, {x: c[x] for x in ("macro_f05", "best_t", "fp_pairs")}, c["fp_by_class"], flush=True)
    log("done")
    log.dump(path("oof_dir") / "world_skew_timing.json")


if __name__ == "__main__":
    main()
