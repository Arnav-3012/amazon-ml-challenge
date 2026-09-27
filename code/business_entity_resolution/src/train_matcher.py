"""M4 step 2: LightGBM pair matcher on features/train -> OOF p for every candidate pair, one global threshold.

- GroupKFold by S1 entity over the pairs: one S1's pairs never span fit and OOF fold.
- Fit rows: all pairs of a seeded `train_entity_frac` of S1 entities (RAM: 78M x 57 float32 does not fit);
  pairs are never subsampled within an entity, so each S1's candidate set and the label prior stay as blocking
  produced them (no calibration correction needed). Early stopping uses held-out *fit* entities, never the OOF fold.
- OOF p is predicted for every train pair (all entities) by the model of its fold.
- Threshold: the one global p cut that maximises OOF macro F0.5 over ALL S1 entities in scope (entities with no
  candidates or with blocking-missed matches count; exact per-entity formula F = 1.25 tp / (0.25 n_true + n_pred)).
Writes models/matcher_fold{k}.txt, models/matcher_meta.json, oof/matcher/part-*.parquet, docs/matching.md.

Run from code/business_entity_resolution/:
  python -m src.train_matcher --features-dir features/sample_train   # quick end-to-end check on a sample
  python -m src.train_matcher
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import GroupKFold

from .features import CATEGORICAL, FEATURES
from .io import CFG, ROOT, load_gt_pairs, path
from .metric import f05
from .normalise import peak_rss_mb

P = CFG["matcher"]
SEED = CFG["seed"]
REPORT = ROOT / "docs" / "matching.md"
META = "matcher_meta.json"


def _parts(d: Path) -> list[Path]:
    parts = sorted(d.glob("part-*.parquet"))
    assert parts, f"no feature parts in {d}"
    return parts


def _X(df: pl.DataFrame) -> np.ndarray:
    return df.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()


def entity_scope(d: Path) -> tuple[pl.DataFrame, np.ndarray]:
    """S1 entities in scope (index e, country for reporting only) and n_true per entity from the ground truth
    (true pairs blocking missed still count: they are recall the submission loses)."""
    ents = pl.read_parquet(d / "s1_ids.parquet").with_row_index("e")
    n_true = (load_gt_pairs().group_by("s1_id").len()
              .join(ents.select("s1_id", "e"), on="s1_id").select("e", "len"))
    arr = np.zeros(ents.height, np.int64)
    arr[n_true["e"].to_numpy()] = n_true["len"].to_numpy()
    return ents, arr


def macro_by_threshold(e: np.ndarray, y: np.ndarray, p: np.ndarray, n_true: np.ndarray, ts) -> np.ndarray:
    n = n_true.size
    tp_all = y.astype(bool)
    out = []
    for t in ts:
        pred = p >= t
        out.append(entity_f05(np.bincount(e[pred & tp_all], minlength=n), np.bincount(e[pred], minlength=n),
                              n_true).mean())
    return np.array(out)


def entity_f05(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 = 1.25 tp / (0.25 n_true + n_pred); singleton: 1 if nothing predicted, else 0."""
    denom = 0.25 * n_true + n_pred
    f = np.divide(1.25 * tp, denom, out=np.zeros(tp.size), where=denom > 0)
    return np.where(n_true == 0, (n_pred == 0).astype(float), f)


def _crosscheck(e, y, pred, n_true, f_vec, rng) -> float:
    """Exact replica (metric.f05) on a seeded subset of entities vs the vectorised formula: max abs difference.
    Truth = captured true pairs (row ids) + one dummy id per true pair blocking missed."""
    pick = rng.choice(n_true.size, min(50_000, n_true.size), replace=False)
    rows = np.flatnonzero(np.isin(e, pick))
    pred_sets = {int(k): set() for k in pick}
    cap = {int(k): set() for k in pick}
    for r, k in zip(rows.tolist(), e[rows].tolist()):
        if pred[r]:
            pred_sets[k].add(r)
        if y[r]:
            cap[k].add(r)
    exact = np.array([f05(pred_sets[k], cap[k] | {("miss", j) for j in range(n_true[k] - len(cap[k]))})
                      for k in pick.tolist()])
    return float(np.abs(exact - f_vec[pick]).max())


def run(d: Path, tag: str) -> None:
    t0 = time.perf_counter()
    parts = _parts(d)
    ents, n_true = entity_scope(d)
    emap = ents.select("s1_id", "e")

    # ---- pass 1: entity index + label per row
    E, Y, sizes = [], [], []
    for pth in parts:
        df = pl.read_parquet(pth, columns=["s1_id", "label"])
        E.append(df["s1_id"].replace_strict(emap["s1_id"], emap["e"], return_dtype=pl.UInt32).to_numpy())
        Y.append(df["label"].to_numpy())
        sizes.append(df.height)
    e, y = np.concatenate(E), np.concatenate(Y).astype(np.uint8)
    del E, Y
    n = e.size
    assert np.all(np.bincount(e[y == 1], minlength=n_true.size) <= n_true), "label count > ground truth"

    # ---- folds (GroupKFold by S1 entity) and the fit-entity sample
    fold = np.empty(n, np.int8)
    gkf = GroupKFold(n_splits=P["n_folds"], shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(gkf.split(np.zeros((n, 1), np.uint8), groups=e)):
        fold[va] = k
    ent_fold = np.full(n_true.size, -1, np.int8)
    ent_fold[e] = fold
    rng = np.random.default_rng(SEED)
    fit_ent = rng.random(n_true.size) < P["train_entity_frac"]
    es_ent = rng.random(n_true.size) < P["es_entity_frac"]
    fit_rows = fit_ent[e]
    n_fit = int(fit_rows.sum())
    print(f"{n:,} pairs, {n_true.size:,} S1, {int(y.sum()):,} positives; fit rows {n_fit:,}", flush=True)

    # ---- fit matrix (preallocated; filled part by part)
    X = np.empty((n_fit, len(FEATURES)), np.float32)
    pos, off = 0, 0
    for pth, sz in zip(parts, sizes):
        msk = fit_rows[off:off + sz]
        if msk.any():
            xp = _X(pl.read_parquet(pth, columns=FEATURES).filter(pl.Series(msk)))
            X[pos:pos + xp.shape[0]] = xp
            pos += xp.shape[0]
        off += sz
    y_fit, f_fit, es_fit = y[fit_rows], fold[fit_rows], es_ent[e[fit_rows]]
    params = dict(P["lgb"], seed=SEED)
    ds = lgb.Dataset(X, label=y_fit, feature_name=FEATURES, categorical_feature=CATEGORICAL,
                     free_raw_data=True, params={"max_bin": params["max_bin"], "verbose": -1})
    ds.construct()
    del X
    t_data = time.perf_counter() - t0

    # ---- one model per fold
    models, best_it, gain = [], [], np.zeros(len(FEATURES))
    mdir = path("models_dir")
    mdir.mkdir(parents=True, exist_ok=True)
    for k in range(P["n_folds"]):
        tk = time.perf_counter()
        fit_idx = np.flatnonzero((f_fit != k) & ~es_fit)
        es_idx = np.flatnonzero((f_fit != k) & es_fit)
        b = lgb.train(params, ds.subset(fit_idx), num_boost_round=P["max_rounds"],
                      valid_sets=[ds.subset(es_idx)], valid_names=["es"],
                      callbacks=[lgb.early_stopping(P["early_stop"], verbose=False), lgb.log_evaluation(100)])
        b.save_model(str(mdir / f"matcher{tag}_fold{k}.txt"), num_iteration=b.best_iteration)
        models.append(b)
        best_it.append(b.best_iteration)
        gain += b.feature_importance("gain", iteration=b.best_iteration)
        print(f"fold {k}: best_iteration {b.best_iteration}, {time.perf_counter() - tk:.0f}s, "
              f"peak RSS {peak_rss_mb()} MB", flush=True)
    t_fit = time.perf_counter() - t0 - t_data

    # ---- OOF prediction for every pair (streamed; also written for M5)
    odir = path("oof_dir") / f"matcher{tag}"
    odir.mkdir(parents=True, exist_ok=True)
    for old in odir.glob("*.parquet"):
        old.unlink()
    p = np.empty(n, np.float32)
    off = 0
    for j, (pth, sz) in enumerate(zip(parts, sizes)):
        df = pl.read_parquet(pth, columns=["s1_id", "match_id", *FEATURES])
        xp, fp = _X(df), fold[off:off + sz]
        pp = np.empty(sz, np.float32)
        for k, b in enumerate(models):
            s = fp == k
            if s.any():
                pp[s] = b.predict(xp[s], num_iteration=b.best_iteration)
        p[off:off + sz] = pp
        df.select("s1_id", "match_id").with_columns(label=pl.Series(y[off:off + sz]), fold=pl.Series(fp),
                                                    p=pl.Series(pp)).write_parquet(odir / f"part-{j:04d}.parquet")
        off += sz
    t_pred = time.perf_counter() - t0 - t_data - t_fit

    # ---- metrics
    lo, hi, step = P["threshold_grid"]
    ts = np.round(np.arange(lo, hi + step / 2, step), 4)
    curve = macro_by_threshold(e, y, p, n_true, ts)
    t_best = float(ts[int(np.argmax(curve))])
    pred = p >= t_best
    f_ent = entity_f05(np.bincount(e[pred & (y == 1)], minlength=n_true.size),
                       np.bincount(e[pred], minlength=n_true.size), n_true)
    f_ceiling = entity_f05(np.bincount(e[y == 1], minlength=n_true.size),
                           np.bincount(e[y == 1], minlength=n_true.size), n_true)
    diff = _crosscheck(e, y, pred, n_true, f_ent, np.random.default_rng(SEED))
    assert diff < 1e-9, f"vectorised F0.5 != metric.f05 (max diff {diff})"
    bins = np.minimum((p * 20).astype(int), 19)
    cal = pl.DataFrame({"bin": bins, "p": p, "y": y}).group_by("bin").agg(
        n=pl.len(), mean_p=pl.col("p").mean(), frac_pos=pl.col("y").mean()).sort("bin")
    ece = float((cal["n"] * (cal["mean_p"] - cal["frac_pos"]).abs()).sum() / n)
    tp = int((pred & (y == 1)).sum())
    m = {"pairs": n, "entities": int(n_true.size), "positives": int(y.sum()), "true_pairs_gt": int(n_true.sum()),
         "pr_auc": float(average_precision_score(y, p)), "log_loss": float(log_loss(y, p.astype(np.float64))),
         "ece": ece, "threshold": t_best, "oof_macro_f05": float(f_ent.mean()),
         "ceiling_macro_f05": float(f_ceiling.mean()), "pair_precision": tp / max(int(pred.sum()), 1),
         "pair_recall_of_gt": tp / max(int(n_true.sum()), 1), "best_iterations": best_it,
         "crosscheck_max_diff": diff, "fit_rows": n_fit,
         "seconds": {"data": round(t_data), "fit": round(t_fit), "oof_predict": round(t_pred),
                     "total": round(time.perf_counter() - t0)}, "peak_rss_mb": peak_rss_mb()}
    ent_tab = ents.select("country").with_columns(
        kind=pl.Series(np.select([n_true == 0, n_true == 1], ["singleton", "1 match"], ">=2 matches")),
        f=pl.Series(f_ent), ceil=pl.Series(f_ceiling),
        empty_pred=pl.Series(np.bincount(e[pred], minlength=n_true.size) == 0))
    meta = {"features": FEATURES, "categorical": CATEGORICAL, "threshold": t_best, "tag": tag,
            "models": [f"matcher{tag}_fold{k}.txt" for k in range(P["n_folds"])], "best_iterations": best_it,
            "features_dir": str(d), "oof": {k: v for k, v in m.items() if k != "seconds"}}
    (mdir / f"matcher{tag}_meta.json").write_text(json.dumps(meta, indent=1))
    imp = sorted(zip(FEATURES, gain / max(gain.sum(), 1e-12)), key=lambda x: -x[1])
    REPORT.write_text(report(m, ts, curve, cal, ent_tab, imp, d), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items()}, indent=1))
    print(f"wrote {REPORT}")


def report(m, ts, curve, cal, ent_tab, imp, d) -> str:
    pct = lambda x: f"{100 * x:.2f}"  # noqa: E731
    L = ["# Matcher v1 (M4) report (generated by `python -m src.train_matcher`; do not hand-edit)", "",
         f"Features: `{d}`, {m['pairs']:,} candidate pairs, {m['entities']:,} S1 entities, "
         f"{m['positives']:,} positive pairs of {m['true_pairs_gt']:,} ground-truth pairs "
         f"(the rest were lost in blocking). Fit on all pairs of {P['train_entity_frac']:.0%} of S1 entities "
         f"({m['fit_rows']:,} rows); {P['n_folds']}-fold GroupKFold by S1; OOF p for every pair. "
         f"Best iterations per fold: {m['best_iterations']}.", "",
         "## OOF summary", "",
         "| metric | value |", "|---|---|",
         f"| **macro F0.5 @ global threshold {m['threshold']}** | **{m['oof_macro_f05']:.4f}** |",
         f"| ceiling (perfect matcher on these candidates) | {m['ceiling_macro_f05']:.4f} |",
         f"| PR-AUC (pairs) | {m['pr_auc']:.4f} |", f"| log-loss (pairs) | {m['log_loss']:.4f} |",
         f"| ECE (20 bins) | {m['ece']:.4f} |",
         f"| pair precision @ threshold | {pct(m['pair_precision'])}% |",
         f"| pair recall vs all GT pairs @ threshold | {pct(m['pair_recall_of_gt'])}% |",
         f"| vectorised F0.5 vs metric.f05 max diff (50k entities) | {m['crosscheck_max_diff']:.1e} |", "",
         "## Macro F0.5 by country and entity type (OOF, at the threshold)", "",
         "Country is used here for reporting only. `ceiling` = perfect matcher on the blocked candidates, so "
         "`ceiling − F0.5` is matcher/threshold loss and `1 − ceiling` is blocking loss.", "",
         "| country | entity type | S1 | share | F0.5 | ceiling | predicted empty |", "|---|---|---|---|---|---|---|"]
    kinds = ["singleton", "1 match", ">=2 matches"]
    tot = ent_tab.height
    for c in sorted(ent_tab["country"].unique().to_list()) + ["all"]:
        sub = ent_tab if c == "all" else ent_tab.filter(pl.col("country") == c)
        for k in kinds + ["all"]:
            s = sub if k == "all" else sub.filter(pl.col("kind") == k)
            if s.height:
                L.append(f"| {c} | {k} | {s.height:,} | {pct(s.height / tot)}% | {s['f'].mean():.4f} | "
                         f"{s['ceil'].mean():.4f} | {pct(s['empty_pred'].mean())}% |")
    L += ["", "## Threshold curve (OOF macro F0.5)", "", "| threshold | macro F0.5 |", "|---|---|"]
    for t, f in zip(ts, curve):
        if abs(round(t * 100) % 5) < 1e-9 or t == m["threshold"]:
            L.append(f"| {t:.2f}{' ←' if t == m['threshold'] else ''} | {f:.4f} |")
    L += ["", "## Calibration (OOF, 20 equal-width bins)", "", "| p bin | pairs | mean p | observed positive rate |",
          "|---|---|---|---|"]
    for r in cal.iter_rows(named=True):
        L.append(f"| {r['bin'] / 20:.2f}–{(r['bin'] + 1) / 20:.2f} | {r['n']:,} | {r['mean_p']:.4f} | "
                 f"{r['frac_pos']:.4f} |")
    L += ["", "## Feature importance (gain share, summed over folds)", "", "| feature | gain share |", "|---|---|"]
    L += [f"| {f} | {pct(v)}% |" for f, v in imp]
    s = m["seconds"]
    L += ["", f"Runtime: data {s['data']}s, fit {s['fit']}s, OOF predict {s['oof_predict']}s, total {s['total']}s; "
              f"peak RSS {m['peak_rss_mb']} MB.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=None, help="default: features/train")
    args = ap.parse_args()
    d = Path(args.features_dir) if args.features_dir else path("features_dir") / "train"
    if not d.is_absolute():
        d = ROOT / d
    run(d, "" if d.name == "train" else f"_{d.name}")


if __name__ == "__main__":
    main()
