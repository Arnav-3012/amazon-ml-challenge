"""M4 baseline decision on OOF p: per record keep only its argmax-p S1 (many-to-one assignment, ties -> lowest
s1_id), then one global threshold t. Scored against the FULL truth of the subset S1s: blocking misses and
S1s with no candidate count (their true ids can never be predicted). The t curve uses a vectorised F0.5
(bincount per S1); the chosen t is re-scored with src.metric.macro_f05 and must agree to 1e-9.
OOF caveat: argmax runs over the subset's S1s only (a record's competitors outside the 20% are absent), so the
argmax effect here is a lower bound of its test-time effect.

Writes oof/decide.json (t, argmax) and docs/matcher.md (CV metrics from oof/cv_metrics.json + this).
Run from code/business_entity_resolution/:  python -m src.decide   (after src.train and src.train --loco)
"""
import json

import numpy as np
import polars as pl

from .block import norm_path
from .io import CFG, StepLog, load_gt, path
from .metric import macro_f05
from .train import subset_ids

MC = CFG["matcher"]


def with_top(df: pl.DataFrame) -> pl.DataFrame:
    return (df.sort(["rec_id", "p", "s1_id"], descending=[False, True, False])
            .with_columns(top=pl.col("rec_id").is_first_distinct()))


def f05_vec(tp: np.ndarray, npred: np.ndarray, ntrue: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / npred, tp / ntrue
        f = 1.25 * p * r / (0.25 * p + r)
    return np.where((npred == 0) & (ntrue == 0), 1.0, np.where(tp == 0, 0.0, f))


class Scorer:
    """Pair rows (code = S1 row in `s1`, p, label, top) vs per-S1 true counts incl. unreachable matches."""

    def __init__(self, pairs: pl.DataFrame, s1: pl.DataFrame):
        d = pairs.join(s1.select("s1_id", "code"), on="s1_id", how="inner")
        self.code, self.p = d["code"].to_numpy(), d["p"].to_numpy()
        self.y, self.top = d["label"].to_numpy().astype(bool), d["top"].to_numpy()
        self.ntrue, self.n = s1["ntrue"].to_numpy(), s1.height

    def kept(self, t: float, argmax: bool) -> np.ndarray:
        return (self.p >= t) & (self.top if argmax else True)

    def per_s1(self, t: float, argmax: bool) -> np.ndarray:
        k = self.kept(t, argmax)
        return f05_vec(np.bincount(self.code[k & self.y], minlength=self.n),
                       np.bincount(self.code[k], minlength=self.n), self.ntrue)

    def curve(self, ts: np.ndarray, argmax: bool) -> np.ndarray:
        return np.array([self.per_s1(t, argmax).mean() for t in ts])


def s1_table(ids: pl.Series, truth: dict) -> pl.DataFrame:
    country = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"])
    ntrue = pl.DataFrame({"s1_id": ids, "ntrue": [len(truth[i]) for i in ids]})
    return (ntrue.join(country, left_on="s1_id", right_on="entity_id", how="left", maintain_order="left")
            .with_row_index("code").with_columns(
                card=pl.when(pl.col("ntrue") == 0).then(pl.lit("singleton")).when(pl.col("ntrue") == 1)
                .then(pl.lit("1")).when(pl.col("ntrue") <= 4).then(pl.lit("2-4")).otherwise(pl.lit("5+"))))


def md(rows: list[dict]) -> list[str]:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
    cols = list(rows[0])
    return (["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows])


def main() -> None:
    log, oof_dir = StepLog(), path("oof_dir")
    truth = load_gt(validate=False)
    ids = subset_ids(MC["subset_fraction"])
    s1 = s1_table(ids, truth)
    oof = with_top(pl.read_parquet(oof_dir / "oof_train.parquet"))
    sc = Scorer(oof, s1)
    log("load", pairs=oof.height, s1=s1.height)

    ts = np.round(np.arange(*CFG["decide"]["t_grid"]), 4)
    cur = {am: sc.curve(ts, am) for am in (True, False)}
    bi = {am: int(np.argmax(c)) for am, c in cur.items()}
    t = float(ts[bi[True]])
    f_best = float(cur[True][bi[True]])
    pred = oof.filter((pl.col("p") >= t) & pl.col("top")).group_by("s1_id").agg("rec_id")
    exact, br = macro_f05({s: set(r) for s, r in pred.iter_rows()}, {i: truth[i] for i in ids})
    assert abs(exact - f_best) < 1e-9, (exact, f_best)
    log("curve + exact check", t=t, f05=round(exact, 6))

    per = sc.per_s1(t, True)
    per_off = sc.per_s1(t, False)
    split = []
    for col in ("card", "country"):
        for g, idx in s1.group_by(col).agg("code").sort(col).rows():
            idx = np.asarray(idx)
            split.append({"split": col, "group": g, "n_s1": len(idx), "F0.5 argmax": float(per[idx].mean()),
                          "F0.5 no argmax": float(per_off[idx].mean())})

    loco_rows = []
    loco_path = oof_dir / "loco_train.parquet"
    if loco_path.exists():
        loco = pl.read_parquet(loco_path)
        for c in sorted(loco["eval_country"].unique()):
            s1c = s1.filter(pl.col("country") == c).drop("code").with_row_index("code")
            lsc = Scorer(with_top(loco.filter(pl.col("eval_country") == c).drop("eval_country")), s1c)
            isc = Scorer(oof.filter(pl.col("s1_id").is_in(s1c["s1_id"].implode())), s1c)
            lc = lsc.curve(ts, True)
            f_in, f_lo = float(isc.per_s1(t, True).mean()), float(lsc.per_s1(t, True).mean())
            loco_rows.append({"eval country": c, "n_s1": s1c.height, "in-dist OOF @t": f_in, "LOCO @t": f_lo,
                              "gap @t": f_lo - f_in, "LOCO own best t": float(ts[int(np.argmax(lc))]),
                              "LOCO @ own best t": float(lc.max())})
        log("loco", countries=len(loco_rows))

    (oof_dir / "decide.json").write_text(json.dumps({"t": t, "argmax": True, "oof_macro_f05": exact,
                                                     "fraction": MC["subset_fraction"]}, indent=1))
    cvm = json.loads((oof_dir / "cv_metrics.json").read_text())
    tm = {n: json.loads(p.read_text()) for n, p in (("features train", path("features_dir") / "train" / "features_timing.json"),
                                                   ("train cv", oof_dir / "train_timing_cv.json"),
                                                   ("train loco", oof_dir / "train_timing_loco.json")) if p.exists()}
    L = ["# Matcher v1 evaluation (M4)",
         f"Generated by `src/decide.py`. Subset: {cvm['n_s1']:,} train S1 ({cvm['fraction']:.0%}, seed {CFG['seed']}), "
         f"{cvm['rows']:,} candidate pairs, {cvm['pos']:,} positive. GroupKFold({MC['n_folds']}) by s1_id; training "
         f"rows = positives + {MC['neg_ratio']}x negatives ({MC['hard_frac']:.0%} hardest by max channel score). "
         "F0.5 is against the full truth of the subset S1s (blocking misses included).", "",
         "## 1. Pair-level OOF",
         f"- PR-AUC {cvm['pr_auc']:.5f}, log-loss {cvm['logloss']:.5f} (p is not calibrated: negatives subsampled).",
         f"- Best iterations per fold {cvm['best_iters']}, mean {cvm['mean_best_iter']:.1f}.", "",
         "Calibration (10 equal-width bins of p):", "", *md(cvm["calibration"]), "",
         "Top-30 features by mean gain:", "", *md(cvm["gain_top30"]), "",
         "## 2. Macro F0.5 vs threshold",
         f"Best t = **{t:.2f}**, OOF macro F0.5 = **{exact:.4f}** with argmax (exact metric agrees); "
         f"without argmax best t = {ts[bi[False]]:.2f}, F0.5 = {cur[False][bi[False]]:.4f}. "
         f"Singletons {br['n_singleton']:,} at F0.5 {br['f05_singleton']:.4f}; non-singletons {br['f05_nonsingleton']:.4f}.",
         "", *md([{"t": float(x), "F0.5 argmax": float(a), "F0.5 no argmax": float(b)}
                  for x, a, b in zip(ts, cur[True], cur[False]) if round(x * 100) % 5 == 0 or x == t]), "",
         f"## 3. Split at t = {t:.2f}", "", *md(split), ""]
    if loco_rows:
        L += ["## 4. Leave-one-country-out",
              "Train on the other country's subset rows at the mean CV best iteration, score this country, decide at "
              "the pooled OOF t. gap = LOCO - in-distribution OOF on the same S1s.", "", *md(loco_rows), ""]
    L += ["## 5. Runtime and peak memory", ""]
    for name, x in tm.items():
        L += [f"**{name}**: total {x['total_s']}s, peak RSS {x['peak_rss_mb']} MB", "",
              *md([{k: s[k] for k in ("step", "s", "peak_rss_mb")} for s in x["steps"]]), ""]
    rp = path("matcher_report")
    rp.write_text("\n".join(L) + "\n")
    log("report", path=str(rp))


if __name__ == "__main__":
    main()
