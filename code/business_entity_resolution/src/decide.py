"""Decision rule on OOF p: per record keep only its argmax-p S1 (many-to-one assignment, ties -> lowest
s1_id), then one global threshold t. Scored against the FULL truth of the subset S1s: blocking misses and
S1s with no candidate count (their true ids can never be predicted). The t curve uses a vectorised F0.5
(bincount per S1); the chosen t is re-scored with src.metric.macro_f05 and must agree to 1e-9.
OOF caveat: argmax runs over the subset's S1s only (a record's competitors outside the 20% are absent), so the
argmax effect here is a lower bound of its test-time effect.

Expected-F0.5 layer (below, `--eval`/`--apply`, evaluated but not used for the submission): after the same per-record argmax, per S1 sort surviving
candidates by p descending and pick the prefix length k (0..K) maximising E[F0.5] under independent
Bernoulli(p_i) draws -- exact via a Poisson-binomial DP, not the single-global-t approximation above. Empty
(k=0) is always a candidate prefix, so a singleton S1 naturally gets an empty prediction when every p is low.

Writes oof/decide.json (t, argmax) and docs/matcher.md (CV metrics from oof/cv_metrics.json + this).
Run from code/business_entity_resolution/:
  python -m src.decide            # global-t rule on src.train's OOF (after src.train and src.train --loco)
  python -m src.decide --eval     # expected-F0.5 layer on OOF (b), per country vs the global t
  python -m src.decide --apply    # expected-F0.5 layer on test candidates -> output/matching_results.tsv
"""
import argparse
import json

import numpy as np
import polars as pl

from .block import norm_path
from .io import CFG, StepLog, load_gt, path, write_candidates
from .metric import macro_f05
from .train import subset_ids

MC, DC = CFG["matcher"], CFG["decide"]


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


def parse_t_country(pairs: list[str] | None) -> dict[str, float]:
    out = {}
    for kv in pairs or []:
        c, v = kv.split("=", 1)
        out[c] = float(v)
    return out


def main(t_country_override: dict[str, float] | None = None, smoke: bool = False) -> None:
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

    # per-country best t (labelled countries only: a country needs >=1 non-singleton S1 to have a curve worth
    # picking, otherwise every t ties at its singleton-only F0.5 and we'd just be encoding noise; France in the
    # train subset has none, so it falls back to the global t as required).
    t_country = {}
    for c, idx in s1.group_by("country").agg("code").rows():
        idx = np.asarray(idx)
        if (s1["ntrue"].to_numpy()[idx] > 0).any():
            remap = {old: new for new, old in enumerate(idx.tolist())}
            mask = np.isin(sc.code, idx)
            sub = Scorer.__new__(Scorer)
            sub.code = np.array([remap[v] for v in sc.code[mask]])
            sub.p, sub.y, sub.top = sc.p[mask], sc.y[mask], sc.top[mask]
            sub.ntrue, sub.n = s1["ntrue"].to_numpy()[idx], len(idx)
            curve_c = sub.curve(ts, True)
            t_country[c] = float(ts[int(np.argmax(curve_c))])

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
    if loco_path.exists() and not smoke:
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

    manual = t_country_override or {}
    effective = {**t_country, **manual}
    source = {c: ("manual" if c in manual else "computed") for c in effective}
    table = [{"country": c, "t": effective[c], "source": source[c]} for c in sorted(effective)]
    table.append({"country": "(other)", "t": t, "source": "global"})
    print(*md(table), sep="\n")

    (oof_dir / "decide.json").write_text(json.dumps({"t": t, "argmax": True, "oof_macro_f05": exact,
                                                     "fraction": MC["subset_fraction"],
                                                     "t_country": effective}, indent=1))
    cvm = json.loads((oof_dir / "cv_metrics.json").read_text())
    tm = {n: json.loads(p.read_text()) for n, p in (("features train", path("features_dir") / "train" / "features_timing.json"),
                                                   ("train cv", oof_dir / "train_timing_cv.json"),
                                                   ("train loco", oof_dir / "train_timing_loco.json")) if p.exists()}
    L = ["# Matcher evaluation (single model)",
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


# ---------------------------------------------------------------------------
# Per-S1 expected-F0.5 prefix selection (exact DP), replacing the single global t.
# ---------------------------------------------------------------------------

def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    """pmf of Sum Bernoulli(p_i), independent, in O(len(p)^2). dp[j] = P(exactly j successes so far)."""
    dp = np.zeros(len(p) + 1)
    dp[0] = 1.0
    for pi in p:
        dp[1:] = dp[1:] * (1 - pi) + dp[:-1] * pi
        dp[0] *= 1 - pi
    return dp


def expected_f05_prefix(p: np.ndarray) -> tuple[int, float]:
    """p: one S1's candidate probabilities, sorted descending. Returns (k*, E[F0.5](k*)) over prefixes
    k=0..len(p), k=0 (empty set) included and scored 1 when the S1 is truly empty (T_total=0).
    Exact: predicting the top-k fixes precision's denominator (k) but both TP (successes in the prefix) and
    T_total = TP + FN_suffix (successes among ALL candidates, prefix+suffix) are random. FN_suffix is
    independent of the prefix draws, so E[F0.5(k)] is a weighted sum over (tp in 0..k, fn in 0..n-k) of
    P(TP=tp) x P(FN_suffix=fn) x f05(tp, k, tp+fn) -- a convolution of two Poisson-binomial pmfs computed once
    (forward DP over the prefix, backward DP over the suffix) and combined per k in O(n) each, O(n^2) total."""
    n = len(p)
    fwd = [poisson_binomial_pmf(p[:k]) for k in range(n + 1)]  # fwd[k][j] = P(TP=j | first k)
    bwd = [poisson_binomial_pmf(p[k:]) for k in range(n + 1)]  # bwd[k][m] = P(FN_suffix=m | last n-k)
    best_k, best_e = 0, float(fwd[0][0] * bwd[0][0])  # k=0: only T_total=0 scores (empty/empty = 1)
    for k in range(1, n + 1):
        dp, suf = fwd[k], bwd[k]
        tp = np.arange(1, k + 1)  # tp=0 always scores 0 for k>0, so start at 1 and skip that row
        fn = np.arange(n - k + 1)
        t = tp[:, None] + fn[None, :]  # T_total per (tp, fn) cell
        f = 1.25 * tp[:, None] / (0.25 * t + k)
        e = float((dp[tp, None] * suf[None, :] * f).sum())
        if e > best_e:
            best_k, best_e = k, e
    return best_k, best_e


def decide_group(p: np.ndarray) -> np.ndarray:
    """p: descending-sorted probs of one S1's surviving candidates. Returns a boolean keep-mask (prefix)."""
    if len(p) > DC["max_cand"]:  # DP cap: fall back to a flat p>=0.5 cut on this S1 only (rare)
        k = int((p >= 0.5).sum())
    else:
        k, _ = expected_f05_prefix(p)
    mask = np.zeros(len(p), bool)
    mask[:k] = True
    return mask


def decide_expected(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_id, rec_id, p (already argmax-per-record). Returns the kept rows (the predicted set per S1)."""
    return (pairs.sort(["s1_id", "p"], descending=[False, True])
            .with_columns(_keep=pl.col("p").map_batches(lambda s: pl.Series(decide_group(s.to_numpy())),
                                                         return_dtype=pl.Boolean).over("s1_id"))
            .filter(pl.col("_keep")).drop("_keep"))


def eval_expected(log: StepLog) -> None:
    """--eval: expected-F0.5 layer on OOF (b) (oof/oof_full.parquet, cv_full's test-density world) vs the
    cv_full.json global-t baseline, overall and per country. Works entirely in code/reck space (cv_full's row
    keys), scored with src.metric.macro_f05 directly -- an offline eval, exactness over speed."""
    from .cv_full import s1_table, truth_keys  # local: cv_full imports decide.f05_vec, avoid a circular import
    O = path("oof_dir")
    s1 = s1_table(1.0)
    oof = pl.read_parquet(O / "oof_full.parquet").filter(pl.col("p_td").is_not_null()).join(
        s1.select(s1k="s1k", code="code"), on="s1k")
    top = with_top(oof.select(s1_id="code", rec_id="reck", p="p_td")).filter("top")
    kept = decide_expected(top.select("s1_id", "rec_id", "p"))
    pred = {a: set(b) for a, b in kept.group_by("s1_id").agg("rec_id").iter_rows()}
    truth = truth_keys(s1)  # code -> set(reck)
    scope_codes = set(oof["code"].unique().to_list())
    truth = {c: set(truth.get(c, ())) for c in scope_codes}
    exact, br = macro_f05(pred, truth)
    log("expected-F0.5 OOF (b)", macro_f05=round(exact, 5), n_s1=br["n_singleton"] + br["n_nonsingleton"])
    cv = json.loads((O / "cv_full.json").read_text())["b_test_density"]
    rows = [{"scope": "overall", "F0.5 expected-F": exact, "F0.5 global t": cv["macro_f05"]}]
    ctry_by_code = dict(zip(s1["code"].to_list(), s1["country"].to_list()))
    for c in sorted(set(s1["country"].to_list())):
        codes = {k for k in scope_codes if ctry_by_code[k] == c}
        f_c, _ = macro_f05({k: pred.get(k, set()) for k in codes}, {k: truth[k] for k in codes})
        rows.append({"scope": c, "F0.5 expected-F": f_c, "F0.5 global t": cv.get(f"f05_{c}")})
    print(*md(rows), sep="\n")
    (O / "decide_expected.json").write_text(json.dumps(
        {"macro_f05": exact, "vs_global_t": cv["macro_f05"], "rows": rows}, indent=1))


def apply_expected(log: StepLog) -> None:
    """--apply: expected-F0.5 layer on test candidates (oof/test_p.parquet, from predict --folds)
    -> output/matching_results.tsv. Same argmax + candidate-set assertions as predict.py.
    Post-DP per-country floor: t_country from oof/decide.json (computed + --t-country manual overrides) drops
    any selected pair whose p falls below its country's floor; countries absent from t_country are untouched, so
    with an empty/missing t_country this filter is a no-op mask (p < null -> false everywhere) and the output is
    unchanged from the pre-floor code path -- structurally, not by a runtime re-check."""
    O = path("oof_dir")
    t_country = {}
    dec_path = O / "decide.json"
    if dec_path.exists():
        t_country = json.loads(dec_path.read_text()).get("t_country") or {}
    scored = pl.read_parquet(O / "test_p.parquet")
    top = with_top(scored).filter("top").select("s1_id", "rec_id", "p")
    matches = decide_expected(top).select("s1_id", "rec_id", "p")

    if t_country:
        country = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "country"])
        tmap = pl.DataFrame({"country": list(t_country), "_floor": list(t_country.values())})
        tagged = matches.join(country, left_on="s1_id", right_on="entity_id", how="left").join(
            tmap, on="country", how="left")
        keep = tagged.filter(pl.col("_floor").is_null() | (pl.col("p") >= pl.col("_floor")))
        dropped = tagged.filter(pl.col("_floor").is_not_null() & (pl.col("p") < pl.col("_floor")))

        summary = []
        for c in sorted(t_country):
            b = tagged.filter(pl.col("country") == c)
            a = keep.filter(pl.col("country") == c)
            d = dropped.filter(pl.col("country") == c)
            summary.append({"country": c, "s1_changed": d["s1_id"].n_unique(), "pairs_dropped": d.height,
                             "mean_matches_before": round(b.height / max(b["s1_id"].n_unique(), 1), 3),
                             "mean_matches_after": round(a.height / max(a["s1_id"].n_unique(), 1), 3)})
        print(*md(summary), sep="\n")
        matches = keep.select("s1_id", "rec_id")
    else:
        matches = matches.select("s1_id", "rec_id")

    cands = pl.scan_parquet(path("interim_dir") / "candidates_test.parquet").select("s1_id", "rec_id")
    outside = matches.lazy().join(cands, on=["s1_id", "rec_id"], how="anti").select(pl.len()).collect().item()
    assert outside == 0, f"{outside} matches not in candidate set"
    assert matches["rec_id"].is_unique().all(), "a record matched to >1 S1"
    s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"])["entity_id"]
    write_candidates(path("matching_results"), matches.lazy(), s1_ids, list_col="matched_entity_ids")
    n_with = matches["s1_id"].n_unique()
    log("expected-F decide + write", matches=matches.height, s1_with_match=n_with, s1_total=s1_ids.len(),
        empty_pct=round(100 * (1 - n_with / s1_ids.len()), 2))


def cli() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--eval", action="store_true", help="expected-F0.5 layer on OOF (b) vs global t")
    g.add_argument("--apply", action="store_true", help="expected-F0.5 layer on test -> matching_results.tsv")
    ap.add_argument("--t-country", action="append", metavar="COUNTRY=t",
                     help="manual per-country t override for main(); repeatable, wins over computed")
    ap.add_argument("--smoke", action="store_true", help="tiny fast pass: skip loco section, print done")
    a = ap.parse_args()
    if a.eval:
        eval_expected(StepLog())
    elif a.apply:
        apply_expected(StepLog())
    else:
        main(parse_t_country(a.t_country), smoke=a.smoke)


if __name__ == "__main__":
    cli()
