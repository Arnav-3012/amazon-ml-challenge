"""M5-D0 diagnostics (docs/m5-strategy-l3.md §3). Read-only: no training, no change to any other module.
Each check writes artifacts/diagnose/d{n}.md; every run re-assembles all existing sections into docs/diagnose_m5.md.

  D0-1 blocking misses: true train pairs absent from candidates_train, by country x native script (raw S2/S3 name
       or address has non-ASCII) x record has address x name / address token_set bucket. Exact block ranks of each
       missed pair in every channel and both directions, recomputed with block.py's own functions from idf_train:
       depth miss (best rank <= 200) / deep (> 200) / vocabulary miss (no shared surviving token in any channel).
       Control: 2,000 hit pairs per country must reproduce their stored X ranks (asserted).
  D0-2 FN anatomy on the M4 OOF (20% subset, t from oof/decide.json): (a) below t, record argmax = this S1,
       (b) record argmax went to another S1; p histograms; oracle macro gain per bucket.
  D0-3 FP anatomy on the M4 OOF: singleton vs extra FP, name/addr token_set buckets, record status.
  D0-4 label-free test density: predicted matches per S1 (output/matching_results.tsv) and share of records whose
       max p > 0.5 (models/lgb_final.txt on a record sample; train side = held-out records, see the section note).
  D0-5 orphan simulation: drop 19% of ALL train S1s (seed 42); for the records of sampled held-out S1s (not in the
       final-fit 50%) re-rank + re-cut blocking ranks (block.finalize) and recompute features.relative; re-predict
       with models/lgb_final.txt (train.py saves no fold models); decide.py rule; macro F0.5 before/after.
  D0-6 per-country OOF macro F0.5 + generator constants (singleton rate, cardinality).
  D0-7 leak sanity: Spearman of S1 vs matched S2/S3 id number and file row. Report only; never a feature.
  D0-8 (M5-1) D0-1's largest miss cell (US, no record address, ASCII, name_sim >= 80): how many US S1s share the
       record's / the true S1's core_name, missed vs hit. Run it BEFORE src.gate --apply replaces candidates_train.
All macro F0.5 values are over EVERY S1 of the evaluated set (zero-candidate S1s count, blocking misses count).

Run from code/business_entity_resolution/:  python -m src.diagnose [--d 1 5]
"""
import argparse
import json
import time
from functools import cache

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from scipy.stats import spearmanr

from .block import (BCFG, CHANNELS, RANK_COLS, SOURCES, _chunks, _fallback, finalize, load, matrix, norm_path,
                    postings, survivors, token_table)
from .decide import f05_vec, md, s1_table, with_top
from .features import REL_COLS, _cp, id_key, relative
from .io import CFG, ROOT, StepLog, load_gt, load_gt_pairs, path, peak_rss_mb
from .normalise import NONASCII
from .train import parts, subset_ids

MC, SEED, M, K = CFG["matcher"], CFG["seed"], BCFG["m"], BCFG["k"]
T = json.loads((path("oof_dir") / "decide.json").read_text())["t"]
TS = np.round(np.arange(*CFG["decide"]["t_grid"]), 4)
SECT, REPORT = path("artifacts_dir") / "diagnose", ROOT / "docs" / "diagnose_m5.md"
DROP_FRAC, MAX_RANK, CTRL = 0.19, 200, 2000
PAIR = ["s1_id", "rec_id"]
S1_SIDE = ["n_cand_s1", *(f"{c}_{x}" for c in REL_COLS for x in ("ds1", "rks1"))]


# ---------- shared ----------
@cache
def truth() -> dict:
    return load_gt(validate=False)


@cache
def gt_pairs() -> pl.DataFrame:
    return load_gt_pairs().select("s1_id", rec_id="match_id")


@cache
def booster() -> tuple[lgb.Booster, list[str]]:
    meta = json.loads((path("models_dir") / "lgb_final.json").read_text())
    return lgb.Booster(model_file=str(path("models_dir") / "lgb_final.txt")), meta["features"]


def predict(df: pl.DataFrame, step: int = 2_000_000) -> np.ndarray:
    bst, feats = booster()
    return np.concatenate([bst.predict(df.slice(i, step).select(pl.col(feats).cast(pl.Float32)).to_numpy())
                           for i in range(0, df.height, step)] or [np.zeros(0)]).astype(np.float32)


def per_s1(code: np.ndarray, keep: np.ndarray, y: np.ndarray, ntrue: np.ndarray, extra_tp: np.ndarray | int = 0):
    """Per-S1 F0.5 over every S1 in ntrue (S1s with no row score 0 unless singleton). extra_tp adds recovered pairs."""
    n = len(ntrue)
    tp, npred = np.bincount(code[keep & y], minlength=n), np.bincount(code[keep], minlength=n)
    return f05_vec(tp + extra_tp, npred + extra_tp, ntrue)


def isin(s: pl.Series, values: pl.Series) -> np.ndarray:
    return s.to_frame().select(pl.first().is_in(values.implode())).to_series().to_numpy()


def sample(s: pl.Series, n: int, seed: int) -> pl.Series:
    s = s.sort()
    return s.gather(np.sort(np.random.default_rng(seed).choice(s.len(), min(n, s.len()), replace=False)))


def bucket(c: str) -> pl.Expr:
    x = pl.col(c)
    return (pl.when(x.is_null()).then(pl.lit("none")).when(x < 50).then(pl.lit("0-50"))
            .when(x < 80).then(pl.lit("50-80")).otherwise(pl.lit("80-100")).alias(c))


def phist(p: np.ndarray) -> list[int]:
    return np.bincount(np.minimum((p * 10).astype(int), 9), minlength=10).tolist()


def pct(a: float, b: float) -> float:
    return float(100 * a / b) if b else float("nan")


def table(df: pl.DataFrame) -> list[str]:
    return md(df.to_dicts()) if df.height else ["(no rows)"]


@cache
def oof() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(s1 table of the 20% subset, OOF rows with code/top/keep/y/pmax); asserts the decide.json score."""
    s1 = s1_table(subset_ids(MC["subset_fraction"]), truth())
    d = (with_top(pl.read_parquet(path("oof_dir") / "oof_train.parquet"))
         .join(s1.select("s1_id", "code"), on="s1_id", how="inner", maintain_order="left")
         .with_columns(y=pl.col("label") == 1, keep=pl.col("top") & (pl.col("p") >= T),
                       pmax=pl.col("p").max().over("rec_id")))
    f = per_s1(d["code"].to_numpy(), d["keep"].to_numpy(), d["y"].to_numpy(), s1["ntrue"].to_numpy()).mean()
    ref = json.loads((path("oof_dir") / "decide.json").read_text())["oof_macro_f05"]
    assert abs(f - ref) < 1e-9, (f, ref)
    return s1, d


# ---------- D0-1 ----------
def true_rank(left: sp.csr_matrix, right_t: sp.csr_matrix, post_len: np.ndarray, col: np.ndarray,
              budget: float) -> np.ndarray:
    """Per row i: block.top_n rank of column col[i] in (left @ right_t)[i] (score desc, ties -> lower column);
    0 when that score is 0. Chunked by the same nnz bound as block.retrieve."""
    binary = sp.csr_matrix((np.ones_like(left.data), left.indices, left.indptr), shape=left.shape)
    rank = np.zeros(left.shape[0], np.int64)
    for lo, hi in _chunks(binary @ post_len, budget):
        P = left[lo:hi] @ right_t
        row = np.repeat(np.arange(hi - lo), np.diff(P.indptr))
        c = col[lo:hi][row]
        own = P.indices == c
        st = np.zeros(hi - lo, np.float32)
        st[row[own]] = P.data[own]
        s = st[row]
        ahead = (P.data > s) | ((P.data == s) & (P.indices < c))
        rank[lo:hi] = np.where(st > 0, 1 + np.bincount(row[ahead], minlength=hi - lo), 0)
    return rank


def pair_ranks(c: str, pairs: pl.DataFrame, idf: pl.LazyFrame) -> pl.DataFrame:
    """{ch}_rr (rank of the S1 among all country S1s for the record) and {ch}_sr (rank of the record among its
    source's records for the S1), exactly as block.main builds them; 0 = no shared surviving token."""
    budget = float(BCFG["nnz_budget"])
    s1 = load("train", 1, c)
    pos1 = s1.select(s1_id="entity_id").with_row_index("i1")
    tab, qw, qw_t, post1 = {}, {}, {}, {}
    for ch in CHANNELS:
        t = (idf.filter((pl.col("country") == c) & (pl.col("ch") == ch))
             .select("tok", "n1", "n2", "n3", "n_country").collect().sort("tok"))
        tab[ch] = token_table(t.drop("n_country"), int(t["n_country"][0]), BCFG["df_cap"])
        k1, _ = survivors(s1[ch], tab[ch], _fallback(ch))
        qw[ch] = matrix(k1, s1.height, tab[ch].height, weighted=True)
        qw_t[ch], post1[ch] = qw[ch].T.tocsr(), postings(k1, tab[ch].height)
    del s1
    out = []
    for n in SOURCES:
        sub = pairs.filter(pl.col("rec_id").str.starts_with(f"S{n}-"))
        if sub.is_empty():
            continue
        rec = load("train", n, c)
        sub = (sub.join(pos1, on="s1_id", how="left", maintain_order="left")
               .join(rec.select(rec_id="entity_id").with_row_index("i2"), on="rec_id", how="left", maintain_order="left"))
        assert sub["i1"].null_count() == 0 and sub["i2"].null_count() == 0, "pair outside its country"
        i1, i2 = sub["i1"].to_numpy().astype(np.int64), sub["i2"].to_numpy().astype(np.int64)
        cols = {}
        for ch in CHANNELS:
            kr, _ = survivors(rec[ch], tab[ch], _fallback(ch))
            r = matrix(kr, rec.height, tab[ch].height, weighted=False)
            cols[f"{ch}_rr"] = true_rank(r[i2], qw_t[ch], post1[ch], i1, budget)
            cols[f"{ch}_sr"] = true_rank(qw[ch][i1], r.T.tocsr(), postings(kr, tab[ch].height), i2, budget)
        out.append(sub.select(PAIR).with_columns(**{k: pl.Series(v) for k, v in cols.items()}))
        del rec
    return pl.concat(out)


def d1(log: StepLog, a) -> tuple[str, list[str]]:
    I = path("interim_dir")
    cand = pl.scan_parquet(I / "candidates_train.parquet").select(*PAIR, "X_rrank", "X_srank", hit=pl.lit(True))
    s1 = pl.scan_parquet(norm_path("train", 1)).select(
        s1_id="entity_id", country="country", n1="core_name", a1=pl.col("addr_tokens").list.join(" "))
    rec = pl.concat([pl.scan_parquet(norm_path("train", n)).select(
        rec_id="entity_id", n2="core_name", a2=pl.col("addr_tokens").list.join(" "), has_addr="has_address",
        native=pl.concat_str("business_name", "business_address", separator=" ").str.contains(NONASCII))
        for n in SOURCES])
    p = (gt_pairs().lazy().join(cand, on=PAIR, how="left").join(s1, on="s1_id", how="left")
         .join(rec, on="rec_id", how="left").with_columns(pl.col("hit").fill_null(False)).collect(engine="streaming")
         .sort(PAIR))  # streaming join order is not deterministic; the control sample below depends on it
    assert p["country"].null_count() == 0 and p["n2"].null_count() == 0
    no_addr = ((p["a1"] == "") | (p["a2"] == "")).to_numpy()
    p = p.with_columns(
        name_sim=pl.Series(_cp(p["n1"].to_list(), p["n2"].to_list(), fuzz.token_set_ratio)),
        addr_sim=pl.Series(np.where(no_addr, np.nan, _cp(p["a1"].to_list(), p["a2"].to_list(), fuzz.token_set_ratio)))
        .fill_nan(None)).drop("n1", "n2", "a1", "a2")
    p = p.with_columns(bucket("name_sim"), bucket("addr_sim"),
                       native=pl.when("native").then(pl.lit("y")).otherwise(pl.lit("n")),
                       has_addr=pl.when("has_addr").then(pl.lit("y")).otherwise(pl.lit("n")))
    log("d1 true pairs + sims", pairs=p.height, missed=int((~p["hit"]).sum()))

    idf = pl.scan_parquet(I / "idf_train.parquet")
    rk = []
    for c in sorted(p["country"].unique()):
        pc = p.filter(pl.col("country") == c)
        todo = pl.concat([pc.filter(~pl.col("hit")).select(PAIR),
                          pc.filter("hit").sample(CTRL, seed=SEED).select(PAIR)])
        rk.append(pair_ranks(c, todo, idf))
        log(f"d1 ranks {c}", pairs=todo.height)
    rk = pl.concat(rk)
    rcols = [f"{ch}_{d}" for ch in CHANNELS for d in ("rr", "sr")]

    def in_cfg(chs) -> pl.Expr:  # would be kept at config m/k by these channels
        return pl.any_horizontal([pl.col(f"{ch}_rr").is_between(1, M) | pl.col(f"{ch}_sr").is_between(1, K)
                                  for ch in chs])
    ctrl = p.filter("hit").join(rk, on=PAIR, how="inner")
    chk = {"control pairs": ctrl.height,
           "control X_rrank mismatches": ctrl.filter(pl.col("X_rrank").is_not_null()
                                                     & (pl.col("X_rrank") != pl.col("X_rr"))).height,
           "control X_srank mismatches": ctrl.filter(pl.col("X_srank").is_not_null()
                                                     & (pl.col("X_srank") != pl.col("X_sr"))).height,
           "control pairs not kept by recomputed X": ctrl.filter(~in_cfg("X")).height}
    miss = p.filter(~pl.col("hit")).join(rk, on=PAIR, how="left")
    chk["missed pairs kept by recomputed X (must be 0)"] = miss.filter(in_cfg("X")).height
    print(chk, flush=True)
    assert chk["control X_rrank mismatches"] == chk["control X_srank mismatches"] == 0, chk
    assert chk["control pairs not kept by recomputed X"] == chk["missed pairs kept by recomputed X (must be 0)"] == 0

    pos = lambda cs: pl.min_horizontal([pl.when(pl.col(x) > 0).then(pl.col(x)) for x in cs]).fill_null(0)
    miss = miss.with_columns(best=pos(rcols), best_X=pos(["X_rr", "X_sr"]), rescue_abc=in_cfg("ABC")).with_columns(
        cls=pl.when(pl.col("best") == 0).then(pl.lit("vocab")).when(pl.col("best") <= MAX_RANK)
        .then(pl.lit("depth")).otherwise(pl.lit("deep")))
    p = p.join(miss.select(*PAIR, "cls"), on=PAIR, how="left")
    n_miss = miss.height

    def seg(keys: list[str], top: int | None = None) -> pl.DataFrame:
        missed = (~pl.col("hit")).sum()
        g = (p.group_by(keys).agg(
            true_pairs=pl.len(), missed=missed,
            **{f"{c} % of seg misses": (pl.col("cls") == c).sum() / missed * 100 for c in ("depth", "deep", "vocab")})
            .with_columns(**{"miss %": pl.col("missed") / pl.col("true_pairs") * 100,
                             "share of all misses %": pl.col("missed") / n_miss * 100}))
        g = g.select(*keys, "true_pairs", "missed", "miss %", "share of all misses %", pl.col("^.* of seg misses$"))
        return g.sort("missed", descending=True).head(top) if top else g.sort(keys)

    def rank_hist(c: str) -> pl.DataFrame:
        x = pl.col(c)
        b = (pl.when(x == 0).then(pl.lit("absent")).when(x <= 10).then(pl.lit("a 1-10")).when(x <= 30)
             .then(pl.lit("b 11-30")).when(x <= 60).then(pl.lit("c 31-60")).when(x <= 100).then(pl.lit("d 61-100"))
             .when(x <= MAX_RANK).then(pl.lit("e 101-200")).otherwise(pl.lit("f >200")))
        return (miss.group_by(b.alias("rank")).len("missed").sort("rank")
                .with_columns(**{"% of misses": pl.col("missed") / n_miss * 100}))

    per_ch = [{"channel": ch, "direction": d, f"rank 1..{MAX_RANK} % of misses": pct(
        miss.filter(pl.col(f"{ch}_{d}").is_between(1, MAX_RANK)).height, n_miss),
        "within config m/k % of misses": pct(miss.filter(pl.col(f"{ch}_{d}").is_between(1, M if d == "rr" else K)).height,
                                             n_miss), "absent % of misses": pct(miss.filter(pl.col(f"{ch}_{d}") == 0).height, n_miss)}
        for ch in CHANNELS for d in ("rr", "sr")]
    L = [f"Train true pairs {p.height:,}; missed by candidates_train (X m{M}/k{K}) {n_miss:,} "
         f"({pct(n_miss, p.height):.2f}%). Ranks recomputed for every missed pair (no sampling).",
         "Classes: **depth** = best rank over all channels x both directions in 1..200; **deep** = scored but > 200; "
         "**vocab** = no shared surviving token in any channel. rr = record-centric rank among the country's S1s "
         "(config m), sr = S1-centric rank among the source's records (config k). Buckets: rapidfuzz token_set_ratio "
         "on core_name / joined addr_tokens; `none` = an address side is empty. native = raw S2/S3 name or address "
         "has a non-ASCII char.", "",
         "### Recompute check", "", *md([{"check": k, "value": v} for k, v in chk.items()]), "",
         "### Miss classes", "", *table(miss.group_by("cls").len("missed").sort("cls").with_columns(
             **{"% of misses": pl.col("missed") / n_miss * 100})),
         f"- kept at config m/k by A, B or C (not selected; select={BCFG['select']}): "
         f"{miss['rescue_abc'].sum():,} ({pct(miss['rescue_abc'].sum(), n_miss):.2f}% of misses)", "",
         "### Best rank of the true pair, any channel/direction", "", *table(rank_hist("best")), "",
         "### Best rank, X only", "", *table(rank_hist("best_X")), "",
         "### Per channel/direction", "", *md(per_ch), ""]
    for k in ("country", "native", "has_addr", "name_sim", "addr_sim"):
        L += [f"### By {k}", "", *table(seg([k])), ""]
    L += ["### Full cross (country x native x has_addr x name_sim x addr_sim), top 25 by misses", "",
          *table(seg(["country", "native", "has_addr", "name_sim", "addr_sim"], top=25)), ""]
    return "Blocking-miss anatomy", L


# ---------- D0-2 ----------
def d2(log: StepLog, a) -> tuple[str, list[str]]:
    s1, d = oof()
    code, y, keep, top = (d[c].to_numpy() for c in ("code", "y", "keep", "top"))
    ntrue = s1["ntrue"].to_numpy()
    F = lambda k, extra=0: float(per_s1(code, k, y, ntrue, extra).mean())
    base = F(keep)
    blk = ntrue - np.bincount(code[y], minlength=len(ntrue))  # true pairs with no candidate row, per S1
    win_kept = (d["pmax"] >= T).to_numpy()
    fa, fb = y & ~keep & top, y & ~top
    b1, b2 = fb & win_kept, fb & ~win_kept

    def reassign(b: np.ndarray) -> float:  # oracle: give each record in b to its true S1, drop the wrong winner
        win = top & keep & isin(d["rec_id"], d.filter(pl.Series(b))["rec_id"])
        return F((keep & ~win) | b) - base
    n_fn = int(blk.sum() + (y & ~keep).sum())
    rows = [{"bucket": "blocking miss (no candidate)", "FN pairs": int(blk.sum()), "S1s": int((blk > 0).sum()),
             "oracle gain": F(keep, blk) - base},
            {"bucket": "(a) below t, argmax = this S1", "FN pairs": int(fa.sum()),
             "S1s": int(np.unique(code[fa]).size), "oracle gain": F(keep | fa) - base},
            {"bucket": "(b1) argmax -> other S1, kept (>= t)", "FN pairs": int(b1.sum()),
             "S1s": int(np.unique(code[b1]).size), "oracle gain": reassign(b1)},
            {"bucket": "(b2) argmax -> other S1, below t", "FN pairs": int(b2.sum()),
             "S1s": int(np.unique(code[b2]).size), "oracle gain": reassign(b2)},
            {"bucket": "(b) total", "FN pairs": int(fb.sum()), "S1s": int(np.unique(code[fb]).size),
             "oracle gain": reassign(fb)}]
    for r in rows:
        r["% of FN"] = pct(r["FN pairs"], n_fn)
    p = d["p"].to_numpy()
    hist = [{"p bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "(a)": x, "(b1)": y1, "(b2)": y2}
            for k, (x, y1, y2) in enumerate(zip(phist(p[fa]), phist(p[b1]), phist(p[b2])))]
    gap = (d["pmax"].to_numpy() - p)[fb]
    q = np.quantile(gap, [0.1, 0.5, 0.9]) if gap.size else [np.nan] * 3
    ctry = s1["country"].to_numpy()[code]
    by_c = [{"country": c, "(a)": int((fa & (ctry == c)).sum()), "(b1)": int((b1 & (ctry == c)).sum()),
             "(b2)": int((b2 & (ctry == c)).sum()), "blocking": int(blk[s1["country"].to_numpy() == c].sum())}
            for c in sorted(set(ctry))]
    L = [f"M4 OOF, {s1.height:,} S1 (20% subset), t = {T}, per-record argmax. Base macro F0.5 {base:.5f} "
         "(asserted = oof/decide.json). Oracle gain = macro F0.5 change if only that bucket were fixed "
         "(b: the record goes to its true S1 and the wrong winner is dropped). Argmax runs over the subset's S1s "
         "only (OOF has no rows for other S1s), so (b) is a lower bound.", "",
         *md(rows), "", "### p histogram (10 bins)", "", *md(hist), "",
         "### (b) margin: p(record argmax) - p(true pair)", "",
         *md([{"n": int(gap.size), "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]),
               "% < 0.1": pct((gap < 0.1).sum(), gap.size), "% < 0.2": pct((gap < 0.2).sum(), gap.size)}]), "",
         "### FN pairs by country", "", *md(by_c), ""]
    return "FN anatomy (OOF)", L


# ---------- D0-3 ----------
def d3(log: StepLog, a) -> tuple[str, list[str]]:
    s1, d = oof()
    code, y, keep = (d[c].to_numpy() for c in ("code", "y", "keep"))
    ntrue = s1["ntrue"].to_numpy()
    F = lambda k: float(per_s1(code, k, y, ntrue).mean())
    base, single = F(keep), ntrue[code] == 0
    fp = keep & ~y
    rows = []
    for name, m in (("singleton FP (S1 has no true match)", fp & single), ("extra FP (non-singleton S1)", fp & ~single)):
        rows.append({"type": name, "FP pairs": int(m.sum()), "S1s": int(np.unique(code[m]).size),
                     "mean p": float(d["p"].to_numpy()[m].mean()) if m.any() else float("nan"),
                     "oracle gain if removed": F(keep & ~m) - base})
    n_single = int((ntrue == 0).sum())
    kept = (d.filter("keep").select(*PAIR, "y", "p", single=pl.Series(single[keep]))
            .with_columns(type=pl.when("y").then(pl.lit("TP")).when("single").then(pl.lit("FP singleton"))
                          .otherwise(pl.lit("FP extra"))))
    sims = (pl.scan_parquet(parts("train")).select(*PAIR, "name_tset", "addr_tset")
            .join(kept.lazy().select(PAIR), on=PAIR, how="semi").collect(engine="streaming")
            .with_columns(pl.col("name_tset", "addr_tset").fill_nan(None)))
    kept = kept.join(sims, on=PAIR, how="left")
    assert kept["name_tset"].null_count() == 0
    ids = s1["s1_id"]
    kept = kept.join(gt_pairs().rename({"s1_id": "true_s1"}), on="rec_id", how="left").with_columns(
        rec_status=pl.when(pl.col("true_s1").is_null()).then(pl.lit("distractor (no true S1)"))
        .when(pl.col("true_s1") == pl.col("s1_id")).then(pl.lit("own true S1"))
        .when(pl.col("true_s1").is_in(ids.implode())).then(pl.lit("true S1 in subset"))
        .otherwise(pl.lit("true S1 outside subset")))
    nt, at = pl.col("name_tset"), pl.col("addr_tset")
    kept = kept.with_columns(
        nb=bucket("name_tset"), ab=bucket("addr_tset"),
        pattern=pl.when(at.is_null()).then(pl.when(nt >= 80).then(pl.lit("name>=80, no addr"))
                                           .otherwise(pl.lit("name<80, no addr")))
        .when((at >= 80) & (nt < 80)).then(pl.lit("same-addr-diff-name"))
        .when((nt >= 80) & (at < 80)).then(pl.lit("same-name-diff-addr"))
        .when(nt >= 80).then(pl.lit("both >=80")).otherwise(pl.lit("both <80")))

    def pivot(keys: list[str]) -> pl.DataFrame:
        t = pl.col("type")
        return (kept.group_by(keys).agg(TP=(t == "TP").sum(), **{"FP singleton": (t == "FP singleton").sum(),
                                                                  "FP extra": (t == "FP extra").sum()})
                .with_columns(**{"FP % of kept": (pl.col("FP singleton") + pl.col("FP extra"))
                                 / (pl.col("TP") + pl.col("FP singleton") + pl.col("FP extra")) * 100}).sort(keys))
    p = d["p"].to_numpy()
    hist = [{"p bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "singleton": x, "extra": z}
            for k, (x, z) in enumerate(zip(phist(p[fp & single]), phist(p[fp & ~single])))]
    L = [f"M4 OOF, t = {T}, argmax. Base macro F0.5 {base:.5f}. {n_single:,} singleton S1s in the subset. "
         "Similarities = the stored name_tset / addr_tset features (token_set_ratio); `none` = an address side empty. "
         "Pattern thresholds: same = >= 80, diff = < 80.", "", *md(rows), "",
         "### name_sim x addr_sim bucket (kept pairs)", "", *table(pivot(["nb", "ab"])), "",
         "### Pattern", "", *table(pivot(["pattern"])), "",
         "### Record status of kept pairs", "", *table(pivot(["rec_status"])), "",
         "### FP p histogram", "", *md(hist), ""]
    return "FP anatomy (OOF)", L


# ---------- D0-4 ----------
def d4(log: StepLog, a) -> tuple[str, list[str]]:
    mr_path, I = path("matching_results"), path("interim_dir")
    if not mr_path.exists():
        return "Test density (label-free)", ["Skipped: output/matching_results.tsv does not exist."]
    ctry = lambda split, n, key: pl.read_parquet(norm_path(split, n), columns=["entity_id", "country"]).rename(
        {"entity_id": key})
    col = pl.col("matched_entity_ids")
    mr = (pl.read_csv(mr_path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
          .select(s1_id="source1_entity_id", n=pl.when(col == "").then(0).otherwise(col.str.count_matches(",") + 1)))
    recs = {s: pl.concat([ctry(s, n, "rec_id") for n in SOURCES]) for s in ("train", "test")}
    has_c = {s: pl.scan_parquet(I / f"candidates_{s}.parquet").select("rec_id").unique().collect()["rec_id"]
             for s in ("train", "test")}

    def rec_stats(s: str) -> pl.DataFrame:
        return recs[s].with_columns(cand=pl.Series(isin(recs[s]["rec_id"], has_c[s]))).group_by("country").agg(
            n_rec=pl.len(), **{"rec with >=1 cand %": pl.col("cand").mean() * 100})
    s1_oof, d = oof()
    oof_n = (s1_oof.select("s1_id", "country").join(d.filter("keep").group_by("s1_id").len("n"), on="s1_id", how="left")
             .group_by("country").agg(**{"OOF pred matches/S1": pl.col("n").fill_null(0).mean()}))
    gt_n = (ctry("train", 1, "s1_id").join(gt_pairs().group_by("s1_id").len("n"), on="s1_id", how="left")
            .group_by("country").agg(n_s1=pl.len(), **{"true matches/S1": pl.col("n").fill_null(0).mean()}))
    tr = (gt_n.join(rec_stats("train"), on="country").join(oof_n, on="country", how="left")
          .with_columns(**{"rec/S1": pl.col("n_rec") / pl.col("n_s1"),
                           "true matched rec %": pl.col("true matches/S1") * pl.col("n_s1") / pl.col("n_rec") * 100})
          .sort("country"))
    te = (mr.join(ctry("test", 1, "s1_id"), on="s1_id", how="left").group_by("country").agg(
        n_s1=pl.len(), **{"pred matches/S1": pl.col("n").mean(), "S1 empty %": (pl.col("n") == 0).mean() * 100,
                          "_m": pl.col("n").sum()})
          .join(rec_stats("test"), on="country")
          .with_columns(**{"rec/S1": pl.col("n_rec") / pl.col("n_s1"),
                           "pred matched rec %": pl.col("_m") / pl.col("n_rec") * 100}).drop("_m").sort("country"))
    log("d4 counts")

    # max p per record: lgb_final on a record sample; train = records whose candidate S1s are all outside the
    # final-fit 50% (lgb_final never saw them)
    final = subset_ids(MC["final_fraction"])
    # is_in(list literal) inside group_by().agg() is broadcast per group (OOM): filter at top level, then anti-join
    cand = pl.scan_parquet(I / "candidates_train.parquet").select(PAIR)
    seen = cand.filter(pl.col("s1_id").is_in(final.implode())).select("rec_id").unique()
    held = (cand.select("rec_id").unique().join(seen, on="rec_id", how="anti")
            .collect(engine="streaming")["rec_id"])
    nb = pl.col("n_cand_rec")
    nbk = (pl.when(nb == 1).then(pl.lit("1")).when(nb == 2).then(pl.lit("2")).when(nb <= 5).then(pl.lit("3-5"))
           .when(nb <= 10).then(pl.lit("6-10")).otherwise(pl.lit("11+")).alias("n_cand"))
    per_rec = {}
    for s, pool in (("test", has_c["test"]), ("train held-out", held)):
        ids = sample(pool, a.sample_rec, SEED)
        split = "test" if s == "test" else "train"
        rows = pl.scan_parquet(parts(split)).filter(pl.col("rec_id").is_in(ids.implode())).collect(engine="streaming")
        rows = rows.select(*PAIR, "n_cand_rec", p=pl.Series(predict(rows)))
        if split == "train":
            rows = rows.join(gt_pairs().with_columns(y=pl.lit(True)), on=PAIR, how="left")
        else:
            rows = rows.with_columns(y=pl.lit(None, pl.Boolean))
        per_rec[s] = (rows.group_by("rec_id").agg(pmax=pl.col("p").max(), n_cand_rec=nb.first(),
                                                  reach=pl.col("y").fill_null(False).any())
                      .join(recs[split], on="rec_id", how="left").with_columns(nbk))
        log(f"d4 max-p {s}", records=ids.len(), rows=rows.height)
    tp = per_rec["train held-out"].join(gt_pairs().select("rec_id", matched=pl.lit(True)), on="rec_id", how="left")
    per_rec["train held-out"] = tp.with_columns(pl.col("matched").fill_null(False))

    def share(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
        extra = ({"true matched %": pl.col("matched").mean() * 100, "true S1 in candidates %": pl.col("reach").mean() * 100}
                 if "matched" in df.columns else {})
        return df.group_by(keys).agg(records=pl.len(), **{"max p > 0.5 %": (pl.col("pmax") > 0.5).mean() * 100,
                                                        f"max p >= t({T}) %": (pl.col("pmax") >= T).mean() * 100},
                                     **extra).sort(keys)
    strat = (share(per_rec["test"], ["n_cand"]).select("n_cand", test_records="records", test_share="max p > 0.5 %")
             .join(share(per_rec["train held-out"], ["n_cand"]).select("n_cand", train_records="records",
                                                                      train_share="max p > 0.5 %"), on="n_cand", how="full",
                   coalesce=True).sort("n_cand")
             .with_columns(test_mix=pl.col("test_records") / pl.col("test_records").sum()))
    rew = float((strat["test_mix"] * strat["train_share"]).sum())
    L = ["(i) Counts. Test: output/matching_results.tsv (M4, lgb_final, t = "
         f"{T}, argmax). Train: ground truth; OOF pred matches/S1 = M4 OOF on the 20% subset.", "",
         "Train", "", *table(tr), "", "Test", "", *table(te), "",
         f"(ii) Max p per S2/S3 record: models/lgb_final.txt on {a.sample_rec:,} sampled records per split (seed {SEED}), "
         "all candidate rows of each sampled record scored. Test pool = records with >= 1 candidate. Train pool = "
         "records whose candidate S1s are ALL outside the final-fit 50% (held out from lgb_final); the M4 OOF cannot "
         "give this number because its rows cover only the 20% subset's S1s, so a record's max p misses ~80% of its "
         "competitors. The held-out pool over-represents records with few candidates: compare by n_cand stratum; "
         "`train reweighted` = train stratum shares weighted by the test n_cand mix.", "",
         *table(share(per_rec["test"], ["country"])), "", *table(share(per_rec["train held-out"], ["country"])), "",
         "By candidate count of the record (max p > 0.5 %)", "", *table(strat), "",
         f"- train reweighted to test n_cand mix: {rew:.2f}%; test: "
         f"{float((per_rec['test']['pmax'] > 0.5).mean() * 100):.2f}%", ""]
    return "Test density (label-free)", L


# ---------- D0-5 ----------
def rebuild(rows: pl.DataFrame, grid: pl.DataFrame, drop: pl.Series, ev: pl.Series) -> tuple[pl.DataFrame, int, pl.DataFrame]:
    """rows = stored features of EVERY candidate of some records; grid = their blockgrid rows. Remove the S1s in
    `drop`, re-rank record-centric ranks among the remaining S1s (grid keeps all rrank <= 10, so ranks <= 10 are
    exact), re-cut with block.finalize (scores / hits / n_channels_hit follow), recompute features.relative per
    record. S1-side columns are recomputed only for `ev` S1s (their rows are complete here), stored otherwise
    (dropping S1s never changes another S1's candidate list). Returns (rows, #new-entrant pairs excluded, entrants)."""
    g = grid.filter(~pl.col("s1_id").is_in(drop.implode())).with_columns(
        *(pl.col(f"{ch}_rrank").rank("ordinal").over("rec_id").cast(pl.Int16) for ch in CHANNELS))
    new = finalize(g.lazy(), M, K, tuple(BCFG["select"])).collect()
    cols = [*RANK_COLS, "n_channels_hit"]
    keep = rows.filter(~pl.col("s1_id").is_in(drop.implode()))
    upd = keep.drop(cols).join(new.select(*PAIR, *cols), on=PAIR, how="left", maintain_order="left")
    assert upd["n_channels_hit"].null_count() == 0, "a remaining candidate fell out of the re-cut"
    entrants = new.join(keep.select(PAIR), on=PAIR, how="anti").select(PAIR)
    keys = upd.select(s1k=id_key("s1_id"), reck=id_key("rec_id"))
    rel = pl.concat([keys.select(n_cand_rec=pl.len().over("reck").cast(pl.Int32),
                                 n_cand_s1=pl.len().over("s1k").cast(pl.Int32)),
                     *(relative(keys, upd[c]) for c in REL_COLS)], how="horizontal")
    is_ev = pl.Series(isin(upd["s1_id"], ev))
    upd = upd.with_columns(*(rel[c] for c in rel.columns if c not in S1_SIDE),
                           *(pl.when(is_ev).then(rel[c]).otherwise(upd[c]).alias(c) for c in S1_SIDE))
    return upd, entrants.height, entrants


def d5(log: StepLog, a) -> tuple[str, list[str]]:
    I, feats = path("interim_dir"), booster()[1]
    all_ids = pl.read_parquet(norm_path("train", 1), columns=["entity_id"])["entity_id"]
    drop = sample(all_ids, round(DROP_FRAC * all_ids.len()), SEED)
    pool = all_ids.filter(~pl.Series(isin(all_ids, subset_ids(MC["final_fraction"]))))
    ev = sample(pool, a.eval_s1, SEED + 5)
    lf = pl.scan_parquet(parts("train"))
    recs = lf.filter(pl.col("s1_id").is_in(ev.implode())).select("rec_id").unique().collect(engine="streaming")["rec_id"]
    rows = lf.filter(pl.col("rec_id").is_in(recs.implode())).collect(engine="streaming")
    grid = (pl.scan_parquet(I / "blockgrid_train_cap1000.parquet").filter(pl.col("rec_id").is_in(recs.implode()))
            .collect(engine="streaming"))
    log("d5 load", eval_s1=ev.len(), records=recs.len(), rows=rows.height, grid_rows=grid.height)

    same, _, _ = rebuild(rows, grid, pl.Series([], dtype=pl.String), ev)  # no drop must reproduce stored features
    X0, X1 = (f.select(pl.col(feats).cast(pl.Float32)).to_numpy() for f in (rows, same))
    assert np.array_equal(X0, X1, equal_nan=True), f"rebuild(no drop) differs in {int((~((X0 == X1) | (np.isnan(X0) & np.isnan(X1)))).any(0).sum())} columns"
    del same, X0, X1
    after, n_entr, entr = rebuild(rows, grid, drop, ev)
    log("d5 rebuild", rows_after=after.height, entrants=n_entr)

    rem = ev.filter(~pl.Series(isin(ev, drop)))
    s1 = s1_table(rem, truth())
    gt = gt_pairs().with_columns(y=pl.lit(True))
    orphan = gt.filter(pl.col("s1_id").is_in(drop.implode()))["rec_id"]

    def score(df: pl.DataFrame, p: np.ndarray) -> dict:
        x = (with_top(df.select(PAIR).with_columns(p=pl.Series(p)))
             .join(gt, on=PAIR, how="left").with_columns(pl.col("y").fill_null(False))
             .join(s1.select("s1_id", "code", "ntrue"), on="s1_id", how="inner", maintain_order="left"))
        code, y, top, pp = (x[c].to_numpy() for c in ("code", "y", "top", "p"))
        ntrue, orph = s1["ntrue"].to_numpy(), isin(x["rec_id"], orphan)
        curve = [per_s1(code, top & (pp >= t), y, ntrue).mean() for t in TS]
        k = top & (pp >= T)
        f = per_s1(code, k, y, ntrue)
        ctry = s1["country"].to_numpy()
        return {"macro F0.5 @t": float(f.mean()), "FP": int((k & ~y).sum()), "FP on orphan records": int((k & ~y & orph).sum()),
                "TP": int((k & y).sum()), "F0.5 singleton": float(f[ntrue == 0].mean()),
                "F0.5 non-singleton": float(f[ntrue > 0].mean()), "best t": float(TS[int(np.argmax(curve))]),
                "macro F0.5 @best t": float(max(curve)),
                **{f"F0.5 @t {c}": float(f[ctry == c].mean()) for c in sorted(set(ctry))}}
    before = score(rows, predict(rows))
    log("d5 predict before", rows=rows.height)
    aft = score(after, predict(after))
    log("d5 predict after", rows=after.height)
    res = [{"condition": "before (all S1s present)", **before}, {"condition": f"after ({DROP_FRAC:.0%} S1s dropped)", **aft}]
    delta = {k: aft[k] - before[k] for k in before if k != "best t"}
    ent_true = entr.join(gt, on=PAIR, how="inner").filter(pl.col("s1_id").is_in(rem.implode())).height
    L = [f"Eval S1 = {ev.len():,} sampled (seed {SEED + 5}) from S1s outside the final-fit "
         f"{MC['final_fraction']:.0%} (lgb_final never trained on them); {rem.len():,} remain after dropping "
         f"{drop.len():,} of {all_ids.len():,} train S1s (seed {SEED}). Scored rows = every candidate of every record "
         f"that touches an eval S1 ({rows.height:,} rows before, {after.height:,} after), so the per-record argmax "
         "sees all competitors; competitor rows of S1s inside the 50% are in-sample for lgb_final. F0.5 over all "
         f"{rem.len():,} remaining eval S1s (zero-candidate ones included). t = {T} (M4 OOF) and the best t of the "
         "same grid. Model = models/lgb_final.txt: train.py saves no fold models, so the OOF models cannot be "
         "reused without retraining.",
         "Recomputed after the drop: A/B/C/X record-centric ranks (re-ranked among remaining S1s, re-cut at "
         f"m={M}/k={K}; scores, hits and n_channels_hit follow the cut), all relative features and n_cand_rec. "
         "Asserted: the same rebuild with no drop reproduces every stored feature exactly.",
         f"Not recomputed: pairs that would ENTER the candidate set after the drop ({n_entr:,}; "
         f"{ent_true:,} are true pairs of remaining eval S1s) have no pair features, so they stay out; ranks "
         "beyond the grid (rrank > 10) are not re-ranked.", "",
         *md(res), "", *md([{"metric": k, "after - before": v} for k, v in delta.items()]), "",
         f"- orphaned records (true S1 dropped) among the {recs.len():,} records: "
         f"{int(isin(recs, orphan).sum()):,}", ""]
    return "Orphan simulation (S1 dropout)", L


# ---------- D0-6 ----------
def d6(log: StepLog, a) -> tuple[str, list[str]]:
    s1, d = oof()
    code, y, keep = (d[c].to_numpy() for c in ("code", "y", "keep"))
    ntrue = s1["ntrue"].to_numpy()
    f = per_s1(code, keep, y, ntrue)
    npred = np.bincount(code[keep], minlength=len(ntrue))
    o = s1.select("country", "ntrue").with_columns(f=pl.Series(f), npred=pl.Series(npred))
    oofc = (o.group_by("country").agg(
        n_s1=pl.len(), **{"OOF macro F0.5 @t": pl.col("f").mean(),
                          "F0.5 singleton": pl.col("f").filter(pl.col("ntrue") == 0).mean(),
                          "F0.5 non-singleton": pl.col("f").filter(pl.col("ntrue") > 0).mean(),
                          "pred matches/S1": pl.col("npred").mean(), "true matches/S1": pl.col("ntrue").mean()})
            .sort("country"))
    cnt = (gt_pairs().with_columns(src=pl.col("rec_id").str.slice(0, 2)).group_by("s1_id")
           .agg(n=pl.len(), n2=(pl.col("src") == "S2").sum(), n3=(pl.col("src") == "S3").sum()))
    card = (pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
            .join(cnt, on="s1_id", how="left").fill_null(0))
    n = pl.col("n")
    recs = pl.concat([pl.read_parquet(norm_path("train", k), columns=["entity_id", "country"]) for k in SOURCES])
    recs = recs.with_columns(matched=pl.Series(isin(recs["entity_id"], gt_pairs()["rec_id"]))).group_by("country").agg(
        n_rec=pl.len(), **{"distractor %": (~pl.col("matched")).mean() * 100})
    gen = (card.group_by("country").agg(
        n_s1=pl.len(), **{"singleton %": (n == 0).mean() * 100, "mean card (all)": n.mean(),
                          "mean card (non-singleton)": n.filter(n > 0).mean(), "S2/S1": pl.col("n2").mean(),
                          "S3/S1": pl.col("n3").mean(), "card 5+ %": (n >= 5).mean() * 100,
                          "card p99": n.quantile(0.99, "nearest").cast(pl.Float64)})
           .join(recs, on="country").with_columns(**{"rec/S1": pl.col("n_rec") / pl.col("n_s1")}).sort("country"))
    L = [f"OOF = M4 OOF (20% subset), t = {T}, argmax.", "", *table(oofc), "",
         "Generator constants, full train ground truth", "", *table(gen), ""]
    return "Per-country OOF + generator constants", L


# ---------- D0-7 ----------
def d7(log: StepLog, a) -> tuple[str, list[str]]:
    def rows(n: int) -> pl.DataFrame:
        f = pl.read_csv(path("train_dir") / f"train_source{n}.tsv", separator="\t", quote_char=None,
                        infer_schema=False, columns=["entity_id"]).with_row_index("row")
        assert f.height == pl.scan_parquet(norm_path("train", n)).select(pl.len()).collect().item(), f"S{n} rows"
        return f
    num = lambda c: pl.col(c).str.split("-").list.last().cast(pl.Int64)
    s1 = rows(1).join(pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"]), on="entity_id")
    rec = pl.concat([rows(n).with_columns(src=pl.lit(f"S{n}")) for n in SOURCES])
    p = (gt_pairs().join(s1.select(s1_id="entity_id", r1="row", country="country"), on="s1_id")
         .join(rec.select(rec_id="entity_id", r2="row", src="src"), on="rec_id")
         .with_columns(k1=num("s1_id"), k2=num("rec_id")))
    out = []
    for keys in (["src"], ["src", "country"]):
        for g, x in sorted(p.group_by(keys), key=lambda t: t[0]):
            r_id = float(spearmanr(x["k1"].to_numpy(), x["k2"].to_numpy()).statistic)
            r_row = float(spearmanr(x["r1"].to_numpy(), x["r2"].to_numpy()).statistic)
            out.append({"group": " / ".join(g), "pairs": x.height, "rho id number": r_id, "rho file row": r_row,
                        "flag |rho| > 0.05": "FLAG" if max(abs(r_id), abs(r_row)) > 0.05 else ""})
    L = [f"{p.height:,} train true pairs. Spearman between the numeric part of the S1 id and of its matched "
         "S2/S3 id, and between their 0-based data-row positions in the raw TSVs. Report only: never a feature.",
         "", *md(out), ""]
    return "Leak sanity (id / row order)", L


# ---------- D0-8 ----------
def d8(log: StepLog, a) -> tuple[str, list[str]]:
    I = path("interim_dir")
    cand = pl.scan_parquet(I / "candidates_train.parquet").select(*PAIR, hit=pl.lit(True))
    us1 = pl.scan_parquet(norm_path("train", 1)).filter(pl.col("country") == "US")
    rec = pl.concat([pl.scan_parquet(norm_path("train", n)).select(
        rec_id="entity_id", n2="core_name", has_addr="has_address",
        native=pl.concat_str("business_name", "business_address", separator=" ").str.contains(NONASCII))
        for n in SOURCES])
    p = (gt_pairs().lazy().join(us1.select(s1_id="entity_id", n1="core_name"), on="s1_id").join(rec, on="rec_id")
         .filter(~pl.col("has_addr") & ~pl.col("native")).join(cand, on=PAIR, how="left")
         .with_columns(pl.col("hit").fill_null(False)).collect(engine="streaming").sort(PAIR))
    n_cell = p.height
    p = p.filter(pl.Series(_cp(p["n1"].to_list(), p["n2"].to_list(), fuzz.token_set_ratio)) >= 80)
    names = us1.group_by("core_name").len("k").collect()
    p = (p.join(names.rename({"core_name": "n2", "k": "same_rec"}), on="n2", how="left")
         .join(names.rename({"core_name": "n1", "k": "same_true"}), on="n1", how="left")
         .with_columns(pl.col("same_rec", "same_true").fill_null(0),
                       status=pl.when("hit").then(pl.lit("hit")).otherwise(pl.lit("missed"))))
    log("d8 cell", pairs=p.height, missed=int((~p["hit"]).sum()))

    def dist(col: str) -> pl.DataFrame:
        x = pl.col(col)
        b = (pl.when(x == 0).then(pl.lit("a 0")).when(x == 1).then(pl.lit("b 1")).when(x == 2).then(pl.lit("c 2"))
             .when(x <= 5).then(pl.lit("d 3-5")).when(x <= 10).then(pl.lit("e 6-10")).when(x <= 50)
             .then(pl.lit("f 11-50")).otherwise(pl.lit("g 51+")))
        return (p.group_by(b.alias("S1s sharing the name")).agg(
            missed=(~pl.col("hit")).sum(), hit=pl.col("hit").sum()).sort("S1s sharing the name")
            .with_columns(**{"missed %": pl.col("missed") / pl.col("missed").sum() * 100,
                             "hit %": pl.col("hit") / pl.col("hit").sum() * 100}))
    summ = (p.group_by("status").agg(
        pairs=pl.len(), **{f"{c} {s}": e for c in ("same_rec", "same_true")
                           for s, e in (("median", pl.col(c).median()), ("% > 1", (pl.col(c) > 1).mean() * 100))})
        .sort("status"))
    L = ["Cell = D0-1's largest miss cell: US, record has no address, no non-ASCII in the raw record, token_set(core "
         f"names) >= 80, missed/hit by the current candidates_train. US true pairs with a no-address ASCII record: "
         f"{n_cell:,}; in the cell (name_sim >= 80): {p.height:,} (D0-1: 190,077, 48,930 missed). same_rec = US S1s "
         "whose core_name equals the RECORD's core_name; same_true = US S1s whose core_name equals the TRUE S1's "
         "core_name (the true S1 itself included).", "", *table(summ), "",
         "### same_rec (record's name)", "", *table(dist("same_rec")), "",
         "### same_true (true S1's name)", "", *table(dist("same_true")), ""]
    return "US no-address name-sim >= 80: S1s sharing the name", L


CHECKS = {1: d1, 2: d2, 3: d3, 4: d4, 5: d5, 6: d6, 7: d7, 8: d8}


def assemble() -> None:
    secs = [SECT / f"d{n}.md" for n in CHECKS if (SECT / f"d{n}.md").exists()]
    head = ["# M5-D0 diagnostics",
            f"Generated by `src/diagnose.py` (seed {SEED}, t = {T}); spec docs/m5-strategy-l3.md §3. "
            "Numbers and tables only.", ""]
    REPORT.write_text("\n".join(head + [s.read_text() for s in secs]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="*", choices=list(CHECKS), default=list(CHECKS))
    ap.add_argument("--eval-s1", type=int, default=20_000, help="D0-5: held-out S1s sampled before the drop")
    ap.add_argument("--sample-rec", type=int, default=60_000, help="D0-4: records sampled per split for max p")
    a = ap.parse_args()
    log = StepLog()
    SECT.mkdir(parents=True, exist_ok=True)
    for n in a.d:
        t0 = time.perf_counter()
        title, L = CHECKS[n](log, a)
        L.append(f"_Runtime {time.perf_counter() - t0:.0f} s; process peak RSS so far {peak_rss_mb():,} MB._")
        (SECT / f"d{n}.md").write_text("\n".join([f"## D0-{n}. {title}", "", *L, ""]))
        assemble()
        log(f"D0-{n} done", s=round(time.perf_counter() - t0))
    log.dump(path("artifacts_dir") / "logs" / "diagnose_timing.json", checks=a.d)


if __name__ == "__main__":
    main()
