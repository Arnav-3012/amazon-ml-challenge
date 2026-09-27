"""M5 mechanisms: which noise transforms carry the OOF (b) loss, and can collective (record-record) evidence fix it.
Read-only; no pipeline change.

World / decision / per-S1 Shapley loss = src.loss_breakdown (cv_full (b) world, t = cv_full.json best_t = 0.75,
per-record argmax ties -> lowest S1 code). A 20% sample of the world's S1s (np default_rng(config seed)) is analysed;
points = 100 x share / n_sample_S1, i.e. macro F0.5 points of the sample (all rows of a table add to the sample loss).
Pairs of sample S1s, status:
  TP   kept (argmax, p >= t) and true;          FP   kept and false (pair = record vs the S1 it was given);
  FN   true, in candidates, not kept;           miss true, not in candidates_train_v1 (blocking-miss).
Each S1's Shapley share of a bucket (FN = matcher-FN, FP = matcher-FP, miss = blocking-miss) is split equally over
its pairs of that status. Tags compare the record with the pair's S1 (raw strings + src.normalise columns);
"only" = the record has it and the S1 does not. distractor = record has no true S1 in the world (only FPs).
  pseudo_brand: record core name shares no name token with the S1's and char-4gram cosine < 0.3.
  num_jitter: first digit run of addr_number differs by 1..99; num_missing: S1 has one, the record not.
  addr_truncated: record has 1..50% as many addr tokens as the S1. state_full_vs_abbr: same canonical
  admin_region, and exactly one side spells it as the canonical (abbreviated) token.
Char-n-gram cosines: sklearn HashingVectorizer (2^20 dims, l2) on normalised text; "name+addr" text =
name_tokens | addr_tokens.
D2 LightGBM: slice = TP/FP pairs with a legal_* or distractor tag; features = top-20 gain features of models/fold_*.txt
(+ tags, never distractor: it is ground truth); GroupKFold(5) by S1, 200 rounds, OOF AUC.
D3 precision: 20% sample of the world's records (default_rng(config seed)); neighbours = records predicted with
p >= 0.95 to any in-world candidate S1 of the record (itself excluded); the most similar one's S1 is "the" guess.

Run from code/business_entity_resolution/:
  python -m src.mechanisms --smoke   # 1% S1 / record samples, same path, writes docs/mechanisms.md (< 3 min)
  python -m src.mechanisms           # 20% samples -> docs/mechanisms.md
Aborts if peak RSS > MAX_RSS_MB (18 GB).
"""
import argparse
import json
from collections import Counter

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .cv_full import drop_mask, s1_table
from .decide import f05_vec, md
from .features import id_key
from .io import CFG, ROOT, StepLog, load_gt_pairs, path, peak_rss_mb
from .loss_breakdown import NONLATIN, shapley

SEED = int(CFG["seed"])
T, P_CONF, MAX_RSS_MB, CHUNK = 0.75, 0.95, 18_000, 500_000  # 24 GB machine; full run peaks ~10.2 GB
FRAC, NP = 0.2, 10  # S1 / record sample fraction, precision partitions; --smoke sets 0.01, 1
REPORT = ROOT / "docs" / "mechanisms.md"
NORM = [path("interim_dir") / f"norm_train_s{n}.parquet" for n in (1, 2, 3)]
E = pl.element()
SIDE = {"n": "business_name", "a": "business_address", "core": "core_name", "legal": "legal_form",
        "nt": "name_tokens", "num": "addr_number", "at": "addr_tokens", "city": "city", "reg": "admin_region"}
TXT = pl.concat_str(pl.col("name_tokens").list.join(" "), pl.col("addr_tokens").list.join(" "), separator=" | ")
UNIT_RE = r"(?i)\b(?:apt|unit|suite|ste|flat|room|floor|fl|shop|plot)\b\.?\s*[\p{L}\p{N}-]+|#\s*\d+"
NULL_RE = r"(?i)<null>|\bn/a\b|\bnull\b"
DOMAIN_RE = r"(?i)www\.|\.c0m\b|\.com\b"
BUCKET_OF = {"miss": 0, "FN": 1, "FP": 2}  # loss_breakdown.BUCKETS order
LOSS_ST = list(BUCKET_OF)
JBINS = [(0, 0), (1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 99), (100, None)]


def guard(step: str) -> None:
    if (rss := peak_rss_mb()) > MAX_RSS_MB:
        raise SystemExit(f"ABORT src.mechanisms: peak RSS {rss} MB > {MAX_RSS_MB} MB at '{step}'. "
                         "Lower CHUNK or raise NP.")


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        guard(step)


def load(files, keys: pl.Series, key: str, **cols: pl.Expr) -> pl.DataFrame:
    return (pl.scan_parquet(files).select(id_key("entity_id").alias(key), **cols)
            .join(pl.LazyFrame({key: keys}), on=key, how="semi").collect(engine="streaming"))


def side(pre: str) -> dict[str, pl.Expr]:
    return {pre + k: pl.col(v) for k, v in SIDE.items()}


def vec(texts: pl.Series, n: int):
    hv = HashingVectorizer(analyzer="char", ngram_range=(n, n), n_features=2 ** 20, alternate_sign=False,
                           dtype=np.float32)
    return hv.transform(texts.fill_null("").to_list())


def rowdot(A, ia: np.ndarray, B, ib: np.ndarray) -> np.ndarray:
    """Row-wise dot of two sparse matrices, A and B pre-vectorized whole (caller's `vec()` calls) --
    CHUNK here only bounds the dot-product's own temporaries, not the vectorization that built A/B."""
    out = np.empty(len(ia), np.float32)
    for k in range(0, len(ia), CHUNK):
        s = slice(k, k + CHUNK)
        out[s] = np.asarray(A[ia[s]].multiply(B[ib[s]]).sum(1)).ravel()
        guard("rowdot")
    return out


def rowdot_text(r_texts: pl.Series, s_texts: pl.Series, n: int) -> np.ndarray:
    """Chunked equivalent of rowdot(vec(r,n), arange, vec(s,n), arange): vectorizes CHUNK rows at a
    time instead of the whole column, so peak RSS is bounded by one chunk's hashed matrix, not all of it."""
    m = len(r_texts)
    out = np.empty(m, np.float32)
    for k in range(0, m, CHUNK):
        s = slice(k, k + CHUNK)
        A = vec(r_texts[s], n)
        B = vec(s_texts[s], n)
        out[s] = np.asarray(A.multiply(B).sum(1)).ravel()
        guard("rowdot_text")
    return out


def num(c: str) -> pl.Expr:
    return pl.col(c).str.extract(r"(\d{1,9})").cast(pl.Int64, strict=False)


def words(c: str) -> pl.Expr:
    return pl.col(c).str.to_lowercase().str.extract_all(r"[\p{L}\p{N}]+")


def tag_exprs() -> dict[str, pl.Expr]:
    c = pl.col
    has = lambda pat: lambda col: c(col).str.contains(pat)
    only_n = lambda f: f("r_n") & ~f("s_n")
    only_a = lambda f: f("r_a") & ~f("s_a")
    ocr_tok = (E.str.contains(r"^[a-z01568]+$") & E.str.contains(r"[01568]") & E.str.contains(r"[a-z].*[a-z]")
               & ~E.str.contains(r"^\d+(?:st|nd|rd|th)$"))
    ocr = lambda col: words(col).list.eval(ocr_tok).list.any()
    caps = lambda col: c(col).str.contains(r"\p{Lu}.*\p{Lu}") & ~c(col).str.contains(r"\p{Ll}")
    dup = lambda col: words(col).list.eval((E == E.shift(1)) & (E.str.len_chars() > 1)).list.any()
    reorder = lambda a, b: (c(a).list.len() > 1) & (c(a) != c(b)) & (c(a).list.sort() == c(b).list.sort())
    rl, sl, rc, sc = c("r_legal"), c("s_legal"), c("r_core"), c("s_core")
    rN, sN = num("r_num"), num("s_num")
    spelled = lambda col, reg: words(col).list.contains(c(reg))
    empty = lambda col: c(col).str.strip_chars() == ""
    return {
        "legal_added": (rl != "") & (sl == ""),
        "legal_removed": (rl == "") & (sl != ""),
        "legal_swapped": (rl != "") & (sl != "") & (rl != sl),
        "token_reorder": reorder("r_nt", "s_nt"),
        "ocr_swap": ocr("r_n") & ~ocr("s_n"),
        "all_caps": only_n(caps),
        "pseudo_brand": (rc != "") & (c("r_nt").list.set_intersection("s_nt").list.len() == 0) & (c("cos4") < 0.3),
        "native_script": only_n(has(NONLATIN)),
        "id_tag": only_n(has(r"(?i)\(\s*id\s*:")),
        "domain": only_n(has(DOMAIN_RE)),
        "noise_prefix": only_n(has(r"^[#*.]+")),
        "dup_token": only_n(dup),
        "name_truncated": (rc != "") & (rc.str.len_chars() < sc.str.len_chars()) & sc.str.starts_with(rc),
        "addr_empty": empty("r_a") & ~empty("s_a"),
        "null_token": only_a(has(NULL_RE)),
        "num_jitter": (rN - sN).abs().is_between(1, 99),
        "num_missing": sN.is_not_null() & rN.is_null(),
        "unit_dropped": has(UNIT_RE)("s_a") & ~has(UNIT_RE)("r_a"),
        "city_differs": (c("r_city") != "") & (c("s_city") != "") & (c("r_city") != c("s_city")),
        "state_full_vs_abbr": (c("r_reg") == c("s_reg")) & (c("r_reg") != "")
                              & (spelled("r_a", "r_reg") ^ spelled("s_a", "s_reg")),
        "addr_reorder": reorder("r_at", "s_at"),
        "addr_truncated": (c("r_at").list.len() > 0) & (2 * c("r_at").list.len() <= c("s_at").list.len()),
        "addr_native": only_a(has(NONLATIN)),
        "distractor": c("distractor"),
    }


TAGS = list(tag_exprs())


def world(log: Log):
    """loss_breakdown's (b) world: S1 table with code, kept predictions, true in-candidate pairs, per-S1 counts."""
    cv = json.loads((path("oof_dir") / "cv_full.json").read_text())["b_test_density"]
    assert abs(cv["best_t"] - T) < 1e-9, cv["best_t"]
    s1 = s1_table(1.0)
    s1 = (s1.filter(pl.Series(~drop_mask(s1.height, SEED + 2000))).select("s1_id", "s1k", "country", "ntrue")
          .with_row_index("code"))
    assert s1.height == cv["n_s1"], (s1.height, cv["n_s1"])
    oof = pl.scan_parquet(path("oof_dir") / "oof_full.parquet")
    code = s1.select("s1k", "code")
    kept = (oof.filter(pl.col("p_td") >= T).select("s1k", "reck", "label", "p_td").collect(engine="streaming")
            .join(code, on="s1k").sort(["reck", "p_td", "code"], descending=[False, True, False])
            .filter(pl.col("reck").is_first_distinct()))
    in_cand = (oof.filter(pl.col("label") == 1).select("s1k", "reck", "p_td").collect(engine="streaming")
               .join(code, on="s1k"))
    n = s1.height
    bc = lambda c, m=None: np.bincount(c if m is None else c[m], minlength=n)
    kc, ky = kept["code"].to_numpy(), kept["label"].to_numpy() == 1
    tp, npred, n_in, ntrue = bc(kc, ky), bc(kc), bc(in_cand["code"].to_numpy()), s1["ntrue"].to_numpy()
    fp, fn_m, fn_b = npred - tp, n_in - tp, ntrue - n_in
    f = f05_vec(tp, npred, ntrue)
    assert abs(f.mean() - cv["macro_f05"]) < 1e-9, (f.mean(), cv["macro_f05"])
    L = shapley(tp, fp, fn_m, fn_b)
    log("world", n_s1=n, kept=kept.height, in_cand=in_cand.height, f05=round(float(f.mean()), 6))
    return oof, s1, code, kept, in_cand, dict(TP=tp, FP=fp, FN=fn_m, miss=fn_b), f, L


def sample_pairs(s1, kept, in_cand, gt, cnt, f, L, log) -> tuple[pl.DataFrame, np.ndarray]:
    n = s1.height
    samp = np.zeros(n, bool)
    samp[np.random.default_rng(SEED).permutation(n)[: round(FRAC * n)]] = True
    ins = lambda df: df.filter(pl.Series(samp[df["code"].to_numpy()]))
    ks, ics = ins(kept), ins(in_cand)
    cols = lambda st: ["s1k", "reck", "code", pl.col("p_td").alias("p"), pl.lit(st).alias("status")]
    TP = ks.filter(pl.col("label") == 1).select(cols("TP"))
    FP = ks.filter(pl.col("label") == 0).select(cols("FP"))
    FN = ics.join(TP, on=["s1k", "reck"], how="anti").select(cols("FN"))
    gts = ins(gt.join(s1.select("s1k", "code"), on="s1k"))
    MISS = (gts.join(ics, on=["s1k", "reck"], how="anti")
            .select("s1k", "reck", "code", p=pl.lit(None, pl.Float32), status=pl.lit("miss")))
    P = pl.concat([TP, FP, FN, MISS]).sort("status", "s1k", "reck")
    st, c = P["status"].to_numpy(), P["code"].to_numpy()
    for s, a in cnt.items():  # pair lists reproduce the per-S1 counts exactly
        assert (np.bincount(c[st == s], minlength=n) == np.where(samp, a, 0)).all(), s
    ns = int(samp.sum())
    w = np.zeros(P.height)
    for s, b in BUCKET_OF.items():
        m = st == s
        w[m] = 100 * L[c[m], b] / ns / cnt[s][c[m]]
    assert np.isclose(w.sum(), 100 * (1 - f[samp]).sum() / ns), (w.sum(), 100 * (1 - f[samp]).mean())
    log("sample pairs", n_s1=ns, pairs=P.height, **{s: int((st == s).sum()) for s in cnt})
    return P.with_columns(w=pl.Series(w, dtype=pl.Float32)), samp


def tag(P: pl.DataFrame, gtw: pl.DataFrame, log) -> pl.DataFrame:
    s = load(NORM[0], P["s1k"].unique(), "s1k", **side("s_"))
    r = load(NORM[1:], P["reck"].unique(), "reck", **side("r_"))
    X = P.join(s, on="s1k").join(r, on="reck")
    assert X.height == P.height, (X.height, P.height)
    # chunked vectorization (not plain rowdot): X.height is the full pair table (~1M+ rows) -- vec() on the
    # whole column here would materialize a (n_pairs, 2**20) sparse matrix twice before any chunking kicks in
    X = X.with_columns(cos4=pl.Series(rowdot_text(X["r_core"], X["s_core"], 4)),
                       distractor=~pl.col("reck").is_in(gtw["reck"].implode()))
    X = X.with_columns(**{k: e.fill_null(False) for k, e in tag_exprs().items()})
    # keep only what D1-D3 read: raw strings and the other normalised columns are dropped here
    X = X.select("s1k", "reck", "status", "w", "r_nt", "r_at", "r_num", "s_num", *TAGS)
    log("tagged")
    return X


def d1(X: pl.DataFrame) -> tuple[list[str], list[str]]:
    M = X.select(TAGS).to_numpy().astype(bool)
    w, st = X["w"].to_numpy().astype(np.float64), X["status"].to_numpy()  # float64 sums only
    W = w.sum()
    rows = []
    for name, m in [*zip(TAGS, M.T), ("(untagged)", ~M.any(1))]:
        lb = {s: float(w[m & (st == s)].sum()) for s in LOSS_ST}
        tot = sum(lb.values())
        rows.append({"tag": name, "n pairs": int(m.sum()), "% of TP": 100 * float(m[st == "TP"].mean()),
                     "loss pts FN": lb["FN"], "loss pts FP": lb["FP"], "loss pts miss": lb["miss"],
                     "total": tot, "% of loss": 100 * tot / W})
    rows.sort(key=lambda r: -r["total"])
    Mf = M.astype(np.float32)
    C, N = Mf.T @ (Mf * w[:, None].astype(np.float32)), Mf.T @ Mf
    i, j = np.triu_indices(len(TAGS), 1)
    pairs = sorted(({"tag A": TAGS[a], "tag B": TAGS[b], "n pairs": int(N[a, b]), "loss pts": float(C[a, b]),
                     "% of loss": 100 * float(C[a, b]) / W} for a, b in zip(i, j)), key=lambda r: -r["loss pts"])
    top3 = [r["tag"] for r in rows if r["tag"] != "(untagged)"][:3]
    return (["## D1. Tags x loss", "",
             "Tags overlap: a pair carries 0..n tags, so tag totals do not add to the loss; `(untagged)` does.", "",
             *md(rows), "", "### Top 15 tag pairs by loss (both tags on the same pair)", "", *md(pairs[:15]), ""],
            top3)


def d2_pseudo(X: pl.DataFrame, oof: pl.LazyFrame, code: pl.DataFrame, log) -> list[str]:
    pb = (X.filter(pl.col("pseudo_brand") & (pl.col("status") != "FP"))
          .select("reck", "status", true="s1k", rtxt=pl.col("r_at").list.join(" ")))
    cand = (oof.select("s1k", "reck").join(pb.lazy().select("reck"), on="reck", how="semi")
            .collect(engine="streaming").join(code, on="s1k").select("reck", "s1k"))
    cand = pl.concat([cand, pb.select("reck", s1k="true")]).unique()
    sa = load(NORM[0], cand["s1k"].unique(), "s1k", stxt=pl.col("addr_tokens").list.join(" "))
    pr = cand.join(pb, on="reck").join(sa, on="s1k")
    idx = np.arange(pr.height)
    pr = pr.with_columns(c=pl.Series(rowdot(vec(pr["rtxt"], 3), idx, vec(pr["stxt"], 3), idx)))
    ct = pl.col("c").filter(pl.col("s1k") == pl.col("true")).first()
    rk = pr.group_by("reck", "true", "status").agg(rank=1 + (pl.col("c") > ct).sum(), n_cand=pl.len())
    rows = [{"status": r["status"], "n pairs": r["n"], "% rank 1": r["r1"], "median rank": r["med"],
             "mean candidates": r["nc"]} for r in
            rk.group_by("status").agg(n=pl.len(), r1=100 * (pl.col("rank") == 1).mean(),
                                      med=pl.col("rank").median(), nc=pl.col("n_cand").mean())
            .sort("status").iter_rows(named=True)]
    log("d2 pseudo_brand", pairs=pb.height, cand_rows=pr.height)
    return ["### pseudo_brand: address-only rank of the true S1", "",
            "True pairs tagged pseudo_brand. Rank = 1 + # in-world candidate S1s of the record with a strictly higher "
            "address char-3gram cosine than the true S1 (the true S1 is added when blocking missed it).", "",
            *md(rows), ""]


def d2_jitter(X: pl.DataFrame) -> list[str]:
    d = X.select("status", d=(num("r_num") - num("s_num")).abs()).drop_nulls("d")
    lab = lambda lo, hi: str(lo) if lo == hi else f"{lo}+" if hi is None else f"{lo}-{hi}"
    rows = []
    for lo, hi in JBINS:
        m = (pl.col("d") >= lo) if hi is None else pl.col("d").is_between(lo, hi)
        rows.append({"|Δ number|": lab(lo, hi), **{
            s: 100 * float(d.filter(pl.col("status") == s).select(m.mean()).item() or 0) for s in
            ("TP", "FP", "FN", "miss")}})
    n = {s: d.filter(pl.col("status") == s).height for s in ("TP", "FP", "FN", "miss")}
    return ["### num_jitter: |Δ street number| (% of pairs with both numbers, per status)", "",
            f"n with both numbers: {n}.", "", *md(rows), ""]


def d2_legal(X: pl.DataFrame, log) -> list[str]:
    fdir = path("features_dir") / "train"
    names = set(pl.scan_parquet(str(fdir / "*.parquet")).collect_schema().names())
    imp = Counter()
    for k in range(5):
        b = lgb.Booster(model_file=str(path("models_dir") / f"fold_{k}.txt"))
        imp.update(dict(zip(b.feature_name(), b.feature_importance("gain"))))
    top = [f for f, _ in imp.most_common() if f in names][:20]
    legal = pl.any_horizontal("legal_added", "legal_removed", "legal_swapped", "distractor")
    sl = X.filter(pl.col("status").is_in(["TP", "FP"]) & legal).select("s1k", "reck", "status", *TAGS)
    fe = (pl.scan_parquet(str(fdir / "*.parquet"))
          .select(s1k=id_key("s1_id"), reck=id_key("rec_id"), *(pl.col(f).cast(pl.Float32) for f in top))
          .join(sl.lazy().select("s1k", "reck"), on=["s1k", "reck"], how="semi").collect(engine="streaming"))
    sl = sl.join(fe, on=["s1k", "reck"])
    y, g = (sl["status"] == "TP").to_numpy(), sl["s1k"].to_numpy()
    tag_feats = [t for t in TAGS if t != "distractor"]  # distractor is ground truth: would leak the label

    def auc(cols: list[str]) -> float:
        Xm = sl.select(pl.col(cols).cast(pl.Float32)).to_numpy()
        p = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(Xm, y, g):
            m = lgb.train({"objective": "binary", "seed": SEED, "deterministic": True, "force_row_wise": True,
                           "verbose": -1}, lgb.Dataset(Xm[tr], y[tr].astype(np.float32)), 200)
            p[va] = m.predict(Xm[va])
        return float(roc_auc_score(y, p))

    ok = 0 < y.sum() < len(y)
    rows = [{"features": "top-20 matcher", "AUC": auc(top) if ok else float("nan")},
            {"features": "top-20 matcher + tags", "AUC": auc(top + tag_feats) if ok else float("nan")}]
    log("d2 legal/distractor", n=sl.height)
    return ["### legal_* / distractor slice: TP vs FP", "",
            f"Slice = TP/FP pairs with a legal_* or distractor tag: n = {sl.height:,}, TP = {int(y.sum()):,}, "
            f"FP = {int((~y).sum()):,} (distractor FPs {int(sl['distractor'].sum()):,}). Top-20 features "
            f"(mean fold gain): {', '.join(top)}.", "", *md(rows), ""]


def d3_fn(X: pl.DataFrame) -> list[str]:
    rt = (X.select("reck", txt=pl.concat_str(pl.col("r_nt").list.join(" "), pl.col("r_at").list.join(" "),
                                             separator=" | ")).unique("reck").sort("reck").with_row_index("i"))
    A = vec(rt["txt"], 3)
    idx = rt.select("reck", "i")
    fn = X.filter(pl.col("status").is_in(["FN", "miss"])).select("s1k", "reck", "status", "w")
    pr = (fn.join(X.filter(pl.col("status") == "TP").select("s1k", nb="reck"), on="s1k")
          .filter(pl.col("reck") != pl.col("nb"))
          .join(idx, on="reck").join(idx.rename({"reck": "nb", "i": "j"}), on="nb"))
    pr = pr.with_columns(c=pl.Series(rowdot(A, pr["i"].to_numpy(), A, pr["j"].to_numpy())))
    mx = fn.join(pr.group_by("s1k", "reck").agg(pl.max("c")), on=["s1k", "reck"], how="left")
    rows = []
    for s in ("FN", "miss"):
        x = mx.filter(pl.col("status") == s)
        cut = lambda t: x.filter(pl.col("c") >= t)
        rows.append({"status": s, "n pairs": x.height, "loss pts": float(x["w"].sum()),
                     "with a TP sibling": x["c"].is_not_null().sum(),
                     "n cos>=0.8": cut(0.8).height, "pts cos>=0.8": float(cut(0.8)["w"].sum()),
                     "n cos>=0.9": cut(0.9).height, "pts cos>=0.9": float(cut(0.9)["w"].sum())})
    return ["## D3. Collective evidence", "", "### FN / miss records vs TP records of the same true S1", "",
            "Max name+addr char-3gram cosine of each FN/miss record to any TP record of its true S1.", "",
            *md(rows), ""]


def d3_precision(oof, code, kept, gtw, log) -> list[str]:
    recs = (oof.select("s1k", "reck").join(code.lazy(), on="s1k", how="semi").select("reck").unique()
            .collect(engine="streaming").sort("reck"))
    R = recs.filter(pl.Series(np.random.default_rng(SEED).random(recs.height) < FRAC))
    conf = kept.filter(pl.col("p_td") >= P_CONF).select("s1k", nb="reck")
    log("precision records", records=R.height, confident=conf.height)
    best = []
    for j in range(NP):
        Rj = R.filter(pl.col("reck") % NP == j)
        rows = (oof.select("s1k", "reck").join(Rj.lazy(), on="reck", how="semi")
                .join(code.lazy().select("s1k"), on="s1k", how="semi").collect(engine="streaming"))
        pr = rows.join(conf, on="s1k").filter(pl.col("reck") != pl.col("nb"))
        # neighbour texts loaded per partition, only the ones this partition touches (bounds RSS)
        tx = load(NORM[1:], Rj["reck"], "reck", txt=TXT).with_row_index("ia")
        pool = load(NORM[1:], pr["nb"].unique(), "nb", txt=TXT).with_row_index("ib")
        pr = pr.join(tx.select("reck", "ia"), on="reck").join(pool.select("nb", "ib"), on="nb")
        cs = rowdot(vec(tx["txt"], 3), pr["ia"].to_numpy(), vec(pool["txt"], 3), pr["ib"].to_numpy())
        best.append(pr.select("reck", "s1k").with_columns(c=pl.Series(cs))
                    .sort(["reck", "c", "s1k"], descending=[False, True, False])
                    .filter(pl.col("reck").is_first_distinct()))
        log(f"precision part {j}", pairs=pr.height)
    state = kept.select("reck", st=pl.when(pl.col("label") == 1).then(pl.lit("predicted, correct"))
                        .otherwise(pl.lit("predicted, wrong")))
    has_true = gtw.select("reck").unique().with_columns(ht=pl.lit(True))
    b = (pl.concat(best).join(gtw.with_columns(ok=pl.lit(True)), on=["s1k", "reck"], how="left")
         .join(state, on="reck", how="left").join(has_true, on="reck", how="left")
         .with_columns(ok=pl.col("ok").fill_null(False), st=pl.coalesce(
             "st", pl.when(pl.col("ht")).then(pl.lit("unpredicted, has true S1"))
             .otherwise(pl.lit("unpredicted, distractor")))))
    out = ["### Precision of 'take the S1 of the most similar confidently matched record'", "",
           f"{R.height:,} sampled records; {b.height:,} have >= 1 neighbour (record predicted with p >= {P_CONF} "
           "to one of its in-world candidate S1s). correct = neighbour's S1 is a true S1 of the record.", ""]
    for t in (0.8, 0.9):
        x = b.filter(pl.col("c") >= t)
        rows = [{"record state now": r["st"], "n": r["n"], "neighbour S1 correct": r["k"],
                 "precision %": 100 * r["k"] / r["n"]} for r in
                x.group_by("st").agg(n=pl.len(), k=pl.col("ok").sum()).sort("st").iter_rows(named=True)]
        rows.append({"record state now": "all", "n": x.height, "neighbour S1 correct": int(x["ok"].sum()),
                     "precision %": 100 * float(x["ok"].mean()) if x.height else float("nan")})
        out += [f"cosine >= {t}:", "", *md(rows), ""]
    return out


def main() -> None:
    global FRAC, NP
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% S1 and record samples, 1 precision partition")
    smoke = ap.parse_args().smoke
    if smoke:
        FRAC, NP = 0.01, 1
    log = Log()
    oof, s1, code, kept, in_cand, cnt, f, L = world(log)
    gt = load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))
    gtw = gt.join(code, on="s1k", how="semi").unique()
    P, samp = sample_pairs(s1, kept, in_cand, gt, cnt, f, L, log)
    del in_cand
    X = tag(P, gtw, log)
    del P
    sec1, top3 = d1(X)
    sec2 = ["## D2. Top tags", "", f"Top 3 tags by loss: {', '.join(top3)}. The three prescribed analyses follow.",
            "", *d2_pseudo(X, oof, code, log), *d2_jitter(X), *d2_legal(X, log)]
    sec3 = d3_fn(X)
    st = X["status"].value_counts().sort("status")
    ns, fs = int(samp.sum()), f[samp]
    del X
    sec3 += d3_precision(oof, code, kept, gtw, log)
    head = ["# Noise mechanisms behind the OOF (b) loss" + (" [SMOKE: 1% samples]" if smoke else ""),
            f"Generated by `src/mechanisms.py` (definitions in its docstring). World: cv_full (b), {s1.height:,} S1s, "
            f"t = {T}, macro F0.5 {f.mean():.6f}. Sample: {ns:,} S1s ({FRAC:.0%}, seed {SEED}), sample macro F0.5 "
            f"{fs.mean():.6f}, loss {100 * (1 - fs.mean()):.4f} pts (points below are sample macro points).", "",
            "Pairs by status: " + ", ".join(f"{s} {k:,}" for s, k in st.iter_rows()) + ".", ""]
    REPORT.write_text("\n".join(head + sec1 + sec2 + sec3 + [f"Peak RSS: {peak_rss_mb()} MB.", ""]))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / f"mechanisms{'_smoke' if smoke else ''}_timing.json")


if __name__ == "__main__":
    main()
