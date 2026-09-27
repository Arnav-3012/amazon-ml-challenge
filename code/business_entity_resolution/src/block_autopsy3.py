"""M5 blocking autopsy 3: can uncapped, untruncated TF-IDF retrieval reach the true S1 of the pairs v1 misses,
and at what cost? (No GBM, no pipeline change.)

Sample + missed pairs = src.block_autopsy's: train S1s with hash(s1_id, seed) % 10 == 0, true pairs not in
candidates_train_v1. Each missed record is scored against ALL train S1s of its own country: cosine of L2-normed
TF-IDF, vocab + IDF fitted on that country's S1s only, chunked S1 (csr) @ dense query block. No S1 x record
matrix is ever held. rank = 1 + #S1 scoring higher + #S1 tied at a lower row index (block.top_n's tie order).
Score 0 (or true S1 in another country) = unreachable.
  V0 addr word unigrams        V1 name char 3-grams, spaces removed
  V2 addr char 3-grams + addr word bigrams (each block L2-normed, then the row)
  V3 name (no spaces) + " " + addr, char 3-grams       V4 = V3 after token_dict on both sides
  V5 = best (min) rank of V1 / V2 / V4
token_dict was mined on 80% of train S1s (src.mine_dict), so every row is also reported on its 20% holdout.
Part 2: record-side top-m of V1/V2/V4 for a seeded sample of train S2/S3 records: new pairs vs v1, row estimates,
gate.ceiling on the S1 sample (from Part-1 ranks: rank <= m == in the record's top-m) and minutes per full split
(sec/query modelled as proportional to the country's S1 count, since test has France and train has none).
Part 3: per record best v1 X_score (0 = no v1 candidate), percentile within country; share of the V5 top-m gain
kept when only the bottom q% records are queried.

Run from code/business_entity_resolution/:
  python -m src.block_autopsy3                  # docs/block_autopsy3.md
  python -m src.block_autopsy3 --p2-n 20000     # smaller Part-2 sample (same extrapolation)
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .block import norm_path
from .block_autopsy import V1_PATH, in_sample
from .decide import md
from .gate import PAIR, ceiling
from .io import CFG, ROOT, StepLog, load_gt_pairs, path, peak_rss_mb
from .mine_dict import HOLD, OUT as DICT_PATH
from .normalise import NONASCII

KS = (1, 3, 5, 10, 20)
MS = (3, 5, 10)
QS = (5, 10, 20, 30, 50, 75, 100)
# variant -> [(text column, analyzer, ngram_range)]; V4 = V3's spec on dict-mapped tokens
SPEC = {"V0": [("addr", "word", (1, 1))], "V1": [("name", "char", (3, 3))],
        "V2": [("addr", "char", (3, 3)), ("addr", "word", (2, 2))],
        "V3": [("joint", "char", (3, 3))], "V4": [("joint", "char", (3, 3))]}
DICT_VARIANTS = {"V4"}
P2_VARIANTS = ("V1", "V2", "V4")
THREADS = min(6, os.cpu_count() or 1)
DENSE_BYTES = 256 << 20  # per thread, for the n_S1 x c result and the vocab x c query block (memory knob only)
MAX_RSS_MB = 10_000
TOK = ["entity_id", "country", "name_tokens", "addr_tokens"]
REPORT = ROOT / "docs" / "block_autopsy3.md"

PART0 = [
    "## 0. v1 retrieval, from the code", "",
    "| channel | tokenizer | df cap | query truncation | min score | IDF rows |", "|---|---|---|---|---|---|",
    "| A | `name_tokens` (core name after legal/honorific strip, `\\S+` split: normalise.py:220, 144-145) + sorted "
    "adjacent bigrams (block.py:58-67, 88-90) + joined name when ≥ 2 tokens (block.py:91-94) | df ≤ 1000 "
    "(config.yaml `df_cap`; block.py:146 `rare`, 157) | none: every linkable token under the cap is kept; "
    "only when zero survive, the 2 lowest-df linkable tokens (fallback, block.py:46-47, 153-156) | none (any "
    "score > 0 is ranked); cut is rank top-m / top-k (block.py:186-196) | S1+S2+S3 of the country, same split "
    "(block.py:130-137, 145) |",
    "| B | `addr_tokens` + one `#<number> <street_core>` key (block.py:84-85, 99) | df ≤ 1000 | none, and "
    "**no fallback** (B not in FALLBACK_CHANNELS, block.py:46): a record with no address token under the cap "
    "has an empty B query | none | same |",
    "| C | phonetic skeletons: `n:` name skeletons + skeleton bigrams, `a:` address skeletons (block.py:100-101; "
    "src.phonetic not read) | df ≤ 1000 | fallback as A | none | same |",
    "| X | `A\\|t`, `B\\|t`, C tokens, + `K\\|<name skel>\\|<addr skel>` pairs (block.py:70-79, 95, 102-104) | "
    "df ≤ 1000 on X's own df table | fallback as A | none; v1 = X rrank ≤ 5 OR X srank ≤ 10 (gate.py:35) | same |",
    "",
    "- Linkable = token in S1 and in S2/S3 (block.py:142); df = records over S1+S2+S3 of the country, df ≥ 2 only "
    "(block.py:127). idf = ln(N_country / df) (block.py:145); test IDF comes from test.",
    "- Score = Σ idf over shared surviving tokens: S1 side idf-weighted, record side binary (block.py:299, 318). "
    "No TF, no length normalisation.",
    "- `nnz_budget` (40M, block.py:177-183, 199-209) only chunks the query rows by an upper bound on product nnz. "
    "It is a memory knob and truncates nothing. There is no N-rarest-token truncation outside the fallback.", ""]


def load_tokens(split: str, n: int, country: str | None = None, ids: pl.Series | None = None,
                extra: tuple[str, ...] = ()) -> pl.DataFrame:
    lf = pl.scan_parquet(norm_path(split, n))
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    if ids is not None:
        lf = lf.filter(pl.col("entity_id").is_in(ids.implode()))
    return lf.select(*TOK, *extra).collect(engine="streaming")


def strings(df: pl.DataFrame, d: dict[str, dict] | None = None) -> pl.DataFrame:
    """name (tokens joined, no spaces), addr (tokens joined by space), joint; token_dict applied first if given."""
    if d:
        df = df.with_columns(pl.col(f"{f}_tokens").list.eval(pl.element().replace(m)) for f, m in d.items())
    name, addr = pl.col("name_tokens").list.join(""), pl.col("addr_tokens").list.join(" ")
    return df.select(name=name, addr=addr, joint=pl.concat_str(name, pl.lit(" "), addr).str.strip_chars())


def vectorize(spec: list, s1: pl.DataFrame, queries: list[pl.DataFrame]) -> tuple[sp.csr_matrix, list]:
    """Fit on the S1 strings; returns S1 matrix and one matrix per query frame (rows L2-normed)."""
    vecs = [TfidfVectorizer(analyzer=a, ngram_range=g, lowercase=False, dtype=np.float32,
                            **({"token_pattern": r"\S+"} if a == "word" else {})) for _, a, g in spec]
    mats = [[v.fit_transform(s1[col].to_list()) for v, (col, _, _) in zip(vecs, spec)]]
    mats += [[v.transform(q[col].to_list()) for v, (col, _, _) in zip(vecs, spec)] for q in queries]
    out = [normalize(sp.hstack(m, format="csr")) if len(m) > 1 else m[0].tocsr() for m in mats]
    return out[0], out[1:]


def products(S: sp.csr_matrix, Q: sp.csr_matrix, fn) -> list:
    """[fn(lo, R)] over chunks of Q rows, R = S @ Q[lo:hi].T as dense n_S1 x c, THREADS at a time, in order.
    scipy's sparse x dense kernel releases the GIL, so the threads run in parallel."""
    c = int(np.clip(DENSE_BYTES // (4 * max(S.shape)), 8, 128))

    def one(lo: int):
        out = fn(lo, S @ Q[lo:lo + c].T.toarray())
        if peak_rss_mb() > MAX_RSS_MB:
            raise MemoryError(f"peak RSS {peak_rss_mb()} MB > {MAX_RSS_MB}: lower DENSE_BYTES or THREADS")
        return out
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(one, range(0, Q.shape[0], c)))


def ranks(S: sp.csr_matrix, Q: sp.csr_matrix, tgt: np.ndarray) -> np.ndarray:
    """Full rank of S1 row tgt[i] for query row i; 0 = unreachable (score 0 or tgt = -1)."""
    def fn(lo: int, R: np.ndarray) -> np.ndarray:
        t = tgt[lo:lo + R.shape[1]]
        ok = t >= 0
        tt, j = np.where(ok, t, 0), np.arange(R.shape[1])
        s = R[tt, j]
        ties = np.array([np.count_nonzero(R[:tt[k], k] == s[k]) for k in j])
        return np.where(ok & (s > 0), 1 + (R > s).sum(0) + ties, 0)
    return np.concatenate(products(S, Q, fn)) if Q.shape[0] else np.zeros(0, np.int64)


def top_m(S: sp.csr_matrix, Q: sp.csr_matrix, m: int) -> pl.DataFrame:
    """Per query row: the m best S1 rows with score > 0 (score desc, ties -> lower row), rank 1-based."""
    def fn(lo: int, R: np.ndarray) -> tuple:
        q, c, r = [], [], []
        for k in range(R.shape[1]):
            col = R[:, k]
            idx = np.flatnonzero(col > 0)
            if idx.size > m:
                idx = idx[col[idx] >= np.partition(col[idx], idx.size - m)[idx.size - m]]
            idx = idx[np.lexsort((idx, -col[idx]))][:m]
            q.append(np.full(idx.size, lo + k)), c.append(idx), r.append(np.arange(1, idx.size + 1))
        return tuple(np.concatenate(x) for x in (q, c, r))
    if not Q.shape[0]:
        return pl.DataFrame(schema={"q": pl.Int64, "c": pl.Int64, "rank": pl.Int64})
    q, c, r = (np.concatenate(x) for x in zip(*products(S, Q, fn)))
    return pl.DataFrame({"q": q, "c": c, "rank": r})


def recall_rows(M: pl.DataFrame, seg: str, e: pl.Expr) -> list[dict]:
    x = M.filter(e)
    rows = []
    for v in (*SPEC, "V5"):
        r = x[f"r_{v}"]
        rows.append({"segment": seg, "pairs": x.height, "variant": v,
                     **{f"R@{k} %": round(float((r <= k).fill_null(False).mean() or 0) * 100, 2) for k in KS},
                     "unreachable %": round(float(r.is_null().mean() or 0) * 100, 2)})
    return rows


def counts(split: str, n: int) -> dict[str, int]:
    return dict(pl.scan_parquet(norm_path(split, n)).group_by("country").len().collect().rows())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-n", type=int, default=200_000, help="Part-2 sampled train S2/S3 records")
    a = ap.parse_args()
    log = StepLog()
    seed = CFG["seed"]
    I = path("interim_dir")

    # ---- missed pairs (same definition as src.block_autopsy) ----
    s1_all = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    S = s1_all.filter(in_sample())
    gt = load_gt_pairs().select("s1_id", rec_id="match_id").filter(in_sample()).join(S.select("s1_id"), on="s1_id")
    ntrue = S.select("s1_id").join(gt.group_by("s1_id").len("ntrue"), on="s1_id", how="left").with_columns(
        pl.col("ntrue").fill_null(0))
    v1 = pl.scan_parquet(V1_PATH).filter(in_sample()).select(*PAIR, v1=pl.lit(True)).collect(engine="streaming")
    T = gt.join(v1, on=PAIR, how="left").with_columns(pl.col("v1").fill_null(False))
    ids = s1_all["s1_id"].sort()  # mine_dict.sides() holdout, same seed and order
    held = ids.gather(np.random.default_rng(seed).permutation(ids.len())[: round(HOLD * ids.len())])
    M = T.filter(~pl.col("v1")).drop("v1").with_columns(held=pl.col("s1_id").is_in(held.implode()))
    log("missed pairs", sample_s1=S.height, true_pairs=T.height, missed=M.height)  # block_autopsy: 28,260

    # ---- records: missed + Part-2 sample ----
    recs = pl.concat([pl.scan_parquet(norm_path("train", n)).select("entity_id") for n in (2, 3)])
    p2_ids = recs.sort(pl.col("entity_id").hash(seed)).head(a.p2_n).collect(engine="streaming")["entity_id"]
    need = pl.concat([M["rec_id"], p2_ids]).unique()
    R = pl.concat([load_tokens("train", n, ids=need, extra=("business_name", "has_address")) for n in (2, 3)])
    R = R.with_columns(name_nonascii_raw=pl.col("business_name").str.contains(NONASCII)).drop("business_name")
    M = M.join(R.select(rec_id="entity_id", rec_country="country", has_address="has_address",
                        name_nonascii_raw="name_nonascii_raw"), on="rec_id", how="left")
    P2 = R.filter(pl.col("entity_id").is_in(p2_ids.implode()))
    d = pl.read_parquet(DICT_PATH)
    tdict = {f: dict(d.filter(pl.col("field") == f).select("s", "t").iter_rows()) for f in ("name", "addr")}
    log("records", missed_recs=M["rec_id"].n_unique(), p2=P2.height, dict_mappings=d.height)

    # ---- Part 1 ranks + Part 2 top-m, per country x variant ----
    rank_parts, top_parts, timing = [], [], []
    for c in sorted(s1_all["country"].unique()):
        s1 = load_tokens("train", 1, country=c)
        idx = s1.select(s1_id="entity_id").with_row_index("t")
        p1 = (M.filter(pl.col("rec_country") == c).select(*PAIR).join(idx, on="s1_id", how="left")
              .join(R.select(rec_id="entity_id", name_tokens="name_tokens", addr_tokens="addr_tokens"), on="rec_id"))
        p2 = P2.filter(pl.col("country") == c)
        tgt = p1["t"].fill_null(-1).cast(pl.Int64).to_numpy()
        txt = {dct: [strings(f, tdict if dct else None) for f in (s1, p1, p2)] for dct in (False, True)}
        out = p1.select(*PAIR)
        for v, spec in SPEC.items():
            s1t, p1t, p2t = txt[v in DICT_VARIANTS]
            Smat, (Q1, Q2) = vectorize(spec, s1t, [p1t, p2t])
            r = ranks(Smat, Q1, tgt)
            out = out.with_columns(pl.Series(f"r_{v}", r)).with_columns(
                pl.when(pl.col(f"r_{v}") > 0).then(pl.col(f"r_{v}")).alias(f"r_{v}"))  # 0 -> null = unreachable
            log(f"{c} {v} part1", s1=Smat.shape[0], vocab=Smat.shape[1], nnz=Smat.nnz, queries=Q1.shape[0])
            if v in P2_VARIANTS:
                t0 = time.perf_counter()
                tm = top_m(Smat, Q2, max(MS))
                sec = time.perf_counter() - t0
                top_parts.append(tm.select(variant=pl.lit(v), rec_id=p2["entity_id"].gather(tm["q"]),
                                           s1_id=s1["entity_id"].gather(tm["c"]), rank="rank"))
                timing.append({"country": c, "variant": v, "queries": Q2.shape[0], "n_s1": Smat.shape[0],
                               "sec": sec})
                log(f"{c} {v} part2", queries=Q2.shape[0], sec=round(sec, 1))
            del Smat, Q1, Q2
        rank_parts.append(out)
        del s1, txt
    M = M.join(pl.concat(rank_parts), on=PAIR, how="left").with_columns(
        r_V5=pl.min_horizontal("r_V1", "r_V2", "r_V4"))
    tops = pl.concat(top_parts)
    log("parts 1-2 scored")

    # ---- Part 1 report ----
    rec1 = recall_rows(M, "all", pl.lit(True)) + recall_rows(M, "dict holdout S1s", pl.col("held"))
    for (c,), _ in sorted(M.group_by("rec_country")):
        rec1 += recall_rows(M, f"country {c}", pl.col("rec_country") == c)
    for ha in (True, False):
        for na in (False, True):
            rec1 += recall_rows(M, f"has_address={ha}, name_nonascii_raw={na}",
                                (pl.col("has_address") == ha) & (pl.col("name_nonascii_raw") == na))

    # ---- Part 2 report ----
    cnt = {(sp_, n): counts(sp_, n) for sp_ in ("train", "test") for n in (1, 2, 3)}
    n_s1 = {sp_: cnt[sp_, 1] for sp_ in ("train", "test")}
    n_rec = {sp_: {k: cnt[sp_, 2].get(k, 0) + cnt[sp_, 3].get(k, 0) for k in n_s1[sp_]} for sp_ in ("train", "test")}
    v1_rows = {sp_: (pl.scan_parquet(I / f"candidates_{sp_}_v1.parquet").select(pl.len()).collect().item()
                     if (I / f"candidates_{sp_}_v1.parquet").exists() else None) for sp_ in ("train", "test")}
    v1_p2 = (pl.scan_parquet(V1_PATH).select(PAIR).join(P2.select(rec_id="entity_id").lazy(), on="rec_id", how="semi")
             .collect(engine="streaming").with_columns(v1=pl.lit(True)))
    tm_df = pl.DataFrame(timing)
    # sec/query = a * n_S1 of the country (brute force is linear in the index size)
    a_v = {v: g["sec"].sum() / (g["queries"] * g["n_s1"]).sum() for (v,), g in tm_df.group_by("variant")}
    load = {sp_: sum(n_rec[sp_][k] * n_s1[sp_][k] for k in n_s1[sp_]) for sp_ in ("train", "test")}
    T2 = T.join(M.select(*PAIR, *(f"r_{v}" for v in (*SPEC, "V5"))), on=PAIR, how="left")
    base = ceiling(ntrue, T2.filter("v1").group_by("s1_id").len("h"))
    rows2 = []
    for v, vs, rcol in [(v, [v], f"r_{v}") for v in P2_VARIANTS] + [("V1∪V2∪V4", list(P2_VARIANTS), "r_V5")]:
        for m in MS:
            new = (tops.filter(pl.col("variant").is_in(vs) & (pl.col("rank") <= m)).select(PAIR).unique()
                   .join(v1_p2, on=PAIR, how="anti").height) / max(P2.height, 1)
            hit = T2.filter(pl.col("v1") | (pl.col(rcol) <= m).fill_null(False)).group_by("s1_id").len("h")
            mins = {sp_: round(sum(a_v[x] for x in vs) * load[sp_] / 60, 1) for sp_ in ("train", "test")}
            rows2.append({"variant": v, "m": m, "new pairs/rec": round(new, 3),
                          **{f"est rows {sp_}": (int(v1_rows[sp_] + new * sum(n_rec[sp_].values()))
                                                 if v1_rows[sp_] is not None else "-") for sp_ in ("train", "test")},
                          **{f"est cand/S1 {sp_}": (round((v1_rows[sp_] + new * sum(n_rec[sp_].values()))
                                                          / sum(n_s1[sp_].values()), 1)
                                                    if v1_rows[sp_] is not None else "-") for sp_ in ("train", "test")},
                          "F0.5 ceiling": round(ceiling(ntrue, hit), 4), "Δ vs v1": round(ceiling(ntrue, hit) - base, 4),
                          "min train": mins["train"], "min test": mins["test"]})
    tm_rows = tm_df.with_columns(sec_per_10k=(pl.col("sec") / pl.col("queries") * 1e4).round(1),
                                 sec=pl.col("sec").round(1)).to_dicts()
    log("part2")

    # ---- Part 3: best v1 X score per record ----
    best = pl.scan_parquet(V1_PATH).group_by("rec_id").agg(best=pl.col("X_score").max())
    allr = (pl.concat([pl.scan_parquet(norm_path("train", n)).select(rec_id="entity_id", country="country")
                       for n in (2, 3)])
            .join(best, on="rec_id", how="left").with_columns(pl.col("best").fill_null(0.0))
            .sort("country", "best", pl.col("rec_id").hash(seed))
            .with_columns(pct=(pl.int_range(1, pl.len() + 1).over("country") / pl.len().over("country")))
            .collect(engine="streaming"))
    mrec = allr.join(M.select("rec_id").unique(), on="rec_id", how="semi")
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    dist = []
    for (c,), g in sorted(allr.group_by("country")):
        for who, x in (("all records", g), ("missed-pair records", mrec.filter(pl.col("country") == c))):
            dist.append({"country": c, "records": who, "n": x.height,
                         "zero v1 %": round(float((x["best"] == 0).mean() or 0) * 100, 2),
                         **{f"p{int(q * 100)}": round(float(x["best"].quantile(q) or 0), 2) for q in qs}})
    G = M.join(allr.select("rec_id", "pct"), on="rec_id", how="left")
    curve = []
    for q in QS:
        row = {"query bottom q% records (per country)": q}
        for m in MS:
            g = G.filter((pl.col("r_V5") <= m).fill_null(False))
            row[f"gain kept % (V5 m={m}, gain {g.height:,} pairs)"] = round(
                float((g["pct"] <= q / 100).mean() or 0) * 100, 2)
        curve.append(row)
    del allr
    log("part3")

    L = ["# Blocking autopsy 3: brute-force TF-IDF on the v1 misses",
         f"Generated by `src/block_autopsy3.py`. Sample: {S.height:,} train S1s (hash(s1_id, seed {seed}) % 10 == 0, "
         f"as in src.block_autopsy), {T.height:,} true pairs, **{M.height:,} missed by v1**. Each missed record is "
         "scored against every train S1 of its country: cosine of L2-normed TF-IDF (sklearn, smooth IDF), vocab + "
         "IDF fitted on that country's S1s, no df cap, no truncation. rank = 1 + #S1 scoring higher + #S1 tied at "
         "a lower row. Unreachable = score 0 or true S1 in another country. V0 addr words; V1 name char 3-grams, "
         "no spaces; V2 addr char 3-grams + addr word bigrams; V3 name+addr char 3-grams; V4 = V3 after "
         "token_dict; V5 = best of V1/V2/V4. token_dict was mined on 80% of train S1s, so the "
         "`dict holdout S1s` row is the leak-free V4/V5 number.", "",
         *PART0,
         "## 1. Recall of the true S1 over all S1s of the country (missed pairs)", "", *md(rec1), "",
         f"## 2. Record-side top-m on {P2.height:,} sampled train S2/S3 records", "",
         "new pairs/rec = top-m pairs not in v1, per sampled record. est rows = v1 rows of the split + new/rec x "
         "S2+S3 records of the split (train rate applied to test). F0.5 ceiling = gate.ceiling on the S1 sample "
         f"with v1 ∪ (Part-1 rank ≤ m). v1 ceiling on the sample: {base:.4f}. Minutes = wall-clock at "
         f"{THREADS} threads, query time only (index build excluded), sec/query = a x n_S1(country) fitted "
         "on train, then summed over the split's countries (France included via its S1 count).", "",
         *md(rows2), "", "Measured query time:", "", *md(tm_rows), "",
         "## 3. Best v1 X score per record", "",
         "best = max X_score over the record's v1 candidates (0 = none). Percentile within country, ties by "
         "hash(rec_id).", "", *md(dist), "",
         "Share of the V5 top-m recall gain kept if only the bottom q% of records (by best v1 X score) are queried:",
         "", *md(curve), "",
         f"Peak RSS: {peak_rss_mb()} MB.", ""]
    REPORT.write_text("\n".join(L))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / "block_autopsy3_timing.json")


if __name__ == "__main__":
    main()
