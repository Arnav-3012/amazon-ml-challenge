"""Blocking channel R: per-country joint name+address char-3gram TF-IDF, queried only for the records the
lexical channels serve worst.

Query set per country: records whose best v1 X_score (candidates_{split}_v1.parquet, 0 if none) is in the bottom
blocking.r_query_pct percent of their country, plus every record with no v1 candidate (an explicit union, so
r_query_pct = 0 still queries them). Ties are broken by hash(rec_id, seed), so the set is reproducible.
R is retrieval only: no df cap, no fallback, no S1-centric direction. S1 vocab + IDF are fitted per country on
that split's S1s (test IDF comes from test), cosine on L2-normed rows, top-3 S1s kept per record. Measured on
train, querying the bottom 10% keeps most of R's recall gain at a fraction of the cost of querying everything.

Run from code/business_entity_resolution/ (needs norm_{split}_s{1,2,3}.parquet, candidates_{split}_v1.parquet
from src.block, token_dict.parquet from src.mine_dict):
  python -m src.block_r --split train
  python -m src.block_r --split test
-> artifacts/interim/blockr_{split}.parquet (s1_id, rec_id, R_score, R_rrank)
"""
import argparse
import os

import numpy as np
import polars as pl
import scipy.sparse as sp

from .block import norm_path
from .io import CFG, StepLog, path, peak_rss_mb
from .mine_dict import OUT as DICT_PATH
from .tfidf import load_tokens, products, strings, vectorize

BCFG = CFG["blocking"]
TOP_M = 3  # gate.R_M must match
JOINT_CHAR3 = [("joint", "char", (3, 3))]  # name (no spaces) + " " + address, after token_dict on both sides
THREADS = os.cpu_count() or 1
MAX_RSS_MB = 14_000


def out_path(split: str):
    return path("interim_dir") / f"blockr_{split}.parquet"


def query_ids(split: str, pct: float) -> pl.DataFrame:
    """rec_id, country of records to query: best v1 X_score in the bottom pct%% of their country, or zero."""
    v1_path = path("interim_dir") / f"candidates_{split}_v1.parquet"
    assert v1_path.exists(), f"{v1_path} missing -- run src.block --split {split} first"
    best = pl.scan_parquet(v1_path).group_by("rec_id").agg(best=pl.col("X_score").max())
    return (pl.concat([pl.scan_parquet(norm_path(split, n)).select(rec_id="entity_id", country="country")
                       for n in (2, 3)])
            .join(best, on="rec_id", how="left").with_columns(pl.col("best").fill_null(0.0))
            .sort("country", "best", pl.col("rec_id").hash(CFG["seed"]))
            .with_columns(pct=(pl.int_range(1, pl.len() + 1).over("country") / pl.len().over("country")))
            .filter((pl.col("pct") <= pct / 100) | (pl.col("best") == 0.0))
            .select("rec_id", "country").collect(engine="streaming"))


def top_m_scored(S: sp.csr_matrix, Q: sp.csr_matrix, m: int) -> pl.DataFrame:
    """Per query row: the m best S1 rows with score > 0 (score desc, ties -> lower row), 1-based rank, score."""
    def fn(lo: int, R: np.ndarray) -> tuple:
        q, c, r, sc = [], [], [], []
        for k in range(R.shape[1]):
            col = R[:, k]
            idx = np.flatnonzero(col > 0)
            if idx.size > m:
                idx = idx[col[idx] >= np.partition(col[idx], idx.size - m)[idx.size - m]]
            idx = idx[np.lexsort((idx, -col[idx]))][:m]
            q.append(np.full(idx.size, lo + k)), c.append(idx)
            r.append(np.arange(1, idx.size + 1)), sc.append(col[idx])
        return tuple(np.concatenate(x) for x in (q, c, r, sc))
    if not Q.shape[0]:
        return pl.DataFrame(schema={"q": pl.Int64, "c": pl.Int64, "rank": pl.Int64, "s": pl.Float32})
    q, c, r, s = (np.concatenate(x) for x in zip(*products(S, Q, fn, THREADS, MAX_RSS_MB)))
    return pl.DataFrame({"q": q, "c": c, "rank": r, "s": s})


def build(split: str, pct: float, log: StepLog) -> None:
    d = pl.read_parquet(DICT_PATH)
    tdict = {f: dict(d.filter(pl.col("field") == f).select("s", "t").iter_rows()) for f in ("name", "addr")}
    q = query_ids(split, pct)
    log("query set", records=q.height, pct=pct)

    parts = []
    for c in sorted(q["country"].unique()):
        ids = q.filter(pl.col("country") == c)["rec_id"]
        s1 = load_tokens(split, 1, country=c)
        rec = pl.concat([load_tokens(split, n, country=c, ids=ids) for n in (2, 3)])
        if not rec.height:
            continue
        s1t, rect = strings(s1, tdict), strings(rec, tdict)
        Smat, (Qmat,) = vectorize(JOINT_CHAR3, s1t, [rect])
        tm = top_m_scored(Smat, Qmat, TOP_M)
        parts.append(tm.select(s1_id=s1["entity_id"].gather(tm["c"]), rec_id=rec["entity_id"].gather(tm["q"]),
                               R_score=pl.col("s"), R_rrank=pl.col("rank").cast(pl.Int16)))
        log(f"{c} R", s1=Smat.shape[0], vocab=Smat.shape[1], queries=Qmat.shape[0], hits=tm.height)
        del Smat, Qmat, s1, rec, s1t, rect
    out = pl.concat(parts) if parts else pl.DataFrame(
        schema={"s1_id": pl.String, "rec_id": pl.String, "R_score": pl.Float32, "R_rrank": pl.Int16})
    assert out.select("s1_id", "rec_id").is_duplicated().sum() == 0, "block_r: duplicate (s1_id, rec_id) rows"
    out.write_parquet(out_path(split))
    log("blockr written", path=str(out_path(split)), rows=out.height, peak_rss_mb=peak_rss_mb())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--pct", type=float, default=BCFG["r_query_pct"])
    a = ap.parse_args()
    log = StepLog()
    build(a.split, a.pct, log)
    log.dump(path("artifacts_dir") / "logs" / f"block_r_timing_{a.split}.json")


if __name__ == "__main__":
    main()
