"""M5-4 blocking channel R: per-country joint name+addr char-3gram TF-IDF, queried only for records the v1/pool
channels are weakest on. Productionises src.block_autopsy3's V4 (docs/block_autopsy3.md): F0.5 ceiling 0.9964
(V1∪V2∪V4, m=10) vs v1's own ceiling, at 78-92% of the reachable gain kept by querying only the bottom
block.r_query_pct % of records (docs/block_autopsy3.md §curve). R is retrieval-only (no df cap, no fallback,
no S1-centric direction): S1 vocab + IDF fitted per country, L2-normalised cosine, top-3 kept.

Query set per country: records with best v1 X_score (candidates_{split}_v1.parquet, 0 if none) in the bottom
r_query_pct percentile of their country, plus every zero-v1 record unconditionally (subset of that percentile
whenever r_query_pct > 0, kept as an explicit union so r_query_pct = 0 still queries them). Percentile/tie
order matches block_autopsy3 Part 3 exactly (hash(rec_id, seed) tiebreak) so the query set is reproducible.

Identical code both splits; test IDF/vocab comes from test S1s only (no leakage).

Run from code/business_entity_resolution/ (needs norm_{split}_s{1,2,3}.parquet, token_dict.parquet,
candidates_{split}_v1.parquet):
  python -m src.block_r --split train
  python -m src.block_r --split test
-> artifacts/interim/blockr_{split}.parquet (s1_id, rec_id, R_score, R_rrank)
"""
import argparse
import os

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import block_autopsy3 as ba3
from .block import norm_path
from .block_autopsy3 import SPEC, load_tokens, products, strings, vectorize
from .io import CFG, StepLog, path, peak_rss_mb
from .mine_dict import OUT as DICT_PATH

BCFG = CFG["blocking"]
DEFAULT_PCT = BCFG.get("r_query_pct", 12)
TOP_M = 3
V4 = SPEC["V4"]
# products() (imported from block_autopsy3) resolves THREADS/DENSE_BYTES/MAX_RSS_MB in that module's own
# globals, so "all perf cores" + this module's RSS budget are set by patching them there, not here.
ba3.THREADS = os.cpu_count() or 1
ba3.MAX_RSS_MB = 14_000


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
    """Like block_autopsy3.top_m, but also keeps the score (needed for the gate's pre-score norm term)."""
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
    q, c, r, s = (np.concatenate(x) for x in zip(*products(S, Q, fn)))
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
        Smat, (Qmat,) = vectorize(V4, s1t, [rect])
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
    ap.add_argument("--pct", type=float, default=DEFAULT_PCT)
    a = ap.parse_args()
    log = StepLog()
    build(a.split, a.pct, log)
    log.dump(path("artifacts_dir") / "logs" / f"block_r_timing_{a.split}.json")


if __name__ == "__main__":
    main()
