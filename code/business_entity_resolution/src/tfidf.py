"""Brute-force TF-IDF retrieval within one country: S1 vocab + IDF, chunked S1 (csr) @ dense query block.

No S1 x record matrix is ever held; each chunk of query rows is scored against every S1 of the country and
reduced by the caller's `fn` before the next chunk. Used by src.block_r (channel R) and src.block_autopsy3.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .block import norm_path
from .io import peak_rss_mb

TOK = ["entity_id", "country", "name_tokens", "addr_tokens"]
DENSE_BYTES = 256 << 20  # per thread, for the n_S1 x c result and the vocab x c query block (memory knob only)


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
    """Fit on the S1 strings; returns S1 matrix and one matrix per query frame (rows L2-normed).
    spec = [(text column, analyzer, ngram_range)]; several entries are hstacked and re-normed."""
    vecs = [TfidfVectorizer(analyzer=a, ngram_range=g, lowercase=False, dtype=np.float32,
                            **({"token_pattern": r"\S+"} if a == "word" else {})) for _, a, g in spec]
    mats = [[v.fit_transform(s1[col].to_list()) for v, (col, _, _) in zip(vecs, spec)]]
    mats += [[v.transform(q[col].to_list()) for v, (col, _, _) in zip(vecs, spec)] for q in queries]
    out = [normalize(sp.hstack(m, format="csr")) if len(m) > 1 else m[0].tocsr() for m in mats]
    return out[0], out[1:]


def products(S: sp.csr_matrix, Q: sp.csr_matrix, fn, threads: int, max_rss_mb: int) -> list:
    """[fn(lo, R)] over chunks of Q rows, R = S @ Q[lo:hi].T as dense n_S1 x c, `threads` at a time, in order.
    scipy's sparse x dense kernel releases the GIL, so the threads run in parallel."""
    c = int(np.clip(DENSE_BYTES // (4 * max(S.shape)), 8, 128))

    def one(lo: int):
        out = fn(lo, S @ Q[lo:lo + c].T.toarray())
        if peak_rss_mb() > max_rss_mb:
            raise MemoryError(f"peak RSS {peak_rss_mb()} MB > {max_rss_mb}: lower DENSE_BYTES or threads")
        return out
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(one, range(0, Q.shape[0], c)))
