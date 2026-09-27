"""M3b blocking v1: candidate (S1, S2/S3) pairs from the M3a normalised caches (artifacts/interim/norm_*.parquet).

Per country group (country is only the partition key: 100% of train true pairs share it; never a feature):
  name      S1 -> pool top-k, char n-gram TF-IDF on core_name
  addr      S1 -> pool top-k, char n-gram TF-IDF on the address; the street part (addr_number + addr_street_core)
            weighs 1, every other address token `addr_rest_weight`. admin_region is never used (the normaliser
            already moved it out of addr_tokens): France S1 carries the region, S2/S3 the departement.
  name_rev  empty-address S2/S3 -> S1 top-k by name (the address channel cannot see them; this keeps them
            from being crowded out of S1-side name top-k lists)
  key_*     exact keys: core_name, core_name|house no., house no.|street, postcode|first name token;
            a key block is used only if n_s1 * n_pool <= key_block_max_pairs (so "subway" doesn't explode)
Retrieval is an inverted index by design (Revision 2a): index = gram -> postings CSR, grams with more than
df_cap postings are stop-grams, each query keeps its top_m rarest grams, so the work per query is bounded by
top_m * df_cap. Scores are cosine restricted to those grams (TF binary, idf over S1+pool, L2 on full vectors).

Run from code/business_entity_resolution/:
  python -m src.blocking --selftest            # inverted-index top-k == brute force on synthetic data
  python -m src.blocking --s1-frac 0.05        # train only on a 5% S1 sample (PC/RR report, fast tuning)
  python -m src.blocking                       # train (report) + test -> output/candidate_pairs.tsv
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp

from .io import CFG, ROOT, load_gt_pairs, path
from .normalise import peak_rss_mb

P = CFG["blocking"]
SEED = CFG["seed"]
REPORT = ROOT / "docs" / "blocking.md"
COLS = ["entity_id", "country", "core_name", "addr_number", "addr_street_core", "addr_tokens", "has_address",
        "is_nonascii_raw"]
# One bit per channel; a pair's `bits` is the OR of the channels that produced it.
CH = {"name": 1, "addr": 2, "name_rev": 4, "key_name": 8, "key_name_num": 16, "key_num_street": 32,
      "key_post_name": 64}
VEC_BITS = CH["name"] | CH["addr"]
KS = (1, 3, 5, 10, 20, 50)  # PC@k curve points (only those <= the configured k are reported)


# ---------------------------------------------------------------- loading
def postcode(full: pl.Expr) -> pl.Expr:
    """Last 5-6 digit token that is not the first token (a leading number is the house number)."""
    return full.str.replace(r"^\S+", "").str.extract_all(r"\b\d{5,6}\b").list.last()


def _load(split: str, n: int, country: str) -> pl.DataFrame:
    df = (pl.scan_parquet(path("interim_dir") / f"norm_{split}_s{n}.parquet")
          .filter(pl.col("country") == country).select(COLS).collect())
    full = pl.col("addr_tokens").list.join(" ")
    return df.with_columns(
        addr_full=full,
        addr_street=pl.concat_str("addr_number", "addr_street_core", separator=" ").str.strip_chars(),
        postcode=postcode(full),
    ).drop("addr_tokens")


def _pool(split: str, country: str) -> pl.DataFrame:
    return pl.concat([_load(split, 2, country), _load(split, 3, country)])


# ---------------------------------------------------------------- vectors
def _grams(text: pl.Series, n: int, w: float) -> pl.DataFrame:
    """(r, h, w): one row per char n-gram occurrence of " text ", h = 64-bit hash of the gram."""
    t = pl.DataFrame({"t": text}).with_row_index("r").with_columns(t=pl.concat_str(pl.lit(" "), "t", pl.lit(" ")))
    return (t.filter(pl.col("t").str.len_chars() >= n)
            .with_columns(o=pl.int_ranges(0, pl.col("t").str.len_chars() - n + 1, dtype=pl.UInt32))
            .explode("o")
            .select("r", h=pl.col("t").str.slice(pl.col("o"), n).hash(SEED), w=pl.lit(w, pl.Float32)))


def _doc_grams(df: pl.DataFrame, ch: str) -> pl.DataFrame:
    """Binary TF per (doc, gram); in the address channel a gram takes its highest part weight."""
    n = P[f"{ch}_ngram"]
    parts = ([(df["core_name"], 1.0)] if ch == "name"
             else [(df["addr_street"], 1.0), (df["addr_full"], P["addr_rest_weight"])])
    return pl.concat([_grams(s, n, w) for s, w in parts]).group_by("r", "h").agg(pl.col("w").max())


def _batches(df: pl.DataFrame):
    for off in range(0, df.height, P["batch_rows"]):
        yield off, df.slice(off, P["batch_rows"])


def _df_counts(df: pl.DataFrame, ch: str, name: str) -> pl.DataFrame:
    parts = [_doc_grams(b, ch).group_by("h").len() for _, b in _batches(df)]
    return pl.concat(parts).group_by("h").agg(pl.col("len").sum().alias(name))


def _vocab(s1: pl.DataFrame, pool: pl.DataFrame, ch: str) -> pl.DataFrame:
    """h -> col id, df in S1 and pool, smoothed idf over S1+pool (sklearn formula)."""
    v = (_df_counts(s1, ch, "df_s1").join(_df_counts(pool, ch, "df_pool"), on="h", how="full", coalesce=True)
         .fill_null(0).sort("h").with_row_index("col"))
    N = s1.height + pool.height
    return v.with_columns(idf=(((N + 1) / (pl.col("df_s1") + pl.col("df_pool") + 1)).log() + 1).cast(pl.Float32))


def _vectors(df: pl.DataFrame, ch: str, vocab: pl.DataFrame, specs: list[tuple]) -> list[tuple]:
    """L2-normalised TF-IDF rows, pruned per spec = (cap_col, top_m | None, row_mask | None).

    cap_col is the df of the side being searched: grams absent from it or above df_cap are dropped.
    top_m keeps each row's top_m highest-weight (= rarest) grams: query side only.
    Returns one (row, col, val) numpy triple per spec; rows index into `df`.
    """
    out = [([], [], []) for _ in specs]
    voc = vocab.select("h", "col", "idf", "df_s1", "df_pool")
    for off, b in _batches(df):
        # sorted after the join so the float row norms sum in a fixed order (joins don't keep order)
        g = (_doc_grams(b, ch).join(voc, on="h").sort("r", "col").with_columns(x=pl.col("w") * pl.col("idf"))
             .with_columns(v=(pl.col("x") / (pl.col("x") ** 2).sum().over("r").sqrt()).cast(pl.Float32))
             .select("r", "col", "v", "df_s1", "df_pool"))
        for (cap_col, top_m, mask), (R, C, V) in zip(specs, out):
            x = g.filter((pl.col(cap_col) > 0) & (pl.col(cap_col) <= P["df_cap"]))
            if mask is not None:
                x = x.filter(pl.col("r").is_in(pl.Series(np.flatnonzero(mask[off:off + b.height]), dtype=pl.UInt32)))
            if top_m:
                x = (x.sort(["r", "v", "col"], descending=[False, True, False])
                     .group_by("r", maintain_order=True).head(top_m))
            R.append(x["r"].to_numpy().astype(np.int32) + off)
            C.append(x["col"].to_numpy().astype(np.int32))
            V.append(x["v"].to_numpy())
    empty = (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    return [tuple(np.concatenate(a) if a else e for a, e in zip(trip, empty)) for trip in out]


# ---------------------------------------------------------------- inverted-index top-k
def _knn(q: tuple, idx: tuple, n_idx: int, n_feat: int, k: int) -> tuple[pl.DataFrame, int]:
    """Top-k index rows per query row by pruned cosine. Returns (q, i, score, rank) and #queries with >=1 gram.

    INV is the inverted index (gram -> postings); Q_chunk @ INV only touches postings of the query's grams.
    Ties at the k boundary break by position in scipy's (deterministic) output, via a stable sort.
    """
    qr, qc, qv = q
    ir, ic, iv = idx
    uq, ql = np.unique(qr, return_inverse=True)  # queries with no surviving gram drop out here
    Q = sp.csr_matrix((qv, (ql, qc)), shape=(uq.size, n_feat), dtype=np.float32)
    INV = sp.csr_matrix((iv, (ic, ir)), shape=(n_feat, n_idx), dtype=np.float32)
    Q.sort_indices()
    INV.sort_indices()
    chunk = P["chunk_rows"]

    def work(s: int):
        S = Q[s:s + chunk] @ INV
        cnt = np.diff(S.indptr)
        rows = np.repeat(np.arange(cnt.size, dtype=np.uint64), cnt)
        top = float(S.data.max()) if S.nnz else 1.0  # 32-bit score quantum relative to the chunk max
        desc = np.uint64(2**32 - 1) - (S.data.astype(np.float64) / top * (2**32 - 1)).astype(np.uint64)
        o = np.argsort((rows << np.uint64(32)) | desc, kind="stable")
        rank = np.arange(o.size) - np.repeat(S.indptr[:-1], cnt)
        keep = rank < k
        sel = o[keep]
        return uq[rows[sel].astype(np.int64) + s], S.indices[sel], S.data[sel], rank[keep]

    with ThreadPoolExecutor(P["threads"]) as ex:
        res = list(ex.map(work, range(0, uq.size, chunk)))
    cols = [np.concatenate([r[j] for r in res]) if res else np.empty(0, np.int64) for j in range(4)]
    return pl.DataFrame({"q": cols[0].astype(np.uint32), "i": cols[1].astype(np.uint32),
                         "score": cols[2].astype(np.float32), "rank": cols[3].astype(np.int16)}), int(uq.size)


# ---------------------------------------------------------------- exact keys
def _key_pairs(s1: pl.DataFrame, pool: pl.DataFrame, key: pl.Expr) -> pl.DataFrame:
    a = s1.select(s1="i", key=key).filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
    b = pool.select(p="i", key=key).filter(pl.col("key").is_not_null() & (pl.col("key") != ""))
    ok = (a.group_by("key").len("na").join(b.group_by("key").len("nb"), on="key")
          .filter(pl.col("na") * pl.col("nb") <= P["key_block_max_pairs"]).select("key"))
    return a.join(ok, on="key").join(b, on="key").select("s1", "p")


def _nonempty_join(*cols: str) -> pl.Expr:
    return pl.when(pl.all_horizontal(pl.col(c) != "" for c in cols)).then(pl.concat_str(*cols, separator="|"))


KEYS = {
    "key_name": pl.col("core_name"),
    "key_name_num": _nonempty_join("core_name", "addr_number"),
    "key_num_street": _nonempty_join("addr_number", "addr_street_core"),
    "key_post_name": pl.when(pl.col("postcode").is_not_null() & (pl.col("core_name") != ""))
    .then(pl.concat_str("postcode", pl.col("core_name").str.extract(r"^(\S+)"), separator="|")),
}


# ---------------------------------------------------------------- one country group
def block_country(s1: pl.DataFrame, pool: pl.DataFrame, q_mask: np.ndarray, stats: dict) -> pl.DataFrame:
    """All channels for one country; returns (s1, p, bits, name_rank, name_score, addr_rank, addr_score).

    q_mask selects the S1 rows used as queries (all rows, or the --s1-frac sample); keys and name_rev are
    computed against the full S1 and filtered to the sample, so sampling doesn't change their blocks.
    """
    frames = []
    no_addr = ~pool["has_address"].to_numpy()
    for ch in ("name", "addr"):
        t = time.perf_counter()
        vocab = _vocab(s1, pool, ch)
        s1_specs = [("df_pool", P["top_m"], q_mask)] + ([("df_s1", None, None)] if ch == "name" else [])
        pool_specs = [("df_pool", None, None)] + ([("df_s1", P["top_m"], no_addr)] if ch == "name" else [])
        s1_vec = _vectors(s1, ch, vocab, s1_specs)
        pool_vec = _vectors(pool, ch, vocab, pool_specs)
        n_feat = vocab.height
        del vocab
        fwd, nq = _knn(s1_vec[0], pool_vec[0], pool.height, n_feat, P[f"k_{ch}"])
        stats[f"{ch}_queries_with_grams"] = nq
        frames.append(fwd.select(s1="q", p="i", ch=pl.lit(CH[ch], pl.UInt8),
                                 **{f"{ch}_rank": "rank", f"{ch}_score": "score"}))
        if ch == "name":
            rev, nq = _knn(pool_vec[1], s1_vec[1], s1.height, n_feat, P["k_name_rev"])
            stats["name_rev_queries"] = int(no_addr.sum())
            stats["name_rev_queries_with_grams"] = nq
            frames.append(rev.select(s1="i", p="q", ch=pl.lit(CH["name_rev"], pl.UInt8)))
        del s1_vec, pool_vec
        stats[f"{ch}_s"] = round(time.perf_counter() - t, 1)
        stats[f"{ch}_peak_rss_mb"] = peak_rss_mb()
    t = time.perf_counter()
    for name, key in KEYS.items():
        frames.append(_key_pairs(s1, pool, key).with_columns(ch=pl.lit(CH[name], pl.UInt8)))
    stats["keys_s"] = round(time.perf_counter() - t, 1)
    cand = pl.concat(frames, how="diagonal_relaxed")
    if not q_mask.all():
        cand = cand.filter(pl.col("s1").is_in(pl.Series(np.flatnonzero(q_mask), dtype=pl.UInt32)))
    cand = (cand.group_by("s1", "p")
            .agg(bits=pl.col("ch").cast(pl.UInt8).unique().sum().cast(pl.UInt8),
                 name_rank=pl.col("name_rank").min(), name_score=pl.col("name_score").max(),
                 addr_rank=pl.col("addr_rank").min(), addr_score=pl.col("addr_score").max())
            .sort("s1", "p"))
    stats["pairs_by_channel"] = {c: int(((cand["bits"] & b) > 0).sum()) for c, b in CH.items()}
    return cand


# ---------------------------------------------------------------- evaluation
def _dist(counts: np.ndarray) -> dict:
    q = np.percentile(counts, [50, 90, 99]) if counts.size else [0, 0, 0]
    return {"mean": round(float(counts.mean()), 2) if counts.size else 0, "p50": int(q[0]), "p90": int(q[1]),
            "p99": int(q[2]), "max": int(counts.max()) if counts.size else 0, "zero": int((counts == 0).sum())}


def _within_k(k: int) -> pl.Expr:
    """Pair survives if the vector channels had it at rank < k, or any non-vector channel produced it."""
    return ((pl.col("name_rank") < k).fill_null(False) | (pl.col("addr_rank") < k).fill_null(False)
            | ((pl.col("bits") & (0x7F ^ VEC_BITS)) > 0))


def evaluate(cand: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, q_mask: np.ndarray,
             gt: pl.DataFrame | None, stats: dict, missed_out: list) -> None:
    n_q = int(q_mask.sum())
    counts = np.zeros(s1.height, np.int64)
    c = cand.group_by("s1").len()
    counts[c["s1"].to_numpy()] = c["len"].to_numpy()
    stats |= {"n_s1": n_q, "n_pool": pool.height, "n_cand": cand.height,
              "rr_country": 1 - cand.height / max(n_q * pool.height, 1), "cand_per_s1": _dist(counts[q_mask])}
    kmax = max(P["k_name"], P["k_addr"])
    stats["cand_at_k"] = {k: int(cand.select(_within_k(k).sum()).item()) for k in KS if k <= kmax}
    if gt is None:
        return
    ids1 = s1.select(s1_id="entity_id", s1="i").filter(pl.Series(q_mask))
    idp = pool.select(match_id="entity_id", p="i", m_has_addr="has_address", m_nonascii="is_nonascii_raw")
    g = (gt.join(ids1, on="s1_id").join(idp, on="match_id")
         .join(cand.select("s1", "p", "bits", "name_rank", "addr_rank"), on=["s1", "p"], how="left")
         .with_columns(found=pl.col("bits").is_not_null(), bits=pl.col("bits").fill_null(0)))
    stats["n_true"] = g.height
    stats["pc"] = g["found"].mean()
    stats["pc_alone"] = {ch: ((g["bits"] & b) > 0).mean() for ch, b in CH.items()}
    stats["pc_only"] = {ch: (g["bits"] == b).mean() for ch, b in CH.items()}
    stats["pc_at_k"] = {k: g.select(_within_k(k).mean()).item() for k in KS if k <= kmax}
    stats["pc_by"] = {f"{col}={v}": (x["found"].mean(), x.height)
                      for col in ("m_has_addr", "m_nonascii") for (v,), x in g.group_by(col) if x.height}
    ent = g.group_by("s1").agg(pl.col("found").all())
    stats["entities_all_found"] = ent["found"].mean()
    stats["entities_none_found"] = 1 - g.group_by("s1").agg(pl.col("found").any())["found"].mean()
    miss = g.filter(~pl.col("found"))
    if miss.height:
        missed_out.append(miss.sort("s1_id", "match_id").sample(min(P["n_missed_examples"], miss.height), seed=SEED)
                          .select("s1_id", "match_id"))


# ---------------------------------------------------------------- report
def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}%"


def _report(results: dict, missed: pl.DataFrame | None, t_total: float, sample: float | None) -> str:
    L = ["# Blocking v1 report (generated by `python -m src.blocking`; do not hand-edit)", "",
         "Params: `" + json.dumps(P) + "`", ""]
    if sample:
        L += [f"**Train on a {sample:.0%} S1 sample** (queries only; index, keys and name_rev use the full data).",
              ""]
    for split, per in results.items():
        L += [f"## {split}", "",
              "| country | S1 | S2+S3 | candidates | cand/S1 mean | p50 | p90 | p99 | max | S1 with 0 cand | "
              "RR (within country) | PC | S1 all matches kept | S1 no match kept |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, s in per.items():
            d = s["cand_per_s1"]
            L.append(f"| {c} | {s['n_s1']:,} | {s['n_pool']:,} | {s['n_cand']:,} | {d['mean']} | {d['p50']} | "
                     f"{d['p90']} | {d['p99']} | {d['max']} | {d['zero']:,} | {s['rr_country']:.6f} | "
                     f"{_pct(s.get('pc'))} | {_pct(s.get('entities_all_found'))} | "
                     f"{_pct(s.get('entities_none_found'))} |")
        n1 = sum(s["n_s1"] for s in per.values())
        npool = sum(s["n_pool"] for s in per.values())
        nc = sum(s["n_cand"] for s in per.values())
        L += ["", f"Global RR vs all S1×(S2∪S3) pairs: **{1 - nc / max(n1 * npool, 1):.6f}** "
                  f"({nc:,} candidates, {nc / max(n1, 1):.1f} per S1). "
                  f"Country pre-filter alone: {n1 * npool / max(sum(s['n_s1'] * s['n_pool'] for s in per.values()), 1):.2f}x.", ""]
        L += ["Channel pair counts / gram coverage / runtime:", "",
              "| country | " + " | ".join(CH) + " | name q w/ grams | addr q w/ grams | name_rev q w/ grams "
              "| name s | addr s | keys s | peak RSS MB |",
              "|---|" + "---|" * (len(CH) + 7)]
        for c, s in per.items():
            L.append(f"| {c} | " + " | ".join(f"{s['pairs_by_channel'][ch]:,}" for ch in CH)
                     + f" | {s['name_queries_with_grams']:,} | {s['addr_queries_with_grams']:,} | "
                       f"{s['name_rev_queries_with_grams']:,}/{s['name_rev_queries']:,} | {s['name_s']} | "
                       f"{s['addr_s']} | {s['keys_s']} | {s['addr_peak_rss_mb']} |")
        L.append("")
        if any("pc" in s for s in per.values()):
            L += ["PC per channel (alone = channel caught it; only = no other channel did):", "",
                  "| country | " + " | ".join(f"{ch} alone / only" for ch in CH) + " |", "|---|" + "---|" * len(CH)]
            for c, s in per.items():
                L.append(f"| {c} | " + " | ".join(f"{_pct(s['pc_alone'][ch])} / {_pct(s['pc_only'][ch])}"
                                                  for ch in CH) + " |")
            ks = list(next(iter(per.values()))["pc_at_k"])
            L += ["", "PC and candidates if the name/addr top-k were cut to k (keys + name_rev kept):", "",
                  "| country | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
            for c, s in per.items():
                L.append(f"| {c} | " + " | ".join(f"{_pct(s['pc_at_k'][k])} ({s['cand_at_k'][k] / s['n_s1']:.1f}/S1)"
                                                  for k in ks) + " |")
            L += ["", "PC by match record type:", "", "| country | slice | PC | true pairs |", "|---|---|---|---|"]
            for c, s in per.items():
                for sl, (pc, n) in sorted(s["pc_by"].items()):
                    L.append(f"| {c} | {sl} | {_pct(pc)} | {n:,} |")
            L.append("")
        else:
            L += ["No labels for this split: PC is not measurable; RR and candidates/S1 only.", ""]
    if missed is not None and missed.height:
        L += ["## Missed true pairs (seeded sample, raw text; `core` = normalised core_name)", "",
              "| S1 name | S1 address | match name | match address | S1 core | match core |",
              "|---|---|---|---|---|---|"]
        esc = lambda v: str(v).replace("|", "\\|")  # noqa: E731
        for r in missed.iter_rows(named=True):
            L.append("| " + " | ".join(esc(r[k]) for k in ("n1", "a1", "n2", "a2", "c1", "c2")) + " |")
        L.append("")
    L.append(f"Total runtime {t_total:.0f}s, peak RSS {peak_rss_mb()} MB.")
    return "\n".join(L) + "\n"


def _missed_rows(missed: list) -> pl.DataFrame | None:
    if not missed:
        return None
    m = pl.concat(missed)
    cols = ["entity_id", "business_name", "business_address", "core_name"]
    raw = pl.concat([pl.scan_parquet(path("interim_dir") / f"norm_train_s{n}.parquet").select(cols)
                     .filter(pl.col("entity_id").is_in(m["s1_id"].to_list() + m["match_id"].to_list()))
                     .collect() for n in (1, 2, 3)])
    r1 = raw.rename({"entity_id": "s1_id", "business_name": "n1", "business_address": "a1", "core_name": "c1"})
    r2 = raw.rename({"entity_id": "match_id", "business_name": "n2", "business_address": "a2", "core_name": "c2"})
    return m.join(r1, on="s1_id").join(r2, on="match_id").sort("s1_id", "match_id")


# ---------------------------------------------------------------- driver
def run_split(split: str, s1_frac: float | None, missed: list) -> dict:
    gt = load_gt_pairs() if split == "train" else None
    countries = sorted(pl.scan_parquet(path("interim_dir") / f"norm_{split}_s1.parquet")
                       .select(pl.col("country").unique()).collect()["country"].to_list())
    per, cands = {}, []
    for country in countries:
        t = time.perf_counter()
        s1 = _load(split, 1, country).with_row_index("i")
        pool = _pool(split, country).with_row_index("i")
        rng = np.random.default_rng(SEED)
        q_mask = rng.random(s1.height) < s1_frac if s1_frac else np.ones(s1.height, bool)
        stats = {}
        cand = block_country(s1, pool, q_mask, stats)
        evaluate(cand, s1, pool, q_mask, gt, stats, missed)
        stats["total_s"] = round(time.perf_counter() - t, 1)
        print(split, country, json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}), flush=True)
        per[country] = stats
        if not s1_frac:
            cands.append(cand.select(s1_id=s1["entity_id"].gather(cand["s1"]),
                                     match_id=pool["entity_id"].gather(cand["p"]),
                                     bits="bits", name_rank="name_rank", name_score="name_score",
                                     addr_rank="addr_rank", addr_score="addr_score"))
        del s1, pool, cand
    if not s1_frac:
        allc = pl.concat(cands)
        allc.write_parquet(path(f"cand_{split}"))
        if split == "test":
            write_candidates(allc)
    return per


def write_candidates(cand: pl.DataFrame) -> None:
    """output/candidate_pairs.tsv: one row per test S1 in file order, sorted unique S2/S3 ids, "" when none."""
    s1 = pl.read_parquet(path("interim_dir") / "norm_test_s1.parquet", columns=["entity_id"]).with_row_index("o")
    lists = cand.group_by("s1_id").agg(pl.col("match_id").unique().sort().str.join(","))
    out = (s1.join(lists, left_on="entity_id", right_on="s1_id", how="left").sort("o")
           .select(source1_entity_id="entity_id", candidate_entity_ids=pl.col("match_id").fill_null("")))
    assert out.height == s1.height and out["source1_entity_id"].is_unique().all()
    p = path("candidate_pairs")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(p, separator="\t", quote_style="never")


def selftest() -> None:
    """_knn on random sparse data must equal brute-force top-k of the same (pruned) vectors."""
    rng = np.random.default_rng(SEED)
    nq, ni, nf, k = 300, 2000, 500, 7
    def rand(n, per):
        r = np.repeat(np.arange(n), per)
        return r, rng.integers(0, nf, r.size).astype(np.int32), rng.random(r.size).astype(np.float32)
    q, idx = rand(nq, 6), rand(ni, 10)
    Qd = sp.csr_matrix((q[2], (q[0], q[1])), shape=(nq, nf)).toarray()
    Id = sp.csr_matrix((idx[2], (idx[0], idx[1])), shape=(ni, nf)).toarray()
    S = Qd @ Id.T
    res, _ = _knn(q, idx, ni, nf, k)
    for r in range(nq):
        got = res.filter(pl.col("q") == r).sort("rank")
        pos = np.flatnonzero(S[r] > 0)
        want = np.sort(S[r, pos])[::-1][:k]
        assert np.allclose(got["score"].to_numpy(), want, rtol=1e-5), r
        assert np.allclose(S[r, got["i"].to_numpy()], got["score"].to_numpy(), rtol=1e-5), r
    print(f"selftest ok: {res.height} pairs, {nq} queries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--s1-frac", type=float, default=None, help="train only, on this share of S1 as queries")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    t0 = time.perf_counter()
    missed = []
    splits = ("train",) if args.s1_frac else ("train", "test")
    results = {s: run_split(s, args.s1_frac, missed) for s in splits}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report(results, _missed_rows(missed), time.perf_counter() - t0, args.s1_frac),
                      encoding="utf-8")
    print(f"wrote {REPORT} in {time.perf_counter() - t0:.0f}s, peak RSS {peak_rss_mb()} MB")


if __name__ == "__main__":
    main()
