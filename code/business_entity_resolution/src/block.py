"""M3b blocking v1: within-country inverted-index retrieval over IDF-weighted, df-capped tokens.

Channels. For each, score(s1, rec) = sum of idf(t) over shared surviving tokens, idf = ln(N_country / df):
  A  core-name tokens (+ adjacent bigrams "a_b"); name_tokens come from core_name, so legal form is unused
  B  address tokens + one exact "#<number> <street_core>" key token where both parts are present
  C  phonetic skeletons of name tokens ("n:", + skeleton bigrams) and address tokens ("a:"), see src.phonetic
  X  combined: A tokens as "A|t", B as "B|t", C as-is, plus (composite) "K|<name skel>|<addr skel>" pair tokens.
     One ranking over all evidence; the K pairs carry the name-AND-address conjunction that single tokens lose
     to the df cap (docs/blocking.md sampled miss causes: 50% all-over-cap, 50% rank cut)
df counts records over S1+S2+S3 of that country in this split (test IDF comes from test). A token is
linkable if it occurs in S1 and in S2/S3. Surviving tokens of a record = its linkable tokens with
df <= df_cap; in A and C (fallback_rarest) a record with none keeps its 2 lowest-df linkable tokens
(ties by token). Bigrams/fallback are config flags read here only, so train and test always match.
Directions per channel and source: record-centric top-m S1 per S2/S3 record, and S1-centric top-k records
per S1 (k per source). A pair is a candidate if any channel keeps it in either direction.
Scores are sparse products (records x vocab) @ (vocab x S1), chunked by an exact upper bound on product
nnz (row cost = sum of the actual posting lengths). No S1 x S2 matrix is ever held. Work runs per
country x per source, and one split per process.

Train keeps ranks up to max(sweep_m) / max(sweep_k) in blockgrid_train_cap{cap}.parquet. candidates_{split}
is that grid cut to config m/k (identical to a direct m/k run: ranks are intrinsic to each direction).

Run from code/business_entity_resolution/:
  python -m src.block --selftest                     # synthetic: sparse pipeline == pure-Python brute force
  python -m src.block --split train --dry            # df + survivor pass: zero-token %, product nnz bound
  python -m src.block --split train --smoke 300000   # first N rows per file; *_smoke outputs; prints nnz/s
  python -m src.block --split train                  # candidates_train + blockgrid_train_cap1000 + idf_train
  python -m src.block --split test                   # candidates_test + output/candidate_pairs.tsv
"""
import argparse
import json
import shutil
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from .io import CFG, path, write_candidates
from .normalise import peak_rss_mb
from .phonetic import add_skeletons

BCFG = CFG["blocking"]
CHANNELS = ("A", "B", "C", "X")
FALLBACK_CHANNELS = ("A", "C", "X")
FALLBACK_N = 2
SOURCES = (2, 3)
LOAD_COLS = ["entity_id", "country", "name_tokens", "addr_tokens", "addr_number", "addr_street_core"]
RANK_COLS = [f"{ch}_{x}" for ch in CHANNELS for x in ("score", "rrank", "srank")]
_TOP_SCHEMA = {"r": pl.Int32, "c": pl.Int32, "s": pl.Float32, "rank": pl.Int16}


def norm_path(split: str, n: int):
    return path("interim_dir") / f"norm_{split}_s{n}.parquet"


def _bigrams(s: pl.Series) -> pl.Series:
    """Adjacent-token bigrams per row; [] for rows with < 2 tokens. With `sorted_bigrams` the pair is
    written in sorted order (min_max), so an adjacent word swap yields the same key."""
    n = (pl.col("t").list.len().cast(pl.Int64) - 1).clip(lower_bound=0)
    x = (s.rename("t").to_frame().with_row_index("_r")
         .select("_r", a=pl.col("t").list.slice(0, n), b=pl.col("t").list.slice(1)).explode("a", "b", empty_as_null=True))  # keeps 0-bigram rows
    a, b = pl.col("a"), pl.col("b")
    if BCFG["sorted_bigrams"]:
        a, b = pl.when(a <= b).then(a).otherwise(b), pl.when(a <= b).then(b).otherwise(a)
    return x.group_by("_r", maintain_order=True).agg(bg=pl.concat_str(a, pl.lit("_"), b).drop_nulls())["bg"]


def _cross(n: pl.Series, a: pl.Series) -> pl.Series:
    """Per row: "K|<n>|<a>" for every (name skeleton, address skeleton) pair; [] when either side is empty.
    One name word or one address word is common (the data reuses both vocabularies independently);
    the pair is rare, so it survives the df cap and ranks the true S1 first."""
    x = (pl.DataFrame({"n": n.list.unique(), "a": a.list.unique()}).with_row_index("_r")
         .explode("n", empty_as_null=True).explode("a", empty_as_null=True).drop_nulls())
    agg = x.group_by("_r").agg(k=pl.concat_str(pl.lit("K|"), "n", pl.lit("|"), "a"))
    base = pl.DataFrame({"_r": pl.int_range(n.len(), dtype=pl.UInt32, eager=True)})
    return (base.join(agg, on="_r", how="left", maintain_order="left")
            .select(pl.col("k").fill_null(pl.lit([], dtype=pl.List(pl.String))))["k"])


def channel_tokens(df: pl.DataFrame, extra: tuple[str, ...] = ()) -> pl.DataFrame:
    """entity_id (+ extra) and the three channel token lists, unique per record."""
    key = (pl.when((pl.col("addr_number") != "") & (pl.col("addr_street_core") != ""))
           .then(pl.concat_str(pl.lit("#"), "addr_number", pl.lit(" "), "addr_street_core")))
    df = add_skeletons(df)
    name, skel = ["name_tokens"], ["name_skel_tokens"]
    if BCFG["bigrams"]:
        df = df.with_columns(_bg=_bigrams(df["name_tokens"]), _sbg=_bigrams(df["name_skel_tokens"]))
        name, skel = name + ["_bg"], skel + ["_sbg"]
    if BCFG["joined_name"]:  # "blair hawaii" -> "blairhawaii" matches a domain-style glued name
        df = df.with_columns(_jn=pl.concat_list(pl.when(pl.col("name_tokens").list.len() >= 2)
                                                .then(pl.col("name_tokens").list.join(""))).list.drop_nulls())
        name = name + ["_jn"]
    k = _cross(df["name_skel_tokens"], df["addr_skel_tokens"]) if BCFG["composite"] else None
    return df.select(
        "entity_id", *extra,
        A=pl.concat_list(name).list.unique(),
        B=pl.concat_list("addr_tokens", key).list.drop_nulls().list.unique(),
        C=pl.concat_list(pl.concat_list(skel).list.eval(pl.lit("n:") + pl.element()),
                         pl.col("addr_skel_tokens").list.eval(pl.lit("a:") + pl.element())).list.unique()
    ).with_columns(X=pl.concat_list(pl.col("A").list.eval(pl.lit("A|") + pl.element()),
                                    pl.col("B").list.eval(pl.lit("B|") + pl.element()), "C",
                                    *([k.alias("_k")] if k is not None else [])).list.unique())


def load(split: str, n: int, country: str, smoke: int = 0, extra: tuple[str, ...] = (),
         ids: pl.Series | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(norm_path(split, n))
    if smoke:
        lf = lf.head(smoke)
    lf = lf.filter(pl.col("country") == country)
    if ids is not None:
        lf = lf.filter(pl.col("entity_id").is_in(ids))
    return channel_tokens(lf.select(*dict.fromkeys(LOAD_COLS + list(extra))).collect(), extra)


def _count(df: pl.DataFrame, n: int) -> dict[str, pl.DataFrame]:
    return {ch: df[ch].explode(empty_as_null=True).drop_nulls().value_counts(name="c").rename({ch: "tok"})
            .with_columns(s=pl.lit(n, pl.Int8)) for ch in CHANNELS}


def _combine(parts: list[dict[str, pl.DataFrame]]) -> dict[str, pl.DataFrame]:
    """Per channel: tok, n1, n2, n3 = records per source containing tok (df >= 2 only), sorted by tok."""
    return {ch: pl.concat([p[ch] for p in parts]).group_by("tok")
            .agg(*(pl.col("c").filter(pl.col("s") == n).sum().cast(pl.Int64).alias(f"n{n}") for n in (1, *SOURCES)))
            .filter(pl.sum_horizontal("n1", "n2", "n3") >= 2).sort("tok") for ch in CHANNELS}


def doc_freq(split: str, country: str, smoke: int) -> tuple[dict[str, pl.DataFrame], int]:
    parts, n_total = [], 0
    for n in (1, *SOURCES):
        df = load(split, n, country, smoke)
        n_total += df.height
        parts.append(_count(df, n))
        del df
    return _combine(parts), n_total


def token_table(t: pl.DataFrame, n_total: int, cap: int) -> pl.DataFrame:
    """Linkable tokens (in S1 and in S2/S3) with tid (sorted by tok), df, idf, rare = df <= cap."""
    return (t.filter((pl.col("n1") > 0) & (pl.col("n2") + pl.col("n3") > 0))
            .with_columns(df=pl.sum_horizontal("n1", "n2", "n3"))
            .with_columns(tid=pl.int_range(pl.len(), dtype=pl.Int32),
                          idf=(pl.lit(n_total, pl.Float64) / pl.col("df")).log().cast(pl.Float32),
                          rare=pl.col("df") <= cap))


def survivors(tokens: pl.Series, tab: pl.DataFrame, fallback: bool) -> tuple[pl.DataFrame, dict]:
    """(r, tid, idf, rare) of the tokens each record keeps + % records with zero survivors (w/o, w/ fallback)."""
    x = (tokens.rename("tok").to_frame().with_row_index("r").explode("tok", empty_as_null=True)
         .join(tab.select("tok", "tid", "idf", "df", "rare"), on="tok", how="inner"))
    keep = pl.col("rare")
    if fallback:
        x = x.sort("r", "df", "tok")
        keep = keep | (~pl.col("rare").any().over("r") & (pl.int_range(pl.len()).over("r") < FALLBACK_N))
    kept = x.filter(keep).select("r", "tid", "idf", "rare")
    n = max(tokens.len(), 1)
    stats = {"rows": tokens.len(),
             "zero_no_fallback_pct": round(100 * (1 - x.filter("rare")["r"].n_unique() / n), 2),
             "zero_with_fallback_pct": round(100 * (1 - x["r"].n_unique() / n), 2) if fallback else None}
    return kept, stats


def matrix(kept: pl.DataFrame, n_rows: int, n_vocab: int, weighted: bool) -> sp.csr_matrix:
    """records x vocab; value = idf (weighted) or 1."""
    data = kept["idf"].to_numpy() if weighted else np.ones(kept.height, np.float32)
    return sp.csr_matrix((data, (kept["r"].to_numpy(), kept["tid"].to_numpy())),
                         shape=(n_rows, n_vocab), dtype=np.float32)


def postings(kept: pl.DataFrame, n_vocab: int, rare_only: bool = False) -> np.ndarray:
    tid = (kept.filter("rare") if rare_only else kept)["tid"].to_numpy()
    return np.bincount(tid, minlength=n_vocab).astype(np.float64)


def _chunks(cost: np.ndarray, budget: float):
    cum, lo = np.cumsum(cost), 0
    while lo < len(cost):
        base = cum[lo - 1] if lo else 0.0
        hi = max(lo + 1, int(np.searchsorted(cum, base + budget, side="right")))
        yield lo, hi
        lo = hi


def top_n(P: sp.csr_matrix, n: int, row0: int) -> pl.DataFrame:
    """Per row of P: the n best columns (score desc, ties -> lower column index), rank 1-based."""
    counts = np.diff(P.indptr)
    df = (pl.DataFrame({"r": np.repeat(np.arange(P.shape[0], dtype=np.int32), counts),
                        "c": P.indices.astype(np.int32), "s": P.data.astype(np.float32)})
          .sort(["r", "s", "c"], descending=[False, True, False]))
    # rows stay grouped in CSR layout after the sort, so position - row start = rank - 1
    rank = np.arange(P.nnz, dtype=np.int64) - np.repeat(P.indptr[:-1].astype(np.int64), counts)
    keep = rank < n
    return (df.filter(pl.Series(keep)).with_columns(pl.col("r") + row0,
                                                    rank=pl.Series((rank[keep] + 1).astype(np.int16))))


def retrieve(left: sp.csr_matrix, right_t: sp.csr_matrix, post_len: np.ndarray, n: int,
             budget: float) -> tuple[pl.DataFrame, int]:
    """Top-n right rows per left row of left @ right_t, in chunks of <= budget upper-bound nnz."""
    binary = sp.csr_matrix((np.ones_like(left.data), left.indices, left.indptr), shape=left.shape)
    cost = binary @ post_len
    out, nnz = [], 0
    for lo, hi in _chunks(cost, budget):
        P = left[lo:hi] @ right_t
        nnz += P.nnz
        out.append(top_n(P, n, lo))
    return (pl.concat(out) if out else pl.DataFrame(schema=_TOP_SCHEMA)), nnz


def _slice(f: pl.DataFrame, lo: int, hi: int) -> pl.DataFrame:
    a, b = np.searchsorted(f["s1"].to_numpy(), [lo, hi])
    return f.slice(int(a), int(b - a))


def merge(rc: dict[str, pl.DataFrame], sc: dict[str, pl.DataFrame], n_s1: int, step: int):
    """Wide per-pair rows (s1, rec, <ch>_score/_rrank/_srank) in S1 slices; rc/sc are long per channel."""
    frames = ([(i, rc[ch].sort("s1", "rec"), "rrank") for i, ch in enumerate(CHANNELS)]
              + [(i, sc[ch], "srank") for i, ch in enumerate(CHANNELS)])  # sc is already sorted by s1
    aggs = []
    for i, ch in enumerate(CHANNELS):
        f = pl.col("ch") == i
        aggs += [pl.col("s").filter(f).max().alias(f"{ch}_score"), pl.col("rrank").filter(f).min().alias(f"{ch}_rrank"),
                 pl.col("srank").filter(f).min().alias(f"{ch}_srank")]
    for lo in range(0, n_s1, step):
        long = pl.concat([_slice(f, lo, lo + step).select("s1", "rec", "s", pl.col("rank").alias(col),
                                                          ch=pl.lit(i, pl.Int8))
                          for i, f, col in frames], how="diagonal")
        if long.height:
            yield long.group_by("s1", "rec").agg(aggs).sort("s1", "rec")


def finalize(lf: pl.LazyFrame, m: int, k: int, select: tuple[str, ...] = CHANNELS) -> pl.LazyFrame:
    """Cut a grid to (m, k): ranks/scores outside it become null. A pair is kept iff a channel in `select`
    keeps it; the other channels' ranks/scores stay as features (null where they didn't hit)."""
    cols, hits = [], {}
    for ch in CHANNELS:
        r, s = pl.col(f"{ch}_rrank"), pl.col(f"{ch}_srank")
        hits[ch] = (r <= m).fill_null(False) | (s <= k).fill_null(False)
        cols += [pl.when(hits[ch]).then(pl.col(f"{ch}_score")).alias(f"{ch}_score"),
                 pl.when(r <= m).then(r).alias(f"{ch}_rrank"), pl.when(s <= k).then(s).alias(f"{ch}_srank")]
    return (lf.filter(pl.any_horizontal(hits[ch] for ch in select))
            .with_columns(*cols, n_channels_hit=pl.sum_horizontal(list(hits.values())).cast(pl.Int8)))


def _fallback(ch: str) -> bool:
    return BCFG["fallback_rarest"] and ch in FALLBACK_CHANNELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"))
    ap.add_argument("--df-cap", type=int, default=BCFG["df_cap"])
    ap.add_argument("--smoke", type=int, default=0, help="first N rows of each file; outputs get _smoke")
    ap.add_argument("--dry", action="store_true", help="df + survivor pass: zero-token %%, product nnz bound")
    ap.add_argument("--selftest", action="store_true", help="synthetic brute-force equivalence check")
    ap.add_argument("--refinalize", action="store_true",
                    help="train only: re-cut the existing blockgrid to config m/k/select -> candidates_train (seconds)")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    assert a.split, "--split is required"
    if a.refinalize:
        assert a.split == "train", "--refinalize needs the train grid"
        out = path("interim_dir")
        finalize(pl.scan_parquet(out / f"blockgrid_train_cap{a.df_cap}.parquet"), BCFG["m"], BCFG["k"],
                 tuple(BCFG["select"])).sink_parquet(out / "candidates_train.parquet")
        n = pl.scan_parquet(out / "candidates_train.parquet").select(pl.len()).collect().item()
        print(f"candidates_train: {n:,} pairs (m={BCFG['m']}, k={BCFG['k']}, select={BCFG['select']})")
        return
    split, cap, budget = a.split, a.df_cap, float(BCFG["nnz_budget"])
    train, main_cap = split == "train", a.df_cap == BCFG["df_cap"]
    m_max = max(BCFG["sweep_m"]) if train else BCFG["m"]
    k_max = max(BCFG["sweep_k"]) if train else BCFG["k"]
    sfx = "_smoke" if a.smoke else ""
    out = path("interim_dir")
    parts = out / f"_parts_{split}_cap{cap}{sfx}"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True)

    steps, surv, clock = [], [], [time.perf_counter()]

    def log(step: str, **kw) -> None:
        now = time.perf_counter()
        row = {"step": step, "s": round(now - clock[0], 1), "peak_rss_mb": peak_rss_mb(), **kw}
        clock[0] = now
        steps.append(row)
        print(row, flush=True)

    t_start = time.perf_counter()
    countries = sorted(pl.scan_parquet(norm_path(split, 1)).select(pl.col("country").unique()).collect()["country"])
    idf_parts = []
    for c in countries:
        dft, n_total = doc_freq(split, c, a.smoke)
        idf_parts += [t.with_columns(country=pl.lit(c), ch=pl.lit(ch), n_country=pl.lit(n_total))
                      for ch, t in dft.items()]
        tab = {ch: token_table(t, n_total, cap) for ch, t in dft.items()}
        nv = {ch: t.height for ch, t in tab.items()}
        del dft
        log(f"{c} df pass", n_country=n_total, linkable_vocab=nv)
        s1 = load(split, 1, c, a.smoke)
        k1, post1, post1_rare = {}, {}, {}
        for ch in CHANNELS:
            k1[ch], st = survivors(s1[ch], tab[ch], _fallback(ch))
            post1[ch], post1_rare[ch] = postings(k1[ch], nv[ch]), postings(k1[ch], nv[ch], rare_only=True)
            surv.append({"country": c, "source": "S1", "channel": ch, **st})
        if not a.dry:
            qw = {ch: matrix(k1[ch], s1.height, nv[ch], weighted=True) for ch in CHANNELS}
            qw_t = {ch: q.T.tocsr() for ch, q in qw.items()}
        del k1
        log(f"{c} S1 survivors/index", n_s1=s1.height)
        for n in SOURCES:
            rec = load(split, n, c, a.smoke)
            rc, sc = {}, {}
            for ch in CHANNELS:
                kr, st = survivors(rec[ch], tab[ch], _fallback(ch))
                post_r = postings(kr, nv[ch])
                cost = post1[ch] * post_r
                top = np.argsort(cost)[::-1][:3]
                surv.append({"country": c, "source": f"S{n}", "channel": ch, **st,
                             "product_nnz_ub": int(cost.sum()),
                             "product_nnz_ub_rare_only": int(post1_rare[ch] @ postings(kr, nv[ch], rare_only=True)),
                             "top_cost_tokens": {tab[ch]["tok"][int(i)]: int(cost[i]) for i in top if cost[i] > 0}})
                print(surv[-1], flush=True)
                if a.dry:
                    continue
                r = matrix(kr, rec.height, nv[ch], weighted=False)
                del kr
                rc[ch], nnz = retrieve(r, qw_t[ch], post1[ch], m_max, budget)
                rc[ch] = rc[ch].rename({"r": "rec", "c": "s1"})
                log(f"{c} S{n} {ch} record-centric", product_nnz=nnz)
                sc[ch], nnz = retrieve(qw[ch], r.T.tocsr(), post_r, k_max, budget)
                sc[ch] = sc[ch].rename({"r": "s1", "c": "rec"})
                log(f"{c} S{n} {ch} S1-centric", product_nnz=nnz)
            if a.dry:
                log(f"{c} S{n} survivors", n_rec=rec.height)
                continue
            n_pairs = 0
            for i, w in enumerate(merge(rc, sc, s1.height, BCFG["s1_range"])):
                n_pairs += w.height
                (w.select(s1_id=s1["entity_id"].gather(w["s1"]), rec_id=rec["entity_id"].gather(w["rec"]), *RANK_COLS)
                 .write_parquet(parts / f"{c}_s{n}_{i:05d}.parquet"))
            n_rec = rec.height
            del rc, sc, rec
            log(f"{c} S{n} merge", n_rec=n_rec, grid_pairs=n_pairs)
        del s1
        if not a.dry:
            del qw, qw_t
    pl.concat(idf_parts).write_parquet(out / f"idf_{split}{sfx}.parquet")
    if a.dry:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60, tbl_width_chars=250):
            print(pl.DataFrame(surv).drop("top_cost_tokens"))
    else:
        lf = pl.scan_parquet(parts / "*.parquet")
        if train:
            lf.sink_parquet(out / f"blockgrid_{split}_cap{cap}{sfx}.parquet")
        if main_cap:
            cands = out / f"candidates_{split}{sfx}.parquet"
            finalize(lf, BCFG["m"], BCFG["k"], tuple(BCFG["select"])).sink_parquet(cands)
            if split == "test" and not a.smoke:
                s1_ids = pl.read_parquet(norm_path(split, 1), columns=["entity_id"])["entity_id"]
                write_candidates(path("candidate_pairs"), pl.scan_parquet(cands).select("s1_id", "rec_id"), s1_ids)
        log("sink outputs")
    shutil.rmtree(parts)
    total = {"split": split, "df_cap": cap, "m_max": m_max, "k_max": k_max, "bigrams": BCFG["bigrams"],
             "fallback_rarest": BCFG["fallback_rarest"], "smoke": a.smoke, "dry": a.dry,
             "total_s": round(time.perf_counter() - t_start, 1), "peak_rss_mb": peak_rss_mb(),
             "steps": steps, "survivors": surv}
    tag = "_dry" if a.dry else sfx
    (out / f"block_timing_{split}_cap{cap}{tag}.json").write_text(json.dumps(total, indent=1))
    print({k: v for k, v in total.items() if k not in ("steps", "survivors")})


def _selftest(cap: int = 8, m: int = 3, k: int = 4) -> None:
    """Synthetic sources through the real pipeline (channel_tokens -> df -> survivors -> chunked products)
    vs an independent pure-Python brute force (own bigrams, skeletons, df, fallback, float32 sums)."""
    from collections import Counter

    from .phonetic import skeleton
    rng = np.random.default_rng(0)
    words = ["ganesh", "traders", "sri", "shivam", "praivet", "private", "eastern", "systems", "sistms", "kumar",
             "summit", "delta", "peak", "rd", "st", "nagar", "keyr", "care"]
    p = np.linspace(4, 1, len(words))
    p /= p.sum()

    def frame(n: int, pre: str) -> pl.DataFrame:
        names = [[str(w) for w in rng.choice(words, rng.integers(1, 4), p=p)] for _ in range(n)]
        addrs = [[str(w) for w in rng.choice(words, rng.integers(0, 3), p=p)] + [str(rng.integers(1, 6))] for _ in range(n)]
        return pl.DataFrame({"entity_id": [f"{pre}{i}" for i in range(n)], "name_tokens": names, "addr_tokens": addrs,
                             "addr_number": [x[-1] for x in addrs], "addr_street_core": [" ".join(x[:-1]) for x in addrs]})
    raw = {1: frame(60, "a"), 2: frame(90, "b"), 3: frame(70, "c")}
    toks = {n: channel_tokens(f) for n, f in raw.items()}
    tables = _combine([_count(f, n) for n, f in toks.items()])
    n_total = sum(f.height for f in raw.values())

    def bf_sets(f: pl.DataFrame) -> dict[str, list[set]]:
        out = {ch: [] for ch in CHANNELS}
        for nm, ad, num, st in f.select("name_tokens", "addr_tokens", "addr_number", "addr_street_core").iter_rows():
            def bgs(ts):
                if not BCFG["bigrams"]:
                    return []
                return [("_".join(sorted((x, y))) if BCFG["sorted_bigrams"] else f"{x}_{y}") for x, y in zip(ts, ts[1:])]
            bg = bgs(nm)
            sk = [skeleton(t) for t in nm]
            sbg = bgs(sk)
            jn = {"".join(nm)} if BCFG["joined_name"] and len(nm) >= 2 else set()
            out["A"].append(set(nm) | set(bg) | jn)
            out["B"].append(set(ad) | ({f"#{num} {st}"} if num and st else set()))
            out["C"].append({"n:" + t for t in sk + sbg} | {"a:" + skeleton(t) for t in ad})
            ks = ({f"K|{x}|{skeleton(y)}" for x in sk for y in ad} if BCFG["composite"] else set())
            out["X"].append({"A|" + t for t in out["A"][-1]} | {"B|" + t for t in out["B"][-1]} | out["C"][-1] | ks)
        return out
    bf = {n: bf_sets(f) for n, f in raw.items()}
    used = {"fallback": 0, "bigram": 0}
    for ch in CHANNELS:
        per_src = {n: Counter(t for s in bf[n][ch] for t in s) for n in raw}
        df = {t: sum(c[t] for c in per_src.values()) for t in set().union(*per_src.values())}
        link = {t for t in df if per_src[1][t] and per_src[2][t] + per_src[3][t]}
        idf = {t: np.float32(np.log(n_total / df[t])) for t in link}

        def keep(s: set) -> list:
            rare = sorted(t for t in s & link if df[t] <= cap)
            if rare or not _fallback(ch):
                return rare
            used["fallback"] += bool(s & link)
            return sorted(sorted(s & link, key=lambda t: (df[t], t))[:FALLBACK_N])
        s1s, s2s = [keep(s) for s in bf[1][ch]], [keep(s) for s in bf[2][ch]]
        used["bigram"] += sum("_" in t for s in s1s + s2s for t in s)

        def score(x: list, y: list) -> np.float32:
            acc = np.float32(0)
            for t in sorted(set(x) & set(y)):  # tid order = token order; scipy accumulates in float32 the same way
                acc = np.float32(acc + idf[t])
            return acc

        def bf_top(rows: list, cols: list, n: int) -> set:
            out = set()
            for i, x in enumerate(rows):
                sc = [(score(x, y), j) for j, y in enumerate(cols) if set(x) & set(y)]
                out |= {(i, j, r + 1, float(s)) for r, (s, j) in enumerate(sorted(sc, key=lambda v: (-v[0], v[1]))[:n])}
            return out

        tab = token_table(tables[ch], n_total, cap)
        assert set(tab["tok"]) == link, ch
        k1, _ = survivors(toks[1][ch], tab, _fallback(ch))
        k2, _ = survivors(toks[2][ch], tab, _fallback(ch))
        q, r = matrix(k1, 60, tab.height, True), matrix(k2, 90, tab.height, False)
        for budget in (1, 25, 1e9):
            rc, _ = retrieve(r, q.T.tocsr(), postings(k1, tab.height), m, budget)
            sc, _ = retrieve(q, r.T.tocsr(), postings(k2, tab.height), k, budget)
            for got, want in ((rc, bf_top(s2s, s1s, m)), (sc, bf_top(s1s, s2s, k))):
                got = {(int(a), int(b), int(c), float(d)) for a, b, d, c in got.select("r", "c", "s", "rank").iter_rows()}
                assert got == want, (ch, budget, len(got), len(want), sorted(got ^ want)[:5])
    assert used["fallback"] > 0 and (used["bigram"] > 0 or not BCFG["bigrams"]), used
    print(f"block self-test OK (cap={cap}, m={m}, k={k}, 3 budgets, channels {CHANNELS}; "
          f"fallback records {used['fallback']}, surviving bigram tokens {used['bigram']})")


if __name__ == "__main__":
    main()
