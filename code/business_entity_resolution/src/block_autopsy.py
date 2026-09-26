"""M5 blocking autopsy: why v1 misses true train pairs, and what cheap unions recover (no GBM).

Sample = train S1s with hash(s1_id, seed) % 10 == 0. v1 = candidates_train_v1.parquet.
Ranks come from blockgrid_train: rrank <= max(blocking.sweep_m), srank <= max(blocking.sweep_k); null = beyond
the grid or never scored. addr_tset = rapidfuzz token_set_ratio over the joined addr_tokens. Name key = joined
name_tokens (core name), exact equality within country; keys held by > NAME_CAP train S1s are skipped.
Unions: (a) v1, (b) v1 | any-channel rrank <= m, (c) (b) | name key, (d) (c) | (B rrank <= ADDR_M and addr_tset >= ADDR_T).

Run from code/business_entity_resolution/:
  python -m src.block_autopsy        # docs/block_autopsy.md
"""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .block import BCFG, CHANNELS, RANK_COLS, norm_path
from .decide import md
from .gate import PAIR, V1, ceiling, grid
from .io import CFG, ROOT, StepLog, load_gt_pairs, path, peak_rss_mb

FRAC = 10
UNION_M = (1, 2, 3, 5)
NAME_CAP = 20
ADDR_M, ADDR_T = 3, 90
M_MAX, K_MAX = max(BCFG["sweep_m"]), max(BCFG["sweep_k"])
V1_PATH = path("interim_dir") / "candidates_train_v1.parquet"
REPORT = ROOT / "docs" / "block_autopsy.md"
RMIN = pl.min_horizontal(*(pl.col(f"{ch}_rrank") for ch in CHANNELS))
KEY, ADDR = pl.col("name_tokens").list.join(" "), pl.col("addr_tokens").list.join(" ")
ANY_CH = pl.any_horizontal([(pl.col(f"{ch}_rrank") <= BCFG["m"]).fill_null(False)
                            | (pl.col(f"{ch}_srank") <= BCFG["k"]).fill_null(False) for ch in CHANNELS])


def in_sample() -> pl.Expr:
    return pl.col("s1_id").hash(CFG["seed"]) % FRAC == 0


def records() -> pl.LazyFrame:
    return pl.concat([pl.scan_parquet(norm_path("train", n)) for n in (2, 3)]).select(
        rec_id="entity_id", country="country", rec_key=KEY, rec_addr=ADDR)


def rec_attrs(pairs: pl.DataFrame) -> pl.DataFrame:
    """rec_key / rec_addr for the records in `pairs` only."""
    return records().join(pairs.select("rec_id").unique().lazy(), on="rec_id", how="semi").select(
        "rec_id", "rec_key", "rec_addr").collect(engine="streaming")


def with_pair_attrs(pairs: pl.DataFrame, S: pl.DataFrame) -> pl.DataFrame:
    """+ name_eq (non-empty core names equal) and addr_tset, row order kept."""
    x = (pairs.join(S.select("s1_id", "key", "s1_addr"), on="s1_id", how="left", maintain_order="left")
         .join(rec_attrs(pairs), on="rec_id", how="left", maintain_order="left")
         .with_columns(pl.col("s1_addr", "rec_addr", "key", "rec_key").fill_null("")))
    tset = cpdist(x["s1_addr"].to_list(), x["rec_addr"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    return x.with_columns(name_eq=(pl.col("key") == pl.col("rec_key")) & (pl.col("key") != ""),
                          addr_tset=pl.Series(tset, dtype=pl.Float32))


def share(df: pl.DataFrame, e: pl.Expr) -> float:
    return round(float(df.select(e.fill_null(False).mean()).item()) * 100, 2) if df.height else 0.0


def main() -> None:
    log = StepLog()
    s1 = pl.scan_parquet(norm_path("train", 1)).select(s1_id="entity_id", country="country", key=KEY, s1_addr=ADDR)
    S = s1.filter(in_sample()).collect(engine="streaming")
    gt = load_gt_pairs().select("s1_id", rec_id="match_id").filter(in_sample()).join(S.select("s1_id"), on="s1_id")
    ntrue = S.select("s1_id").join(gt.group_by("s1_id").len("ntrue"), on="s1_id", how="left").with_columns(
        pl.col("ntrue").fill_null(0))
    log("sample", s1=S.height, true_pairs=gt.height)

    # v1 in the sample + which rule actually produced it
    V = pl.scan_parquet(V1_PATH).filter(in_sample()).select(*PAIR, *RANK_COLS).collect(engine="streaming")
    rule = {"X only (gate.py V1)": share(V, V1), "any channel at blocking m/k": share(V, ANY_CH)}
    V = V.select(*PAIR, v1=pl.lit(True))
    G = (grid("train").filter(in_sample() & (RMIN <= max(UNION_M))).select(*PAIR, rmin=RMIN, B_rrank="B_rrank")
         .collect(engine="streaming"))
    okkey = (s1.filter(pl.col("key") != "").group_by("country", "key").len("n")
             .filter(pl.col("n") <= NAME_CAP).select("country", "key"))
    N = (S.lazy().join(okkey, on=["country", "key"], how="semi").select("s1_id", "country", rec_key="key")
         .join(records().select("rec_id", "country", "rec_key"), on=["country", "rec_key"])
         .select(*PAIR).unique().with_columns(nk=pl.lit(True)).collect(engine="streaming"))
    log("sets", v1=V.height, grid_rmin=G.height, name_key=N.height)

    U = (V.join(G, on=PAIR, how="full", coalesce=True).join(N, on=PAIR, how="full", coalesce=True)
         .with_columns(pl.col("v1", "nk").fill_null(False)))
    del V, G, N
    B = with_pair_attrs(U.filter(pl.col("B_rrank") <= ADDR_M).select(PAIR), S)
    U = U.join(B.filter(pl.col("addr_tset") >= ADDR_T).select(*PAIR, ad=pl.lit(True)), on=PAIR, how="left").with_columns(
        pl.col("ad").fill_null(False))
    del B
    log("union frame", rows=U.height)

    # true pairs: grid ranks + set flags
    T = (gt.lazy().join(grid("train").select(*PAIR, *RANK_COLS), on=PAIR, how="left").collect(engine="streaming")
         .join(U.select(*PAIR, "v1", "nk", "ad"), on=PAIR, how="left")
         .with_columns(pl.col("v1", "nk", "ad").fill_null(False), rmin=RMIN))

    def b(m): return pl.col("v1") | (pl.col("rmin") <= m).fill_null(False)
    def c(m): return b(m) | pl.col("nk")
    def d(m): return c(m) | pl.col("ad")
    sets = [("(a) v1", "-", pl.col("v1"))] + [("(b) v1 ∪ rec top-m", m, b(m)) for m in UNION_M] \
        + [("(c) (b) ∪ name key", m, c(m)) for m in UNION_M] + [(f"(d) (c) ∪ addr top-{ADDR_M} tset≥{ADDR_T}", m, d(m))
                                                               for m in UNION_M]
    rows, base = [], U.filter(pl.col("v1")).height
    for name, m, e in sets:
        hit = T.filter(e)
        n_c = U.filter(e.fill_null(False)).height
        rows.append({"set": name, "m": m, "PC %": share(T, e),
                     "F0.5 ceiling": round(ceiling(ntrue, hit.group_by("s1_id").len("h")), 4),
                     "cand/S1": round(n_c / S.height, 2), "+cand/S1 vs v1": round((n_c - base) / S.height, 2)})
    log("unions")

    # Q1: missed true pairs
    M = with_pair_attrs(T.filter(~pl.col("v1")), S)
    has_v1 = (pl.scan_parquet(V1_PATH).select("rec_id").join(M.select("rec_id").unique().lazy(), on="rec_id", how="semi")
              .unique().collect(engine="streaming").with_columns(rec_has_v1=pl.lit(True)))
    M = M.join(has_v1, on="rec_id", how="left").with_columns(zero_v1=pl.col("rec_has_v1").is_null())
    no_grid = pl.all_horizontal(pl.col(col).is_null() for col in RANK_COLS)
    facts = {"missed true pairs": M.height, "missed % of true": share(T, ~pl.col("v1")),
             "record has ZERO v1 candidates %": share(M, pl.col("zero_v1")),
             f"addr_tset ≥ {ADDR_T} %": share(M, pl.col("addr_tset") >= ADDR_T),
             "exact core-name equal %": share(M, pl.col("name_eq")),
             "name equal OR addr_tset ≥ 90 %": share(M, pl.col("name_eq") | (pl.col("addr_tset") >= ADDR_T)),
             "absent from grid (no channel, no direction) %": share(M, no_grid),
             **{f"any-channel record-side rank ≤ {m} %": share(M, pl.col("rmin") <= m) for m in (1, 2, 3, 5, M_MAX)}}
    rb = [(1, 1), (2, 2), (3, 3), (4, 5), (6, M_MAX)]
    sb = [(1, 10), (11, 30), (31, K_MAX)]
    ranks = []
    for ch in CHANNELS:
        r, s = pl.col(f"{ch}_rrank"), pl.col(f"{ch}_srank")
        ranks.append({"channel": ch,
                      **{f"rrank {lo}-{hi} %": share(M, r.is_between(lo, hi)) for lo, hi in rb},
                      f"rrank >{M_MAX}/none %": share(M, r.is_null()),
                      **{f"srank {lo}-{hi} %": share(M, s.is_between(lo, hi)) for lo, hi in sb},
                      f"srank >{K_MAX}/none %": share(M, s.is_null())})
    ex = (M.sort(pl.col("rec_id").hash(CFG["seed"])).head(15)
          .select("key", "rec_key", "s1_addr", "rec_addr", pl.col("addr_tset").round(0), "zero_v1",
                  *(f"{ch}_rrank" for ch in CHANNELS), *(f"{ch}_srank" for ch in CHANNELS)).to_dicts())
    log("missed", n=M.height)

    L = ["# Blocking autopsy (M5)",
         f"Generated by `src/block_autopsy.py`. Sample: {S.height:,} train S1s (hash(s1_id, seed {CFG['seed']}) % "
         f"{FRAC} == 0), {T.height:,} true pairs. Ranks from blockgrid_train (rrank ≤ {M_MAX}, srank ≤ {K_MAX}; "
         "null = beyond grid or never scored). addr_tset = token_set_ratio of joined addr_tokens; name key = joined "
         f"core-name tokens, exact, within country, keys with > {NAME_CAP} train S1s skipped. "
         "Ceiling = gate.ceiling (perfect matcher, singletons = 1) over sample S1s; cand/S1 over all sample S1s.", "",
         "## 1. True pairs missed by v1", "", *md([{"fact": k, "value": v} for k, v in facts.items()]), "",
         "Rank of the true pair per channel (% of missed pairs); rrank = record's rank of its true S1, "
         "srank = S1's rank of the record (per source).", "", *md(ranks), "",
         "Examples (15, hash order):", "", *md(ex), "",
         "## 2. v1 retrieval rule", "",
         f"- `gate.py:35` V1 = `X_rrank <= {BCFG['m']} OR X_srank <= {BCFG['k']}`: channel X only, "
         "record-centric top-m OR S1-centric top-k (k per source).",
         "- `block.py:240-243` finalize uses the same OR per channel (and any channel across). So v1 is a union of both "
         "directions: neither S1-centric nor record-centric alone, and not mutual/AND.",
         f"- Share of v1 rows (sample) satisfying each rule: {rule}.", "",
         "## 3. Unions without the GBM", "", *md(rows), "",
         f"Peak RSS: {peak_rss_mb()} MB.", ""]
    REPORT.write_text("\n".join(L))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / "block_autopsy_timing.json")


if __name__ == "__main__":
    main()
