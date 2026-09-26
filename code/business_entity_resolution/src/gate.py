"""M5-1 blocking v2: a wider candidate pool, gated per S1 by a cheap pre-score (blocking outputs only, no matcher).

Pool = block grid cut by block.finalize(m, gate.pool_k, all channels): per channel A/B/C/X, record-centric top-m
S1s U S1-centric top-pool_k records. Both grids hold these ranks (train m<=10/k<=60, test max(sweep_m)/pool_k),
so train and test pools are the same function of the data.
  norm_ch = max(score / the S1's best score in ch, score / the record's best score in ch); 0 if ch did not keep
            the pair. The S1's best is its S1-centric rank-1 row (in the pool by construction); the record's best is
            its record-centric rank-1 row (read from the grid).
  pre     = max over channels of norm_ch + n_channels_hit (at the pool m/k).
  gr_pre  = ordinal rank inside the S1 by pre desc, X_score desc, rec_id. gr_v1 = the same with the v1 set
            (X rrank <= blocking.m or X srank <= blocking.k: the M3b candidates) ranked first.
Candidates = rows with gr_{gate.variant} <= gate.n: at most n per S1 on either split.

Run from code/business_entity_resolution/ (needs blockgrid_{split}_cap{df_cap}.parquet from src.block):
  python -m src.gate --split train   # artifacts/interim/pool_train_m{m}/ for m in sweep_m (rows ranked <= max(sweep_n))
  python -m src.gate --split test
  python -m src.gate --eval          # docs/blocking_v2.md: PC, perfect-matcher F0.5 ceiling, cand/S1 per (m, variant, n)
  python -m src.gate --apply         # gate.m/n/variant -> candidates_{train,test}.parquet + output/candidate_pairs.tsv
"""
import argparse
import shutil

import numpy as np
import polars as pl

from .block import BCFG, CHANNELS, RANK_COLS, finalize, norm_path
from .decide import md
from .io import CFG, ROOT, StepLog, load_gt_pairs, path, write_candidates

GCFG = CFG["gate"]
PAIR = ["s1_id", "rec_id"]
VARIANTS = ("pre", "v1")
N_MAX = max(GCFG["sweep_n"])
REPORT = ROOT / "docs" / "blocking_v2.md"
V1 = ((pl.col("X_rrank") <= BCFG["m"]) | (pl.col("X_srank") <= BCFG["k"])).fill_null(False)


def grid(split: str) -> pl.LazyFrame:
    return pl.scan_parquet(path("interim_dir") / f"blockgrid_{split}_cap{BCFG['df_cap']}.parquet")


def pool_dir(split: str, m: int):
    return path("interim_dir") / f"pool_{split}_m{m}"


def in_pool(m: int) -> pl.Expr:
    return pl.any_horizontal([(pl.col(f"{ch}_rrank") <= m).fill_null(False)
                              | (pl.col(f"{ch}_srank") <= GCFG["pool_k"]).fill_null(False) for ch in CHANNELS])


def rec_best(split: str) -> pl.DataFrame:
    """rec_id -> {ch}_rbest = score of the record's record-centric rank-1 S1 in channel ch."""
    one = {ch: pl.col(f"{ch}_rrank") == 1 for ch in CHANNELS}
    return (grid(split).filter(pl.any_horizontal(list(one.values())))
            .group_by("rec_id").agg(*(pl.col(f"{ch}_score").filter(one[ch]).max().alias(f"{ch}_rbest")
                                      for ch in CHANNELS))
            .collect(engine="streaming"))


def gate_bucket(lf: pl.LazyFrame, best: pl.DataFrame) -> pl.DataFrame:
    norm = [pl.max_horizontal(pl.col(f"{ch}_score") / pl.col(f"{ch}_score").max().over("s1_id"),
                              pl.col(f"{ch}_score") / pl.col(f"{ch}_rbest")).fill_null(0) for ch in CHANNELS]
    x = (lf.join(best.lazy(), on="rec_id", how="left")
         .with_columns(pre=(pl.max_horizontal(norm) + pl.col("n_channels_hit")).cast(pl.Float32), v1=V1,
                       n_pool=pl.len().over("s1_id").cast(pl.Int32))
         .drop(*(f"{ch}_rbest" for ch in CHANNELS)).collect())
    for name, first in (("gr_pre", []), ("gr_v1", [pl.col("v1")])):
        by = [*first, pl.col("pre"), pl.col("X_score").fill_null(0), pl.col("rec_id")]
        x = (x.sort([pl.col("s1_id"), *by], descending=[False, *[True] * (len(by) - 1), False])
             .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over("s1_id").alias(name)))
    return x.filter(pl.min_horizontal("gr_pre", "gr_v1") <= N_MAX)


def build(split: str, log: StepLog) -> None:
    best = rec_best(split)
    log(f"{split} record best scores", records=best.height)
    B, K, ms = GCFG["buckets"], GCFG["pool_k"], GCFG["sweep_m"]
    for m in ms:
        shutil.rmtree(pool_dir(split, m), ignore_errors=True)
        pool_dir(split, m).mkdir(parents=True)
    src = finalize(grid(split), max(ms), K, CHANNELS)
    for b in range(B):
        base = src.filter(pl.col("s1_id").hash(CFG["seed"]) % B == b).collect()
        for m in ms:  # finalize at a smaller m of the max-m pool == finalize of the grid at that m
            x = gate_bucket(finalize(base.lazy(), m, K, CHANNELS), best)
            x.write_parquet(pool_dir(split, m) / f"part-{b:03d}.parquet")
            log(f"{split} bucket {b} m{m}", pool_rows=x.height)


def ceiling(ntrue: pl.DataFrame, hits: pl.DataFrame) -> float:
    """Perfect-matcher macro F0.5 over every train S1: P = 1, R = hits / ntrue; singletons (empty) = 1."""
    t = ntrue.join(hits, on="s1_id", how="left").with_columns(pl.col("h").fill_null(0))
    n, h = t["ntrue"].to_numpy(), t["h"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = h / n
        f = np.where(n == 0, 1.0, 1.25 * r / (0.25 + r))
    return float(f.mean())


def evaluate(log: StepLog) -> None:
    s1 = {sp: pl.read_parquet(norm_path(sp, 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
          for sp in ("train", "test")}
    gt = load_gt_pairs().select("s1_id", rec_id="match_id").join(s1["train"], on="s1_id")
    ntrue = s1["train"].join(gt.group_by("s1_id").len("ntrue"), on="s1_id", how="left").select(
        "s1_id", pl.col("ntrue").fill_null(0))
    ref = (gt.lazy().join(grid("train").select(*PAIR, *(c for c in RANK_COLS if "rank" in c)), on=PAIR, how="left")
           .with_columns(v1=V1, **{f"pool_m{m}": in_pool(m) for m in GCFG["sweep_m"]}).collect(engine="streaming"))
    log("eval: true pairs vs grid", pairs=ref.height)

    def counts(split: str) -> dict:
        return (grid(split).select(v1=V1.sum(), **{f"pool_m{m}": in_pool(m).sum() for m in GCFG["sweep_m"]})
                .collect(engine="streaming").row(0, named=True))
    tot = {sp: counts(sp) for sp in ("train", "test") if (path("interim_dir") / f"blockgrid_{sp}_cap{BCFG['df_cap']}.parquet").exists()}
    log("eval: grid counts")

    def row(name: str, m, n, f: pl.DataFrame, hit: pl.Expr, tr_cand: float, te: dict) -> dict:
        h = f.select("s1_id", "country", "v1", h=hit)
        pc_c = {f"PC {c} %": float(g["h"].mean() * 100) for (c,), g in sorted(h.group_by("country"))}
        return {"set": name, "m": m, "n": n, "PC %": float(h["h"].mean() * 100), **pc_c,
                "F0.5 ceiling": ceiling(ntrue, h.group_by("s1_id").agg(pl.col("h").sum())),
                "v1 hits kept %": float(h.filter("v1")["h"].mean() * 100), "train cand/S1": tr_cand, **te}

    rows = []
    ntr, nte = s1["train"].height, s1["test"].height
    rows.append(row("v1 (X m5/k10)", BCFG["m"], "-", ref, pl.col("v1"), tot["train"]["v1"] / ntr, {}))
    for m in GCFG["sweep_m"]:
        te = {"test cand/S1": tot["test"][f"pool_m{m}"] / nte} if "test" in tot else {}
        rows.append(row("pool, ungated", m, "-", ref, pl.col(f"pool_m{m}"), tot["train"][f"pool_m{m}"] / ntr, te))
    for m in GCFG["sweep_m"]:
        pools = {sp: pl.scan_parquet(pool_dir(sp, m) / "*.parquet") for sp in ("train", "test")
                 if pool_dir(sp, m).exists()}
        f = (ref.lazy().join(pools["train"].select(*PAIR, "gr_pre", "gr_v1"), on=PAIR, how="left")
             .collect(engine="streaming"))
        assert f.height == ref.height, "duplicate pairs in the pool"
        size = {sp: s1[sp].join(lf.group_by("s1_id").agg(pl.col("n_pool").first()).collect(engine="streaming"),
                                on="s1_id", how="left").with_columns(pl.col("n_pool").fill_null(0))
                for sp, lf in pools.items()}
        for v in VARIANTS:
            for n in GCFG["sweep_n"]:
                c = {sp: np.minimum(d["n_pool"].to_numpy(), n) for sp, d in size.items()}
                te = {}
                if "test" in c:
                    ctry = size["test"]["country"].to_numpy()
                    te = {"test cand/S1": float(c["test"].mean()), "test p99": float(np.quantile(c["test"], 0.99)),
                          "test max": int(c["test"].max()),
                          **{f"test {k} cand/S1": float(c["test"][ctry == k].mean()) for k in sorted(set(ctry))}}
                rows.append(row(f"gated {v}", m, n, f, (pl.col(f"gr_{v}") <= n).fill_null(False),
                                float(c["train"].mean()), te))
        log(f"eval m{m}")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    rows = [{k: r.get(k, "") for k in cols} for r in rows]
    L = ["# Blocking v2 evaluation (M5-1)",
         f"Generated by `src/gate.py`. Pool = all channels at record-centric m / S1-centric k = {GCFG['pool_k']}; "
         "gated = at most n candidates per S1 by the pre-score (`pre`) or v1 set first, then pre-score (`v1`). "
         "PC and the perfect-matcher F0.5 ceiling are over all train true pairs / S1s (singletons = 1). "
         "v1 hits kept % = share of the true pairs v1 finds that the set keeps. Test cand/S1 over every test S1 "
         "(zero-candidate S1s count).", "", *md(rows), ""]
    REPORT.write_text("\n".join(L))
    log("eval report", path=str(REPORT))


def dry_estimate(rows: dict, n_features: int, log: StepLog) -> None:
    """rows[split] x n_features x 4B (float32) feature-matrix estimate, printed before any write."""
    for split, r in rows.items():
        gb = r * n_features * 4 / 1e9
        print(f"dry estimate {split}: {r:,} rows x {n_features} features x 4B = {gb:.2f} GB")
    log("dry estimate", n_features=n_features, **{f"{s}_rows": r for s, r in rows.items()})


def apply(log: StepLog) -> None:
    from .sibling import SIB_FEATS, sib_dir  # local import: optional path, avoid a hard dependency
    from .train import feature_cols

    m, n, v = GCFG["m"], GCFG["n"], GCFG["variant"]
    assert m in GCFG["sweep_m"] and n <= N_MAX and v in VARIANTS, (m, n, v)
    sib_on, delta = GCFG.get("sibling", {}).get("enabled", False), GCFG.get("sibling", {}).get("delta")
    if sib_on:
        assert delta is not None, "gate.sibling.enabled but gate.sibling.delta is not set"
    I = path("interim_dir")

    pools, counts = {}, {}
    for split in ("train", "test"):
        v1_bak = I / f"candidates_{split}_v1.parquet"
        assert v1_bak.exists(), f"{v1_bak} missing -- v1 backup must exist before --apply (run once on the v1 config first)"
        pool = pl.scan_parquet(pool_dir(split, m) / "*.parquet").filter(pl.col(f"gr_{v}") <= n).select(
            *PAIR, *RANK_COLS, "n_channels_hit")
        if sib_on:
            schema = pool.collect_schema()
            new = pl.scan_parquet(sib_dir(split, delta) / "part-000.parquet").select(*PAIR)
            dup = new.join(pool.select(*PAIR), on=PAIR, how="inner").collect(engine="streaming")
            assert dup.height == 0, f"{split}: sibling pairs already in pool (join bug), {dup.height} rows"
            # rank/score/n_channels_hit columns don't exist for a sibling-only pair (it never scored in A/B/C/X);
            # null them at the pool's own dtypes so pl.concat(how="diagonal") doesn't hit a schema mismatch
            filler = {c: pl.lit(None, dtype=schema[c]) for c in (*RANK_COLS, "n_channels_hit")}
            pool = pl.concat([pool, new.with_columns(**filler)], how="diagonal")
        pools[split] = pool.collect(engine="streaming")
        counts[split] = pools[split].height

    n_features = len(feature_cols("train")) + (len(SIB_FEATS) if sib_on else 0)
    dry_estimate(counts, n_features, log)

    for split in ("train", "test"):
        dst = I / f"candidates_{split}.parquet"
        pool = pools[split]
        assert pool.select(PAIR).is_duplicated().sum() == 0, f"{split}: duplicate (s1_id, rec_id) pairs in pool"
        if sib_on:
            # sibling.build(split, delta, ...) already wrote features.parquet (all_features on the in-memory
            # cand it computed) -- apply() has no cand of its own, so read the cached file instead of re-deriving it
            sib_feats = pl.scan_parquet(sib_dir(split, delta) / "features.parquet")
            pool = pool.join(sib_feats, on=PAIR, how="left").with_columns(
                pl.col("sib_hit").fill_null(False), pl.col("sib_anchor_margin").fill_null(0.0),
                pl.col("n_sib_anchors").fill_null(0), pl.col("sib_name_tset").fill_null(0.0))
        pool.write_parquet(dst)
        per = pl.scan_parquet(dst).group_by("s1_id").len().collect(engine="streaming")["len"]
        st = {"rows": int(per.sum()), "s1": per.len(), "max_per_s1": int(per.max())}
        if not sib_on:
            assert st["max_per_s1"] <= n, st  # sibling can add rows past n for a record's S1 candidates by design
        log(f"candidates_{split}", m=m, n=n, variant=v, sibling=sib_on, **st)

    s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"])["entity_id"]
    cand_test = pl.scan_parquet(I / "candidates_test.parquet").select(PAIR)
    write_candidates(path("candidate_pairs"), cand_test, s1_ids)
    parquet_pairs = cand_test.collect(engine="streaming").unique(PAIR)
    tsv_pairs = (pl.read_csv(path("candidate_pairs"), separator="\t")
                 .filter(pl.col("candidate_entity_ids") != "")
                 .select(s1_id="source1_entity_id", rec_id=pl.col("candidate_entity_ids").str.split(","))
                 .explode("rec_id"))
    assert parquet_pairs.height == tsv_pairs.height and \
        parquet_pairs.join(tsv_pairs, on=PAIR, how="anti").height == 0, \
        "candidate_pairs.tsv row set != candidates_test.parquet row set"
    log("candidate_pairs.tsv", path=str(path("candidate_pairs")), rows=parquet_pairs.height)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=("train", "test"))
    g.add_argument("--eval", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    log = StepLog()
    mode = "eval" if a.eval else "apply" if a.apply else a.split
    if a.eval:
        evaluate(log)
    elif a.apply:
        apply(log)
    else:
        build(a.split, log)
    log.dump(path("artifacts_dir") / "logs" / f"gate_timing_{mode}.json")


if __name__ == "__main__":
    main()
