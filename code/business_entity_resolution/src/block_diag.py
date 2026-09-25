"""M3b quick blocking diagnosis (train, minutes not hours): PC by segment, PC ceiling of the rank grid,
m x k budget curve. Streaming joins only; no token loading.

  python -m src.block_diag
"""
import time

import polars as pl

from .block import BCFG, CHANNELS, norm_path
from .io import load_gt_pairs, path

t0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - t0:6.0f}s] {msg}", flush=True)


def main() -> None:
    I = path("interim_dir")
    gt = load_gt_pairs()
    s1 = pl.scan_parquet(norm_path("train", 1)).select(s1_id="entity_id", country="country")
    rec = pl.concat([pl.scan_parquet(norm_path("train", n)).select(
        match_id="entity_id", native="is_nonascii_raw", has_addr="has_address") for n in (2, 3)])
    grid = pl.scan_parquet(I / "blockgrid_train_cap1000.parquet").select(
        "s1_id", "rec_id", mr=pl.min_horizontal(*(f"{c}_rrank" for c in CHANNELS)),
        ms=pl.min_horizontal(*(f"{c}_srank" for c in CHANNELS)),
        *(pl.col(f"X_{r}").alias(f"x{r[0]}") for r in ("rrank", "srank")))
    cand = pl.scan_parquet(I / "candidates_train.parquet").select(
        "s1_id", "rec_id", **{f"h{c}": pl.col(f"{c}_score").is_not_null() for c in CHANNELS})
    p = (gt.lazy().join(s1, on="s1_id").join(rec, on="match_id")
         .join(cand, left_on=["s1_id", "match_id"], right_on=["s1_id", "rec_id"], how="left")
         .join(grid, left_on=["s1_id", "match_id"], right_on=["s1_id", "rec_id"], how="left")
         .with_columns(pl.col("^h[A-Z]$").fill_null(False), src=pl.col("match_id").str.slice(0, 2))
         .with_columns(hit=pl.any_horizontal(*(f"h{c}" for c in CHANNELS)), hABC=pl.col("hA") | pl.col("hB") | pl.col("hC"),
                       in_grid=pl.col("mr").is_not_null() | pl.col("ms").is_not_null())
         .collect(engine="streaming"))
    log(f"joined {p.height:,} true pairs")

    def pc(e):
        return (e.mean() * 100).round(2)
    seg = (p.group_by("country", "src", "native").agg(
        pl.len().alias("pairs"), pc(pl.col("hit")).alias("PC"), pc(pl.col("in_grid")).alias("grid ceiling"),
        *(pc(pl.col(f"h{c}")).alias(c) for c in CHANNELS), pc(pl.col("hABC")).alias("A∪B∪C"))
        .sort("country", "src", "native"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print("\nPC by country x source x native script (%, config m/k; grid ceiling = m<=10 or k<=60)")
        print(seg)
        print(p.group_by("country").agg(pl.len(), pc(pl.col("hit")).alias("PC"), pc(pl.col("in_grid")).alias("grid ceiling"),
                                        pc(pl.col("has_addr")).alias("match has addr")).sort("country"))
    miss = p.filter(~pl.col("hit"))
    log(f"missed {miss.height:,}: in grid (rank cut) {miss['in_grid'].sum():,} | "
        f"not scored by any channel within grid {(~miss['in_grid']).sum():,}")

    n_s1 = pl.scan_parquet(norm_path("train", 1)).select(pl.len()).collect().item()
    n_single = n_s1 - p["s1_id"].n_unique()

    def f05_ceiling(hit: pl.Expr) -> float:
        """Macro F0.5 with a perfect matcher on these candidates: P=1, R = found/true per S1; singletons = 1."""
        r = p.group_by("s1_id").agg(r=hit.cast(pl.Float64).mean())["r"]
        return round((n_single + (1.25 * r / (0.25 + r)).fill_nan(0).sum()) / n_s1, 4)
    g = grid.group_by("mr", "ms", "xr", "xs").len().collect(engine="streaming")
    log("grid rank histogram done")
    rows = []
    for name, r, sc in (("all channels", "mr", "ms"), ("X only", "xr", "xs")):
        for m in BCFG["sweep_m"]:
            for k in BCFG["sweep_k"]:
                cut = (pl.col(r) <= m).fill_null(False) | (pl.col(sc) <= k).fill_null(False)
                rows.append({"set": name, "m": m, "k": k, "PC %": round(p.select(cut.mean()).item() * 100, 2),
                             "F0.5 ceiling": f05_ceiling(cut), "cand/S1": round(g.filter(cut)["len"].sum() / n_s1, 1)})
    with pl.Config(tbl_rows=-1):
        print("\nBudget curve (train):")
        print(pl.DataFrame(rows))
    for r, label in (("mr", "best single channel"), ("xr", "X combined")):
        print(f"\nrank of true S1 for its record ({label}), cum % of ALL true pairs:")
        print(p.filter(pl.col(r).is_not_null()).group_by(r).len().sort(r).with_columns(
            cum_pct=(pl.col("len").cum_sum() / p.height * 100).round(2)))
    log("done")


if __name__ == "__main__":
    main()
