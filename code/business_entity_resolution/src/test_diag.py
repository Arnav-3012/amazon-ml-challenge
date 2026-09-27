"""Test-vs-train score-distribution diagnosis (read-only) -> docs/test_diag.md.
Per country: S1 mix, predicted matches/S1, % empty, candidates/S1, best-p histogram, % best p in [0.3, 0.75),
% records whose best S1 has p > 0.75. Test = oof/test_p (predict --folds) + output/matching_results.tsv;
train = OOF (b) p_td (rows with p_td, i.e. the test-density world). France: raw examples.

Run from code/business_entity_resolution/:  python -m src.test_diag
"""
import json

import numpy as np
import pandas as pd
import polars as pl

from .block import norm_path
from .features import id_key
from .io import CFG, ROOT, path, peak_rss_mb

LO, HI = 0.3, 0.75
BINS = np.linspace(0, 1, 11)
N_EX = 40


def country(split: str) -> pl.DataFrame:
    return (pl.scan_parquet(norm_path(split, 1)).select("country", s1_id="entity_id")
            .with_columns(s1k=id_key("s1_id")).collect())


def summarise(pairs: pl.LazyFrame, s1: pl.DataFrame, n_pred: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """pairs: s1k, reck, p (scored rows); s1: s1k, country (the S1 universe); n_pred: s1k, n_pred."""
    per_s1 = pairs.group_by("s1k").agg(n_cand=pl.len(), best=pl.col("p").max()).collect(engine="streaming")
    per_rec = (pairs.group_by("reck").agg(best=pl.col("p").max(), s1k=pl.col("s1k").first()).collect(engine="streaming")
               .join(s1.select("s1k", "country"), on="s1k", how="left"))  # candidates never cross countries
    d = (s1.select("s1k", "country").join(per_s1, on="s1k", how="left").join(n_pred, on="s1k", how="left")
         .with_columns(pl.col("n_cand", "n_pred").fill_null(0)))
    tab = (d.group_by("country").agg(
        s1=pl.len(), pred_per_s1=pl.col("n_pred").mean(), empty_pct=100 * (pl.col("n_pred") == 0).mean(),
        cand_per_s1=pl.col("n_cand").mean(), no_cand_pct=100 * (pl.col("n_cand") == 0).mean(),
        mid_pct=100 * pl.col("best").is_between(LO, HI, closed="left").sum() / pl.col("best").is_not_null().sum())
        .join(per_rec.group_by("country").agg(rec_best_gt_hi_pct=100 * (pl.col("best") > HI).mean(),
                                                n_rec=pl.len()), on="country", how="left")
        .with_columns(share_pct=100 * pl.col("s1") / pl.col("s1").sum()).sort("s1", descending=True))
    hist = {}
    for (c,), g in d.filter(pl.col("best").is_not_null()).group_by("country"):
        h, _ = np.histogram(g["best"].to_numpy(), BINS)
        hist[c] = np.round(100 * h / h.sum(), 2)
    hist = pl.DataFrame({"bin": [f"{a:.1f}-{b:.1f}" for a, b in zip(BINS[:-1], BINS[1:])], **{c: hist[c] for c in tab["country"]}})
    return tab, hist


def md(df: pl.DataFrame) -> str:
    df = df.with_columns(pl.col(pl.Float64, pl.Float32).round(3))
    rows = [df.columns, ["---"] * df.width, *df.rows()]
    return "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)


def test_part() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    s1 = country("test")
    pairs = pl.scan_parquet(path("oof_dir") / "test_p.parquet").select(
        "s1_id", "rec_id", "p", s1k=id_key("s1_id"), reck=id_key("rec_id"))
    n_tp = pairs.select(pl.len()).collect().item()
    n_c = pl.scan_parquet(path("interim_dir") / "candidates_test.parquet").select(pl.len()).collect().item()
    assert n_tp == n_c, f"test_p rows {n_tp:,} != candidates_test {n_c:,}"
    mr = pd.read_csv(path("matching_results"), sep="\t", dtype=str, keep_default_na=False)
    n_pred = pl.DataFrame({"s1k": pl.Series(mr["source1_entity_id"]).str.replace_all(r"\D", "").cast(pl.Int64),
                           "n_pred": [len(x.split(",")) if x else 0 for x in mr["matched_entity_ids"]]})
    assert mr.shape[0] == s1.height, "matching_results rows != test S1"
    tab, hist = summarise(pairs.select("s1k", "reck", "p"), s1, n_pred)

    fr = s1.filter(pl.col("country") == "France")
    fr_pairs = pairs.join(fr.lazy().select("s1k"), on="s1k", how="semi").select("s1_id", "rec_id", "p")
    pick = fr.sample(N_EX, seed=CFG["seed"])["s1_id"]
    top3 = (fr_pairs.filter(pl.col("s1_id").is_in(pick.implode())).collect(engine="streaming")
            .sort(["s1_id", "p"], descending=[False, True]).group_by("s1_id", maintain_order=True).head(3))
    mid = (fr_pairs.filter(pl.col("p").is_between(LO, HI, closed="left")).collect(engine="streaming")
           .sample(N_EX, seed=CFG["seed"]).sort("p", descending=True))
    raw = raw_text(pl.concat([pick, top3["rec_id"], mid["s1_id"], mid["rec_id"]]).unique())
    missing = fr.filter(pl.col("s1_id").is_in(pick.implode()) & ~pl.col("s1_id").is_in(top3["s1_id"].implode()))
    return tab, hist, attach(top3, raw, missing["s1_id"]), attach(mid, raw)


def raw_text(ids: pl.Series) -> dict[str, str]:
    out = {}
    for n in (1, 2, 3):
        df = (pl.scan_parquet(norm_path("test", n)).select("entity_id", "business_name", "business_address")
              .filter(pl.col("entity_id").is_in(ids.implode())).collect())
        out |= {e: f"{nm} \\| {ad}" for e, nm, ad in df.iter_rows()}
    return out


def attach(df: pl.DataFrame, raw: dict[str, str], no_cand: pl.Series | None = None) -> str:
    lines = ["| S1 (name \\| address) | candidate (name \\| address) | p |", "| --- | --- | --- |"]
    for s, r, p in df.iter_rows():
        lines.append(f"| `{s}` {raw[s]} | `{r}` {raw[r]} | {p:.3f} |")
    for s in (no_cand if no_cand is not None else []):
        lines.append(f"| `{s}` {raw.get(s, '?')} | (no candidates) | - |")
    return "\n".join(lines)


def train_part() -> tuple[pl.DataFrame, pl.DataFrame, float]:
    cv = json.loads((path("oof_dir") / "cv_full.json").read_text())["b_test_density"]
    t = cv["best_t"]
    pairs = (pl.scan_parquet(path("oof_dir") / "oof_full.parquet").select("s1k", "reck", p="p_td")
             .filter(pl.col("p").is_not_null()))
    # argmax-per-record then p >= t  ==  among rows with p >= t, argmax per record (ties -> lowest s1k, as with_top)
    n_pred = (pairs.filter(pl.col("p") >= t).sort(["reck", "p", "s1k"], descending=[False, True, False])
              .unique("reck", keep="first").group_by("s1k").agg(n_pred=pl.len()).collect(engine="streaming"))
    world = pairs.select("s1k").unique().collect(engine="streaming")  # (b) S1 universe = S1s with a p_td row
    s1 = country("train").join(world, on="s1k", how="semi")
    tab, hist = summarise(pairs, s1, n_pred)
    return tab, hist, t, cv["n_s1"]


def main() -> None:
    tt, th, fr_top3, fr_mid = test_part()
    rss_test = peak_rss_mb()
    rt, rh, t, n_b = train_part()
    doc = f"""# Test diagnosis (stage-1 fold-mean p vs train OOF (b))

Generated by `python -m src.test_diag`. Threshold t = {t} (cv_full (b)). "mid" = best p in [{LO}, {HI}).
Train universe = S1s with a p_td row in OOF (b) ({rt['s1'].sum():,}; cv_full n_s1 = {n_b:,}), so train no_cand_pct = 0
by construction. Record country = country of any of its candidate S1s. Peak RSS {peak_rss_mb()} MB (test part {rss_test} MB).

## Per country: test
{md(tt)}

## Per country: train OOF (b)
{md(rt)}

## Best p per S1: histogram (% of S1s with >= 1 candidate): test
{md(th)}

## Best p per S1: histogram: train OOF (b)
{md(rh)}

## France: {N_EX} random S1s, top-3 candidates
{fr_top3}

## France: {N_EX} random pairs with p in [{LO}, {HI})
{fr_mid}
"""
    (ROOT / "docs" / "test_diag.md").write_text(doc)
    print(tt, rt, f"peak RSS {peak_rss_mb()} MB", sep="\n")


if __name__ == "__main__":
    main()
