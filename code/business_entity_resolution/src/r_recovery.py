"""M5-4 R recovery: which v1-missed true pairs does channel R (src.block_r) bring back, and would an exact core-key
channel K add more on top of v1 ∪ R? Read-only; no pipeline change.

World / 20%-S1 sample / pair status (TP/FN/miss) / loss weight `w` = src.mechanisms (cv_full (b) world, t = 0.75,
sample macro F0.5 points). Tags = mechanisms.tag(); "untagged" = none of mechanisms.TAGS (its D1 definition).
Populations / core key / tie size = src.tiebreak: A = addr_empty, else B = dict-remapped name_tokens of record ==
of S1 and record address non-empty, else "other"; tie size = world S1s with the record's country and core key.
  v1       = candidates_train_v1.parquet rows passing gate.V1 (X m5/k10, the M3b set).
  R        = blockr_train.parquet rows with R_rrank <= gate.R_M.
  missed   = true pairs of sample S1s (status TP/FN/miss) not in v1. They can still be TP/FN: the (b) world was
             scored on the gated pool, which extends v1. Only status "miss" carries blocking loss, so
             "miss pts recovered" = w of miss pairs that R (or K) contains, at a perfect matcher.
  K        = every train S2/S3 record x world S1s of its country with an equal, non-empty core key, kept only when
             that tie set has <= K_TIE S1s. Evaluated only: rows added beyond v1 ∪ R, precision = added rows that
             are GT pairs / added rows.

Run from code/business_entity_resolution/ (needs cv_full (b) OOF, candidates_train_v1.parquet, blockr_train.parquet):
  python -m src.r_recovery --smoke   # 1% S1 sample + 1% of records for K, same path incl. docs/r_recovery.md
  python -m src.r_recovery           # 20% sample -> docs/r_recovery.md
Aborts if peak RSS > 12 GB (tiebreak.Log guard).
"""
import argparse

import polars as pl

import src.mechanisms as M
from . import gate as G
from .decide import md
from .features import id_key, load_token_dict
from .io import ROOT, load_gt_pairs, path, peak_rss_mb
from .mechanisms import NORM, SEED, TAGS, load, sample_pairs, world
from .tiebreak import Log, core_key, remapped  # importing tiebreak also sets mechanisms.MAX_RSS_MB = 12 GB

REPORT = ROOT / "docs" / "r_recovery.md"
KEYS = ["s1k", "reck"]
K_TIE = 5
SHOW_TAGS = ["addr_empty", "pseudo_brand", "native_script", "domain", "city_differs", "legal_removed"]
BREAKS, BINS = [0, 1, 5, 20], ["0", "1", "2-5", "6-20", "21+"]
REC_MOD = 1  # K record sample: reck % REC_MOD == 0; --smoke sets 100


def keyed(file, *filters: pl.Expr) -> pl.LazyFrame:
    lf = pl.scan_parquet(file)
    return (lf.filter(*filters) if filters else lf).select(s1k=id_key("s1_id"), reck=id_key("rec_id"))


def flag(df: pl.DataFrame, lf: pl.LazyFrame, name: str) -> pl.DataFrame:
    """df + bool column `name` = (s1k, reck) in lf. The semi-join builds on df's keys, never on lf's."""
    hit = lf.join(df.lazy().select(KEYS), on=KEYS, how="semi").unique().with_columns(pl.lit(True).alias(name))
    return df.join(hit.collect(engine="streaming"), on=KEYS, how="left").with_columns(pl.col(name).fill_null(False))


def stats(d: pl.DataFrame, by: str) -> dict:
    miss = d.filter(pl.col("status") == "miss")
    pts, rec = float(miss["w"].sum()), float(miss.filter(pl.col(by))["w"].sum())
    return {"n missed": d.height, "% recovered": 100 * float(d[by].mean()) if d.height else float("nan"),
            "n world-miss": miss.height, "miss pts": pts, "miss pts recovered": rec,
            "% miss pts recovered": 100 * rec / pts if pts else float("nan")}


def part2(T: pl.DataFrame, gtw: pl.DataFrame, pool: pl.DataFrame, dm: dict, log: Log) -> tuple[list[str], dict]:
    Mv = T.filter(~pl.col("in_v1"))
    X = M.tag(Mv.select("s1k", "reck", "code", "p", "status", "w"), gtw, log).select(*KEYS, *TAGS)
    s = load(NORM[0], Mv["s1k"].unique(), "s1k", s_core=remapped("name_tokens", dm).list.join(" "))
    r = load(NORM[1:], Mv["reck"].unique(), "reck", r_a=pl.col("business_address"), rc=pl.col("country"),
             r_core=remapped("name_tokens", dm).list.join(" "), ckey=core_key("name_tokens", dm))
    sizes = pool.group_by("country", "ckey").agg(n_tie=pl.len()).rename({"country": "rc"})
    X = (Mv.join(X, on=KEYS).join(s, on="s1k").join(r, on="reck")
         .join(sizes, on=["rc", "ckey"], how="left").with_columns(pl.col("n_tie").fill_null(0)))
    assert X.height == Mv.height, (X.height, Mv.height)
    X = X.with_columns(
        untagged=~pl.any_horizontal(TAGS),
        pop=pl.when(pl.col("addr_empty")).then(pl.lit("A"))
        .when((pl.col("r_core") == pl.col("s_core")) & (pl.col("r_a").str.strip_chars() != "")).then(pl.lit("B"))
        .otherwise(pl.lit("other")),
        bin=pl.col("n_tie").cut(BREAKS, labels=BINS).cast(pl.String))
    overall = {"group": "all v1-missed", **stats(X, "in_R")}
    tags = [{"tag": t, **stats(X.filter(pl.col(t)), "in_R")} for t in [*SHOW_TAGS, "untagged"]]
    pops = [{"population": p, "tie size": b, **stats(d, "in_R")}
            for p in ("A", "B", "other") for b in [*BINS, "all"]
            if (d := X.filter(pl.col("pop") == p, (pl.col("bin") == b) if b != "all" else pl.lit(True))).height]
    log("part2", missed=X.height, recovered=int(X["in_R"].sum()))
    return (["## 2. v1-missed true pairs recovered by R", "", *md([overall]), "",
             "### By tag (a pair can carry several tags)", "", *md(tags), "",
             "### By tiebreak population x tie size", "",
             "A = addr_empty; B = core-identical with record address; other = neither. Tie size over all world S1s.",
             "", *md(pops), ""], overall)


def part3(T: pl.DataFrame, pool: pl.DataFrame, v1: pl.LazyFrame, R: pl.LazyFrame, dm: dict, n_world: int,
          log: Log) -> list[str]:
    ties = (pool.filter(pl.col("ckey") != "")
            .with_columns(n_tie=pl.len().over("country", "ckey")).filter(pl.col("n_tie") <= K_TIE))
    recs = (pl.concat([pl.scan_parquet(f).select(reck=id_key("entity_id"), country="country",
                                                 ckey=core_key("name_tokens", dm)) for f in NORM[1:]])
            .filter(pl.col("reck") % REC_MOD == 0).collect(engine="streaming"))
    K = recs.join(ties.select("s1k", "country", "ckey"), on=["country", "ckey"]).select(KEYS).unique()
    del recs
    log("K built", rows=K.height)
    K = flag(flag(K, v1, "in_v1"), R, "in_R")
    new = K.filter(~pl.col("in_v1") & ~pl.col("in_R")).select(KEYS)
    gt = load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id")).unique()
    n_true = new.join(gt, on=KEYS, how="semi").height
    T = flag(T.filter(~pl.col("in_v1") & ~pl.col("in_R")), new.lazy(), "in_K")
    add = {"K rows": K.height, "K rows beyond v1 ∪ R": new.height,
           "added rows / world S1" + (f" (x{REC_MOD} record sample)" if REC_MOD > 1 else ""): new.height / n_world,
           "added rows that are true": n_true, "precision %": 100 * n_true / new.height if new.height else float("nan")}
    samp = {"group": "sample true pairs not in v1 ∪ R", **stats(T, "in_K"),
            "n TP/FN in K": int(T.filter(pl.col("in_K") & (pl.col("status") != "miss")).height)}
    log("part3", **{k: v for k, v in add.items() if isinstance(v, int)})
    return ["## 3. Exact-key channel K (evaluate only)", "",
            f"K = record x world S1s of its country with an equal non-empty core key, tie size <= {K_TIE}. "
            "Precision over every added row (all records" + (f", 1/{REC_MOD} sample" if REC_MOD > 1 else "") +
            "), true = GT pair. Loss pts over the 20% S1 sample.", "", *md([add]), "", *md([samp]), ""]


def main() -> None:
    global REC_MOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% S1 sample, 1%% of records for K, same path incl. report")
    smoke = ap.parse_args().smoke
    if smoke:
        M.FRAC, REC_MOD = 0.01, 100
    log = Log()
    oof, s1, code, kept, in_cand, cnt, f, L = world(log)
    del oof
    gt = load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))
    gtw = gt.join(code, on="s1k", how="semi").unique()
    P, samp = sample_pairs(s1, kept, in_cand, gt, cnt, f, L, log)
    del in_cand, kept, gt
    T = P.filter(pl.col("status") != "FP")
    assert not T.select(KEYS).is_duplicated().any(), "duplicate true pairs in the sample"
    I = path("interim_dir")
    v1 = keyed(I / "candidates_train_v1.parquet", G.V1)
    R = keyed(G.blockr_path("train"), pl.col("R_rrank") <= G.R_M)
    T = flag(flag(T, v1, "in_v1"), R, "in_R")
    dm = load_token_dict() or {}
    pool = load(NORM[0], s1["s1k"], "s1k", country=pl.col("country"), ckey=core_key("name_tokens", dm))
    log("sample flags", true=T.height, in_v1=int(T["in_v1"].sum()), in_R=int(T["in_R"].sum()), pool=pool.height)

    n_r = R.select(pl.len()).collect().item()
    n_r_v1 = v1.join(R, on=KEYS, how="semi").select(pl.len()).collect(engine="streaming").item()
    n_s1 = pl.scan_parquet(NORM[0]).select(pl.len()).collect().item()
    rrow = {"R rows": n_r, "R rows in v1": n_r_v1, "R rows new vs v1": n_r - n_r_v1,
            "new rows / train S1": (n_r - n_r_v1) / n_s1}
    log("R rows", **rrow)
    s2, overall = part2(T, gtw, pool, dm, log)
    s3 = part3(T, pool, v1, R, dm, s1.height, log)

    fs = f[samp]
    head = ["# Channel R recovery of v1 misses + exact-key channel K" + (" [SMOKE: 1% samples]" if smoke else ""),
            "", f"Generated by `src/r_recovery.py` (definitions in its docstring). World: cv_full (b), "
            f"{s1.height:,} S1s, t = {M.T}; sample {int(samp.sum()):,} S1s ({M.FRAC:.0%}, seed {SEED}), loss "
            f"{100 * (1 - fs.mean()):.4f} sample macro pts. {T.height:,} true pairs in the sample, "
            f"{int(T['in_v1'].sum()):,} in v1, {overall['n missed']:,} missed by v1.", "",
            "## 1. Rows R adds (all train)", "", *md([rrow]), ""]
    REPORT.write_text("\n".join(head + s2 + s3 + [f"Peak RSS: {peak_rss_mb()} MB.", ""]))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / f"r_recovery{'_smoke' if smoke else ''}_timing.json")


if __name__ == "__main__":
    main()
