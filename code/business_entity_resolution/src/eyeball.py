"""Standalone M5 eyeball report: raw-string error buckets for manual reading. Read-only, no pipeline changes.

Reads only: oof/oof_train.parquet + models/lgb_final.txt (M4 OOF preds + pred_contrib), ground truth
(src.io.load_gt), artifacts/interim/norm_train_s{1,2,3}.parquet (raw name/address/country -- verified against
the raw train TSVs via src.io.load_source), artifacts/interim/candidates_train_v1.parquet (v1 blocking ranks,
for the "not in v1" buckets), models/lgb_final.json (feature list), artifacts/interim/idf_train.parquet +
src.mine_dict.shared (vocab/depth miss tagging, section 6). Never imports or calls block.py/features.py
logic -- data only.

Writes docs/eyeball.md. Six sections, raw strings verbatim (no truncation), pipe-delimited rows, seed 42,
country split ~50/50 India/US within each sampled bucket. No analysis/commentary in the output.

Run from code/business_entity_resolution/:  python -m src.eyeball
Needs oof/oof_train.parquet + models/lgb_final.txt first (python -m src.train && python -m src.train --final).
"""
import json
import time

import numpy as np
import polars as pl
import lightgbm as lgb

from .io import CFG, ROOT, load_gt, load_source, path
from .mine_dict import FIELDS as VD_FIELDS, shared

SEED = CFG["seed"]
OUT = ROOT / "docs" / "eyeball.md"


def booster() -> tuple[lgb.Booster, list[str]]:
    """Same load as diagnose.py's booster(): feature list from lgb_final.json, not a fresh schema scan --
    this is what the saved model actually trained on."""
    meta = json.loads((path("models_dir") / "lgb_final.json").read_text())
    return lgb.Booster(model_file=str(path("models_dir") / "lgb_final.txt")), meta["features"]


# ---------- data loading (read-only) ----------

def raw_lookup() -> pl.DataFrame:
    """entity_id -> country, business_name, business_address (raw) for S1+S2+S3, from norm caches."""
    parts = [pl.read_parquet(path("interim_dir") / f"norm_train_s{n}.parquet",
                              columns=["entity_id", "country", "business_name", "business_address"])
             for n in (1, 2, 3)]
    return pl.concat(parts)


def verify_raw_matches_source(lut: pl.DataFrame) -> None:
    """norm cache's raw business_name/business_address must equal the raw TSV verbatim (no silent drift)."""
    for n in (1, 2, 3):
        src = pl.from_pandas(load_source("train", n))
        j = lut.join(src.select("entity_id", "business_name", "business_address"), on="entity_id",
                      how="inner", suffix="_src")
        bad = j.filter((pl.col("business_name") != pl.col("business_name_src"))
                        | (pl.col("business_address") != pl.col("business_address_src")))
        assert bad.height == 0, f"S{n}: {bad.height} raw-string mismatches between norm cache and source TSV"


def lut_index(lut: pl.DataFrame) -> dict:
    """entity_id -> (business_name, business_address, country), for O(1) row lookups in the section writers."""
    return {r["entity_id"]: (r["business_name"], r["business_address"], r["country"])
            for r in lut.iter_rows(named=True)}


def s1_row_str(idx: dict, s1_id: str) -> str:
    name, addr, country = idx[s1_id]
    return f"{name} | {addr} | {country}"


def rec_row_str(idx: dict, rec_id: str) -> str:
    name, addr, country = idx[rec_id]
    return f"{name} | {addr} | {country}"


def balanced_sample(df: pl.DataFrame, n: int, country_col: str = "country") -> pl.DataFrame:
    """~n/2 India, ~n/2 US by country_col (data values "India" / "US"), seed 42; if one side is short, tops up
    from the other side's unsampled rows (never a duplicate row)."""
    df = df.with_row_index("_bs")
    ind, us = df.filter(pl.col(country_col) == "India"), df.filter(pl.col(country_col) == "US")
    a_n = min(n // 2, ind.height)
    b_n = min(n - a_n, us.height)
    a_n = min(n - b_n, ind.height)  # India takes back any US shortfall
    return pl.concat([ind.sample(a_n, seed=SEED), us.sample(b_n, seed=SEED)]).drop("_bs")


def load_oof() -> pl.DataFrame:
    """oof/oof_train.parquet, unmodified. pred_contrib is NOT computed here -- section3 computes it only for
    its own ~60-row sample (see top3_contrib_for), since pred_contrib over the full OOF set is O(n_rows *
    n_trees * n_features) and every other section ignores it."""
    return pl.read_parquet(path("oof_dir") / "oof_train.parquet").with_row_index("_oof")


def top3_contrib_for(oof_rows: pl.DataFrame, bst: lgb.Booster, feats: list[str]) -> dict[int, str]:
    """_oof -> top-3 |pred_contrib| feature=value string, computed only for the given (already-filtered/
    sampled) oof_rows -- same features, same booster, same math as before, just not run over every OOF row.
    Keyed by _oof so callers look up per row regardless of their own row order."""
    X = (pl.scan_parquet(str(path("features_dir") / "train" / "part-*.parquet"))
         .filter(pl.col("rec_id").is_in(oof_rows["rec_id"].implode()))  # stream-filter before the join
         .join(oof_rows.lazy().select("_oof", "s1_id", "rec_id"), on=["s1_id", "rec_id"], how="inner")
         .select("_oof", *feats).collect(engine="streaming"))
    assert X["_oof"].n_unique() == X.height == oof_rows.height, "sampled OOF rows must map 1:1 to feature rows"
    contrib = bst.predict(X.select(feats).to_numpy().astype(np.float32), pred_contrib=True)[:, :-1]  # drop bias
    top3_idx = np.argsort(-np.abs(contrib), axis=1)[:, :3]
    return {k: ", ".join(f"{feats[j]}={contrib[i, j]:+.3f}" for j in top3_idx[i])
            for i, k in enumerate(X["_oof"].to_list())}


# ---------- section writers ----------

def section1(f, feats: list[str]) -> None:
    f.write("## 1. Full model feature list\n\n")
    for c in feats:
        f.write(f"- {c}\n")
    f.write("\n")


def section2(f, oof: pl.DataFrame, lut: pl.DataFrame, idx: dict, gt: dict) -> None:
    """60 extra-FPs: p>0.9, name_tset>=80, addr_tset>=80, predicted top S1 wrong (or record is a distractor
    -- true S1 not in this candidate set at all)."""
    f.write("## 2. Extra-FPs (p>0.9, name_tset>=80, addr_tset>=80)\n\n")
    f.write("S1 raw | record raw | record's TRUE S1 raw (or DISTRACTOR)\n")
    f.write("---|---|---\n")
    top = (oof.sort(["rec_id", "p", "s1_id"], descending=[False, True, False])
           .unique(subset=["rec_id"], keep="first"))
    fp = top.filter((pl.col("p") > 0.9) & (pl.col("label") == 0))
    tset = pl.read_parquet(str(path("features_dir") / "train" / "part-*.parquet"),
                            columns=["s1_id", "rec_id", "name_tset", "addr_tset"])
    fp = fp.join(tset, on=["s1_id", "rec_id"], how="left").filter(
        (pl.col("name_tset") >= 80) & (pl.col("addr_tset") >= 80))
    fp = fp.join(lut.select("entity_id", "country"), left_on="rec_id", right_on="entity_id", how="left")
    fp = balanced_sample(fp, 60)
    true_s1 = {rec: s1 for s1, matched in gt.items() for rec in matched}  # one pass, not a gt scan per row
    for r in fp.iter_rows(named=True):
        rec_true = true_s1.get(r["rec_id"])
        true_str = s1_row_str(idx, rec_true) if rec_true else "DISTRACTOR"
        f.write(f"{s1_row_str(idx, r['s1_id'])} | {rec_row_str(idx, r['rec_id'])} | {true_str}\n")
    f.write("\n")


def section3(f, oof: pl.DataFrame, lut: pl.DataFrame, idx: dict, bst: lgb.Booster, feats: list[str]) -> None:
    """60 FN(a): true pairs (label=1) scored p in [0.3, 0.8) -- top-3 pred_contrib features. pred_contrib is
    computed only on this 60-row sample (see top3_contrib_for), not the full OOF set."""
    f.write("## 3. FN(a): true pairs, p in [0.3, 0.8)\n\n")
    f.write("S1 raw | record raw | p | top-3 features by pred_contrib\n")
    f.write("---|---|---|---\n")
    fn = oof.filter((pl.col("label") == 1) & (pl.col("p") >= 0.3) & (pl.col("p") < 0.8))
    fn = fn.join(lut.select("entity_id", "country"), left_on="rec_id", right_on="entity_id", how="left")
    fn = balanced_sample(fn, 60)
    top3 = top3_contrib_for(fn, bst, feats)
    for r in fn.iter_rows(named=True):
        f.write(f"{s1_row_str(idx, r['s1_id'])} | {rec_row_str(idx, r['rec_id'])} | "
                 f"{r['p']:.4f} | {top3[r['_oof']]}\n")
    f.write("\n")


def section4(f, oof: pl.DataFrame, lut: pl.DataFrame, idx: dict, gt: dict) -> None:
    """30 singleton-FPs: record's true S1 is empty (singleton) but model predicted a match, same thresholds
    as section 2."""
    f.write("## 4. Singleton-FPs (p>0.9, name_tset>=80, addr_tset>=80)\n\n")
    f.write("S1 raw | record raw | record's TRUE S1 raw (or DISTRACTOR)\n")
    f.write("---|---|---\n")
    matched_recs = {rec for v in gt.values() for rec in v}
    top = (oof.sort(["rec_id", "p", "s1_id"], descending=[False, True, False])
           .unique(subset=["rec_id"], keep="first"))
    fp = top.filter((pl.col("p") > 0.9) & (~pl.col("rec_id").is_in(list(matched_recs))))
    tset = pl.read_parquet(str(path("features_dir") / "train" / "part-*.parquet"),
                            columns=["s1_id", "rec_id", "name_tset", "addr_tset"])
    fp = fp.join(tset, on=["s1_id", "rec_id"], how="left").filter(
        (pl.col("name_tset") >= 80) & (pl.col("addr_tset") >= 80))
    fp = fp.join(lut.select("entity_id", "country"), left_on="rec_id", right_on="entity_id", how="left")
    fp = balanced_sample(fp, 30)
    for r in fp.iter_rows(named=True):
        f.write(f"{s1_row_str(idx, r['s1_id'])} | {rec_row_str(idx, r['rec_id'])} | DISTRACTOR\n")
    f.write("\n")


def top1_by_rec(v1: pl.DataFrame) -> dict:
    """rec_id -> top-1 s1_id by X_score, computed once (v1 is ~800MB; never re-filter it per row)."""
    top = (v1.select("rec_id", "s1_id", "X_score").sort("X_score", descending=True)
           .unique(subset=["rec_id"], keep="first"))
    return dict(top.select("rec_id", "s1_id").iter_rows())


def section5(f, lut: pl.DataFrame, idx: dict, gt: dict, v1: pl.DataFrame, wrong_lut: dict) -> None:
    """30 US no-address blocking misses: record has no address, true pair missing from v1 candidates,
    rank-1 wrong S1 by v1's X_score for that record."""
    f.write("## 5. US no-address blocking misses\n\n")
    f.write("record raw | true S1 raw | rank-1 wrong S1 raw\n")
    f.write("---|---|---\n")
    no_addr = lut.filter((pl.col("country") == "US") & (pl.col("business_address") == ""))
    pairs = [(s1, rec) for s1, matched in gt.items() for rec in matched]
    tp = pl.DataFrame({"s1_id": [p[0] for p in pairs], "rec_id": [p[1] for p in pairs]})
    tp = tp.join(no_addr.select(rec_id="entity_id"), on="rec_id", how="inner")
    v1_pairs = v1.select("s1_id", "rec_id").unique()
    missed = tp.join(v1_pairs.with_columns(_in=pl.lit(True)), on=["s1_id", "rec_id"], how="left").filter(
        pl.col("_in").is_null())
    sample = missed.sample(min(30, missed.height), seed=SEED)
    for r in sample.iter_rows(named=True):
        wrong = wrong_lut.get(r["rec_id"])
        wrong_str = s1_row_str(idx, wrong) if wrong else "(no v1 candidate for this record)"
        f.write(f"{rec_row_str(idx, r['rec_id'])} | {s1_row_str(idx, r['s1_id'])} | {wrong_str}\n")
    f.write("\n")


def vocab_depth_misses(gt: dict, v1: pl.DataFrame) -> pl.DataFrame:
    """True pairs not in candidates_train_v1, tagged vocab/depth by mine_dict's exact definition: vocab = the
    pair shares NO name/addr token with df <= blocking.df_cap; depth = shares >= 1 such token but was still
    missed (a ranking/depth-cut failure, not a vocab gap). Reuses mine_dict.shared logic but vectorizes the
    surviving-token lookup to avoid the cartesian join explosion."""
    s1 = pl.read_parquet(path("interim_dir") / "norm_train_s1.parquet",
                          columns=["entity_id", "country", "name_tokens", "addr_tokens"])
    a = s1.select("country", s1_id="entity_id",
                  **{f"a_{f}": pl.col(f"{f}_tokens").list.unique() for f in VD_FIELDS})
    b = pl.concat([pl.read_parquet(path("interim_dir") / f"norm_train_s{n}.parquet",
                                    columns=["entity_id", "name_tokens", "addr_tokens"]) for n in (2, 3)]
                  ).select(rec_id="entity_id",
                           **{f"b_{f}": pl.col(f"{f}_tokens").list.unique() for f in VD_FIELDS})

    pairs = [(s1_id, rec) for s1_id, matched in gt.items() for rec in matched]
    tp = pl.DataFrame({"s1_id": [p[0] for p in pairs], "rec_id": [p[1] for p in pairs]}).join(
        a, on="s1_id").join(b, on="rec_id").with_columns(pid=pl.int_range(pl.len(), dtype=pl.Int64))

    v1_pairs = v1.select("s1_id", "rec_id").unique().with_columns(_in=pl.lit(True))
    missed = tp.join(v1_pairs, on=["s1_id", "rec_id"], how="left").filter(pl.col("_in").is_null())

    # mine_dict's exact surviving-token table and shared() test (tokens in BOTH sides' lists, df <= cap)
    surv = (pl.scan_parquet(path("interim_dir") / "idf_train.parquet")
            .filter(pl.col("ch").is_in(list(VD_FIELDS.values())))
            .filter(pl.sum_horizontal("n1", "n2", "n3") <= CFG["blocking"]["df_cap"])
            .select("country", "ch", "tok").collect(engine="streaming"))
    return shared(missed, surv, "").with_columns(vocab=~pl.col("surv"))


def section6(f, lut: pl.DataFrame, idx: dict, gt: dict, v1: pl.DataFrame, wrong_lut: dict) -> None:
    """40 vocab misses + 40 depth misses among true pairs absent from v1, tagged by mine_dict's definition
    (see vocab_depth_misses)."""
    missed = vocab_depth_misses(gt, v1)
    vocab, depth = missed.filter(pl.col("vocab")), missed.filter(~pl.col("vocab"))

    for title, bucket in (("Vocab misses", vocab), ("Depth misses", depth)):
        f.write(f"## 6. {title} (true pairs not in v1)\n\n")
        f.write("record raw | true S1 raw | rank-1 wrong S1 raw\n")
        f.write("---|---|---\n")
        sample = balanced_sample(bucket, 40)
        for r in sample.iter_rows(named=True):
            wrong = wrong_lut.get(r["rec_id"])
            wrong_str = s1_row_str(idx, wrong) if wrong else "(no v1 candidate for this record)"
            f.write(f"{rec_row_str(idx, r['rec_id'])} | {s1_row_str(idx, r['s1_id'])} | {wrong_str}\n")
        f.write("\n")


def main() -> None:
    t0 = time.perf_counter()

    def log(step: str) -> None:
        now = time.perf_counter()
        print(f"[{now - t0:7.1f}s] {step}", flush=True)

    lut = raw_lookup()
    log("raw_lookup")
    verify_raw_matches_source(lut)
    log("verify_raw_matches_source")
    idx = lut_index(lut)
    log("lut_index")
    gt = load_gt()
    log("load_gt")
    bst, feats = booster()
    log("booster")
    oof = load_oof()
    log("load_oof")
    v1 = pl.read_parquet(path("interim_dir") / "candidates_train_v1.parquet")
    log("read candidates_train_v1")
    wrong_lut = top1_by_rec(v1)
    log("top1_by_rec")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Eyeball report (M5, seed 42)\n\n")
        section1(f, feats)
        log("section1")
        section2(f, oof, lut, idx, gt)
        log("section2")
        section3(f, oof, lut, idx, bst, feats)
        log("section3 (pred_contrib on 60 rows)")
        section4(f, oof, lut, idx, gt)
        log("section4")
        section5(f, lut, idx, gt, v1, wrong_lut)
        log("section5")
        section6(f, lut, idx, gt, v1, wrong_lut)
        log("section6")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
