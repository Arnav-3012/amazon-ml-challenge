"""Standalone ID-dump side-outputs for src.e0. Read-only, no pipeline change, does not touch block_autopsy.py
or eyeball2.py -- reuses their own helpers so the sets are identical to what those scripts would report.

1) autopsy_missed.parquet: block_autopsy's missed-true-pairs set M (same sample: hash(s1_id, seed) % 10 == 0,
   true pairs absent from candidates_train_v1) + raw name/address text on both sides, has_addr /
   name_nonascii_raw of the record side, for e0's native-script and no-address subsets. Asserts row count
   == 28,260 (block_autopsy.md 2026-09-26 run).
2) eyeball2_pairs.parquet: 20k TP + 20k FP (seed) of eyeball2's own predicted set (oof_full.parquet p_td,
   per-record argmax, p >= T -- same rule as eyeball2.partition/cv_full.json b_test_density) + raw text on
   both sides, so e0.py needs no id_key/s1k/reck knowledge.

Run from code/business_entity_resolution/:  python -m src.dump_ids
"""
import numpy as np
import polars as pl

from .block import norm_path
from .block_autopsy import V1_PATH, in_sample
from .eyeball2 import T
from .features import id_key
from .gate import PAIR
from .io import CFG, StepLog, load_gt_pairs, path

SEED = CFG["seed"]
N_SAMPLE = 20_000
EXPECT_MISSED = 28_260
OUT_MISSED = path("interim_dir") / "autopsy_missed.parquet"
OUT_EYEBALL2 = path("interim_dir") / "eyeball2_pairs.parquet"


def missed_pairs() -> pl.DataFrame:
    """block_autopsy's M: true pairs of the sample not in v1, + raw text, country, has_addr, nonascii."""
    s1 = pl.scan_parquet(norm_path("train", 1)).select(s1_id="entity_id")
    S = s1.filter(in_sample()).collect(engine="streaming")
    gt = load_gt_pairs().select("s1_id", rec_id="match_id").filter(in_sample()).join(S, on="s1_id")
    v1 = pl.scan_parquet(V1_PATH).filter(in_sample()).select(*PAIR).collect(engine="streaming").with_columns(
        v1=pl.lit(True))
    M = gt.join(v1, on=PAIR, how="left").filter(pl.col("v1").is_null()).select(*PAIR)
    s1_txt = pl.scan_parquet(norm_path("train", 1)).select(
        s1_id="entity_id", s1_name="business_name", s1_addr="business_address")
    rec_txt = (pl.concat([pl.scan_parquet(norm_path("train", n)) for n in (2, 3)])
               .select(rec_id="entity_id", rec_name="business_name", rec_addr="business_address",
                       country="country", has_addr="has_address", nonascii="is_nonascii_raw"))
    return (M.join(s1_txt.collect(engine="streaming"), on="s1_id", how="left")
            .join(rec_txt.collect(engine="streaming"), on="rec_id", how="left"))


def eyeball2_pairs() -> pl.DataFrame:
    """20k TP + 20k FP (seed) of eyeball2's predicted set: per-record argmax of p_td, p >= T, + raw text."""
    x = (pl.scan_parquet(path("oof_dir") / "oof_full.parquet").select("s1k", "reck", "label", p="p_td")
         .filter(pl.col("p").is_not_null())
         .sort(["reck", "p", "s1k"], descending=[False, True, False])
         .with_columns(rk=pl.int_range(pl.len()).over("reck")).filter(pl.col("rk") == 0)
         .filter(pl.col("p") >= T).select("s1k", "reck", "label", "p").collect(engine="streaming"))
    tp, fp = x.filter(pl.col("label") == 1), x.filter(pl.col("label") == 0)
    out = []
    for part, n, salt in ((tp, N_SAMPLE, 0), (fp, N_SAMPLE, 1)):
        take = min(n, part.height)
        idx = part.sort("reck").with_row_index("_i")  # deterministic order before the seeded draw
        keep = np.random.default_rng(SEED + salt).choice(idx.height, take, replace=False)
        out.append(idx.filter(pl.col("_i").is_in(keep)).drop("_i"))
    E = pl.concat(out)

    s1_txt = pl.scan_parquet(norm_path("train", 1)).select(
        s1k=id_key("entity_id"), s1_name="business_name", s1_addr="business_address")
    rec_txt = (pl.concat([pl.scan_parquet(norm_path("train", n)) for n in (2, 3)])
               .select(reck=id_key("entity_id"), rec_name="business_name", rec_addr="business_address",
                       country="country"))
    return (E.join(s1_txt.collect(engine="streaming"), on="s1k", how="left")
            .join(rec_txt.collect(engine="streaming"), on="reck", how="left"))


def main() -> None:
    log = StepLog()
    M = missed_pairs()
    log("missed_pairs", n=M.height)
    assert M.height == EXPECT_MISSED, f"missed pairs {M.height} != expected {EXPECT_MISSED} -- sample drifted"
    M.write_parquet(OUT_MISSED)

    E = eyeball2_pairs()
    log("eyeball2_pairs", tp=int((E["label"] == 1).sum()), fp=int((E["label"] == 0).sum()))
    E.write_parquet(OUT_EYEBALL2)
    print(f"wrote {OUT_MISSED} ({M.height:,} rows), {OUT_EYEBALL2} ({E.height:,} rows)")


if __name__ == "__main__":
    main()
