"""krish-v2 PREPARED, NOT SHIPPED (CLAUDE.md: never hard-code country; needs Krish's approval).

Label-free per-country threshold: for every country c, t_c = the smallest t >= global t at which c's predicted
empty-S1 rate on test reaches the mean empty rate of the training countries (US, India) at the global t; US/India themselves keep t. Rationale:
the generator constants are identical in US and India train (singleton 5.58%, 3.46 matches/S1), so an unseen
country predicted empty far less often than they are is over-matching. Countries already at/above the target keep t.

Reads oof_dir/test_p.parquet (stage-1 test p), cv_full.json best t (b); writes
output_krish/matching_results_countryt.tsv + experiments/country_t.json. Run from repo root with krish_env.sh sourced:
  .venv/bin/python experiments/country_t.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src.block import norm_path  # noqa: E402
from src.decide import with_top  # noqa: E402
from src.io import path, write_candidates  # noqa: E402

TRAIN_COUNTRIES = ["US", "India"]


def stats(top: pl.DataFrame, s1: pl.DataFrame, t: float) -> pl.DataFrame:
    m = top.filter(pl.col("p") >= pl.lit(t) if not isinstance(t, str) else pl.col("p") >= pl.col(t))
    per = s1.join(m.group_by("s1_id").len("k"), on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
    return per.group_by("country").agg(n_s1=pl.len(), empty_pct=(pl.col("k") == 0).mean() * 100,
                                       pred_per_s1=pl.col("k").mean()).sort("country")


def main() -> None:
    O = path("oof_dir")
    t = json.loads((O / "cv_full.json").read_text())["b_test_density"]["best_t"]
    s1 = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    top = with_top(pl.read_parquet(O / "test_p.parquet")).filter("top").join(s1, on="s1_id")
    base = stats(top, s1, t)
    target = float(base.filter(pl.col("country").is_in(TRAIN_COUNTRIES))["empty_pct"].mean())
    tc = {}
    for c in base["country"]:
        tc[c] = t
        if c in TRAIN_COUNTRIES:  # the reference countries keep the OOF-tuned t
            continue
        for tt in np.round(np.arange(t, 0.99, 0.01), 2):
            e = stats(top.filter(pl.col("country") == c), s1.filter(pl.col("country") == c), float(tt))["empty_pct"][0]
            tc[c] = float(tt)
            if e >= target:
                break
    top = top.with_columns(tc=pl.col("country").replace_strict(tc, return_dtype=pl.Float64))
    after = stats(top, s1, "tc")
    m = top.filter(pl.col("p") >= pl.col("tc")).select("s1_id", "rec_id")
    out = path("output_dir").parent / "output_krish" / "matching_results_countryt.tsv"
    write_candidates(out, m.lazy(), s1["s1_id"], list_col="matched_entity_ids")
    res = {"global_t": t, "target_empty_pct": target, "t_by_country": tc,
           "before": base.to_dicts(), "after": after.to_dicts(), "out": str(out)}
    Path(__file__).with_name("country_t.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
