"""krish-v2 Phase 4: average the stage-1 OOF of several cv_full seed runs (same folds and (b) eval world) and score it.

Runs: oof_full{tag}.parquet for each --tags entry ("" = the base run, "_s1" = CV_SEED_SHIFT=1, ...). Rows must align
(same _i); p_std and p_td are averaged. Scored with cv_full.score, same scope as cv_full (b).
-> oof_dir/cv_full_ens.json (a/b scores, best t, gain over the base run, keep = gain_b >= min_gain).
Run from code/business_entity_resolution/:  python -m src.seed_avg --tags "" _s1
"""
import argparse
import json

import numpy as np
import polars as pl

from .cv_full import SEED, Data, drop_mask, s1_table, score, truth_keys
from .io import StepLog, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--min-gain", type=float, default=0.001)
    a = ap.parse_args()
    log, O = StepLog(), path("oof_dir")
    oofs = [pl.read_parquet(O / f"oof_full{t}.parquet", columns=["_i", "p_std", "p_td"]) for t in a.tags]
    for o in oofs[1:]:
        assert o["_i"].equals(oofs[0]["_i"]), "OOF rows differ between runs"
        assert (o["p_td"].is_null() == oofs[0]["p_td"].is_null()).all(), "(b) eval worlds differ between runs"
    p_std = np.mean([o["p_std"].to_numpy() for o in oofs], axis=0).astype(np.float32)
    p_td = np.mean([o["p_td"].fill_null(np.nan).to_numpy() for o in oofs], axis=0).astype(np.float32)
    s1 = s1_table(0.02 if "_smoke" in a.tags[0] else 1.0)
    d = Data(s1)
    assert np.array_equal(d.i, oofs[0]["_i"].to_numpy())
    log("loaded", runs=len(oofs), rows=len(p_td))
    base = json.loads((O / f"cv_full{a.tags[0]}.json").read_text())
    res = {"tags": a.tags,
           "a_standard": score(s1, d, p_std, np.ones(s1.height, bool)),
           "b_test_density": score(s1, d, p_td, ~drop_mask(s1.height, SEED + 2000), exact=truth_keys(s1))}
    res["gain_b"] = res["b_test_density"]["macro_f05"] - base["b_test_density"]["macro_f05"]
    res["keep"] = bool(res["gain_b"] >= a.min_gain)
    res["features"] = base["features"]
    (O / "cv_full_ens.json").write_text(json.dumps(res, indent=1))
    for k in ("a_standard", "b_test_density"):
        print(k, {x: v for x, v in res[k].items() if x != "curve"}, flush=True)
    log("scored", b=round(res["b_test_density"]["macro_f05"], 5), gain_b=round(res["gain_b"], 5), keep=res["keep"])


if __name__ == "__main__":
    main()
