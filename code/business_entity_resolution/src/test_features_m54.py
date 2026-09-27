"""Unit test for the M5-4 feature additions (src.features: addr_tset_alias/locality_rel, cov_*, num_range_hit/
num_absdiff_bucket, rec_name_novel/rec_name_all_novel, legal_*, twin_hits) and the locality alias mining helpers
in src.mine_dict. No artifacts required: normalises 10 raw rows drawn from docs/eyeball2.md's top-loss tables
in-process and asserts on the resulting feature values directly (not the full src.features pipeline, which
needs artifacts/interim parquet caches this test doesn't build).

Run from code/business_entity_resolution/:  python -m src.test_features_m54
"""
import pandas as pd
import polars as pl

from .features import (addr_tset_alias, cov, legal_rel, locality_rel, num_range, rec_name_novel, s1_name_vocab)
from .mine_dict import locality_toks
from .normalise import normalise

# entity_id, business_name, business_address, country -- pairs (S1, S2/S3) from docs/eyeball2.md
ROWS = [
    ("S1-1", "First Transit Dynamics Company", "570 Quarry Place Court, Reisterstown, MD", "US"),
    ("S2-1", "First Transit Dynamics Co.", "570 Quarry Place Ct, Reisterstown, Maryland", "US"),  # TP, exact addr
    ("S1-2", "Greater Noida Services Private Limited",
     "G-81, Delta-2, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh", "India"),
    ("S2-2", "Greater Noida Pvt Ltd Services", "G-81, Delta-2, Gautam Buddha Nagar, UP", "India"),  # TP, legal reordered
    ("S1-3", "Oncology Center Inc", "486 S Hill Road, Northumberland County, VA", "US"),
    ("S2-3", "Oncology Inc Center", "554-558 South Hill Road, Heathsville, Virginia", "US"),  # FN, num range vs 486
    ("S1-4", "Emerald Industries LLC", "1156 Front Street, Conway, AR", "US"),
    ("S2-4", "Emerald Llc Industries", "", "US"),  # FN, no address on the record
    ("S1-5", "Metech Group", "3411 Chokecherry Road, Lincoln, MT", "US"),
    ("S2-5", "Metech LLC Enterprises", "", "US"),  # FN, legal dropped + no address
]


def build():
    df = pd.DataFrame(ROWS, columns=["entity_id", "business_name", "business_address", "country"])
    n = normalise(pl.from_pandas(df))
    return {r["entity_id"]: r for r in n.to_dicts()}


def pair(n: dict, s1_id: str, rec_id: str) -> tuple[dict, dict]:
    return n[s1_id], n[rec_id]


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


def main() -> None:
    n = build()

    # ---- cov (group [cov]) ----
    a, b = pair(n, "S1-1", "S2-1")
    cov_rec, cov_s1 = cov(pl.Series([a["addr_tokens"]]), pl.Series([b["addr_tokens"]]))
    check("cov identical addr -> 1.0 both sides", cov_rec[0] == 1.0 and cov_s1[0] == 1.0,
          f"{cov_rec[0]}, {cov_s1[0]}")
    a4, b4 = pair(n, "S1-4", "S2-4")
    cov_rec4, cov_s14 = cov(pl.Series([a4["addr_tokens"]]), pl.Series([b4["addr_tokens"]]))
    check("cov: empty record addr -> 0.0 (no div-by-zero)", cov_rec4[0] == 0.0 and cov_s14[0] == 0.0)

    # ---- num_range (group [num2]) ----
    a3, b3 = pair(n, "S1-3", "S2-3")
    r = num_range(pl.Series([[]]), pl.Series([["554-558"]]), pl.Series([a3["addr_number"]]),
                 pl.Series([b3["addr_number"]]))
    check("num_range_hit: S1 number 486 inside record's 554-558? no", r["num_range_hit"][0] == 0)
    r2 = num_range(pl.Series([["484-488"]]), pl.Series([[]]), pl.Series(["486"]), pl.Series(["487"]))
    check("num_range_hit: 487 falls inside 484-488", r2["num_range_hit"][0] == 1)
    check("num_absdiff_bucket: |486-487|=1 -> bucket 1", r2["num_absdiff_bucket"][0] == 1)
    r3 = num_range(pl.Series([[]]), pl.Series([[]]), pl.Series([""]), pl.Series(["554"]))
    check("num_absdiff_bucket: missing S1 number -> bucket 6", r3["num_absdiff_bucket"][0] == 6)

    # ---- legal_rel (group [legal]) ----
    check("legal_rel: LLC vs LLC -> equal(1)",
          legal_rel(pl.Series(["llc"]), pl.Series(["llc"]))[0] == 1)
    check("legal_rel: Inc vs '' -> dropped_on_rec(3)",
          legal_rel(pl.Series(["inc"]), pl.Series([""]))[0] == 3)
    check("legal_rel: LLC vs Metech's '' record with LLC-in-name (S1-5 has no legal_form -> added_on_rec)",
          legal_rel(pl.Series([""]), pl.Series(["llc"]))[0] == 2)
    check("legal_rel: both none -> 0", legal_rel(pl.Series([""]), pl.Series([""]))[0] == 0)

    # ---- rec_name_novel (group [novel]) ----
    vocab = s1_name_vocab(pl.DataFrame({"country": ["US"], "name_toks": [["oncology", "center", "inc"]]}))
    novel = rec_name_novel(pl.Series([["oncology", "center", "brandnewword"]]), pl.Series(["US"]), vocab)
    check("rec_name_novel: 1 of 3 tokens novel -> ~0.333", abs(novel["rec_name_novel"][0] - 1 / 3) < 1e-6)
    check("rec_name_all_novel: not all novel -> 0", novel["rec_name_all_novel"][0] == 0)
    novel_all = rec_name_novel(pl.Series([["zzzz", "yyyy"]]), pl.Series(["US"]), vocab)
    check("rec_name_all_novel: all novel -> 1", novel_all["rec_name_all_novel"][0] == 1)

    # ---- locality alias (group [alias]) ----
    loc_map = {"reisterstown": "reisterstown", "delta": "delta"}  # identity stub, just exercises the code path
    alias_sim = addr_tset_alias(pl.Series([a["addr_tokens"]]), pl.Series([b["addr_tokens"]]), loc_map)
    check("addr_tset_alias runs and returns [0,100]", 0.0 <= alias_sim[0] <= 100.0, str(alias_sim[0]))
    check("addr_tset_alias: None map -> -1 sentinel",
          addr_tset_alias(pl.Series([a["addr_tokens"]]), pl.Series([b["addr_tokens"]]), None)[0] == -1.0)
    rel = locality_rel(pl.Series([["reisterstown"]]), pl.Series([["reisterstown"]]), {"x": "y"})
    check("locality_rel: raw shared locality token -> equal(1)", rel[0] == 1)
    rel2 = locality_rel(pl.Series([["springfield"]]), pl.Series([["springfeeld"]]),
                        {"springfeeld": "springfield"})
    check("locality_rel: only matches after remap -> alias(2)", rel2[0] == 2)
    rel3 = locality_rel(pl.Series([[]]), pl.Series([[]]), {"a": "b"})
    check("locality_rel: neither side has a locality token -> missing(0)", rel3[0] == 0)

    # ---- mine_dict.locality_toks (unigrams + adjacent bigrams) ----
    lt = locality_toks(pl.Series([["quarry", "place", "reisterstown"]]))
    toks = lt.to_list()[0]
    check("locality_toks: keeps unigrams", "reisterstown" in toks, str(toks))
    check("locality_toks: adds adjacent bigrams", any("_" in t for t in toks), str(toks))

    print(f"\n{sum(1 for _ in ROWS)} rows normalised OK; all M5-4 feature checks passed.")


if __name__ == "__main__":
    main()
