"""M5-1b sibling channel: recover blocking misses where a record has no/weak address but another record of
the same true S1 was retrieved confidently (docs/diagnose_m5.md D0-1 Full cross: US no-addr/name>=80/addr=none
is 17.4% of all misses, India's equivalent 9.3%; vocab-class misses inside those cells mean name tokens alone
don't survive the existing channels either -- this is confidence propagation across a name-similarity edge,
not a new channel score).

Anchors: a record r' whose v2 top-1 S1 s has pre-score margin (top1 - top2) >= gate.sib_margin. Margin is a
calibration-free confidence proxy: the pool already has pre-score per (s1, rec); "confident" = the runner-up
S1 is far behind, not that the score is high in absolute terms.
Sibling edges: record<->record, same country, sharing an exact core_name OR token_set_ratio >= 90 inside a
shared rarest-token block (reuses gate's tokenisation via the S2/S3 norm table -- no pairwise loop; blocks
> 200 records are skipped as chains, matching the "no O(n^2)" rule).
Expand: for record r linked to anchor r', propose (s, r) if absent from the v2 pool. Cap <= 3 sibling S1s per
r, highest anchor margin first, so one noisy anchor cannot flood a record's candidate list.
Emits sib_hit / sib_anchor_margin / n_sib_anchors / sib_name_tset for v2 pairs too (0 default): a real feature,
not just an extra channel, so the matcher can learn to distrust it.

Run from code/business_entity_resolution/:
  python -m src.sibling --split train --sweep         # per-country, per delta in gate.sib_margin sweep list ->
                                                        # artifacts/interim/sibling_train_d{delta}/ + eval rows
  python -m src.sibling --split test  --delta 0.10     # single delta, both splits share the chosen delta
  python -m src.sibling --eval                          # docs/blocking_v2.md: append "## Sibling channel"
                                                        # (added rows, cand/S1 delta, precision, PC, ceiling
                                                        # vs the v2 baseline 0.9885, US/India no-addr cell gain)

Keep rule (not automatic): ceiling gain >= +0.001 at <= +5 cand/S1, else log and do not union.
Union happens only inside `gate.py --apply` (reads sibling_{split}_d{delta}/ when gate.sibling.enabled is
true) so candidate_pairs.tsv stays exactly the scored set -- sibling.py itself never writes candidates_*.
"""
import argparse
import shutil

import numpy as np
import polars as pl
from rapidfuzz.fuzz import token_set_ratio
from rapidfuzz.process import cdist

from .block import CHANNELS, finalize, norm_path
from .gate import GCFG, N_MAX, VARIANTS, gate_bucket, grid, pool_dir, rec_best
from .io import CFG, ROOT, StepLog, load_gt_pairs, path

SCFG = CFG["gate"]  # sib_margin lives under gate: in config.yaml, alongside m/n/variant
PAIR = ["s1_id", "rec_id"]
NAME_BLOCK_MAX = 200  # skip name-token blocks bigger than this: a "chain" (near-stopword core_name), not signal
SIB_CAP = 3           # sibling-derived S1s per record, highest anchor margin first
TSET_MIN = 90
REPORT = ROOT / "docs" / "blocking_v2.md"


def sib_dir(split: str, delta: float):
    return path("interim_dir") / f"sibling_{split}_d{delta:.2f}"


def anchor_margins(split: str, m: int, log: StepLog) -> pl.DataFrame:
    """rec_id -> (s1_id, margin), margin = top1 - top2 pre-score over ALL records -- delta-independent (the
    611M-row blockgrid scan + gate_bucket join/sort/collect happens here, ONCE; anchors() just filters this
    by delta). Reuses gate_bucket's `pre` score (same pool the v1/v2 candidates come from) so an anchor's
    confidence is measured on the same scale the rest of the pipeline already trusts.
    gate_bucket's `pre` expr reads n_channels_hit, which only block.finalize() adds -- gate_bucket must
    never be called on a raw grid()/blockgrid scan (see gate.py's own build(), which always finalizes first)."""
    best = rec_best(split)
    log(f"{split} rec_best", records=best.height)
    # finalize adds n_channels_hit (gate_bucket's `pre` expr reads it) and cuts the grid to (m, pool_k), same
    # as gate.py's own build() -- gate_bucket must never see the raw blockgrid scan directly
    x = gate_bucket(finalize(grid(split), m, GCFG["pool_k"], CHANNELS), best)  # margin needs the runner-up, no n-cut yet
    log(f"{split} gate_bucket", rows=x.height)
    top2 = (x.sort(["rec_id", "pre"], descending=[False, True])
            .group_by("rec_id", maintain_order=True)
            .agg(s1_id=pl.col("s1_id").first(), top1=pl.col("pre").first(),
                 top2=pl.col("pre").slice(1, 1).first()))  # null for a single-row group (no runner-up) -- fill_null below
    log(f"{split} top2 groupby", records=top2.height)
    out = top2.with_columns(margin=(pl.col("top1") - pl.col("top2").fill_null(0))).select("rec_id", "s1_id", "margin")
    log(f"{split} anchor margins done", records=out.height)
    return out


def anchors(margins: pl.DataFrame, delta: float) -> pl.DataFrame:
    """anchor_margins(split, m), cut at delta -- the only delta-dependent step."""
    return margins.filter(pl.col("margin") >= delta)


def name_edges(split: str) -> pl.DataFrame:
    """Sparse record<->record edges within a country: exact core_name match, or token_set_ratio >= TSET_MIN
    inside a shared rarest-token block. No pairwise loop over all records -- only within a shared block, and
    blocks > NAME_BLOCK_MAX are dropped as chains (a near-stopword name would make every record "similar")."""
    recs = pl.concat([
        pl.scan_parquet(norm_path(split, s)).select(
            "entity_id", "country", "core_name",
            rarest=pl.col("name_tokens").list.eval(pl.element()).list.first())  # cheapest proxy for rarest token
        for s in (2, 3)
    ]).rename({"entity_id": "rec_id"}).collect(engine="streaming")

    exact = (recs.group_by(["country", "core_name"]).agg(pl.col("rec_id"))
             .filter(pl.col("rec_id").list.len().is_between(2, NAME_BLOCK_MAX)))
    exact_edges = (exact.explode("rec_id", empty_as_null=True).rename({"rec_id": "a"})
                   .join(exact.explode("rec_id", empty_as_null=True).rename({"rec_id": "b"}), on=["country", "core_name"])
                   .filter(pl.col("a") != pl.col("b")).select("country", "a", "b")
                   .with_columns(tset=pl.lit(100.0)))

    fuzzy_blocks = (recs.group_by(["country", "rarest"]).agg(pl.col("rec_id"), pl.col("core_name"))
                    .filter(pl.col("rec_id").list.len().is_between(2, NAME_BLOCK_MAX)))
    sizes = fuzzy_blocks["rec_id"].list.len().to_numpy()
    print(f"name blocks: {len(sizes)}, size histogram (log2 buckets): "
          f"{dict(zip(*np.unique(np.floor(np.log2(np.maximum(sizes, 1))).astype(int), return_counts=True)))}"
          if len(sizes) else "name blocks: 0")
    rows = []
    for country, block_recs, names in fuzzy_blocks.select("country", "rec_id", "core_name").iter_rows():
        m = cdist(names, names, scorer=token_set_ratio, score_cutoff=TSET_MIN, workers=-1, dtype=np.uint8)
        i, j = np.nonzero(np.triu(m, k=1))  # upper triangle only: skip diagonal and mirrored pairs
        if len(i):
            recs_arr = np.asarray(block_recs)
            rows.append(pl.DataFrame({"country": country, "a": recs_arr[i], "b": recs_arr[j],
                                       "tset": m[i, j].astype(np.float64)}))
    fuzzy_edges = pl.concat(rows) if rows else \
        pl.DataFrame(schema={"country": pl.Utf8, "a": pl.Int64, "b": pl.Int64, "tset": pl.Float64})
    return pl.concat([exact_edges, fuzzy_edges]).unique(["a", "b"])


SIB_FEATS = ["sib_hit", "sib_anchor_margin", "n_sib_anchors", "sib_name_tset"]


def sibling_support(split: str, delta: float, margins: pl.DataFrame, edges: pl.DataFrame, log: StepLog) -> pl.DataFrame:
    """(s1_id, rec_id, sib_anchor_margin, n_sib_anchors, sib_name_tset), capped SIB_CAP per rec_id, for every
    pair with sibling support -- whether or not it's already in the v2 pool. `new` (added-only) is `expand`'s
    anti-join of this against the pool. `edges` (name_edges(split)) and `margins` (anchor_margins(split, m, log))
    are both delta-independent -- callers compute them ONCE and pass them in; only the anchors() cut is
    per-delta (a cheap in-memory filter, not a 611M-row blockgrid rescan)."""
    anc = anchors(margins, delta)
    log(f"{split} d{delta} anchors/edges", anchors=anc.height, edges=edges.height)
    linked = pl.concat([
        edges.rename({"a": "rec_id", "b": "nbr"}),
        edges.rename({"b": "rec_id", "a": "nbr"}).select("country", "rec_id", "nbr", "tset"),
    ]).join(anc.rename({"rec_id": "nbr"}), on="nbr")  # nbr is the anchor record

    return (linked.group_by(["rec_id", "s1_id"])
            .agg(sib_anchor_margin=pl.col("margin").max(), n_sib_anchors=pl.len().cast(pl.Int32),
                 sib_name_tset=pl.col("tset").max())
            .sort(["rec_id", "sib_anchor_margin"], descending=[False, True])
            .with_columns(pl.int_range(1, pl.len() + 1).over("rec_id").alias("_rk"))
            .filter(pl.col("_rk") <= SIB_CAP).drop("_rk"))


def expand(cand: pl.DataFrame, split: str, m: int, log: StepLog) -> pl.DataFrame:
    """(s1_id, rec_id, sib_anchor_margin, n_sib_anchors, sib_name_tset) for pairs NOT already in the v2 pool."""
    existing = pl.scan_parquet(pool_dir(split, m) / "*.parquet").select(*PAIR).collect(engine="streaming")
    new = cand.join(existing, on=PAIR, how="anti").with_columns(sib_hit=pl.lit(True))
    log(f"{split} new pairs (not already in v2)", rows=new.height)
    return new


def all_features(cand: pl.DataFrame, split: str, m: int) -> pl.DataFrame:
    """sib_* for EVERY v2 pool pair (0 default) union every sibling-supported pair (added or not) -- keyed on
    (s1_id, rec_id), one row per pair. Consumed by the features stage so sib_* reaches the model as columns,
    not just as extra candidate rows."""
    cand = cand.with_columns(sib_hit=pl.lit(True))
    pool = pl.scan_parquet(pool_dir(split, m) / "*.parquet").select(*PAIR).collect(engine="streaming")
    return (pool.join(cand, on=PAIR, how="full", coalesce=True)
            .with_columns(pl.col("sib_hit").fill_null(False), pl.col("sib_anchor_margin").fill_null(0.0),
                          pl.col("n_sib_anchors").fill_null(0), pl.col("sib_name_tset").fill_null(0.0))
            .select(*PAIR, *SIB_FEATS))


def build(split: str, delta: float, margins: pl.DataFrame, edges: pl.DataFrame, log: StepLog) -> None:
    print(f"build start: split={split} delta={delta}", flush=True)
    m = GCFG["m"]
    cand = sibling_support(split, delta, margins, edges, log)
    log(f"{split} d{delta} sibling_support", rows=cand.height)
    d = sib_dir(split, delta)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    expand(cand, split, m, log).write_parquet(d / "part-000.parquet")
    print(f"build wrote part-000.parquet: split={split} delta={delta}", flush=True)
    all_features(cand, split, m).write_parquet(d / "features.parquet")
    print(f"build wrote features.parquet: split={split} delta={delta}", flush=True)


def evaluate(log: StepLog) -> None:
    """Train-only: for each sib_margin candidate, added rows / cand-S1 delta / precision / PC / ceiling vs the
    v2 baseline (0.9885), plus the gain restricted to the US/India no-address cells (D0-1's biggest miss cells).
    Keep rule (manual, not auto-applied): ceiling +>=0.001 at <=+5 cand/S1."""
    from .gate import ceiling  # local import: eval-only helper, avoid pulling gate's argparse main in
    s1 = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    gt = load_gt_pairs().select("s1_id", rec_id="match_id").join(s1, on="s1_id")
    ntrue = s1.join(gt.group_by("s1_id").len("ntrue"), on="s1_id", how="left").select(
        "s1_id", pl.col("ntrue").fill_null(0))
    base_hits = (pl.scan_parquet(pool_dir("train", GCFG["m"]) / "*.parquet")
                 .filter(pl.col(f"gr_{GCFG['variant']}") <= GCFG["n"]).select(*PAIR)
                 .join(gt.lazy(), on=PAIR, how="inner").collect(engine="streaming"))
    base_ceiling = ceiling(ntrue, base_hits.group_by("s1_id").len("h"))
    no_addr_s1 = set(gt.join(  # crude proxy for D0-1's no-address cell without re-reading normalise internals
        pl.scan_parquet(norm_path("train", 2)).select(entity_id="entity_id", has_addr=pl.col("addr_tokens").list.len() > 0)
        .rename({"entity_id": "rec_id"}).collect(engine="streaming").filter(~pl.col("has_addr"))["rec_id"], on="rec_id")["s1_id"])

    rows = []
    for delta in SCFG["sib_margin"]:
        d = sib_dir("train", delta)
        if not d.exists():
            continue
        new = pl.scan_parquet(d / "*.parquet").collect(engine="streaming")
        hit = new.join(gt, on=PAIR, how="inner")
        combined = pl.concat([base_hits.select(*PAIR), hit.select(*PAIR)]).unique()
        comb_ceiling = ceiling(ntrue, combined.group_by("s1_id").len("h"))
        cell_gain = (ceiling(ntrue.filter(pl.col("s1_id").is_in(no_addr_s1)),
                              combined.filter(pl.col("s1_id").is_in(no_addr_s1)).group_by("s1_id").len("h"))
                     - ceiling(ntrue.filter(pl.col("s1_id").is_in(no_addr_s1)),
                               base_hits.filter(pl.col("s1_id").is_in(no_addr_s1)).group_by("s1_id").len("h")))
        added_per_s1 = new.group_by("s1_id").len()["len"]
        rows.append({"delta": delta, "added_rows": new.height, "precision": float(hit.height / max(new.height, 1)),
                     "cand_per_s1_delta": float(added_per_s1.mean() or 0.0),
                     "ceiling_v2": base_ceiling, "ceiling_with_sibling": comb_ceiling,
                     "ceiling_gain": comb_ceiling - base_ceiling, "no_addr_cell_gain": cell_gain,
                     "keep": (comb_ceiling - base_ceiling) >= 0.001 and (added_per_s1.mean() or 0.0) <= 5})
        log(f"sibling eval delta={delta}", **rows[-1])

    from .decide import md
    L = ["", "## Sibling channel (M5-1b)",
         "Train-only. Added rows/precision/ceiling are v2 (m/n/variant from config) union sibling, at each "
         "gate.sib_margin candidate. Keep rule: ceiling gain >= +0.001 at <= +5 cand/S1 -- not auto-applied; "
         "union only happens via `gate.py --apply` when gate.sibling.enabled is set by hand after reading this.",
         "", *md(rows), ""]
    with REPORT.open("a") as f:
        f.write("\n".join(L))
    log("sibling eval report appended", path=str(REPORT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"))
    ap.add_argument("--delta", type=float)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    log = StepLog()
    if a.eval:
        evaluate(log)
    elif a.sweep:
        assert a.split, "--sweep needs --split"
        # both delta-independent: compute once, reuse across the whole sweep instead of once per delta
        edges = name_edges(a.split)
        log(f"{a.split} name edges", edges=edges.height)
        margins = anchor_margins(a.split, GCFG["m"], log)
        for delta in SCFG["sib_margin"]:
            build(a.split, delta, margins, edges, log)
    else:
        assert a.split and a.delta is not None, "need --split and --delta (or --sweep / --eval)"
        build(a.split, a.delta, anchor_margins(a.split, GCFG["m"], log), name_edges(a.split), log)
    log.dump(path("artifacts_dir") / "logs" / f"sibling_timing_{a.split or 'eval'}.json")


if __name__ == "__main__":
    main()
