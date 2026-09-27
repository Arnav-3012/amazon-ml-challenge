"""M5 loss breakdown: where the OOF (b) F0.5 loss comes from, per S1 and per segment. (No model, no pipeline change.)

World = cv_full's (b) test-density world: S1s kept by drop_mask(seed+2000), p = p_td, decide rule = cv_full.score
(argmax per record over rows present, ties -> lowest S1 code, then p >= t with t = cv_full.json b best_t).
Per S1, pair counts: tp, fp, fn_m (true, in candidates, not predicted: p < t or lost the record's argmax) and
fn_b (true, not in candidates_train_v1 = the rows oof_full scores). Loss 1 - F0.5 is split exactly (sums to the
loss) by the Shapley value over three fixes: blocking (fn_b -> tp), matcher-FN (fn_m -> tp), matcher-FP (fp -> 0).
Points = 100 x sum(share) / n_S1 in the world, i.e. macro F0.5 points; all buckets x segments add up to 100(1 - F).
Segments are S1-side:
  sibling density = # other train S1s of the country whose address token set overlaps the S1's by
    >= 80% of the larger set (5 |A∩B| >= 4 max(|A|, |B|)); empty address -> 0. All train S1s count, not only (b).
  name duplicated = another train S1 of the country with the same non-empty name_tokens.
  non-Latin name = business_name has a letter outside the Latin script.

Run from code/business_entity_resolution/:
  python -m src.loss_breakdown      # docs/loss_breakdown.md
"""
import json
from itertools import combinations
from math import ceil

import numpy as np
import polars as pl

from .block import norm_path
from .cv_full import drop_mask, s1_table
from .decide import f05_vec, md
from .features import id_key
from .io import CFG, ROOT, StepLog, load_gt_pairs, path, peak_rss_mb

SEED = CFG["seed"]
MAX_RSS_MB = 6_000
PAIR_BUDGET = 10_000_000  # candidate set pairs per sibling-join chunk (memory knob only)
NONLATIN = r"[^\P{L}\p{Latin}]"  # a letter that is not Latin script
BUCKETS = ("blocking-miss", "matcher-FN", "matcher-FP")
REPORT = ROOT / "docs" / "loss_breakdown.md"


def guard() -> None:
    if peak_rss_mb() > MAX_RSS_MB:
        raise MemoryError(f"peak RSS {peak_rss_mb()} MB > {MAX_RSS_MB}")


def shapley(tp: np.ndarray, fp: np.ndarray, fn_m: np.ndarray, fn_b: np.ndarray) -> np.ndarray:
    """n x 3 loss shares (blocking, matcher-FN, matcher-FP); rows sum to 1 - F0.5(actual)."""
    def f(fix: frozenset) -> np.ndarray:
        t = tp + (fn_b if 0 in fix else 0) + (fn_m if 1 in fix else 0)
        n_fp = 0 if 2 in fix else fp
        n_fn = (0 if 0 in fix else fn_b) + (0 if 1 in fix else fn_m)
        return f05_vec(t, t + n_fp, t + n_fn)
    F = {frozenset(s): f(frozenset(s)) for r in range(4) for s in combinations(range(3), r)}
    w = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}  # |S|! (3 - |S| - 1)! / 3!
    return np.stack([sum(w[len(s)] * (F[s | {i}] - F[s]) for s in F if i not in s) for i in range(3)], 1)


def siblings(s1: pl.DataFrame, log: StepLog) -> pl.DataFrame:
    """(entity_id, sib) via a prefix-filtered set-similarity self-join on distinct address token sets."""
    t = s1.select("entity_id", "country", toks=pl.col("addr_tokens").list.unique().list.sort()).with_columns(
        key=pl.col("toks").list.join("\x1f"), size=pl.col("toks").list.len())
    sets = (t.filter(pl.col("size") > 0).group_by("country", "key").agg(pl.first("toks", "size"), mult=pl.len())
            .sort("country", "key").with_row_index("sid").with_columns(pl.col("sid").cast(pl.Int64)))
    ex = sets.select("sid", "country", "size", tok="toks").explode("tok")
    ex = (ex.join(ex.group_by("country", "tok").len("df"), on=["country", "tok"]).sort("sid", "df", "tok")
          .with_columns(pos=pl.int_range(pl.len()).over("sid")))
    need = (4 * pl.col("size") + 4) // 5  # ceil(0.8 size): overlap needed even against an equal-size set
    pre = ex.filter(pl.col("pos") < pl.col("size") - need + 1).select("sid", "country", "tok", "size")
    est = int(pre.group_by("country", "tok").len().select((pl.col("len").cast(pl.Int64) ** 2).sum()).item())
    n_chunks = max(1, ceil(est / PAIR_BUDGET))
    log("sibling prefix", sets=sets.height, prefix_rows=pre.height, est_pairs=est, chunks=n_chunks)
    toks = ex.select("sid", "tok")
    found = []
    for c in range(n_chunks):
        cand = (pre.filter(pl.col("sid") % n_chunks == c)
                .join(pre, on=["country", "tok"], suffix="_r")
                .filter((pl.col("sid") < pl.col("sid_r"))
                        & (5 * pl.min_horizontal("size", "size_r") >= 4 * pl.max_horizontal("size", "size_r")))
                .select("sid", "sid_r", "size", "size_r").unique(["sid", "sid_r"]))
        ov = (cand.join(toks, on="sid").join(toks, left_on=["sid_r", "tok"], right_on=["sid", "tok"])
              .group_by("sid", "sid_r", "size", "size_r").len("ov"))
        found.append(ov.filter(5 * pl.col("ov") >= 4 * pl.max_horizontal("size", "size_r")).select("sid", "sid_r"))
        guard()
    hits = pl.concat(found)
    mult = sets.select("sid", "mult")
    nb = (pl.concat([hits, hits.select(sid="sid_r", sid_r="sid")])
          .join(mult, left_on="sid_r", right_on="sid").group_by("sid").agg(nb=pl.col("mult").sum()))
    sets = sets.join(nb, on="sid", how="left").select(
        "country", "key", sib=pl.col("mult") - 1 + pl.col("nb").fill_null(0))
    log("siblings", verified_set_pairs=hits.height)
    return t.join(sets, on=["country", "key"], how="left").select("entity_id", pl.col("sib").fill_null(0))


def main() -> None:
    log = StepLog()
    I = path("interim_dir")
    t = json.loads((path("oof_dir") / "cv_full.json").read_text())["b_test_density"]
    T, F_REF = t["best_t"], t["macro_f05"]

    # ---- (b) world S1s ----
    s1 = s1_table(1.0)
    # re-index: codes must be 0..n-1 over the world for bincount (sorted order kept, so argmax ties are unchanged)
    s1 = (s1.filter(pl.Series(~drop_mask(s1.height, SEED + 2000))).select("s1_id", "s1k", "country", "ntrue")
          .with_row_index("code"))
    assert s1.height == t["n_s1"], (s1.height, t["n_s1"])

    # ---- candidates: oof_full rows must be exactly candidates_train_v1 ----
    oof = pl.scan_parquet(path("oof_dir") / "oof_full.parquet")
    v1 = pl.scan_parquet(I / "candidates_train_v1.parquet")
    n_oof, n_v1 = (x.select(pl.len()).collect().item() for x in (oof, v1))
    gt = load_gt_pairs()
    gt_in_v1 = v1.select("s1_id", "rec_id").join(
        gt.lazy().select("s1_id", rec_id="match_id"), on=["s1_id", "rec_id"], how="semi").select(pl.len())
    n_gt_v1, n_pos = gt_in_v1.collect(engine="streaming").item(), oof.select(pl.col("label").sum()).collect().item()
    assert n_oof == n_v1 and n_gt_v1 == n_pos, (n_oof, n_v1, n_gt_v1, n_pos)
    log("candidates", rows=n_oof, true_in_v1=n_pos)

    # ---- per-S1 counts in the world ----
    code = s1.select("s1k", "code")
    kept = (oof.filter(pl.col("p_td") >= T).select("s1k", "reck", "label", "p_td").collect(engine="streaming")
            .join(code, on="s1k").sort(["reck", "p_td", "code"], descending=[False, True, False])
            .filter(pl.col("reck").is_first_distinct()))
    in_cand = (oof.filter(pl.col("label") == 1).select("s1k", "p_td").collect(engine="streaming")
               .join(code, on="s1k"))
    guard()
    n = s1.height
    bc = lambda c, m=None: np.bincount(c if m is None else c[m], minlength=n)
    kc, ky = kept["code"].to_numpy(), kept["label"].to_numpy() == 1
    tp, npred = bc(kc, ky), bc(kc)
    ic, ip = in_cand["code"].to_numpy(), in_cand["p_td"].to_numpy()
    n_in = bc(ic)
    ntrue = s1["ntrue"].to_numpy()
    fp, fn_m, fn_b = npred - tp, n_in - tp, ntrue - n_in
    assert (fn_m >= 0).all() and (fn_b >= 0).all()
    f = f05_vec(tp, npred, ntrue)
    assert abs(f.mean() - F_REF) < 1e-9, (f.mean(), F_REF)
    L = shapley(tp, fp, fn_m, fn_b)
    assert np.allclose(L.sum(1), 1 - f)
    pairs = [{"true pairs of world S1s": "not in candidates (blocking-miss)", "pairs": int(fn_b.sum())},
             {"true pairs of world S1s": "in candidates, p < t", "pairs": int((ip < T).sum())},
             {"true pairs of world S1s": "in candidates, p >= t, lost the record argmax",
              "pairs": int(fn_m.sum() - (ip < T).sum())},
             {"true pairs of world S1s": "predicted (tp)", "pairs": int(tp.sum())},
             {"true pairs of world S1s": "false positives (fp)", "pairs": int(fp.sum())}]
    del kept, in_cand
    log("attributed", f05=round(float(f.mean()), 6))

    # ---- segments ----
    raw = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country", "business_name", "name_tokens",
                                                           "addr_tokens", "has_address"])
    sib = siblings(raw, log)
    name = pl.col("name_tokens").list.join(" ")
    seg = (raw.with_columns(nk=name).with_columns(
        name_dup=(pl.col("nk") != "") & (pl.len().over("country", "nk") > 1),
        nonlatin=pl.col("business_name").str.contains(NONLATIN))
           .join(sib, on="entity_id").select(s1_id="entity_id", has_address="has_address", nonlatin="nonlatin",
                                             name_dup="name_dup", sib="sib"))
    del raw, sib
    card = pl.col("ntrue")
    S = (s1.join(seg, on="s1_id", how="left", maintain_order="left").with_columns(
        singleton=pl.when(card == 0).then(pl.lit("singleton")).otherwise(pl.lit("non-singleton")),
        cardinality=pl.when(card == 0).then(pl.lit("0")).when(card == 1).then(pl.lit("1"))
        .when(card <= 3).then(pl.lit("2-3")).when(card <= 6).then(pl.lit("4-6")).otherwise(pl.lit("7+")),
        sibling_density=pl.when(pl.col("sib") == 0).then(pl.lit("0")).when(pl.col("sib") == 1).then(pl.lit("1"))
        .otherwise(pl.lit("2+")),
        has_address=pl.col("has_address").cast(pl.String), non_latin_name=pl.col("nonlatin").cast(pl.String),
        name_dup_in_country=pl.col("name_dup").cast(pl.String))
         .with_columns(pl.Series(f"L_{b}", L[:, i] * 100 / n) for i, b in enumerate(BUCKETS))
         .with_columns(f=pl.Series(f)))
    assert S["sib"].null_count() == 0
    log("segments")

    DIMS = ["country", "singleton", "cardinality", "has_address", "non_latin_name", "sibling_density",
            "name_dup_in_country"]
    lc = [f"L_{b}" for b in BUCKETS]
    total = float(S.select(pl.sum_horizontal(lc).sum()).item())
    tot_rows = [{"bucket": b, "loss (F0.5 pts)": float(S[c].sum()), "share %": float(S[c].sum() / total * 100)}
                for b, c in zip(BUCKETS, lc)] + [{"bucket": "total", "loss (F0.5 pts)": total, "share %": 100.0}]
    sections, cells = [], []
    for d in DIMS:
        g = (S.group_by(d).agg(pl.len().alias("n_S1"), pl.col("f").mean().alias("segment F0.5"),
                               *(pl.sum(c) for c in lc))
             .with_columns(total=pl.sum_horizontal(lc)).sort("total", descending=True))
        rows = [{"segment": r[d], "n_S1": r["n_S1"], "% S1": float(r["n_S1"] / n * 100),
                 "segment F0.5": r["segment F0.5"], **{b: r[c] for b, c in zip(BUCKETS, lc)},
                 "total pts": r["total"], "% of loss": float(r["total"] / total * 100)} for r in g.iter_rows(named=True)]
        sections += [f"### {d}", "", *md(rows), ""]
        cells += [{"bucket": b, "dimension": d, "segment": r["segment"], "n_S1": r["n_S1"], "loss pts": r[b],
                   "% of loss": r[b] / total * 100} for r in rows for b in BUCKETS]
    top = sorted(cells, key=lambda r: -r["loss pts"])[:15]

    out = ["# OOF (b) loss breakdown",
           f"Generated by `src/loss_breakdown.py`. World: cv_full (b), {n:,} S1s, t = {T}, macro F0.5 "
           f"{f.mean():.6f} (reproduces oof/cv_full.json {F_REF:.6f}). Loss = 100 x (1 - F0.5) = {total:.4f} "
           "macro F0.5 points. Per S1, 1 - F0.5 is split exactly by the Shapley value over three fixes: "
           "blocking-miss (true pair not in candidates_train_v1 -> found), matcher-FN (true pair in candidates but "
           "p < t or lost its record's argmax -> found), matcher-FP (false positives -> removed). Singleton with a "
           "prediction: all loss is matcher-FP. Points are shares of the macro score, so every table's rows sum "
           "to its bucket totals. candidates_train_v1 rows = oof_full rows "
           f"({n_oof:,}), true pairs in v1 = oof positives ({n_pos:,}).", "",
           "Segments are the S1's own fields. Sibling density = # other train S1s of the country whose address "
           "token set overlaps by >= 80% of the larger set (all train S1s, not only the world); no address -> 0. "
           "Name duplicated = another train S1 of the country with the same non-empty normalised name. "
           "Non-Latin = business_name has a letter outside the Latin script.", "",
           "## 1. Totals", "", *md(tot_rows), "", "Pair counts:", "", *md(pairs), "",
           "## 2. By segment", "", *sections,
           "## 3. Top 15 bucket x segment cells", "",
           "Cells overlap across dimensions (each S1 is in one segment per dimension).", "", *md(top), "",
           f"Peak RSS: {peak_rss_mb()} MB.", ""]
    REPORT.write_text("\n".join(out))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / "loss_breakdown_timing.json")


if __name__ == "__main__":
    main()
