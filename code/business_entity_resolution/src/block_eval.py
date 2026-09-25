"""M3b blocking evaluation (train) -> docs/blocking.md.

Sections: phonetic-skeleton check, PC by segment x channel, budget curve (m, k, df cap), candidate volume
+ RR, runtime/memory per step, missed true pairs by likely cause + 50 examples.
Needs (run from code/business_entity_resolution/):
  python -m src.block --split train                 # cap = config df_cap: candidates + grid + idf
  python -m src.block --split train --df-cap 500    # optional cap-sweep rows (missing caps are reported as such)
  python -m src.block --split train --df-cap 5000
  python -m src.block_eval
Shared-token flags are recomputed per country x source on the matched records only (not the full sources).
"""
import argparse
import json
import sys
import time
import traceback

import polars as pl

from .block import BCFG, CHANNELS, SOURCES, load, norm_path
from .io import ROOT, load_gt_pairs, path
from .normalise import peak_rss_mb

SEED = 42
N_EXAMPLES = 50
OUT = ROOT / "docs" / "blocking.md"
CAP, M, K = BCFG["df_cap"], BCFG["m"], BCFG["k"]
EXTRA = ("business_name", "business_address", "core_name", "legal_form", "has_address", "is_nonascii_raw")
# flag groups per channel: which shared tokens count for which flag, as a function of the token expr
# (a bare pl.col("tok") for the explode/join path, or pl.element() inside list.eval)
GROUPS = {"A": {"name": lambda t: t.is_not_null()},
          "B": {"addr": lambda t: ~t.str.starts_with("#"), "key": lambda t: t.str.starts_with("#")},
          "C": {"name_skel": lambda t: t.str.starts_with("n:"), "addr_skel": lambda t: t.str.starts_with("a:")}}
FLAGS = [f"{g}_{x}" for gs in GROUPS.values() for g in gs for x in ("any", "rare")]


def yn(e: pl.Expr) -> pl.Expr:
    return pl.when(e).then(pl.lit("yes")).otherwise(pl.lit("no"))


def ranks(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select("s1_id", "rec_id", mr=pl.min_horizontal(*(f"{c}_rrank" for c in CHANNELS)),
                     ms=pl.min_horizontal(*(f"{c}_srank" for c in CHANNELS)))


def grid_path(cap: int):
    return path("interim_dir") / f"blockgrid_train_cap{cap}.parquet"


def pair_ranks(keys: pl.DataFrame, cap: int) -> pl.DataFrame:
    return (keys.lazy().join(ranks(pl.scan_parquet(grid_path(cap))), left_on=["s1_id", "match_id"],
                             right_on=["s1_id", "rec_id"], how="left", maintain_order="left")
            .collect(engine="streaming"))


def base_pairs(sample: int = 0) -> pl.DataFrame:
    """True pairs + segment attributes + channel hit flags (config candidates) + grid ranks (config cap).
    `sample` draws N true pairs (seeded) BEFORE the candidates/grid joins, so those joins run against N
    rows instead of the full ~7.6M -- the two big frames (278M-row candidates, 500M-row grid) otherwise
    dominate runtime regardless of how small the eventual sample is. Same rows either way (same seed on
    the same frame), so this only changes how fast we get there, not which pairs are picked.
    `card` (true-match count per S1) is computed on the FULL pair set before sampling: it's a property
    of the S1 entity's whole ground truth, not of the sampled rows, so sampling first would silently
    undercount cardinality for any S1 whose matches split across the sample boundary."""
    full = load_gt_pairs()
    card = full.group_by("s1_id").agg(card=pl.len())
    p = full.sample(min(sample, full.height), seed=SEED) if sample else full
    s1 = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country", "legal_form"])
    cand = pl.scan_parquet(path("interim_dir") / "candidates_train.parquet").select(
        "s1_id", "rec_id", *(pl.col(f"{c}_score").is_not_null().alias(f"h{c}") for c in CHANNELS))
    p = (p.join(card, on="s1_id").join(s1, left_on="s1_id", right_on="entity_id")
         .with_columns(src=pl.col("match_id").str.slice(0, 2),
                       card_bin=pl.when(pl.col("card") >= 5).then(pl.lit("5+")).otherwise(pl.col("card").cast(pl.String))))
    p = (p.lazy().join(cand, left_on=["s1_id", "match_id"], right_on=["s1_id", "rec_id"], how="left",
                       maintain_order="left").collect(engine="streaming")
         .with_columns(pl.col(*(f"h{c}" for c in CHANNELS)).fill_null(False)))
    return pair_ranks(p, CAP)


def shared_flags(x: pl.DataFrame, idf_c: pl.DataFrame) -> pl.DataFrame:
    """Per pair id: any shared token / any shared token with df <= CAP, per flag group.
    No explode/join/group_by: a per-channel rare-token frozenset gives an O(1) membership test, applied
    elementwise inside `list.eval` directly on the (already small, per country x source) shared-token
    lists. This replaces an eager explode + hash-join + group_by per channel (12x total) with one lazy
    `with_columns` pass, cutting both the row-count blowup from exploding and the join overhead."""
    cols = []
    for ch, groups in GROUPS.items():
        rare_set = frozenset(idf_c.filter((pl.col("ch") == ch) & (pl.col("df") <= CAP))["tok"].to_list())
        shared = pl.col(ch).list.set_intersection(pl.col(f"{ch}_r"))
        for g, cond in groups.items():
            match = cond(pl.element())
            cols += [shared.list.eval(match).list.any().fill_null(False).alias(f"{g}_any"),
                     shared.list.eval(match & pl.element().is_in(rare_set)).list.any().fill_null(False)
                     .alias(f"{g}_rare")]
    return x.select("pid", *cols)


def token_pairs(p: pl.DataFrame, t0: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Adds shared-token flags + cause per pair; returns (all pairs, missed pairs with display columns)."""
    idf = pl.read_parquet(path("interim_dir") / "idf_train.parquet").with_columns(
        df=pl.sum_horizontal("n1", "n2", "n3"))
    keep, missed = [], []
    groups = p.group_by("country", "src", maintain_order=True)
    n_groups = p.select("country", "src").n_unique()
    for i, ((c, src), g) in enumerate(groups, 1):
        tag = f"2/8 token_pairs [{i}/{n_groups}] {c} {src} ({g.height:,} pairs)"
        n = int(src[1])
        _log(t0, f"{tag}: load S1 tokens")
        s1t = load("train", 1, c, extra=EXTRA, ids=g["s1_id"].unique())
        _log(t0, f"{tag}: load {src} tokens")
        rt = load("train", n, c, extra=EXTRA, ids=g["match_id"].unique())
        _log(t0, f"{tag}: join")
        x = (g.drop("legal_form").join(s1t, left_on="s1_id", right_on="entity_id")
             .join(rt, left_on="match_id", right_on="entity_id", suffix="_r").with_row_index("pid"))
        del s1t, rt
        _log(t0, f"{tag}: shared-token flags")
        x = x.with_columns(shared_flags(x, idf.filter(pl.col("country") == c)).drop("pid"))
        any_rare = pl.any_horizontal(*(f"{g}_rare" for gs in GROUPS.values() for g in gs))
        any_shared = pl.any_horizontal(*(f"{g}_any" for gs in GROUPS.values() for g in gs))
        in_grid = pl.col("mr").is_not_null() | pl.col("ms").is_not_null()
        x = x.with_columns(
            hit=pl.any_horizontal(*(f"h{c}" for c in CHANNELS)),
            native=yn(pl.col("is_nonascii_raw_r")), has_addr=yn(pl.col("has_address_r")),
            legal_drop=pl.when(pl.col("legal_form") == "").then(pl.lit("S1 has none"))
            .otherwise(yn(pl.col("legal_form_r") == "")),
            cause=pl.when(any_rare & in_grid).then(pl.lit(f"rank cut, inside grid (m<={max(BCFG['sweep_m'])} or k<={max(BCFG['sweep_k'])})"))
            .when(any_rare).then(pl.lit("rank cut, beyond grid"))
            .when(any_shared).then(pl.lit("shared tokens all over df cap (unlinked, or fallback-linked + rank cut)"))
            .when(~pl.col("has_address_r")).then(pl.lit("no shared token, match has no address"))
            .when(pl.col("is_nonascii_raw_r")).then(pl.lit("no shared token, native script"))
            .otherwise(pl.lit("no shared token, ASCII")))
        keep.append(x.select("country", "src", "card_bin", "native", "has_addr", "legal_drop", *(f"h{c}" for c in CHANNELS),
                             "hit", "mr", "ms", "cause", *FLAGS))
        missed.append(x.filter(~pl.col("hit")).select(
            "cause", "country", "src", "business_name", "business_address", "business_name_r", "business_address_r",
            "core_name", "core_name_r", "B", "B_r", "C", "C_r"))
        _log(t0, f"{tag}: done, peak {peak_rss_mb()} MB")
    return pl.concat(keep), pl.concat(missed)


def pct(e: pl.Expr) -> pl.Expr:
    return (e.mean() * 100).round(2)


def pc_aggs() -> list[pl.Expr]:
    a, b, c = pl.col("hA"), pl.col("hB"), pl.col("hC")
    return [pl.len().alias("pairs"), pct(a).alias("A %"), pct(b).alias("B %"), pct(c).alias("C %"),
            pct(a | b).alias("A∪B %"), pct(a | b | c).alias("A∪B∪C %"),
            (((a | b | c).mean() - (a | b).mean()) * 100).round(2).alias("C marginal pp"),
            pct(pl.col("hX")).alias("X %"), pct(pl.col("hit")).alias("all = PC %")]


def md(df: pl.DataFrame) -> list[str]:
    def cell(v) -> str:
        if isinstance(v, float):
            s = f"{v:,.2f}"
        elif isinstance(v, int) and not isinstance(v, bool):
            s = f"{v:,}"
        else:
            s = ", ".join(v) if isinstance(v, list) else str(v)
        return s.replace("|", "\\|")
    return (["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
            + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.iter_rows()])


def sizes() -> pl.DataFrame:
    """Per country: N1 and N2+N3 (train)."""
    n = [pl.scan_parquet(norm_path("train", s)).group_by("country").agg(pl.len().alias(f"n{s}")).collect()
         for s in (1, *SOURCES)]
    return n[0].join(n[1], on="country", how="full", coalesce=True).join(n[2], on="country", how="full", coalesce=True).fill_null(0)


def timing(cap: int) -> dict | None:
    f = path("interim_dir") / f"block_timing_train_cap{cap}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _log(t0: float, msg: str) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="evaluate a seeded sample of N true pairs (fast)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    _log(t0, "1/8 base_pairs: start")
    p = base_pairs(a.sample)
    _log(t0, f"1/8 base_pairs: done, {p.height:,} rows")
    pairs, missed = token_pairs(p, t0)
    del p
    n_pairs = pairs.height
    _log(t0, f"2/8 token_pairs: done, {n_pairs:,} pairs, {missed.height:,} missed")
    sz = sizes()
    n_s1 = int(sz["n1"].sum())
    L = ["# Blocking v1 evaluation (M3b)",
         f"Generated by `src/block_eval.py` (train, seed {SEED}"
         + (f", SAMPLE of {a.sample:,} true pairs: pair %s are sampled, volume/RR are exact" if a.sample else "")
         + f"). Config: df_cap={CAP}, m={M} (record-centric "
         f"top-m S1 per record per channel), k={K} (S1-centric top-k records per S1 per channel per source). "
         "Channels: A core-name tokens, B address tokens + exact number+street key, C phonetic skeletons "
         "(name `n:` + address `a:`). Score = sum of shared IDF (per country, S1+S2+S3 of the split). "
         f"Bigrams={BCFG['bigrams']}, rarest-2 fallback (A, C)={BCFG['fallback_rarest']}. Legal form is not used. PC = true pairs in candidates / all true pairs. "
         "Native script, has-address and legal form dropped describe the matched S2/S3 record.", ""]

    _log(t0, "3/8 phonetic table")
    L += ["## 1. Phonetic skeleton check (all train true pairs)",
          f"% of true pairs sharing ≥1 rare token (df ≤ {CAP}). normal = the A/B token itself; skel = its "
          "phonetic skeleton; name tokens include adjacent bigrams when `blocking.bigrams` is on. skel-only = rare "
          "skeleton shared but no rare normal token shared.", ""]
    ph = (pairs.group_by("country", "native").agg(
        pl.len().alias("pairs"), pct(pl.col("name_rare")).alias("name normal %"),
        pct(pl.col("name_skel_rare")).alias("name skel %"), pct(pl.col("addr_rare")).alias("addr normal %"),
        pct(pl.col("addr_skel_rare")).alias("addr skel %"),
        pct(pl.col("name_rare") | pl.col("addr_rare") | pl.col("key_rare")).alias("any normal %"),
        pct(pl.col("name_skel_rare") | pl.col("addr_skel_rare")).alias("any skel %"),
        pct((pl.col("name_skel_rare") | pl.col("addr_skel_rare"))
            & ~(pl.col("name_rare") | pl.col("addr_rare") | pl.col("key_rare"))).alias("skel-only %"))
        .sort("country", "native"))
    L += md(ph) + [""]

    _log(t0, "4/8 PC by segment")
    segs = [("all", pl.lit("all")), ("country", pl.col("country")),
            ("country × native script", pl.concat_str("country", pl.lit(" / native "), "native")),
            ("cardinality (true matches of S1)", pl.col("card_bin")), ("native script", pl.col("native")),
            ("match has address", pl.col("has_addr")), ("legal form dropped", pl.col("legal_drop")),
            ("source", pl.col("src"))]
    seg = pl.concat([pairs.group_by(e.alias("value")).agg(pc_aggs()).sort("value")
                     .select(pl.lit(name).alias("segment"), pl.all()) for name, e in segs])
    L += ["## 2. Pair completeness by segment and channel (config m, k, df_cap)",
          "Channel columns = PC if only that channel's kept pairs were candidates (both directions). "
          "**C marginal pp** = PC(A∪B∪C) − PC(A∪B); the India / native yes row is the key number.", ""]
    L += md(seg) + [""]

    _log(t0, "5/8 budget curve (scans the grid parquet)")
    grid = (ranks(pl.scan_parquet(grid_path(CAP))).group_by("mr", "ms").len().collect(engine="streaming"))

    def cut(m: int, k: int) -> pl.Expr:
        return (pl.col("mr") <= m).fill_null(False) | (pl.col("ms") <= k).fill_null(False)
    rows = []
    for m in BCFG["sweep_m"]:
        for k in BCFG["sweep_k"]:
            n_c = int(grid.filter(cut(m, k))["len"].sum())
            rows.append({"m": m, "k": k, "PC %": round(pairs.select(pct(cut(m, k))).item(), 2),
                         "candidates / S1": round(n_c / n_s1, 1), "total pairs": n_c})
    L += [f"## 3. Budget curve", f"### 3a. m × k at df_cap={CAP} (from `blockgrid_train_cap{CAP}.parquet`)", ""]
    L += md(pl.DataFrame(rows)) + [""]
    rows = []
    for cap in BCFG["sweep_df_cap"]:
        if not grid_path(cap).exists():
            rows.append({"df_cap": cap, "PC %": None, "candidates / S1": None, "block runtime s": None,
                         "peak RSS MB": None, "note": f"not run: python -m src.block --split train --df-cap {cap}"})
            continue
        pr = pairs if cap == CAP else pair_ranks(load_gt_pairs(), cap)
        g = ranks(pl.scan_parquet(grid_path(cap))).filter(cut(M, K)).select(pl.len()).collect(engine="streaming").item()
        tm = timing(cap) or {}
        rows.append({"df_cap": cap, "PC %": round(pr.select(pct(cut(M, K))).item(), 2),
                     "candidates / S1": round(g / n_s1, 1), "block runtime s": tm.get("total_s"),
                     "peak RSS MB": tm.get("peak_rss_mb"), "note": ""})
    L += [f"### 3b. df cap at m={M}, k={K} (runtime/peak RSS = the whole `src.block` run incl. the m/k grid)", ""]
    L += md(pl.DataFrame(rows)) + [""]

    _log(t0, "6/8 candidate volume (scans candidates parquet)")
    cands = pl.scan_parquet(path("interim_dir") / "candidates_train.parquet")
    per_s1 = (pl.scan_parquet(norm_path("train", 1)).select("entity_id", "country")
              .join(cands.group_by("s1_id").len(), left_on="entity_id", right_on="s1_id", how="left")
              .with_columns(pl.col("len").fill_null(0)).collect(engine="streaming"))
    vol = (per_s1.group_by("country").agg(
        pl.len().alias("S1"), pl.col("len").sum().alias("candidate pairs"),
        pl.col("len").mean().round(1).alias("cand / S1 mean"), pl.col("len").quantile(0.99).alias("cand / S1 p99"),
        pct(pl.col("len") == 0).alias("S1 with 0 cand %"))
        .join(sz, on="country").with_columns(
        (100 * (1 - pl.col("candidate pairs") / (pl.col("n1") * (pl.col("n2") + pl.col("n3"))))).round(4)
        .alias("RR within country %")).drop("n1", "n2", "n3").sort("country"))
    tot = int(per_s1["len"].sum())
    all_pairs = n_s1 * int(sz["n2"].sum() + sz["n3"].sum())
    within = int((sz["n1"] * (sz["n2"] + sz["n3"])).sum())
    L += ["## 4. Candidate volume (config)", ""] + md(vol) + [""]
    L += [f"- Total candidate pairs {tot:,}; per S1 mean {per_s1['len'].mean():.1f}, p99 {per_s1['len'].quantile(0.99):.0f}.",
          f"- RR vs all S1 × (S2∪S3) pairs: {100 * (1 - tot / all_pairs):.4f}%; vs same-country pairs: "
          f"{100 * (1 - tot / within):.4f}%.", ""]

    _log(t0, "7/8 runtime table")
    tm = timing(CAP)
    L += ["## 5. Runtime and memory (`src.block`, config cap)",
          "peak RSS = process high-water mark after the step (cumulative, monotone). product nnz = entries of "
          "the sparse score products actually computed in that step (before top-n).", ""]
    if tm:
        L += md(pl.DataFrame([{"step": s["step"], "s": s["s"], "peak RSS MB": s["peak_rss_mb"],
                               "product nnz": s.get("product_nnz"), "grid pairs": s.get("grid_pairs")}
                              for s in tm["steps"]])) + ["", f"- Total {tm['total_s']}s, peak RSS {tm['peak_rss_mb']} MB.", ""]
    else:
        L += ["(no timing file)", ""]

    _log(t0, "8/8 missed-pair causes + examples")
    by_cause = (pairs.filter(~pl.col("hit")).group_by("cause", "country").len()
                .pivot("country", index="cause", values="len").fill_null(0)
                .with_columns(total=pl.sum_horizontal(pl.exclude("cause")))
                .with_columns((100 * pl.col("total") / pl.col("total").sum()).round(2).alias("% of missed"),
                              (100 * pl.col("total") / n_pairs).round(3).alias("% of all pairs"))
                .sort("total", descending=True))
    L += ["## 6. Missed true pairs by likely cause",
          f"Cause order (first match wins): a rare (df ≤ {CAP}) token is shared in some channel → the pair "
          "scored but was cut by m/k (inside/beyond the sweep grid); shared tokens exist but all have df > cap; "
          "nothing shared in any channel (split by empty match address / native script / ASCII).", ""]
    L += md(by_cause) + [""]
    counts = missed.group_by("cause").len().sort("len", "cause", descending=[True, False])
    ex = pl.concat([missed.filter(pl.col("cause") == c).sample(min(n, max(2, round(N_EXAMPLES * n / missed.height))), seed=SEED)
                    for c, n in counts.iter_rows()]).head(N_EXAMPLES)

    def skel(col: str, pre: str) -> pl.Expr:
        return pl.col(col).list.eval(pl.element().filter(pl.element().str.starts_with(pre)).str.slice(2)).list.sort().list.join(" ")
    ex = ex.select(
        "cause", "country",
        S1=pl.concat_str("business_name", pl.lit(" ‖ "), "business_address"),
        match=pl.concat_str("src", pl.lit(": "), "business_name_r", pl.lit(" ‖ "), "business_address_r"),
        core_name=pl.concat_str("core_name", pl.lit(" → "), "core_name_r"),
        addr_norm=pl.concat_str(pl.col("B").list.sort().list.join(" "), pl.lit(" → "), pl.col("B_r").list.sort().list.join(" ")),
        name_skel=pl.concat_str(skel("C", "n:"), pl.lit(" → "), skel("C_r", "n:")))
    L += [f"### {ex.height} sampled missed pairs (stratified by cause, seed {SEED}; `‖` separates name and address, "
          "`→` S1 → match; normalised address = B tokens incl. the `#number street` key)", ""]
    L += md(ex) + ["", f"Eval runtime {time.perf_counter() - t0:.0f}s, peak RSS {peak_rss_mb()} MB."]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    _log(t0, f"DONE wrote {OUT}, peak {peak_rss_mb()} MB")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print(f"FAILED, peak {peak_rss_mb()} MB", flush=True)
        sys.exit(1)
