"""France deep-dive: why does France underperform vs US+India on test (b)-style diagnostics?

Reads only: src.normalise (legal-form regex, LEX, transliteration), artifacts/interim/norm_test_s{1,2,3}.parquet
(the parquets normalise.py writes), raw dataset/test/test_source{1,2,3}.tsv, oof/test_p.parquet (s1_id, rec_id, p
over test candidate pairs). No pipeline output touched; writes only docs/france_deep.md.

country is joined from S1's own raw/normalised country (never used as a model feature elsewhere; here it only
splits the report). Record argmax = per rec_id the candidate row(s) with max p (ties kept, rare). Predicted
match = argmax row with p >= t. All test_p rows are candidate pairs already (post-blocking), so "% S1 zero
predictions" counts S1s with no surviving argmax-kept row, not blocking misses (no ground truth on test).

Run from code/business_entity_resolution/:
  python -m src.france_deep --smoke   # 1% of S1s, same code path, writes docs/france_deep.md (< 3 min)
  python -m src.france_deep           # full test set
"""
import argparse
import random
import re
from collections import Counter

import polars as pl

from .io import CFG, ROOT, StepLog, load_source, path, peak_rss_mb
from .normalise import LEX

T = 0.75
SEED = int(CFG["seed"])
OUT = ROOT / "docs" / "france_deep.md"
NORM = {n: path("interim_dir") / f"norm_test_s{n}.parquet" for n in (1, 2, 3)}
FIVE_DIGIT_RE = r"\b\d{5}\b"


def md_table(df: pl.DataFrame, fmt: str) -> str:
    """Markdown table without pandas/tabulate; floats formatted with `fmt` (e.g. '.3f')."""
    cell = lambda v: format(v, fmt) if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), "|".join(["---"] * df.width)]
    return "\n".join(rows + [" | ".join(cell(v) for v in r) for r in df.iter_rows()])
FRANCE_MARKER_RE = r"\(\s*france\s*\)"
MAX_RSS_MB = 18_000  # 24 GB machine; argmax over 60.9M test rows peaks ~10.3 GB


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        rss = self.rows[-1]["peak_rss_mb"]
        if rss > MAX_RSS_MB:
            raise SystemExit(f"STOP: peak RSS {rss} MB > {MAX_RSS_MB} MB after '{step}'")


def _edit_le2(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 2:
        return False
    if a == b:
        return True
    # cheap DP, strings are short tokens
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[lb] <= 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% of S1s, same code path, <3 min")
    args = ap.parse_args()
    log = Log()
    rng = random.Random(SEED)
    lines = ["# France deep dive: France vs US+India (test)\n"]

    # ---- load S1 (country lives here; join key for everything else) ----
    s1 = pl.read_parquet(NORM[1])
    raw1 = pl.from_pandas(load_source("test", 1))
    print("actual country values (raw S1):", raw1["country"].unique().sort().to_list())
    assert set(raw1["country"].unique()) <= {"US", "India", "France"}, "unexpected country in test S1"
    log("load_s1", n=s1.height)

    if args.smoke:
        keep_ids = set(s1["entity_id"].sample(fraction=0.01, seed=SEED).to_list())
        s1 = s1.filter(pl.col("entity_id").is_in(keep_ids))
        raw1 = raw1.filter(pl.col("entity_id").is_in(keep_ids))

    s1_country = s1.select("entity_id", "country").rename({"entity_id": "s1_id"})
    n_s1_by_country = s1_country.group_by("country").agg(n_s1=pl.len())

    # ---- test_p: join country lazily. Full run has no S1 filter (all countries needed for section A's
    # aggregates) -- 60M+ rows, so keep it a LazyFrame and only collect grouped aggregates / the France-only
    # slice, never the raw per-row table (that was the OOM: collecting all 60M joined rows eagerly). ----
    p_lf = pl.scan_parquet(path("oof_dir") / "test_p.parquet")
    if args.smoke:
        p_lf = p_lf.filter(pl.col("s1_id").is_in(keep_ids))
    p_lf = p_lf.join(s1_country.lazy(), on="s1_id", how="inner")
    log("load_test_p", n=p_lf.select(pl.len()).collect(engine="streaming").item())

    argmax_lf = (p_lf.sort(["rec_id", "p", "s1_id"], descending=[False, True, False])
                 .filter(pl.col("rec_id").is_first_distinct()))
    # France-only slice of argmax is small enough to collect eagerly -- every section past A only reads this
    fr_argmax = argmax_lf.filter(pl.col("country") == "France").collect(engine="streaming")
    log("argmax", n=fr_argmax.height)

    # ============================================================== A) volumes / density / p distribution
    # all group-bys below run lazily/streaming over p_lf/argmax_lf -- the full 60M-row table is never
    # collected, only these small per-(country[,s1_id]) aggregates are
    lines.append("## A) Volume, density, prediction rate at t=0.75\n")
    recs_per_s1 = p_lf.group_by(["country", "s1_id"]).agg(n_rec=pl.len()).collect(engine="streaming")
    matches_per_s1 = (argmax_lf.filter(pl.col("p") >= T).group_by(["country", "s1_id"])
                      .agg(n_match=pl.len()).collect(engine="streaming"))
    vol = (n_s1_by_country.join(recs_per_s1.group_by("country").agg(mean_rec_per_s1=pl.col("n_rec").mean()),
                                on="country")
           .join(matches_per_s1.group_by("country").agg(total_matches=pl.col("n_match").sum()), on="country")
           .with_columns(matches_per_s1=pl.col("total_matches") / pl.col("n_s1"))
           .join(s1_country.join(matches_per_s1.select("s1_id"), on="s1_id", how="anti")
                 .group_by("country").agg(zero_pred_s1=pl.len()), on="country", how="left")
           .with_columns(pct_zero_pred=100 * pl.col("zero_pred_s1").fill_null(0) / pl.col("n_s1")))
    lines.append(vol.select("country", "n_s1", "mean_rec_per_s1", "matches_per_s1", "pct_zero_pred")
                 .pipe(md_table, ".3f"))
    lines.append("")

    share_mid = (argmax_lf.select("country", "p").with_columns(mid=pl.col("p").is_between(0.3, 0.9))
                 .group_by("country").agg(pct_mid=100 * pl.col("mid").mean()).collect(engine="streaming"))
    lines.append("Share of records with max p in [0.3, 0.9]:\n")
    lines.append(md_table(share_mid, ".2f"))
    lines.append("")

    # France-only t where France matches/S1 == US+India rate
    us_in_rate = float(vol.filter(pl.col("country") != "France")
                       .select((pl.col("total_matches").sum() / pl.col("n_s1").sum())).item())
    n_s1_fr = int(n_s1_by_country.filter(pl.col("country") == "France")["n_s1"].item())
    ts = sorted(set(x / 1000 for x in range(20, 991, 5)))
    best_t, best_diff = None, None
    for t in ts:
        rate = fr_argmax.filter(pl.col("p") >= t).height / n_s1_fr
        diff = abs(rate - us_in_rate)
        if best_diff is None or diff < best_diff:
            best_t, best_diff = t, diff
    lines.append(f"US+India matches/S1 = {us_in_rate:.4f}. France t matching that rate: **t={best_t:.3f}** "
                f"(France matches/S1 at t=0.75 = {vol.filter(pl.col('country') == 'France')['matches_per_s1'].item():.4f}).\n")
    del matches_per_s1, recs_per_s1
    log("section_A_done")

    # ============================================================== B) legal forms
    lines.append("## B) Legal-form tokens: first/last, caught vs not\n")
    legal_re = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, LEX["France"].legal), key=len, reverse=True))
                          + r")\b")
    s2 = pl.read_parquet(NORM[2]).filter(pl.col("country") == "France")
    s3 = pl.read_parquet(NORM[3]).filter(pl.col("country") == "France")
    if args.smoke:
        s2, s3 = s2.sample(fraction=0.01, seed=SEED), s3.sample(fraction=0.01, seed=SEED)
    fr_s1 = s1.filter(pl.col("country") == "France")
    fr_raw1 = raw1.filter(pl.col("country") == "France")

    def edge_tokens(names: pl.Series, which: str) -> Counter:
        c = Counter()
        for n in names.to_list():
            toks = n.split()
            if toks:
                c[toks[0] if which == "first" else toks[-1]] += 1
        return c

    raw2 = pl.from_pandas(load_source("test", 2))
    raw3 = pl.from_pandas(load_source("test", 3))
    if args.smoke:
        raw2, raw3 = raw2.filter(pl.col("entity_id").is_in(s2["entity_id"])), \
            raw3.filter(pl.col("entity_id").is_in(s3["entity_id"]))
    fr_raw2 = raw2.filter(pl.col("country") == "France")
    fr_raw3 = raw3.filter(pl.col("country") == "France")

    for label, names in (("France S1 (raw)", fr_raw1["business_name"].str.to_lowercase()),
                         ("France records (raw, S2+S3)",
                          pl.concat([fr_raw2["business_name"], fr_raw3["business_name"]]).str.to_lowercase())):
        for which in ("first", "last"):
            top = edge_tokens(names, which).most_common(40)
            caught = [(tok, cnt, bool(legal_re.fullmatch(tok))) for tok, cnt in top]
            lines.append(f"**{label} — top 40 {which} tokens** (token, count, caught_by_legal_regex):\n")
            lines.append(", ".join(f"{t}({c},{'Y' if k else 'N'})" for t, c, k in caught))
            lines.append("")
    # % France names ending in an uncaught frequent token (freq = top-40 last-token list, not caught)
    last_counter = edge_tokens(fr_raw1["business_name"].str.to_lowercase(), "last")
    uncaught_freq = {t for t, _ in last_counter.most_common(40) if not legal_re.fullmatch(t)}
    pct_uncaught_end = 100 * fr_raw1["business_name"].str.to_lowercase().str.split(" ").list.last() \
        .is_in(uncaught_freq).mean()
    lines.append(f"% France S1 names ending in an uncaught frequent last-token: **{pct_uncaught_end:.2f}%**\n")
    log("section_B_done", peak=peak_rss_mb())

    # ============================================================== C) "(France)" marker
    lines.append("## C) \"(France)\" marker\n")
    s1_marker = fr_raw1["business_name"].str.to_lowercase().str.contains(FRANCE_MARKER_RE)
    rec_marker_n = (s2["business_name"].str.to_lowercase().str.contains(FRANCE_MARKER_RE).sum()
                    + s3["business_name"].str.to_lowercase().str.contains(FRANCE_MARKER_RE).sum())
    rec_marker_share = 100 * rec_marker_n / (s2.height + s3.height)
    lines.append(f"Share with marker: France S1 = {100 * s1_marker.mean():.3f}%, "
                f"France records = {rec_marker_share:.3f}%\n")
    fr_top1 = fr_argmax.join(fr_raw1.select(s1_id="entity_id", has_marker=s1_marker), on="s1_id")
    med = fr_top1.group_by("has_marker").agg(median_p=pl.col("p").median(), n=pl.len())
    lines.append(md_table(med, ".4f"))
    lines.append("")
    log("section_C_done")

    # ============================================================== D) numbers / 5-digit postal tokens
    lines.append("## D) 5-digit tokens (postal codes)\n")
    pct_s1_5d = 100 * fr_raw1["business_address"].str.contains(FIVE_DIGIT_RE).mean()
    rec_addr = pl.concat([fr_raw2["business_address"], fr_raw3["business_address"]])
    pct_rec_5d = 100 * rec_addr.str.contains(FIVE_DIGIT_RE).mean()
    lines.append(f"% France S1 addresses with a 5-digit token: {pct_s1_5d:.2f}%; "
                f"% France record addresses: {pct_rec_5d:.2f}%\n")

    rec_all = pl.concat([fr_raw2.select("entity_id", "business_name", "business_address"),
                        fr_raw3.select("entity_id", "business_name", "business_address")]) \
        .rename({"entity_id": "rec_id", "business_name": "rec_name", "business_address": "rec_address"})
    s1_all = fr_raw1.select(s1_id="entity_id", s1_name="business_name", s1_address="business_address")
    fr_pairs = fr_argmax.join(s1_all, on="s1_id").join(rec_all, on="rec_id")
    fr_pairs = fr_pairs.with_columns(s1_5d=pl.col("s1_address").str.extract(FIVE_DIGIT_RE, 0))
    with_5d = fr_pairs.filter(pl.col("s1_5d").is_not_null())
    with_5d = with_5d.with_columns(
        rec_has_s1_5d=pl.struct(["rec_address", "s1_5d"]).map_elements(
            lambda s: s["s1_5d"] in s["rec_address"], return_dtype=pl.Boolean))
    for lo, hi, tag in ((0.9, 1.01, "p>=0.9"), (0.3, 0.9, "p in [0.3,0.9)")):
        sub = with_5d.filter(pl.col("p").is_between(lo, hi, closed="left" if hi == 0.9 else "both"))
        pct = 100 * sub["rec_has_s1_5d"].mean() if sub.height else float("nan")
        lines.append(f"Among argmax pairs with {tag} (n={sub.height}): record contains S1's 5-digit token "
                    f"in **{pct:.2f}%**")
    lines.append("")
    log("section_D_done")

    # ============================================================== E) abbreviations (token-pair mining)
    lines.append("## E) Abbreviation-like token pairs on high-confidence France pairs (p>=0.98)\n")
    hi = fr_argmax.filter(pl.col("p") >= 0.98).join(
        s1.filter(pl.col("country") == "France").select(s1_id="entity_id", s1_name="name_tokens",
                                                         s1_addr="addr_tokens"), on="s1_id") \
        .join(pl.concat([s2, s3]).select(rec_id="entity_id", rec_name="name_tokens", rec_addr="addr_tokens"),
              on="rec_id")
    lines.append(f"n pairs at p>=0.98: {hi.height}\n")

    def token_pair_counts(s1_col: str, rec_col: str) -> Counter:
        c = Counter()
        for s1t, rt in zip(hi[s1_col].to_list(), hi[rec_col].to_list()):
            s1set, rset = set(s1t), set(rt)
            only_s1, only_r = s1set - rset, rset - s1set
            for a in only_r:  # rec-only
                for b in only_s1:  # s1-only
                    if a == b:
                        continue
                    if a.startswith(b) or b.startswith(a) or _edit_le2(a, b):
                        c[(a, b)] += 1
        return c

    for label, s1_col, rec_col in (("name", "s1_name", "rec_name"), ("address", "s1_addr", "rec_addr")):
        counts = token_pair_counts(s1_col, rec_col)
        top = counts.most_common(50)
        lines.append(f"**{label}** top 50 (rec_token, s1_token, count):\n")
        lines.append(", ".join(f"({a},{b},{n})" for (a, b), n in top) or "(none)")
        lines.append("")
    log("section_E_done", peak=peak_rss_mb())

    # ============================================================== F) accents
    lines.append("## F) Accent folding examples (raw -> normalised)\n")
    fr_s1_accented = fr_s1.filter(pl.col("is_nonascii_raw")).join(
        fr_raw1.select(entity_id="entity_id", raw_name="business_name"), on="entity_id")
    for row in fr_s1_accented.select("raw_name", "core_name").head(10).iter_rows(named=True):
        lines.append(f"- S1 raw=`{row['raw_name']}` -> norm=`{row['core_name']}`")
    fr_rec_accented = pl.concat([s2, s3]).filter(pl.col("is_nonascii_raw"))
    for row in fr_rec_accented.select("business_name", "core_name").head(10).iter_rows(named=True):
        lines.append(f"- rec raw=`{row['business_name']}` -> norm=`{row['core_name']}`")
    lines.append("")
    log("section_F_done")

    # ============================================================== G) 30 random mid-confidence pairs
    lines.append("## G) 30 random France pairs with p in [0.3, 0.9)\n")
    mid_pairs = fr_pairs.filter(pl.col("p").is_between(0.3, 0.9, closed="left"))
    sample_idx = rng.sample(range(mid_pairs.height), k=min(30, mid_pairs.height))
    for i in sample_idx:
        r = mid_pairs.row(i, named=True)
        lines.append(f"- p={r['p']:.3f} | S1 `{r['s1_name']}` | `{r['s1_address']}`  "
                    f"vs rec `{r['rec_name']}` | `{r['rec_address']}`")
    lines.append("")
    log("section_G_done", peak_rss_mb=peak_rss_mb())

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}, peak_rss_mb={peak_rss_mb()}")


if __name__ == "__main__":
    main()
