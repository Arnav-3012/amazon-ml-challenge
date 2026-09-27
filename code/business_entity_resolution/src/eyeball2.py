"""Standalone M5 eyeball 2: top-loss errors of OOF (b) at its decision rule + "sibling at the same address"
quantification on ALL FPs vs ALL TPs. Read-only, no pipeline changes.

Reads only: oof/oof_full.parquet (s1k, reck, label, p_td = OOF (b)), GT (src.io.load_gt_pairs),
artifacts/interim/norm_train_s{1,2,3}.parquet (tokens, addr_number, raw strings), artifacts/interim/
candidates_train.parquet (row count only: asserts oof_full is its OOF).

(b) world = rows with non-null p_td (the dropped 19% of S1s are null). Decision = cv_full.score's rule:
per-record argmax of p_td (ties -> lowest s1k), predicted iff p >= T. The FP count is asserted equal to
oof/cv_full.json b_test_density.fp_pairs, so this is the same decision the (b) metric scored.

Sibling of a predicted pair (rec, s): another in-world candidate S1 s' of rec whose unique address tokens
share >= 80% with s's: |A_s & A_s'| / max(|A_s|, |A_s'|) >= 0.8; the highest-p such s' is "the" sibling.
  name_in_sib: >= 1 record name token absent from s's name tokens appears in the sibling's name tokens.
  num_closer:  |num(rec) - num(sib)| < |num(rec) - num(s)|, first digit run of addr_number, all three present.

Writes docs/eyeball2.md. Peak RSS guarded at 6 GB (processed in NP reck-hash partitions).
Run from code/business_entity_resolution/:  python -m src.eyeball2
"""
import json

import polars as pl

from .cv_full import drop_mask, s1_table
from .features import id_key
from .io import CFG, ROOT, StepLog, load_gt_pairs, path

T, NP, N_SHOW, MAX_RSS_MB = 0.75, 8, 60, 6144
OUT = ROOT / "docs" / "eyeball2.md"
NORM = [path("interim_dir") / f"norm_train_s{n}.parquet" for n in (1, 2, 3)]


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        if self.rows[-1]["peak_rss_mb"] > MAX_RSS_MB:
            raise SystemExit(f"STOP: peak RSS {self.rows[-1]['peak_rss_mb']} MB > {MAX_RSS_MB} MB after '{step}'")


def num(col: str) -> pl.Expr:
    return pl.col(col).str.extract(r"(\d{1,15})").cast(pl.Int64, strict=False)


def s1_side() -> pl.DataFrame:
    return pl.read_parquet(NORM[0], columns=["entity_id", "name_tokens", "addr_tokens", "addr_number"]).select(
        s1k=id_key("entity_id"), nt=pl.col("name_tokens").list.unique(), at=pl.col("addr_tokens").list.unique(),
        num=num("addr_number"))


def rec_side(j: int) -> pl.DataFrame:
    return (pl.scan_parquet(NORM[1:]).select(reck=id_key("entity_id"), rnt=pl.col("name_tokens").list.unique(),
                                             rnum=num("addr_number"), country="country")
            .filter(pl.col("reck") % NP == j).collect(engine="streaming"))


def partition(j: int, s1: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """-> (flags of every predicted pair, FP p>0.9 rows, FN p<0.5 rows) for records with reck % NP == j."""
    x = (pl.scan_parquet(path("oof_dir") / "oof_full.parquet")
         .select("s1k", "reck", "label", p="p_td")
         .filter(pl.col("p").is_not_null() & (pl.col("reck") % NP == j))
         .sort(["reck", "p", "s1k"], descending=[False, True, False])
         .with_columns(rk=pl.int_range(pl.len()).over("reck"))
         .collect(engine="streaming"))
    top = x.filter(pl.col("rk") == 0).drop("rk")
    second = x.filter(pl.col("rk") == 1).select("reck", s1k2="s1k", p2="p")
    pred = top.filter(pl.col("p") >= T)

    # sibling: highest-p other candidate whose address shares >= 80% with the predicted S1's
    at = s1.select("s1k", "at")
    sib = (x.filter(pl.col("rk") > 0).select("reck", sk="s1k", sp="p")
           .join(pred.select("reck", "s1k"), on="reck")
           .join(at, on="s1k").join(at.rename({"s1k": "sk", "at": "sat"}), on="sk")
           .filter(pl.col("at").list.set_intersection("sat").list.len()
                   >= 0.8 * pl.max_horizontal(pl.col("at").list.len(), pl.col("sat").list.len()))
           .filter(pl.col("at").list.len() > 0)
           .sort(["reck", "sp", "sk"], descending=[False, True, False]).unique("reck", keep="first")
           .select("reck", "sk"))
    del x
    s1f = s1.select("s1k", "nt", "num")
    unmatched = pl.col("rnt").list.set_difference("nt")
    flags = (pred.join(sib, on="reck", how="left").join(rec_side(j), on="reck", how="left")
             .join(s1f, on="s1k", how="left")
             .join(s1f.rename({"s1k": "sk", "nt": "snt", "num": "snum"}), on="sk", how="left")
             .select("country", fp=pl.col("label") == 0, has_sib=pl.col("sk").is_not_null(),
                     name_in_sib=unmatched.list.set_intersection("snt").list.len().fill_null(0) > 0,
                     num_elig=pl.all_horizontal(pl.col("rnum", "num", "snum").is_not_null()),
                     num_closer=((pl.col("rnum") - pl.col("snum")).abs()
                                 < (pl.col("rnum") - pl.col("num")).abs()).fill_null(False)))
    fp = (pred.filter((pl.col("label") == 0) & (pl.col("p") > 0.9)).join(second, on="reck", how="left")
          .select("reck", "s1k", "p", ck="s1k2", cp="p2"))
    # FN competitor = best in-world candidate that is not the true S1 (top-1, or top-2 when top-1 is the true one)
    fn = (pl.scan_parquet(path("oof_dir") / "oof_full.parquet").select("s1k", "reck", "label", p="p_td")
          .filter((pl.col("reck") % NP == j) & (pl.col("label") == 1) & (pl.col("p") < 0.5))
          .collect(engine="streaming")
          .join(top.select("reck", s1k1="s1k", p1="p"), on="reck").join(second, on="reck", how="left")
          .select("reck", "s1k", "p", ck=pl.when(pl.col("s1k1") != pl.col("s1k")).then("s1k1").otherwise("s1k2"),
                  cp=pl.when(pl.col("s1k1") != pl.col("s1k")).then("p1").otherwise("p2")))
    return flags, fp, fn


def stats(flags: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    pct = lambda e: (100 * e.mean()).round(2)
    return (flags.group_by(by).agg(
        n=pl.len(), pct_has_sib=pct(pl.col("has_sib")),
        pct_name_in_sib=pct(pl.col("name_in_sib")),
        pct_name_in_sib_given_sib=pct(pl.col("name_in_sib").filter("has_sib")),
        pct_num_closer=pct(pl.col("num_closer")),
        n_num_elig=(pl.col("has_sib") & pl.col("num_elig")).sum(),
        pct_num_closer_given_elig=pct(pl.col("num_closer").filter(pl.col("has_sib") & pl.col("num_elig"))))
        .with_columns(cls=pl.when("fp").then(pl.lit("FP")).otherwise(pl.lit("TP"))).drop("fp")
        .sort(by[1:] + ["cls"]).select("cls", *by[1:], pl.exclude("cls", *by[1:])))


def md_table(df: pl.DataFrame) -> str:
    rows = [" | ".join(df.columns), "|".join(["---"] * df.width)]
    return "\n".join(rows + [" | ".join(str(v) for v in r) for r in df.iter_rows()]) + "\n\n"


def raw_rows(keys: set[int]) -> dict[int, str]:
    esc = lambda c: pl.col(c).str.replace_all("|", r"\|", literal=True)
    lut = (pl.scan_parquet(NORM).select(k=id_key("entity_id"), s=pl.concat_str(
        esc("business_name"), esc("business_address"), "country", separator=" \\| "))
           .filter(pl.col("k").is_in(list(keys))).collect(engine="streaming"))
    return dict(lut.iter_rows())


def main() -> None:
    log = Log()
    # candidates_train row order == oof_full _i (cv_full reads features/train in candidates order)
    c = pl.scan_parquet(path("interim_dir") / "candidates_train.parquet").select(pl.len()).collect().item()
    o = pl.scan_parquet(path("oof_dir") / "oof_full.parquet")
    assert o.select(pl.len()).collect().item() == c, "oof_full rows != candidates_train rows"
    cv = json.loads((path("oof_dir") / "cv_full.json").read_text())["b_test_density"]
    assert abs(cv["best_t"] - T) < 1e-9, cv["best_t"]

    s1 = s1_side()
    log("s1 side", s1=s1.height)
    parts = []
    for j in range(NP):
        parts.append(partition(j, s1))
        log(f"partition {j}")
    flags, fp, fn = (pl.concat([p[i] for p in parts]) for i in range(3))
    del parts, s1
    log("partitions done", pred=flags.height, fp_gt09=fp.height, fn_lt05=fn.height)
    assert int(flags["fp"].sum()) == cv["fp_pairs"], (int(flags["fp"].sum()), cv["fp_pairs"])

    top_fp = fp.sort(["p", "reck"], descending=[True, False]).head(N_SHOW)
    top_fn = fn.sort(["p", "reck"]).head(N_SHOW)
    shown = pl.concat([top_fp, top_fn])
    true_s1 = dict(load_gt_pairs().select(r=id_key("match_id"), s=id_key("s1_id"))
                   .filter(pl.col("r").is_in(shown["reck"].implode())).iter_rows())
    s1t = s1_table(1.0)  # same drop mask as cv_full.run's (b) world
    in_world = set(s1t["s1k"].filter(~pl.Series(drop_mask(s1t.height, CFG["seed"] + 2000))).to_list())
    raw = raw_rows(set(shown["reck"]) | set(shown["s1k"]) | set(shown["ck"].drop_nulls()) | set(true_s1.values()))
    log("raw lookup", keys=len(raw))

    def true_str(r: int) -> str:
        s = true_s1.get(r)
        return "DISTRACTOR" if s is None else raw[s] + ("" if s in in_world else " [S1 dropped in (b): distractor]")

    def listing(df: pl.DataFrame, title: str, head: str) -> str:
        out = [f"## {title}\n\n", f"{head} | p | record raw | record's TRUE S1 raw | best competing S1 raw | competitor p\n",
               "---|---|---|---|---|---\n"]
        for r in df.iter_rows(named=True):
            comp = raw[r["ck"]] if r["ck"] is not None else "(no other in-world candidate)"
            cp = f"{r['cp']:.4f}" if r["cp"] is not None else ""
            out.append(f"{raw[r['s1k']]} | {r['p']:.4f} | {raw[r['reck']]} | {true_str(r['reck'])} | {comp} | {cp}\n")
        return "".join(out) + "\n"

    n_fp, n_tp = int(flags["fp"].sum()), int((~flags["fp"]).sum())
    with open(OUT, "w") as f:
        f.write(f"# Eyeball 2: OOF (b) top-loss errors + same-address siblings (t = {T})\n\n"
                "Generated by `python -m src.eyeball2`; definitions in the module docstring. "
                f"(b) world = non-null p_td; predicted = per-record argmax with p >= {T}. "
                f"FP = {n_fp:,} (== cv_full.json fp_pairs), TP = {n_tp:,}; "
                f"FPs with p > 0.9 = {fp.height:,}; FNs (true in-world pair, p < 0.5) = {fn.height:,}.\n\n"
                "Row format for raw columns: name \\| address \\| country.\n\n")
        f.write("## 1. Siblings at the same address: all FPs vs all TPs\n\n")
        f.write(md_table(stats(flags, ["fp"])))
        f.write("### By record country\n\n")
        f.write(md_table(stats(flags, ["fp", "country"])))
        f.write(listing(top_fp, f"2. Top-loss FPs (p > 0.9, highest p first, {N_SHOW})", "predicted S1 raw"))
        f.write(listing(top_fn, f"3. Top-loss FNs (true pair p < 0.5, lowest p first, {N_SHOW})", "true S1 raw"))
    log("written")
    log.dump(ROOT / "artifacts" / "logs" / "eyeball2_timing.json")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
