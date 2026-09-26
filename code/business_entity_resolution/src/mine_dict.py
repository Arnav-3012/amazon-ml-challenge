"""Token-dictionary probe (standalone, no pipeline change): can a mined token map turn blocking vocab misses into
pairs that share a token with their true S1?

Inputs (train only): norm_train_s{1,2,3}.parquet name/addr tokens, the ground truth, candidates_train_v1.parquet
(the set D0-1 measured misses against; D0-1 never persisted its miss list), idf_train.parquet (df per token).
Holdout: 20% of train S1s (seeded) never feed the mining; every % below is on their missed pairs only.

1) native: India pairs whose S2/S3 name (or address) has a non-ASCII raw string. Tokens are already transliterated
   by normalise ("marketimg"), so a mapping = record token -> S1 token, same field, both unshared in the pair.
2) abbr: US + India only, both directions: short alphabetic token (<= 5 chars) on one side -> a token >= 2 chars
   longer with the same first letter on the other side, same field; kept only if the short is a subsequence of
   the long ("bldg" -> "building"), which drops transliteration noise.
Keep a mapping if its top-1 target co-occurs in >= SUPPORT pairs and in >= SHARE of the source's pairs.
Share = pairs where src and tgt are both unshared / pairs where src is unshared (1.0 = deterministic).

Vocab miss (proxy for D0-1 "vocab") = missed pair sharing no name (A) or addr (B) token with df <= blocking.df_cap
(C skeleton / X composite channels ignored; the full-data proxy count is printed next to D0-1's 106,630).
After the dict: shares >= 1 token (any df), and >= 1 surviving token (df <= cap, df of the token as it is today).

Run from code/business_entity_resolution/:  python -m src.mine_dict
-> artifacts/interim/token_dict.parquet (field, kind, s, t, co, n_src, share) + docs/mine_dict.md
"""
import numpy as np
import polars as pl

from .block import norm_path
from .decide import md
from .io import CFG, ROOT, StepLog, load_gt_pairs, path

NONASCII = r"[^\x00-\x7f]"
SUPPORT, SHARE, HOLD, SHORT_MAX = 20, 0.8, 0.2, 5
CAP = CFG["blocking"]["df_cap"]
FIELDS = {"name": "A", "addr": "B"}  # field -> blocking channel holding its tokens in idf_train
OUT = path("interim_dir") / "token_dict.parquet"
REPORT = ROOT / "docs" / "mine_dict.md"
ALPHA = r"^[a-z]+$"


def sides() -> tuple[pl.DataFrame, pl.DataFrame, pl.Series]:
    """a (S1 side), b (S2/S3 side, + native flags), held S1 ids -- shared by pairs() and neg_pairs()."""
    tok = {f"{f}": pl.col(f"{f}_tokens").list.unique() for f in FIELDS}
    s1 = pl.read_parquet(norm_path("train", 1), columns=["entity_id", "country", "name_tokens", "addr_tokens"])
    a = s1.select("country", s1_id="entity_id", **{f"a_{f}": e for f, e in tok.items()})
    b = pl.concat([pl.read_parquet(norm_path("train", s), columns=["entity_id", "name_tokens", "addr_tokens",
                                                                   "business_name", "business_address"])
                   for s in (2, 3)]).select(
        rec_id="entity_id", **{f"b_{f}": e for f, e in tok.items()},
        name_nat=pl.col("business_name").str.contains(NONASCII),
        addr_nat=pl.col("business_address").str.contains(NONASCII))
    ids = s1["entity_id"].sort()
    held = ids.gather(np.random.default_rng(CFG["seed"]).permutation(ids.len())[: round(HOLD * ids.len())])
    return a, b, held


def pairs(a: pl.DataFrame, b: pl.DataFrame, held: pl.Series) -> pl.DataFrame:
    """One row per true train pair: unique name/addr token lists per side, native flags, held, hit (in v1)."""
    v1 = (pl.scan_parquet(path("interim_dir") / "candidates_train_v1.parquet").select("s1_id", "rec_id")
          .with_columns(hit=pl.lit(True)).collect(engine="streaming"))
    return (load_gt_pairs().select("s1_id", rec_id="match_id").join(a, on="s1_id").join(b, on="rec_id")
            .join(v1, on=["s1_id", "rec_id"], how="left")
            .with_columns(pl.col("hit").fill_null(False), held=pl.col("s1_id").is_in(held.implode()),
                          pid=pl.int_range(pl.len(), dtype=pl.Int64)))


def neg_pairs(a: pl.DataFrame, b: pl.DataFrame, held: pl.Series) -> pl.DataFrame:
    """Held-out v1 candidate pairs that are NOT ground truth: same shape as pairs(), for a precision check
    (does the dict invent a shared token between records that don't actually match)."""
    gt = load_gt_pairs().select("s1_id", rec_id="match_id").with_columns(true=pl.lit(True))
    v1 = (pl.scan_parquet(path("interim_dir") / "candidates_train_v1.parquet").select("s1_id", "rec_id")
          .filter(pl.col("s1_id").is_in(held.implode())).collect(engine="streaming"))
    return (v1.join(gt, on=["s1_id", "rec_id"], how="left").filter(pl.col("true").is_null())
            .join(a, on="s1_id").join(b, on="rec_id")
            .with_columns(pid=pl.int_range(pl.len(), dtype=pl.Int64)))


def unshared(p: pl.DataFrame, f: str, side: str, name: str) -> pl.DataFrame:
    """(pid, name) for each token of `side` absent from the other side of the pair, field f."""
    me, other = (f"a_{f}", f"b_{f}") if side == "a" else (f"b_{f}", f"a_{f}")
    return p.select("pid", pl.col(me).list.set_difference(pl.col(other)).alias(name)).explode(name).drop_nulls()


def top1(src: pl.DataFrame, tgt: pl.DataFrame, on: list[str], keep: pl.Expr | None = None) -> pl.DataFrame:
    """Per source token s: its most co-occurring target t, co = pairs with both, share = co / pairs with s."""
    n = src.group_by("s").len("n_src")
    co = src.join(tgt, on=on)
    if keep is not None:
        co = co.filter(keep)
    return (co.group_by("s", "t").len("co").join(n, on="s").with_columns(share=pl.col("co") / pl.col("n_src"))
            .sort(["s", "co", "t"], descending=[False, True, False]).group_by("s", maintain_order=True).first())


def mine_native(m: pl.DataFrame) -> pl.DataFrame:
    out = []
    for f in FIELDS:
        q = m.filter((pl.col("country") == "India") & pl.col(f"{f}_nat"))
        out.append(top1(unshared(q, f, "b", "s"), unshared(q, f, "a", "t"), ["pid"]).with_columns(field=pl.lit(f)))
    return pl.concat(out).with_columns(kind=pl.lit("native"))


def is_subseq(s: str, t: str) -> bool:
    it = iter(t)
    return all(c in it for c in s)


def mine_abbr(m: pl.DataFrame) -> pl.DataFrame:
    out = []
    for f in FIELDS:
        for short, long in (("b", "a"), ("a", "b")):
            s = (unshared(m, f, short, "s").filter(pl.col("s").str.len_chars() <= SHORT_MAX, pl.col("s").str.contains(ALPHA))
                 .with_columns(k=pl.col("s").str.slice(0, 1)))
            t = (unshared(m, f, long, "t").filter(pl.col("t").str.len_chars() >= 3, pl.col("t").str.contains(ALPHA))
                 .with_columns(k=pl.col("t").str.slice(0, 1)))
            out.append(top1(s, t, ["pid", "k"], pl.col("t").str.len_chars() >= pl.col("s").str.len_chars() + 2)
                       .with_columns(field=pl.lit(f)))
    # a short token mined in both directions: keep its better-supported target
    x = pl.concat(out).sort(["field", "s", "co"], descending=[False, False, True]).unique(["field", "s"], keep="first")
    keep = pl.Series([is_subseq(s, t) for s, t in x.select("s", "t").iter_rows()], dtype=pl.Boolean)
    return x.filter(keep).with_columns(kind=pl.lit("abbr"))


def apply(p: pl.DataFrame, d: pl.DataFrame) -> pl.DataFrame:
    """Canonicalise both sides of every pair with the field's map (single pass, no chaining)."""
    for f in FIELDS:
        m = dict(d.filter(pl.col("field") == f).select("s", "t").iter_rows())
        p = p.with_columns(pl.col(f"a_{f}", f"b_{f}").list.eval(pl.element().replace(m)))
    return p


def shared(p: pl.DataFrame, surv: pl.DataFrame, sfx: str) -> pl.DataFrame:
    """p + any{sfx} (>= 1 shared token) + surv{sfx} (>= 1 shared token with df <= cap), over both fields."""
    rows = pl.concat([p.select("pid", "country", ch=pl.lit(ch), tok=pl.col(f"a_{f}").list.set_intersection(f"b_{f}"))
                      .explode("tok").drop_nulls() for f, ch in FIELDS.items()])
    hit_any, hit_surv = rows["pid"].unique(), rows.join(surv, on=["country", "ch", "tok"])["pid"].unique()
    return p.with_columns(**{f"any{sfx}": pl.col("pid").is_in(hit_any.implode()),
                             f"surv{sfx}": pl.col("pid").is_in(hit_surv.implode())})


def main() -> None:
    log = StepLog()
    a, b, held = sides()
    p = pairs(a, b, held)
    log("pairs", rows=p.height, held=int(p["held"].sum()), missed=int((~p["hit"]).sum()))
    surv = (pl.scan_parquet(path("interim_dir") / "idf_train.parquet").filter(pl.col("ch").is_in(list(FIELDS.values())))
            .filter(pl.sum_horizontal("n1", "n2", "n3") <= CAP).select("country", "ch", "tok").collect(engine="streaming"))
    log("surviving tokens", rows=surv.height)

    m = p.filter(~pl.col("held"))
    raw = pl.concat([mine_native(m), mine_abbr(m.filter(pl.col("country").is_in(["US", "India"])))], how="diagonal")
    log("mined candidates", rows=raw.height)
    cand = raw.filter(pl.col("co") >= SUPPORT)  # distribution population: every source with a supported top-1
    d = cand.filter(pl.col("share") >= SHARE).select("field", "kind", "s", "t", "co", "n_src", "share")
    d.write_parquet(OUT)
    log("dict", mappings=d.height, native=int((d["kind"] == "native").sum()), abbr=int((d["kind"] == "abbr").sum()))

    miss = shared(p.filter(~pl.col("hit")), surv, "_before")
    vocab_full = int((~miss["surv_before"]).sum())
    ev = miss.filter("held").with_columns(
        vocab=~pl.col("surv_before"),
        nat=(pl.col("country") == "India") & (pl.col("name_nat") | pl.col("addr_nat")))
    dicts = {"native": d.filter(pl.col("kind") == "native"), "abbr": d.filter(pl.col("kind") == "abbr"), "both": d}
    for k, dk in dicts.items():
        ev = ev.join(shared(apply(ev.select("pid", "country", *(f"{s}_{f}" for s in "ab" for f in FIELDS)), dk),
                            surv, f"_{k}").select("pid", f"any_{k}", f"surv_{k}"), on="pid")
    log("eval", held_misses=ev.height, vocab_proxy_full=vocab_full)

    pct = lambda s, c: float(s[c].mean() * 100) if s.height else float("nan")
    segs = {"vocab (proxy)": pl.col("vocab"), "India-native vocab": pl.col("vocab") & pl.col("nat"),
            "India-native, all misses": pl.col("nat"), "vocab ∪ India-native (KEY)": pl.col("vocab") | pl.col("nat")}
    rows = []
    for name, e in segs.items():
        s = ev.filter(e)
        rows.append({"segment": name, "held misses": s.height, "full-data est": round(s.height / HOLD),
                     "any before %": pct(s, "any_before"), "surv before %": pct(s, "surv_before"),
                     **{f"surv after {k} %": pct(s, f"surv_{k}") for k in dicts},
                     "any after both %": pct(s, "any_both")})

    # precision: held-out v1 candidate pairs that are NOT ground truth. Only those with no surviving shared
    # token today can gain a false one from the dict -- % of THOSE that do is the dict's false-positive rate.
    neg = shared(neg_pairs(a, b, held), surv, "_before")
    neg = neg.join(shared(apply(neg.select("pid", "country", *(f"{s}_{f}" for s in "ab" for f in FIELDS)),
                                dicts["both"]), surv, "_both").select("pid", "surv_both"), on="pid")
    fp = neg.filter(~pl.col("surv_before"))
    rows.append({"segment": "precision: held v1 negatives, new surv token", "held misses": fp.height,
                "full-data est": round(fp.height / HOLD), "any before %": float("nan"), "surv before %": 0.0,
                **{f"surv after {k} %": float("nan") for k in dicts if k != "both"},
                "surv after both %": pct(fp, "surv_both"), "any after both %": float("nan")})
    log("precision", v1_negatives_held=neg.height, no_surv_before=fp.height,
        new_surv_after_both_pct=rows[-1]["surv after both %"])

    edges = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001]
    dist = []
    for k in ("native", "abbr"):
        sh = cand.filter(pl.col("kind") == k)["share"].to_numpy()
        h = np.histogram(sh, edges)[0]
        dist.append({"kind": k, "sources co>=20": int(sh.size), "kept (share>=0.8)": int((sh >= SHARE).sum()),
                     **{f"[{lo:g},{min(hi, 1):g}{')' if hi < 1 else ']'}": int(c) for lo, hi, c in zip(edges, edges[1:], h)},
                     "median share": float(np.median(sh)) if sh.size else float("nan")})
    sample = pl.concat([d.filter(pl.col("kind") == k).sort("co", descending=True).head(15) for k in ("native", "abbr")])

    L = ["# Token dictionary probe (src/mine_dict.py)", "",
         f"Mined on 80% of train S1s' true pairs; evaluated on the missed pairs (not in candidates_train_v1) of the "
         f"held-out 20%. Keep: top-1 co >= {SUPPORT} and share >= {SHARE}. Vocab proxy = no shared name/addr token "
         f"with df <= {CAP}; full-data proxy count {vocab_full:,} vs D0-1 vocab 106,630 (D0-1 also counts C/X "
         "channels). surv = shares >= 1 token with df <= cap (df as it is today).", "",
         f"**Mappings:** {d.height:,} (native {dicts['native'].height:,}, abbr {dicts['abbr'].height:,}).", "",
         "## Is it deterministic? top-1 share over sources with co >= 20", "", *md(dist), "",
         "## KEY: held-out misses that share a token after the dict", "", *md(rows), "",
         "## 30 sample mappings (top co per kind)", "", *md(sample.to_dicts()), ""]
    REPORT.write_text("\n".join(L))
    log("report", path=str(REPORT))
    log.dump(path("interim_dir") / "mine_dict_timing.json")


if __name__ == "__main__":
    main()
