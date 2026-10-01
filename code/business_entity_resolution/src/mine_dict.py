"""Mines token maps from train pairs; a pipeline step (after src.block --split train, before src.block_r).
token_dict feeds channel R and the name_tset_dict feature; locality_alias feeds addr_tset_alias / locality_rel.
The report also measures how many blocking vocab misses the map turns into pairs sharing a token with their S1.

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

3) locality: address-token alias map (src.features.locality_rel/addr_tset_alias consume it), mined separately
   from native/abbr -- gated on STRONG pairs (street+number agree, or addr_tset >= LOC_TSET), not on the
   native/abbr unshared-token setup. Leak guard: mined only on hash(s1_id, seed=42) % 5 != 0 (80% of train S1s,
   same split as everything else here); the other 20% is held out for the lift check below, never mined on.
   Differing (S1 locality token or adjacent bigram, record locality token/bigram) pairs, symmetric (each
   direction mined independently, both kept). Keep if support >= LOC_SUPPORT and share >= LOC_SHARE (same
   top1() machinery). If holdout lift (mean addr_tset_alias - addr_tset on aliased-hit pairs) is < 50% of the
   mined-set lift, LOC_SUPPORT is raised to 60 and mining reruns once.

Run from code/business_entity_resolution/:  python -m src.mine_dict
-> artifacts/interim/token_dict.parquet (field, kind, s, t, co, n_src, share)
-> artifacts/interim/locality_alias.parquet (s, t, co, n_src, share)
-> docs/mine_dict.md
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .block import _bigrams, norm_path
from .decide import md
from .io import CFG, ROOT, StepLog, load_gt_pairs, path

NONASCII = r"[^\x00-\x7f]"
SUPPORT, SHARE, HOLD, SHORT_MAX = 20, 0.8, 0.2, 5
CAP = CFG["blocking"]["df_cap"]
FIELDS = {"name": "A", "addr": "B"}  # field -> blocking channel holding its tokens in idf_train
OUT = path("interim_dir") / "token_dict.parquet"
REPORT = ROOT / "docs" / "mine_dict.md"
ALPHA = r"^[a-z]+$"

LOC_SUPPORT, LOC_SHARE, LOC_TSET = 30, 0.6, 80  # task spec; LOC_SUPPORT retried at 60 if holdout lift too weak
LOC_OUT = path("interim_dir") / "locality_alias.parquet"


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


def loc_split(ids: pl.Series) -> tuple[pl.Series, pl.Series]:
    """80/20 train-S1 split for locality mining, hash(s1_id, seed=42) % 5 != 0 (deterministic, no RNG state)."""
    h = ids.hash(seed=CFG["seed"])
    return ids.filter(h % 5 != 0), ids.filter(h % 5 == 0)


def loc_pairs() -> pl.DataFrame:
    """One row per true train pair with addr_tokens both sides + addr_tset, street_core, addr_number, s1_id."""
    s1 = pl.read_parquet(norm_path("train", 1),
                         columns=["entity_id", "addr_tokens", "addr_number", "addr_street_core"])
    a = s1.select("entity_id", a_tok="addr_tokens", a_num="addr_number", a_street="addr_street_core")
    rec = pl.concat([pl.read_parquet(norm_path("train", n),
                                     columns=["entity_id", "addr_tokens", "addr_number", "addr_street_core"])
                     for n in (2, 3)])
    b = rec.select(rec_id="entity_id", b_tok="addr_tokens", b_num="addr_number", b_street="addr_street_core")
    p = (load_gt_pairs().select("s1_id", rec_id="match_id")
         .join(a, left_on="s1_id", right_on="entity_id").join(b, on="rec_id"))
    n1, n2 = p["a_tok"].list.join(" ").to_list(), p["b_tok"].list.join(" ").to_list()
    tset = cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)  # row-aligned, not full cdist
    return p.with_columns(addr_tset=pl.Series(tset),
                          strong=(pl.col("a_street") == pl.col("b_street")) & (pl.col("a_num") == pl.col("b_num"))
                          & (pl.col("a_street") != ""))


def locality_toks(tok: pl.Series) -> pl.Series:
    """Unique unigrams + adjacent bigrams of an addr_tokens list (bigram key = block._bigrams, sorted-word form)."""
    return pl.concat_list(tok, _bigrams(tok)).list.unique()


def mine_locality(p: pl.DataFrame, support: int) -> pl.DataFrame:
    """(s, t) locality alias pairs, symmetric: mined independently a->b and b->a, both kept.
    strong = street+number agree, or addr_tset >= LOC_TSET (loc_pairs marks `strong`; addr_tset >= LOC_TSET
    is applied here since it's a threshold, not a boolean column)."""
    q = p.filter(pl.col("strong") | (pl.col("addr_tset") >= LOC_TSET))
    q = q.with_columns(a_addr=locality_toks(q["a_tok"]), b_addr=locality_toks(q["b_tok"])).with_row_index("pid")
    return pl.concat([top1(unshared(q, "addr", "a", "s"), unshared(q, "addr", "b", "t"), ["pid"]),
                      top1(unshared(q, "addr", "b", "s"), unshared(q, "addr", "a", "t"), ["pid"])]
                     ).filter(pl.col("co") >= support)


def apply_locality(a_tok: pl.Series, b_tok: pl.Series, d: pl.DataFrame) -> np.ndarray:
    """addr_tset recomputed after mapping both sides' locality tokens through d {s: t}; -1.0 sentinel if d empty."""
    if d.height == 0:
        return np.full(a_tok.len(), -1.0, np.float32)
    m = dict(d.select("s", "t").iter_rows())
    ra = a_tok.list.eval(pl.element().replace(m)).list.join(" ").to_list()
    rb = b_tok.list.eval(pl.element().replace(m)).list.join(" ").to_list()
    return cpdist(ra, rb, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)


def locality_lift(p: pl.DataFrame, d: pl.DataFrame, held: pl.Series) -> tuple[float, float, dict]:
    """mean(addr_tset_alias - addr_tset) on rows whose alias mapping actually changes a token, split by held.
    Returns (mined_set_lift, holdout_lift, top30-sample dict) for the report / the 50% retry check."""
    if d.height == 0:
        return 0.0, 0.0, {}
    m = d["s"].to_list()
    p = p.with_columns(hit=(pl.col("a_tok").list.eval(pl.element().is_in(m)).list.any()
                            | pl.col("b_tok").list.eval(pl.element().is_in(m)).list.any()))
    hitp = p.filter("hit")
    alias = apply_locality(hitp["a_tok"], hitp["b_tok"], d)
    hitp = hitp.with_columns(addr_tset_alias=pl.Series(alias), lift=pl.Series(alias) - pl.col("addr_tset"))
    hitp = hitp.with_columns(held=pl.col("s1_id").is_in(held.implode()))
    mined = hitp.filter(~pl.col("held"))
    hold = hitp.filter("held")
    return (float(mined["lift"].mean()) if mined.height else 0.0,
            float(hold["lift"].mean()) if hold.height else 0.0,
            {"mined_hit_rows": mined.height, "holdout_hit_rows": hold.height})


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

    # ---- locality alias map (independent mining/eval; see module docstring §3) ----
    lp = loc_pairs()
    mine_ids, loc_held = loc_split(lp["s1_id"].unique())
    lp_mine = lp.filter(pl.col("s1_id").is_in(mine_ids.implode()))
    log("locality pairs", rows=lp.height, mine_s1=mine_ids.len(), held_s1=loc_held.len(),
        strong=int(lp_mine["strong"].sum()), strong_or_tset80=int((lp_mine["strong"] | (lp_mine["addr_tset"] >= LOC_TSET)).sum()))

    loc_support = LOC_SUPPORT
    loc_d = mine_locality(lp_mine, loc_support)
    loc_d = loc_d.filter(pl.col("share") >= LOC_SHARE).select("s", "t", "co", "n_src", "share")
    mined_lift, hold_lift, lift_meta = locality_lift(lp, loc_d, loc_held)
    log("locality dict", mappings=loc_d.height, support=loc_support, mined_lift=round(mined_lift, 3),
        holdout_lift=round(hold_lift, 3), **lift_meta)
    if mined_lift > 0 and hold_lift < 0.5 * mined_lift:
        loc_support = 60
        loc_d = mine_locality(lp_mine, loc_support).filter(pl.col("share") >= LOC_SHARE).select(
            "s", "t", "co", "n_src", "share")
        mined_lift, hold_lift, lift_meta = locality_lift(lp, loc_d, loc_held)
        log("locality dict (retry support=60)", mappings=loc_d.height, mined_lift=round(mined_lift, 3),
            holdout_lift=round(hold_lift, 3), **lift_meta)
    loc_d.write_parquet(LOC_OUT)
    loc_sample = loc_d.sort("co", descending=True).head(30)

    L = ["# Token dictionary (src/mine_dict.py)", "",
         f"Mined on 80% of train S1s' true pairs; evaluated on the missed pairs (not in candidates_train_v1) of the "
         f"held-out 20%. Keep: top-1 co >= {SUPPORT} and share >= {SHARE}. Vocab proxy = no shared name/addr token "
         f"with df <= {CAP}; full-data proxy count {vocab_full:,} vs D0-1 vocab 106,630 (D0-1 also counts C/X "
         "channels). surv = shares >= 1 token with df <= cap (df as it is today).", "",
         f"**Mappings:** {d.height:,} (native {dicts['native'].height:,}, abbr {dicts['abbr'].height:,}).", "",
         "## Is it deterministic? top-1 share over sources with co >= 20", "", *md(dist), "",
         "## KEY: held-out misses that share a token after the dict", "", *md(rows), "",
         "## 30 sample mappings (top co per kind)", "", *md(sample.to_dicts()), "",
         "## Locality alias map (src/mine_dict.py §3)", "",
         f"Mined on hash(s1_id, seed={CFG['seed']}) % 5 != 0 (80% of train S1s); support >= {loc_support}, "
         f"share >= {LOC_SHARE}, gated on street+number agreement or addr_tset >= {LOC_TSET}. "
         f"**Mappings:** {loc_d.height:,}. Lift (mean addr_tset_alias - addr_tset on rows an alias touches): "
         f"mined-set {mined_lift:+.2f}, holdout {hold_lift:+.2f} "
         f"({'OK' if mined_lift <= 0 or hold_lift >= 0.5 * mined_lift else 'BELOW 50% of mined-set lift'}).", "",
         "## 30 sample locality mappings (top co)", "", *md(loc_sample.to_dicts()), ""]
    REPORT.write_text("\n".join(L))
    log("report", path=str(REPORT))
    log.dump(path("interim_dir") / "mine_dict_timing.json")


if __name__ == "__main__":
    main()
