"""Structure probe of the RAW train data (no pipeline artifacts): ID tags, match cardinality, pseudo-brands,
distractor transforms, domains, within-source duplicates. Writes docs/structure.md.

Reads only dataset/train/*.tsv via src.io (load_source, load_gt_pairs) + the legal-form lexicons of src.normalise.
Sample: first FRAC of a seeded permutation of S1 rows (config seed) -> their true S2/S3 records, plus a seeded
FRAC of the distractor records (records in no GT list). Nearest-S1 / address-rank searches run against ALL S1
(--smoke: the sampled S1 only) with char_wb 3-gram TF-IDF (hashed to 2^20, grams in > MAX_DF of the pool get
idf 0 so products stay sparse, l2-normalised), capped at NQ queries per group.

Text prep ("clean"): lowercase, l.l.c. -> llc, non-letter/digit runs -> one space. legal = sorted canonical legal
forms found with normalise's per-country regex (country only routes the lexicon, as in normalise); core = clean
name minus legal words.

Run from code/business_entity_resolution/:
  python -m src.structure --smoke   # 1% sample, sampled-S1 search pool, < 3 min
  python -m src.structure           # 20% sample, all-S1 search pool
"""
import argparse

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .features import id_key
from .io import CFG, ROOT, StepLog, load_gt_pairs, load_source
from .normalise import DEFAULT_LEX, DIGIT_MAP, LEX, _words_re

OUT = ROOT / "docs" / "structure.md"
SEED, MAX_RSS_MB, MAX_DF, NF, CH, TF_CH = CFG["seed"], 18000, 0.002, 2**20, 128, 200_000
TAG_RE, BARE_RE = r"\(ID:\s*\[?(\d+)", r"\d{6,}"
DOM_HAS = r"(?i)www\.|\.c[o0]m"
DOM_TLD = r"\.(?:c[o0]m|net|org|in|co|fr)"
DOM_RE = rf"(?i)(?:www\.)?([\p{{L}}\p{{N}}-]+){DOM_TLD}\b"
E = pl.element()
CTOK = pl.col("core").str.extract_all(r"\S+")
RAW = pl.concat_str("name", "addr", "country", separator=" ‖ ")
HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=NF, alternate_sign=False, norm=None,
                       dtype=np.float32)


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        if self.rows[-1]["peak_rss_mb"] > MAX_RSS_MB:
            raise SystemExit(f"STOP: peak RSS {self.rows[-1]['peak_rss_mb']} MB > {MAX_RSS_MB} MB after '{step}'")


def pct(e: pl.Expr | str) -> pl.Expr:
    return ((pl.col(e) if isinstance(e, str) else e).cast(pl.Float64).mean() * 100).round(2)  # keeps e's name


def md(df: pl.DataFrame) -> str:
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v).replace("|", r"\|")
    rows = [" | ".join(df.columns), "|".join(["---"] * df.width)]
    return "\n".join(rows + [" | ".join(map(fmt, r)) for r in df.iter_rows()]) + "\n\n"


def load(n: int) -> pl.DataFrame:
    return pl.from_pandas(load_source("train", n)).select(
        k=id_key("entity_id"), eid="entity_id", name="business_name", addr="business_address", country="country")


def clean(c: str) -> pl.Expr:
    return (pl.col(c).str.to_lowercase().str.replace_all(r"\b(\p{L})\.", "${1}")
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())


def derive(df: pl.DataFrame) -> pl.DataFrame:
    """+ nm, ad (clean), legal, core, anum / nnum (digit runs of address / name). Row order NOT kept."""
    df = df.with_columns(nm=clean("name"), ad=clean("addr"))
    parts = []
    for (c,), p in df.partition_by("country", as_dict=True).items():
        lex = LEX.get(c, DEFAULT_LEX).legal
        lre = _words_re(lex)
        parts.append(p.with_columns(
            legal=pl.col("nm").str.extract_all(lre).list.eval(E.replace_strict(lex, return_dtype=pl.String))
            .list.unique().list.sort().list.join(" "),
            core=pl.col("nm").str.replace_all(lre, " ").str.replace_all(r"\s+", " ").str.strip_chars()))
    return pl.concat(parts).with_columns(anum=pl.col("ad").str.extract_all(r"\d+").list.join(" "),
                                         nnum=pl.col("nm").str.extract_all(r"\d+").list.join(" "))


def text(df: pl.DataFrame, *cols: str) -> pl.Series:
    return df.select(pl.concat_str(*cols, separator=" ")).to_series()


def tf(s: pl.Series) -> sp.csr_matrix:
    return sp.vstack([HV.transform(s.slice(i, TF_CH).to_list()) for i in range(0, len(s), TF_CH)], format="csr")


class Index:
    """Char-3gram TF-IDF over the S1 pool, stored transposed (gram x S1) so a query batch costs ~sum of df."""

    def __init__(self, s: pl.Series) -> None:
        x = tf(s)
        n, df = x.shape[0], np.bincount(x.indices, minlength=NF)
        self.idf = np.where(df > MAX_DF * n, 0, np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        self.xt = self.vec(x).T.tocsr()

    def vec(self, x: sp.csr_matrix) -> sp.csr_matrix:
        x.data *= self.idf[x.indices]
        x.eliminate_zeros()
        return normalize(x, copy=False)

    def search(self, s: pl.Series, true_pos: np.ndarray | None = None):
        """-> (argmax pool pos or -1, max cosine, cosine to true_pos) per query."""
        q = self.vec(tf(s))
        n = q.shape[0]
        nn, top, tru = np.full(n, -1), np.zeros(n, np.float32), np.zeros(n, np.float32)
        for i in range(0, n, CH):
            m = q[i:i + CH] @ self.xt
            j = np.arange(m.shape[0])
            mx = m.max(axis=1).toarray().ravel()
            nn[i + j], top[i + j] = np.where(mx > 0, np.asarray(m.argmax(axis=1)).ravel(), -1), mx
            if true_pos is not None:
                tru[i + j] = np.asarray(m[j, true_pos[i:i + CH]]).ravel()
        return nn, top, tru


# ---------------------------------------------------------------- 1) ID tags
def rels(t: pl.Expr, n1: int) -> dict[str, pl.Expr]:
    a, b = pl.col("num"), t.cast(pl.String)
    ai, bi = a.cast(pl.Int64, strict=False), t.cast(pl.Int64, strict=False)
    d = (ai - bi).abs()
    return {"equal": ai == bi, "contains": a.str.contains(b, literal=True) | b.str.contains(a, literal=True),
            "pm1": d <= 1, "pm1000": d <= 1000, "reverse": ai == b.str.reverse().cast(pl.Int64, strict=False),
            "last4": ai % 10_000 == bi % 10_000, "mod_nS1": ai % n1 == bi % n1}


def q1(recs: pl.DataFrame, s1: pl.DataFrame, gts: pl.DataFrame) -> str:
    x = recs.select("k", "eid", "name", "label", tag=pl.col("name").str.extract(TAG_RE, 1),
                    bare=pl.col("name").str.replace_all(r"\(ID:[^)]*\)?", "").str.extract_all(BARE_RE))
    prev = x.group_by("label").agg(n=pl.len(), pct_id_tag=pct(pl.col("tag").is_not_null()),
                                   pct_bare6=pct(pl.col("bare").list.len() > 0)).sort("label")
    s1c = s1.select(s1k="k", s1row="row", s1id=pl.col("eid").str.slice(3), s1raw=RAW)
    t = x.filter("label").join(gts.select("s1k", k="rk"), on="k").join(s1c, on="s1k")
    keep = ["kind", "num", "eid", "name", "s1k", "s1row", "s1id", "s1raw"]
    long = pl.concat([
        t.filter(pl.col("tag").is_not_null()).with_columns(kind=pl.lit("id_tag"), num="tag").select(keep),
        t.explode("bare").drop_nulls("bare").with_columns(kind=pl.lit("bare6"), num="bare").select(keep)])
    rr = np.random.default_rng(SEED + 10).integers(0, s1.height, long.height)
    long = long.with_columns(r_row=pl.Series(rr, dtype=pl.Int64), r_id=s1["eid"].str.slice(3).gather(rr))
    true_t = {"s1_id": pl.col("s1id"), "s1_row0": pl.col("s1row"), "s1_row1": pl.col("s1row") + 1,
              "self_id": pl.col("eid").str.slice(3)}
    rand_t = {"s1_id": pl.col("r_id"), "s1_row0": pl.col("r_row"), "s1_row1": pl.col("r_row") + 1}

    def block(g: pl.DataFrame, kind: str, arm: str, tm: dict) -> pl.DataFrame:
        return pl.concat([g.select(kind=pl.lit(kind), target=pl.lit(tn), arm=pl.lit(arm), n=pl.len(),
                                   **{r: pct(e.fill_null(False)) for r, e in rels(te, s1.height).items()})
                          for tn, te in tm.items()])

    tab = pl.concat([block(g, kind, arm, tm) for (kind,), g in long.partition_by("kind", as_dict=True).items()
                     for arm, tm in (("true S1", true_t), ("random S1", rand_t))]) if long.height else long
    tags = long.filter(pl.col("kind") == "id_tag").with_columns(v=pl.col("num").cast(pl.Int64, strict=False))
    multi = tags.group_by("s1k").agg(n=pl.len(), u=pl.col("num").n_unique()).filter(pl.col("n") >= 2)
    stats = (f"ID-tag numbers: n={tags.height:,}, unique={tags['num'].n_unique():,}, min={tags['v'].min()}, "
             f"median={tags['v'].median()}, max={tags['v'].max()}, max digits={tags['num'].str.len_chars().max()}. "
             f"S1s with >=2 tagged true records: {multi.height:,}; all tags equal within the S1: "
             f"{multi.select(pct(pl.col('u') == 1)).item() if multi.height else 'n/a'}%. "
             f"Share of tag numbers used under >1 distinct S1: "
             f"{tags.group_by('num').agg(pl.col('s1k').n_unique()).select(pct(pl.col('s1k') > 1)).item()}%.\n\n")
    ex = long.sample(min(10, long.height), seed=SEED).select("kind", "num", record_name="name", s1_id="s1id",
                                                             s1_row0="s1row", s1_raw="s1raw")
    return ("## 1. ID tags in record names\n\n`(ID: n)` = regex `" + TAG_RE + "`; bare6 = any 6+-digit run "
            "outside the ID tag. "
            "Relations of the number vs the TRUE S1 (entity-id digits / 0- or 1-based row in train_source1) and "
            "vs a uniformly random S1 (baseline); self_id = the record's own id digits. Cells = % of numbers.\n\n"
            + md(prev) + md(tab) + stats + "### Examples\n\n" + md(ex))


# ---------------------------------------------------------------- 2) cardinality
def q2(s1: pl.DataFrame, cnt: pl.DataFrame) -> str:
    c = (s1.with_columns(nm_dup=pl.col("nm").is_duplicated()).filter("ins")
         .join(cnt, left_on="k", right_on="s1k", how="left").with_columns(pl.col("n2", "n3").fill_null(0))
         .with_columns(single=(pl.col("n2") + pl.col("n3")) == 0))
    cap = lambda col: pl.when(pl.col(col) >= 5).then(pl.lit("5+")).otherwise(pl.col(col).cast(pl.String))
    ct = (c.select(S2=cap("n2"), n3=cap("n3")).group_by("S2", "n3").len()
          .pivot(on="n3", index="S2", values="len").fill_null(0).sort("S2"))
    ct = ct.select(pl.col("S2").alias("S2 count \\ S3 count"), *sorted(x for x in ct.columns if x != "S2"))
    s = c.select(n=pl.len(), both=pct((pl.col("n2") > 0) & (pl.col("n3") > 0)),
                 s2_only=pct((pl.col("n2") > 0) & (pl.col("n3") == 0)),
                 s3_only=pct((pl.col("n2") == 0) & (pl.col("n3") > 0)), none=pct("single"),
                 max_s2=pl.col("n2").max(), max_s3=pl.col("n3").max(), mean_s2=pl.col("n2").mean().round(3),
                 mean_s3=pl.col("n3").mean().round(3))
    props = c.group_by("single").agg(
        n=pl.len(), name_chars=pl.col("name").str.len_chars().mean().round(1),
        name_tokens=pl.col("nm").str.count_matches(r"\S+").mean().round(2),
        addr_chars=pl.col("addr").str.len_chars().mean().round(1), pct_legal=pct(pl.col("legal") != ""),
        pct_nonascii=pct(pl.col("name").str.contains(r"[^\x00-\x7F]") | pl.col("addr").str.contains(r"[^\x00-\x7F]")),
        pct_addr_empty=pct(pl.col("ad") == ""), pct_name_digit=pct(pl.col("name").str.contains(r"\d")),
        pct_addr_digit=pct(pl.col("ad").str.contains(r"\d")), pct_name_dup_in_S1=pct("nm_dup"),
        id_digits=pl.col("eid").str.len_chars().mean().round(2) - 3,
        mean_row_pct=(100 * pl.col("row").mean() / s1.height).round(1)).sort("single")
    ctry = (c.group_by("single", "country").len().with_columns(pct=(100 * pl.col("len") / pl.col("len").sum()
                                                                     .over("single")).round(2)).sort("single", "country"))
    ex = c.filter("single").sample(min(10, int(c["single"].sum())), seed=SEED).select("eid", raw=RAW)
    return ("## 2. Match cardinality per sampled S1\n\n" + md(s) + "Joint distribution (rows S2 count, cols S3 "
            "count, cells = #S1):\n\n" + md(ct) + "Singletons (0 matches) vs matched S1:\n\n" + md(props)
            + md(ctry) + "### Singleton examples\n\n" + md(ex))


# ---------------------------------------------------------------- 3) pseudo-brands
def q3(recs: pl.DataFrame, vocab: pl.DataFrame) -> tuple[str, pl.DataFrame]:
    """-> (markdown, per-record pseudo flag)."""
    tk = (recs.select("k", "country", tok=CTOK).explode("tok").filter(pl.col("tok").str.len_chars() >= 3)
          .join(vocab.with_columns(inv=pl.lit(True)), on=["country", "tok"], how="left"))
    per = tk.group_by("k").agg(n_tok=pl.len(), n_inv=pl.col("inv").sum())
    r = recs.join(per, on="k", how="left").with_columns(
        pseudo=((pl.col("n_tok") > 0) & (pl.col("n_inv") == 0)).fill_null(False))
    split = r.group_by("label").agg(n=pl.len(), n_pseudo=pl.col("pseudo").sum(), pct_pseudo=pct("pseudo")).sort("label")
    n_ps = int(r["pseudo"].sum())
    pt = r.filter("pseudo").select(pct("label")).item() if n_ps else None

    nov = (tk.filter(pl.col("inv").is_null() & (pl.col("tok").str.len_chars() >= 5))
           .join(r.filter("pseudo").select("k", "label"), on="k"))
    vt = vocab.filter(pl.col("tok").str.len_chars() >= 5).select("tok").unique()

    def affixes(df: pl.DataFrame, kind: str) -> pl.DataFrame:
        f = (lambda L: pl.col("tok").str.slice(0, L)) if kind == "prefix" else (lambda L: pl.col("tok").str.slice(-L))
        return pl.concat([df.with_columns(affix=f(L)) for L in (3, 4)])

    tabs = []
    for kind in ("prefix", "suffix"):
        v = affixes(vt, kind).group_by("affix").agg(pct_s1_vocab=(100 * pl.len() / vt.height).round(3))
        tabs.append(affixes(nov, kind).unique(["k", "affix"]).group_by("affix")
                    .agg(n_rec=pl.len(), n_true=pl.col("label").sum())
                    .with_columns(pct_true=(100 * pl.col("n_true") / pl.col("n_rec")).round(1),
                                  pct_pseudo_recs=(100 * pl.col("n_rec") / max(n_ps, 1)).round(2))
                    .join(v, on="affix", how="left").sort(["n_rec", "affix"], descending=[True, False]).head(30)
                    .rename({"affix": kind}))
    return (f"## 3. Pseudo-brands\n\nPseudo = record with >=1 core token (len>=3, legal words removed) and none of "
            f"them in ANY S1 name of its country (full train S1). {n_ps:,} pseudo records; % true matches among "
            f"them = {pt}%.\n\n" + md(split)
            + "Affixes (3-4 chars) of novel tokens (len>=5) in pseudo names; n_rec = pseudo records carrying it; "
              "pct_s1_vocab = % of S1 vocabulary tokens (len>=5) with the same affix.\n\n" + md(tabs[0]) + md(tabs[1]),
            r.select("k", "pseudo"))


def q3_addr(recs: pl.DataFrame, pseudo: pl.DataFrame, pool: pl.DataFrame, gts: pl.DataFrame, idx: Index,
            nq: int) -> str:
    t = (recs.filter("label").join(pseudo, on="k").join(gts.select("s1k", k="rk"), on="k")
         .join(pool.select("pos", s1k="k", s1_raw=RAW), on="s1k"))
    rows, ex = [], None
    for name, g in (("pseudo", t.filter("pseudo")), ("non-pseudo", t.filter(~pl.col("pseudo")))):
        if not g.height:
            continue
        g = g.sample(min(nq, g.height), seed=SEED)
        nn, top, tru = idx.search(g["ad"], g["pos"].to_numpy())
        hit = (tru > 0) & (tru >= top - 1e-6)
        rows.append({"true records": name, "n": g.height, "pct_addr_rank1": round(100 * hit.mean(), 2),
                     "pct_rec_addr_empty": round(100 * (g["ad"] == "").mean(), 2),
                     "median_true_addr_cos": round(float(np.median(tru)), 3)})
        if name == "pseudo":
            ex = g.with_columns(addr_rank1=pl.Series(hit)).head(10).select(record=RAW, true_s1="s1_raw",
                                                                           addr_rank1="addr_rank1")
    return ("For true pseudo records: does the address alone (char-3gram TF-IDF vs every S1 address in the pool) "
            "rank the true S1 first (ties count as first)? Non-pseudo true records as baseline.\n\n"
            + md(pl.DataFrame(rows)) + ("### Pseudo examples\n\n" + md(ex) if ex is not None else ""))


# ---------------------------------------------------------------- 4) distractors
def tags() -> dict[str, pl.Expr]:
    L, Ls, an, ans = pl.col("legal"), pl.col("legal_s"), pl.col("anum"), pl.col("anum_s")
    rt, st = CTOK, pl.col("core_s").str.extract_all(r"\S+")
    add, drop = rt.list.set_difference(st).list.len() > 0, st.list.set_difference(rt).list.len() > 0
    return {"legal_add": (L != "") & (Ls == ""), "legal_remove": (L == "") & (Ls != ""),
            "legal_swap": (L != "") & (Ls != "") & (L != Ls), "legal_same": (L != "") & (L == Ls),
            "name_identical": pl.col("nm") == pl.col("nm_s"), "core_identical": pl.col("core") == pl.col("core_s"),
            "core_tok_added_only": add & ~drop, "core_tok_dropped_only": drop & ~add, "core_tok_replaced": add & drop,
            "name_num_change": pl.col("nnum") != pl.col("nnum_s"),
            "addr_num_change": (an != "") & (ans != "") & (an != ans), "addr_num_missing": (an == "") & (ans != ""),
            "addr_identical": (pl.col("ad") != "") & (pl.col("ad") == pl.col("ad_s")),
            "addr_empty_rec": pl.col("ad") == "", "cos_ge_0.9": pl.col("cos") >= 0.9,
            "cos_0.7_0.9": pl.col("cos").is_between(0.7, 0.9, closed="left"), "cos_lt_0.7": pl.col("cos") < 0.7}


def q4(recs: pl.DataFrame, pool: pl.DataFrame, gts: pl.DataFrame, cnt: pl.DataFrame, idx: Index, nq: int) -> str:
    side = pool.select("pos", s1k_s="k", raw_s=RAW, **{f"{c}_s": c for c in ("nm", "ad", "legal", "core", "anum", "nnum")})
    d = recs.filter(~pl.col("label"))
    d = d.sample(min(nq, d.height), seed=SEED)
    t = recs.filter("label").join(gts.select("s1k", k="rk"), on="k").join(pool.select("pos", s1k="k"), on="s1k")
    t = t.sample(min(nq, t.height), seed=SEED)
    dn, dc, _ = idx.search(text(d, "nm", "ad"))
    tn, _, tt = idx.search(text(t, "nm", "ad"), t["pos"].to_numpy())
    d = d.with_columns(pos=pl.Series(dn), cos=pl.Series(dc)).filter(pl.col("pos") >= 0).join(side, on="pos")
    t = t.with_columns(nn=pl.Series(tn), cos=pl.Series(tt)).join(side, on="pos")
    T = tags()
    arms = {"distractor→nearest": d, "true→true S1": t,
            "distractor (cos>=0.7)": d.filter(pl.col("cos") >= 0.7), "true (cos>=0.7)": t.filter(pl.col("cos") >= 0.7)}
    R = {a: x.select(**{k: pct(e) for k, e in T.items()}).row(0, named=True) if x.height else {}
         for a, x in arms.items()}
    tab = (pl.DataFrame([{"tag": k, **{a: R[a].get(k) for a in arms}} for k in T], strict=False)
           .with_columns(diff=pl.col("distractor→nearest") - pl.col("true→true S1"),
                         diff_cos07=pl.col("distractor (cos>=0.7)") - pl.col("true (cos>=0.7)"))
           .sort(pl.col("diff").abs(), descending=True))
    q = lambda s: " / ".join(f"{s.quantile(p):.3f}" for p in (0.1, 0.5, 0.9)) if len(s) else "n/a"
    single = cnt.select(s1k_s="s1k", n_true=pl.col("n2") + pl.col("n3"))
    ds = d.join(single, on="s1k_s", how="left").select(pct(pl.col("n_true").is_null())).item()
    base = pool.join(single, left_on="k", right_on="s1k_s", how="left").select(pct(pl.col("n_true").is_null())).item()
    head = (f"Distractors sampled {d.height:,} (with a nearest S1), true pairs {t.height:,}. Name+address cosine "
            f"p10/p50/p90: distractor→nearest {q(d['cos'])}; true→true S1 {q(t['cos'])}. True records whose nearest "
            f"pool S1 IS the true S1: {t.select(pct(pl.col('nn') == pl.col('pos'))).item()}%. Distractors whose "
            f"nearest S1 is a singleton: {ds}% (pool singleton rate {base}%).\n\n"
            "Tag rates (%), sorted by |distractor − true|:\n\n")
    tl = pl.concat_list([pl.when(e).then(pl.lit(k)) for k, e in T.items() if not k.startswith("cos")])
    ex = d.sample(min(10, d.height), seed=SEED).select(record=RAW, nearest_s1="raw_s", cos=pl.col("cos").round(3),
                                                       tags=tl.list.drop_nulls().list.join(","))
    return "## 4. Distractors vs true records\n\n" + head + md(tab) + "### Distractor examples\n\n" + md(ex)


# ---------------------------------------------------------------- 5) domains
def dom_rules(lab: pl.Expr, nm: str, core: str) -> dict[str, pl.Expr]:
    t, c = pl.col(nm).str.extract_all(r"\S+"), pl.col(core).str.extract_all(r"\S+")
    cj, f = c.list.join(""), c.list.first()
    ini = lambda x: x.list.eval(E.str.slice(0, 1)).list.join("")
    return {"concat_all": lab == t.list.join(""), "concat_core": lab == cj, "first_word": lab == f,
            "first2_concat": lab == c.list.head(2).list.join(""), "initials_all": lab == ini(t),
            "initials_core": lab == ini(c),
            "prefix_of_concat_core": (lab.str.len_chars() >= 3) & cj.str.starts_with(lab) & (lab != cj),
            "first_word_plus_more": lab.str.starts_with(f) & (lab != f)}


def q5(recs: pl.DataFrame, s1: pl.DataFrame, gts: pl.DataFrame) -> str:
    x = recs.filter(pl.col("name").str.contains(DOM_HAS) | pl.col("addr").str.contains(DOM_HAS)).with_columns(
        in_name=pl.col("name").str.contains(DOM_HAS),
        name_is_domain=pl.col("name").str.strip_chars().str.contains(rf"(?i)^(?:www\.)?[\p{{L}}\p{{N}}-]+{DOM_TLD}$"),
        lab=pl.coalesce(pl.col("name").str.extract(DOM_RE, 1), pl.col("addr").str.extract(DOM_RE, 1))
        .str.to_lowercase().str.replace_all("-", ""))
    x = x.with_columns(labfix=pl.col("lab").str.replace_many(list(DIGIT_MAP), list(DIGIT_MAP.values())))
    n_lab = recs.group_by("label").len()
    counts = (x.group_by("label").agg(n=pl.len(), pct_in_name=pct("in_name"), pct_name_is_domain=pct("name_is_domain"),
                                      pct_label_parsed=pct(pl.col("lab").is_not_null()))
              .join(n_lab, on="label").with_columns(pct_of_records=(100 * pl.col("n") / pl.col("len")).round(3))
              .drop("len").sort("label"))
    t = (x.filter(pl.col("label") & pl.col("lab").is_not_null()).join(gts.select("s1k", k="rk"), on="k")
         .join(s1.select(s1k="k", nm_s="nm", core_s="core", s1_raw=RAW), on="s1k").with_row_index("i"))
    rr = np.random.default_rng(SEED + 11).integers(0, s1.height, t.height)
    t = t.with_columns(nm_r=s1["nm"].gather(rr), core_r=s1["core"].gather(rr))

    def arm(nm: str, core: str) -> tuple[dict, pl.DataFrame]:
        """-> (% per rule, t + one bool column per rule)."""
        L, LF = pl.col("lab"), pl.col("labfix")
        rs = {k: (a | b).fill_null(False) for (k, a), b in zip(dom_rules(L, nm, core).items(),
                                                                dom_rules(LF, nm, core).values())}
        ct = (t.select("i", "lab", "labfix", tok=pl.col(core).str.extract_all(r"\S+")).explode("tok")
              .filter(pl.col("tok").str.len_chars() >= 3)
              .group_by("i").agg(tokhit=(pl.col("lab").str.contains(pl.col("tok"), literal=True)
                                         | pl.col("labfix").str.contains(pl.col("tok"), literal=True)).any()))
        rs["contains_core_token"] = pl.col("tokhit").fill_null(False)
        hits = t.join(ct, on="i", how="left").with_columns(**rs)
        hits = hits.with_columns(any_rule=pl.any_horizontal(list(rs)))
        return hits.select(pct(k) for k in [*rs, "any_rule"]).row(0, named=True), hits

    if not t.height:
        return "## 5. Domains\n\nNo true records with a parsed domain label.\n\n" + md(counts)
    (tru, hits), (rnd, _) = arm("nm_s", "core_s"), arm("nm_r", "core_r")
    tab = pl.DataFrame([{"rule": k, "pct_true_S1": tru[k], "pct_random_S1": rnd[k]} for k in tru])
    tl = pl.concat_list([pl.when(k).then(pl.lit(k)) for k in tru if k != "any_rule"])
    ex = (hits.sample(min(10, hits.height), seed=SEED)
          .select("name", "lab", "s1_raw", rules=tl.list.drop_nulls().list.join(",")))
    return ("## 5. Domains (`www.` / `.com` / `.c0m`)\n\nLabel = text before the TLD, lowercased, hyphens removed; "
            "each rule is tried on the label as-is and with normalise.DIGIT_MAP look-alikes fixed (0->o, 1->l, ...). "
            "S1 tokens are its clean name (all) / core (legal words removed). Random-S1 baseline for chance hits.\n\n"
            + md(counts) + md(tab) + "### Examples\n\n" + md(ex))


# ---------------------------------------------------------------- 6) within-source duplicates
def q6(recs: pl.DataFrame, gts: pl.DataFrame, idx: Index) -> str:
    tr = recs.filter("label").select("k", "src", raw=RAW, h_raw=pl.concat_str("name", "addr").hash(),
                                     h_clean=pl.concat_str("nm", "ad").hash(),
                                     txt=pl.concat_str("nm", "ad", separator=" ")).with_row_index("p")
    X = idx.vec(tf(tr["txt"]))
    g = gts.select("s1k", k="rk").join(tr.select("k", "p", "src", "h_raw", "h_clean"), on="k")
    pp = g.join(g, on="s1k", suffix="_b").filter(pl.col("p") < pl.col("p_b"))
    a, b = pp["p"].to_numpy(), pp["p_b"].to_numpy()
    cos = np.concatenate([np.asarray(X[a[i:i + TF_CH]].multiply(X[b[i:i + TF_CH]]).sum(axis=1)).ravel()
                          for i in range(0, len(a), TF_CH)]) if len(a) else np.zeros(0, np.float32)
    pp = pp.with_columns(cos=pl.Series(cos, dtype=pl.Float32), kind=pl.when(pl.col("src") == pl.col("src_b"))
                         .then(pl.format("within S{}", "src")).otherwise(pl.lit("cross-source")))
    tab = pp.group_by("kind").agg(
        n_pairs=pl.len(), pct_cos_ge_0_9=pct(pl.col("cos") >= 0.9), pct_cos_ge_0_95=pct(pl.col("cos") >= 0.95),
        pct_clean_identical=pct(pl.col("h_clean") == pl.col("h_clean_b")),
        pct_raw_identical=pct(pl.col("h_raw") == pl.col("h_raw_b")),
        p10=pl.col("cos").quantile(0.1).round(3), p50=pl.col("cos").median().round(3),
        p90=pl.col("cos").quantile(0.9).round(3)).sort("kind")
    per = (pp.filter(pl.col("src") == pl.col("src_b")).group_by("s1k", "src").agg(hit=(pl.col("cos") >= 0.9).any())
           .group_by("src").agg(n_groups=pl.len(), pct_groups_with_near_copy=pct("hit")).sort("src"))
    ex = (pp.filter((pl.col("src") == pl.col("src_b")) & (pl.col("cos") >= 0.9)))
    ex = (ex.sample(min(10, ex.height), seed=SEED).join(tr.select("p", rec_a="raw"), on="p")
          .join(tr.select(p_b="p", rec_b="raw"), on="p_b").select("kind", pl.col("cos").round(3), "rec_a", "rec_b"))
    return ("## 6. Within-source duplicates\n\nAll pairs of true records of the same sampled S1; cosine = clean "
            "name+address char-3gram TF-IDF (idf from the S1 pool). Near-copy = cos >= 0.9.\n\n" + md(tab)
            + "(S1, source) groups with >=2 true records in that source:\n\n" + md(per)
            + "### Near-copy examples (same source)\n\n" + md(ex))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% sample, search pool = sampled S1 only")
    args = ap.parse_args()
    frac, nq = (0.01, 1_000) if args.smoke else (0.2, 20_000)
    log = Log()

    s1 = load(1).with_row_index("row")
    take = np.zeros(s1.height, bool)
    take[np.random.default_rng(SEED).permutation(s1.height)[: int(frac * s1.height)]] = True
    s1 = s1.with_columns(ins=pl.Series(take))
    log("s1", n=s1.height, sampled=int(take.sum()))

    gt = load_gt_pairs().select(s1k=id_key("s1_id"), rk=id_key("match_id"),
                                src=pl.col("match_id").str.slice(1, 1).cast(pl.Int8))
    cnt = gt.group_by("s1k").agg(n2=(pl.col("src") == 2).sum(), n3=(pl.col("src") == 3).sum())
    n_multi = int(gt["rk"].is_duplicated().sum())
    gts = gt.join(s1.filter("ins").select(s1k="k"), on="s1k")
    parts = []
    for n in (2, 3):
        r = load(n)
        dm = np.random.default_rng(SEED + n).random(r.height) < frac
        parts += [r.join(gts.select(k="rk"), on="k", how="semi").with_columns(label=pl.lit(True)),
                  r.filter(pl.Series(dm)).join(gt.select(k="rk"), on="k", how="anti").with_columns(label=pl.lit(False))]
        del r
        log(f"s{n} loaded")
    del gt
    recs = derive(pl.concat(parts).with_columns(src=pl.col("eid").str.slice(1, 1).cast(pl.Int8)))
    del parts
    s1 = derive(s1).sort("row")
    vocab = s1.select("country", tok=pl.col("nm").str.extract_all(r"\S+")).explode("tok").drop_nulls().unique()
    pool = ((s1.filter("ins") if args.smoke else s1).with_row_index("pos").with_columns(pl.col("pos").cast(pl.Int64)))
    log("derived", recs=recs.height, true=int(recs["label"].sum()), vocab=vocab.height, pool=pool.height)

    sec = [q1(recs, s1, gts)]
    log("q1")
    sec.append(q2(s1, cnt))
    log("q2")
    md3, pseudo = q3(recs, vocab)
    del vocab
    idx = Index(pool["ad"])
    log("addr index", nnz=idx.xt.nnz)
    sec.append(md3 + q3_addr(recs, pseudo, pool, gts, idx, nq))
    del idx
    log("q3")
    idx = Index(text(pool, "nm", "ad"))
    log("name+addr index", nnz=idx.xt.nnz)
    sec.append(q4(recs, pool, gts, cnt, idx, nq))
    log("q4")
    sec.append(q5(recs, s1, gts))
    log("q5")
    sec.append(q6(recs, gts, idx))
    log("q6")

    mode = "SMOKE (1%, pool = sampled S1)" if args.smoke else "full (20%, pool = all train S1)"
    head = (f"# Structure probe of the raw train data\n\nGenerated by `python -m src.structure"
            f"{' --smoke' if args.smoke else ''}`; definitions in the module docstring. Mode: {mode}. "
            f"Sampled S1 {int(take.sum()):,} / {s1.height:,}; true records of sampled S1 {int(recs['label'].sum()):,}; "
            f"sampled distractors {int((~recs['label']).sum()):,}; search pool {pool.height:,} S1; query cap {nq:,}/group. "
            f"Records listed under >1 S1 in the GT: {n_multi:,}. Raw strings are name ‖ address ‖ country.\n\n")
    OUT.write_text(head + "".join(sec))
    log("written")
    log.dump(ROOT / "artifacts" / "logs" / f"structure{'_smoke' if args.smoke else ''}_timing.json")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
