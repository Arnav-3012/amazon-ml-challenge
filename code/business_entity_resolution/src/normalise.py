"""M3a normaliser v1: vectorised polars rules that invert the noise operators mined by src.noise_ops.

`normalise(df, off)` is pure: raw source rows (entity_id, business_name, business_address, country) in,
raw columns kept untouched + the normalised columns in OUT_COLS out, row order preserved. `off` disables
named rules one at a time (ablation in src.normalise_eval).
Country only routes lexicons (legal forms, street types, admin regions); it is never an output feature.
Polars regex is the Rust `regex` crate (no lookaround, no backrefs), so token-level rules run on lists.

Run from code/business_entity_resolution/:
  python -m src.normalise --smoke 20000   # first 20k rows per file, prints rate + samples, writes nothing
  python -m src.normalise                 # all 6 files -> artifacts/interim/norm_{split}_s{n}.parquet
"""
import argparse
import json
import re
import time
from dataclasses import dataclass

import polars as pl
from anyascii import anyascii

from .io import load_source, path, peak_rss_mb  # noqa: F401  (peak_rss_mb re-exported for block/normalise_eval)

# Rule names in execution order (see docs/normalise.md). Structural steps (lowercase, punct/whitespace
# strip, address component parse) always run; everything listed here can be ablated.
NAME_RULES = ("anyascii", "alias_split", "id_tag", "acronym_dots", "domain_strip", "trailing_phone", "digit_fix",
              "dedupe_adjacent", "legal_form", "honorifics")
# Reverted after the M3a ablation (docs/decisions_mistakes.md): country_marker, amp_and, landmark.
ADDR_RULES = ("number_prefix", "street_type", "ordinal", "null_token", "admin_region")
RULES = NAME_RULES + ADDR_RULES  # anyascii applies to both fields

OUT_COLS = ["entity_id", "business_name", "business_address", "country",
            "core_name", "legal_form", "aliases", "name_tokens",
            "addr_number", "addr_street_core", "addr_tokens", "admin_region", "city",
            "has_address", "is_nonascii_raw"]

# ---- patterns (all applied after lowercase + anyascii) ----
# Mined: "<generated brand> <kw> <real name>"; the S1 name is always the right-hand side.
ALIAS_KW = r"formerly known as|formerly|doing business as|trading as|d/b/a|dba|t/a|a/k/a|aka|f/k/a|fka|n[eé]e"
ALIAS_RE = rf"^(?P<alias>.+?)\s+(?:{ALIAS_KW})\s+(?P<core>.+)$"
# "(ID: 22383)" and "#98825" tags: 1 S1 name vs ~16k per S2/S3 file carry "#NNN" -> injected.
ID_TAG_RE = r"\(\s*id\s*:?\s*\d+\s*\)|#\s?\d{3,}"
DOMAIN_SUFFIX_RE = r"\s*\|\s*(?:www\.)?\S+\.(?:com|net|org|in)\b"  # "Basera Advisors | www.baseraadv.com"
APOS_RE = r"['`]"  # gabriela's -> gabrielas, not "gabriela s"

# 0/1/5/8/6 are the digits seen inside words in S2 names (2/3/9 ~0); 1 always stands for l.
DIGIT_MAP = {"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"}
PHONE_RE = r"\s\d[\d ]{6,}$"  # trailing run of >=7 digit/space chars, after punct strip
# Leading run only. All six measured as injected (noise_ops: India 1.3-1.7% add vs <=5/80k S1; US `the`
# 1.33% vs 0.13%); `m s` = M/s, M/S., M.S. prefix (29k S3 names). `shree` is organic (612 S1) -> kept.
HONORIFICS = ["m s", "smt", "shri", "sri", "mr", "dr", "the"]
HONORIFIC_RE = r"^(?:(?:" + "|".join(HONORIFICS) + r") )+"
NULLS = ["null", "n a", "na", "none", "nil"]  # whole address components (N/A, <NULL> after punct strip)
# Injected junk components: US "PO BOX 2624"/"PMB 12" (3.1% of pairs), India "Lucknow Hq Region".
JUNK_COMP_RE = r"^(?:po box|p o box|pmb)\b| region$"
ADDR_PREFIX_RE = r"\b(?:(?:h|house|door|d) )?(?:no|ndeg) ?(\d)"  # n° -> anyascii -> ndeg; # vanishes with punct
HN_PREFIX_RE = r"\bhn "  # "HN 753 E-1", "HN G-103": always a prefix
FLOORLIKE_RE = r"^0*\d+[a-z]? (?:floor|flr|fl|unit|apt|apartment|suite|ste|room|shop|plot|flat|block|sector)\b"
NUM_STREET_RE = r"^0*(\d+)[a-z]?(?: (?:bis|ter))? (.*\p{L}.*)$"  # leading zeros / letter suffix are noise
NONASCII = r"[^\x00-\x7F]"
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
            "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th"}


def _forms(spec: str) -> dict[str, str]:
    """'tx=texas|od=odisha=orissa' -> {form: canonical}; the first form of each item is canonical."""
    out = {}
    for item in spec.split("|"):
        forms = [f.strip() for f in item.split("=")]
        out |= dict.fromkeys(forms, forms[0])
    return out


@dataclass(frozen=True)
class Lex:
    legal: dict[str, str]
    street: dict[str, str]
    region: dict[str, str]


_LEGAL_US = _forms("llc|inc=incorporated=lnc|corp=corporation|co=company|ltd=limited|lp|llp|pllc|pc|plc")
# India: anyascii of Devanagari legal words (praivet, piraivet, praibhet, elelpi) + typos/truncations
# (noise_ops legal spelling pairs; the long tail of one-off 'private' typos is left unmapped, ~0.4% of pairs).
_LEGAL_IN = _forms("pvt=private=praivet=piraivet=praibhet=praivrr=pra|ltd=limited=limitet=limirrd=limtid=li|"
                   "llp=elelpi|co=company|inc=lnc|corp=corporation|llc")
_LEGAL_FR = _forms("sarl|sas|sasu|sa|eurl|sci|snc|ei|selarl")

_STREET_US = _forms("st=street=saint|rd=road|dr=drive|ave=avenue=av|ln=lane|blvd=boulevard|ct=court|"
                    "cir=circle|pl=place|ter=terrace|hwy=highway|trl=trail")
_STREET_IN = _forms("rd=road|st=street|nagar=ngr")
_STREET_FR = _forms("rue=r|ave=avenue=av|blvd=boulevard=bd|route=rte|allee=all|place=pl|impasse=imp|"
                    "chemin=ch=chem|st=saint|ste=sainte")

_REGION_US = _forms(
    "al=alabama|ak=alaska|az=arizona|ar=arkansas|ca=california|co=colorado|ct=connecticut|de=delaware|"
    "dc=district of columbia|fl=florida|ga=georgia|hi=hawaii|id=idaho|il=illinois|in=indiana|ia=iowa|"
    "ks=kansas|ky=kentucky|la=louisiana|me=maine|md=maryland|ma=massachusetts|mi=michigan|mn=minnesota|"
    "ms=mississippi|mo=missouri|mt=montana|ne=nebraska|nv=nevada|nh=new hampshire|nj=new jersey|"
    "nm=new mexico|ny=new york|nc=north carolina|nd=north dakota|oh=ohio|ok=oklahoma|or=oregon|"
    "pa=pennsylvania|ri=rhode island|sc=south carolina|sd=south dakota|tn=tennessee|tx=texas|ut=utah|"
    "vt=vermont|va=virginia|wa=washington|wv=west virginia|wi=wisconsin|wy=wyoming|pr=puerto rico")
# India: full=abbr, then anyascii of native-script state names (noise_ops state variant table).
_REGION_IN = _forms(
    "ap=andhra pradesh=amdhrprdes|ar=arunachal pradesh|as=assam|br=bihar|cg=chhattisgarh=ct|ga=goa|"
    "gj=gujarat=gujrat|hr=haryana=hriyana|hp=himachal pradesh|jh=jharkhand|ka=karnataka=krnatk|"
    "kl=kerala=keralam=kerlm|mp=madhya pradesh=mdhy prdes|mh=maharashtra=mharastr|mn=manipur|ml=meghalaya|"
    "mz=mizoram|nl=nagaland|od=odisha=orissa=or=od isa|pb=punjab=pmjab|rj=rajasthan=rajsthan|sk=sikkim|"
    "tn=tamil nadu=tmilnatu|tg=telangana=ts=telmgan|tr=tripura|up=uttar pradesh=uttr prdes|"
    "uk=uttarakhand=uttaranchal=ut|wb=west bengal=pscimbng|dl=delhi=dilli|jk=jammu and kashmir|la=ladakh|"
    "ch=chandigarh|py=puducherry=pondicherry|"
    "an=andaman and nicobar islands|ld=lakshadweep|dn=dadra and nagar haveli and daman and diu=dd")
# S1 carries the region, S2/S3 the departement (different hierarchy levels): both leave addr_tokens.
_REGION_FR = _forms(
    "auvergne rhone alpes|bourgogne franche comte|bretagne|centre val de loire|corse|grand est|"
    "hauts de france|ile de france|normandie|nouvelle aquitaine|occitanie|pays de la loire|"
    "provence alpes cote d azur|aisne|nord|oise|pas de calais|somme|charente|charente maritime|correze|"
    "creuse|dordogne|gironde|landes|lot et garonne|pyrenees atlantiques|deux sevres|vienne|haute vienne|"
    "loire atlantique|maine et loire|mayenne|sarthe|vendee")

LEX = {"US": Lex(_LEGAL_US, _STREET_US, _REGION_US),
       "India": Lex(_LEGAL_IN, _STREET_IN, _REGION_IN),
       "France": Lex(_LEGAL_FR, _STREET_FR, _REGION_FR)}
# Unseen country: legal forms only (the union has no conflicting canonicals); street/region maps collide.
DEFAULT_LEX = Lex(_LEGAL_US | _LEGAL_IN | _LEGAL_FR, {}, {})

E = pl.element()


def _ascii(s: pl.Series) -> pl.Series:
    # lean: Python loop, but only non-ASCII values (0-28% per file) pay the anyascii call
    return pl.Series(s.name, [v if v.isascii() else anyascii(v) for v in s.to_list()], dtype=pl.String)


def _fold(col: str, translit: bool) -> pl.Expr:
    e = pl.col(col).str.to_lowercase()
    return e.map_batches(_ascii, return_dtype=pl.String).str.to_lowercase() if translit else e


def _punct(e: pl.Expr) -> pl.Expr:
    """Brackets/punct -> space, whitespace collapsed (one regex: the run absorbs existing spaces)."""
    return e.str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def _tokens(e: pl.Expr) -> pl.Expr:
    return e.str.extract_all(r"\S+")


def _digit_token() -> pl.Expr:
    """Word with >=2 letters whose only digits are look-alikes (gonza1ez, 5hifflet); ordinals excluded."""
    return (E.str.contains(r"^[a-z01568]+$") & E.str.contains(r"[01568]")
            & E.str.contains(r"[a-z].*[a-z]") & ~E.str.contains(r"^\d+(?:st|nd|rd|th)$"))


def _words_re(words) -> str:
    return r"\b(?:" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b"


def _squash(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r" {2,}", " ").str.strip_chars()


NULL_STR = pl.lit(None, dtype=pl.String)

# Speed rule: inside list.eval only elementwise expressions (flattened, vectorised). filter/shift/cum_min
# there run once per row (~10us each); drops are when(...)->null + native list.drop_nulls instead.


def _dedupe_adjacent(df: pl.DataFrame) -> pl.DataFrame:
    x = df.select("_i", t=pl.col("_n").str.split(" ")).explode("t")
    # single letters exempt: "W & W Minerals" -> "w w minerals" is a real name, injected dups are words
    keep = (pl.col("t").ne_missing(pl.col("t").shift(1)) | pl.col("_i").ne(pl.col("_i").shift(1)).fill_null(True)
            | (pl.col("t").str.len_chars() == 1))
    agg = x.filter(keep).group_by("_i").agg(_n2=pl.col("t").drop_nulls().str.join(" "))
    return df.join(agg, on="_i", how="left").with_columns(_n=pl.col("_n2").fill_null("")).drop("_n2")


def _normalise_one(df: pl.DataFrame, lex: Lex, off: frozenset[str]) -> pl.DataFrame:
    def on(rule: str) -> bool:
        return rule not in off

    n, a = pl.col("_n"), pl.col("_a")
    df = df.with_columns(_n=_fold("business_name", on("anyascii")), _a=_fold("business_address", on("anyascii")))

    # ---- name (kept as a single-spaced string; tokens only where a per-token test is needed) ----
    if on("alias_split"):
        g = n.str.extract_groups(ALIAS_RE)
        df = df.with_columns(_alias=g.struct.field("alias"), _n=pl.coalesce(g.struct.field("core"), n))
    else:
        df = df.with_columns(_alias=NULL_STR)
    if on("id_tag"):
        df = df.with_columns(_n=n.str.replace_all(ID_TAG_RE, " "))
    if on("acronym_dots"):  # l.l.c. -> llc, e.u.r.l. -> eurl (before punct would split them into letters)
        df = df.with_columns(_n=n.str.replace_all(r"\b(\p{L})\.", "${1}"))
    if on("domain_strip"):
        df = df.with_columns(_n=n.str.replace_all(DOMAIN_SUFFIX_RE, " "))
        df = df.with_columns(_n=n.str.replace_all(r"\bwww\.|\.(?:com|net|org|in)\b", " "))
    df = df.with_columns(_n=_punct(n.str.replace_all(APOS_RE, "")), _alias=_punct(pl.col("_alias")))
    if on("trailing_phone"):
        df = df.with_columns(_n=n.str.replace(PHONE_RE, ""))
    if on("digit_fix"):  # after punct: token boundaries need it gone (and-5hifflet)
        fixed = E.str.replace_many(list(DIGIT_MAP), list(DIGIT_MAP.values()))
        df = df.with_columns(_n=n.str.split(" ").list.eval(pl.when(_digit_token()).then(fixed).otherwise(E))
                             .list.join(" "))
    if on("dedupe_adjacent"):
        df = _dedupe_adjacent(df)
    if on("legal_form"):
        legal_re = _words_re(lex.legal)
        core = _squash(n.str.replace_all(legal_re, ""))
        core = pl.when(core != n).then(core.str.replace(r" and$", "")).otherwise(core)  # "Ability & Co"
        found = (n.str.extract_all(legal_re).list.eval(E.replace_strict(lex.legal, return_dtype=pl.String))
                 .list.unique().list.sort().list.join(" "))
        keep = core != ""  # a name made only of legal words stays as-is
        df = df.with_columns(legal_form=pl.when(keep).then(found).otherwise(pl.lit("")),
                             _n=pl.when(keep).then(core).otherwise(n))
    else:
        df = df.with_columns(legal_form=pl.lit(""))
    if on("honorifics"):
        core = n.str.replace(HONORIFIC_RE, "")
        df = df.with_columns(_n=pl.when(core != "").then(core).otherwise(n))
    df = df.with_columns(core_name=n, name_tokens=_tokens(n),
                         aliases=pl.concat_list(pl.when(pl.col("_alias") != "").then(pl.col("_alias"))
                                                .otherwise(NULL_STR)).list.drop_nulls())

    # ---- address: commas kept as the component separator until the component split ----
    df = df.with_columns(_a=a.str.replace_all(APOS_RE, "").str.replace_all(r"[^\p{L}\p{N},]+", " ")
                         .str.replace_all(r"\s*,[\s,]*", " , ").str.strip_chars(" ,"))
    if on("number_prefix"):
        df = df.with_columns(_a=a.str.replace_all(ADDR_PREFIX_RE, "${1}").str.replace_all(HN_PREFIX_RE, ""))
    tok_map = (lex.street if on("street_type") else {}) | (ORDINALS if on("ordinal") else {})
    if tok_map:
        df = df.with_columns(_a=a.str.split(" ").list.eval(E.replace(tok_map)).list.join(" "))
    comp = E.str.replace_all(r"\s+", " ").str.strip_chars()
    bad = (comp == "") | (comp.is_in(NULLS) | comp.str.contains(JUNK_COMP_RE) if on("null_token") else False)
    df = df.with_columns(_c=a.str.split(",").list.eval(pl.when(bad).then(NULL_STR).otherwise(comp)).list.drop_nulls())
    c, r = pl.col("_c"), pl.col("_r")
    if on("admin_region") and lex.region:
        found = c.list.eval(E.replace_strict(lex.region, default=None, return_dtype=pl.String)).list.drop_nulls()
        rest = c.list.eval(pl.when(E.is_in(list(lex.region))).then(NULL_STR).otherwise(E)).list.drop_nulls()
        df = df.with_columns(admin_region=found.list.unique().list.sort().list.join(" "), _r=rest)
    else:
        df = df.with_columns(admin_region=pl.lit(""), _r=c)

    def num_street(k: int) -> pl.Expr:
        # both groups match together, so the first non-null of each list comes from the same component
        grp = pl.when(E.str.contains(FLOORLIKE_RE)).then(NULL_STR).otherwise(E.str.extract(NUM_STREET_RE, k))
        return r.list.eval(grp).list.drop_nulls().list.first().fill_null("")

    return df.with_columns(
        addr_number=num_street(1), addr_street_core=num_street(2),
        addr_tokens=_tokens(r.list.join(" ")),
        city=r.list.eval(pl.when(E.str.contains(r"\d")).then(NULL_STR).otherwise(E))
        .list.drop_nulls().list.last().fill_null(""),
        has_address=c.list.len() > 0,
        is_nonascii_raw=pl.col("business_name").str.contains(NONASCII)
        | pl.col("business_address").str.contains(NONASCII),
    ).select("_i", *OUT_COLS)


def normalise(df: pl.DataFrame, off: frozenset[str] = frozenset()) -> pl.DataFrame:
    unknown = set(off) - set(RULES)
    assert not unknown, f"unknown rules: {unknown}"
    df = df.with_row_index("_i")
    parts = [_normalise_one(p, LEX.get(c, DEFAULT_LEX), frozenset(off))
             for (c,), p in df.partition_by("country", as_dict=True).items()]
    return pl.concat(parts).sort("_i").drop("_i")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="first N rows per file; print only, write nothing")
    args = ap.parse_args()
    out = path("interim_dir")
    out.mkdir(parents=True, exist_ok=True)
    timing = []
    for split in ("train", "test"):
        for n in (1, 2, 3):
            t0 = time.perf_counter()
            raw = pl.from_pandas(load_source(split, n, nrows=args.smoke or None))
            t1 = time.perf_counter()
            df = normalise(raw)
            t2 = time.perf_counter()
            row = {"file": f"{split}_s{n}", "rows": df.height, "load_s": round(t1 - t0, 1),
                   "normalise_s": round(t2 - t1, 1), "peak_rss_mb": peak_rss_mb()}
            if args.smoke:
                row["projected_normalise_s_per_5.3M_rows"] = round((t2 - t1) / df.height * 5.3e6)
                print(row)
                with pl.Config(tbl_cols=-1, fmt_str_lengths=50, tbl_rows=6):
                    print(df.select("business_name", "core_name", "legal_form", "aliases", "addr_number",
                                    "addr_street_core", "admin_region", "city").sample(6, seed=42))
                continue
            df.write_parquet(out / f"norm_{split}_s{n}.parquet")
            row["total_s"] = round(time.perf_counter() - t0, 1)
            timing.append(row)
            print(row, flush=True)
    if not args.smoke:
        (out / "norm_timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
