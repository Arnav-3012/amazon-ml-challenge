"""M4 pair features for EVERY candidate pair of a split -> artifacts/features/{split}/part-*.parquet.

Country-agnostic: no country column is written. Country is used only to look up the per-country IDF that
blocking already used (idf_{split}.parquet), never as a value.

Pass 1, chunks of `features.chunk_rows` rows of candidates_{split}.parquet (file order):
  name (core_name): rapidfuzz ratio / token_sort / token_set / partial_ratio / Jaro-Winkler (cpdist, all
    cores); IDF-weighted Jaccard over tokens, sorted bigrams (the blocking A keys) and phonetic skeletons;
    best name-variant token_set when either side has aliases (NaN otherwise); |len| diff; non-ASCII per side.
  legal form code; country-marker code ("(France)" style suffix present on neither/one/both names).
  address: token_set, IDF Jaccard (tokens, skeletons), number code, street_core equality, has-address code.
  blocking: {A,B,C,X}_{score,rrank,srank} as written by block.finalize (null = channel did not keep it),
    n_channels_hit. is_s3.
  IDF Jaccard = sum idf(shared) / sum idf(union), idf = ln(N_country / df); a token missing from the idf table
  has df = 1 (the table keeps df >= 2), so it weighs ln N_country and can never be shared.
Pass 2, whole split, one value column at a time: for REL_COLS, v - max over the record's candidate S1s,
  v - max over the S1's candidates, rank within record and within S1 (1 = best, ties min), margin to the
  record's best OTHER S1 (> 0 only for the record's unique argmax); n_cand_rec, n_cand_s1. Computed over all
  candidates of the split: features on a 20% S1 subset would hide ~80% of each record's competing S1s,
  a train/test skew. Null A/B/C scores (channel cut) and missing addresses count as 0 here only.

Codes (int8): legal/number 0 both missing, 1 one missing, 2 equal, 3 different; marker/has_addr = number of
sides with it (0/1/2).

Memory (15.7 GB Windows machine): record tables are built in `features.prep_batch`-row batches (prep is row-wise),
pass 1 runs in `features.chunk_rows`-pair chunks and finds each pair's record rows by searchsorted on sorted int64
id keys + gather (no per-chunk hash join against the 10M-row S2/S3 table), and pass 2 runs one country at a time: a
pair's S1 and record always share a country and each country is one contiguous run of the candidate file (both
asserted), so no record/S1 window group crosses a block. map_tokens sorts before aggregating, so token-id list
order and the W float sum are the same on every run (the unsorted version was not). Progress: ASCII tqdm bars on
stdout with the process peak RSS, one "[features] ..." line before each stage.

Run from code/business_entity_resolution/:
  python -m src.features --selftest                     # batched/chunked/lookup/per-country == one pass (synthetic)
  python -m src.features --split train --smoke 300000   # first 300k candidate rows -> artifacts/features/train_smoke
  python -m src.features --split train
  python -m src.features --split test
"""
import argparse
import shutil
import sys

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from tqdm import tqdm

from .block import RANK_COLS, _bigrams, norm_path
from .io import CFG, StepLog, path, peak_rss_mb
from .phonetic import add_skeletons

ID_COLS = ["s1_id", "rec_id"]
SIB_COLS = ["sib_hit", "sib_anchor_margin", "n_sib_anchors", "sib_name_tset"]  # from gate.py --apply, sibling.enabled only
KEY_COLS = ["s1k", "reck"]  # int64 forms of the ids, pass 2 only; not written to the final parts
FIELDS = {  # IDF-Jaccard field -> (token column, idf channel, token prefix in that channel)
    "name": ("name_tokens", "A", ""),
    "nbg": ("_bg", "A", ""),
    "nskel": ("name_skel_tokens", "C", "n:"),
    "addr": ("addr_tokens", "B", ""),
    "askel": ("addr_skel_tokens", "C", "a:"),
}
REL_COLS = ["X_score", "A_score", "B_score", "C_score", "name_tset", "addr_tset"]
MARKER_RE = r"(?i)\(\s*(?:fr\p{L}nce|india|usa?)\s*\)"
FCFG = CFG["features"]
# unmatched_tokens' per-entity token set: name + address tokens, lowercased, unique (first-seen order). Built once
# per entity in prep instead of per pair in pass 1; replaces the name_toks + addr_toks list columns (753 -> 523 MB
# live on train S2/S3); pass 1 only ever takes set() of it.
U_TOKS = (pl.col("name_tokens").list.concat("addr_tokens").list.eval(pl.element().str.to_lowercase())
          .list.unique(maintain_order=True))
ENT_COLS = ["entity_id", "country", "business_name", "core_name", "legal_form", "aliases", "name_tokens",
            "addr_tokens", "addr_number", "addr_street_core", "has_address", "is_nonascii_raw"]


def out_dir(split: str, smoke: int = 0):
    return path("features_dir") / f"{split}{'_smoke' if smoke else ''}"


def id_key(col: str) -> pl.Expr:
    """'S2-437567938' -> 2437567938 (injective: io.load_source asserts the 'S{n}-' prefix, n is one digit)."""
    return pl.col(col).str.replace_all(r"\D", "").cast(pl.Int64, strict=True)


def vocab(split: str) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, np.ndarray]:
    """Per channel (country, tok, tid, w) with w = ln(N_country / df); lnN per country; w by tid."""
    f = path("interim_dir") / f"idf_{split}.parquet"
    # only the A/B/C rows are decoded (row-group filter); a lazy filter over the whole table (X is most of it)
    # reached a 3 GB working set for 162 MB of result. File order is kept, so tid is unchanged.
    abc = pl.from_arrow(pq.read_table(f, columns=["ch", "country", "tok", "n1", "n2", "n3", "n_country"],
                                      filters=[("ch", "in", ["A", "B", "C"])]))
    v = (abc.select("ch", "country", "tok",
                    w=(pl.col("n_country").cast(pl.Float64) / pl.sum_horizontal("n1", "n2", "n3")).log().cast(pl.Float32))
         .with_row_index("tid"))
    lnn = (abc.select("country", "n_country").unique()
           .select("country", lnN=pl.col("n_country").cast(pl.Float64).log().cast(pl.Float32)))
    return ({ch: v.filter(pl.col("ch") == ch).drop("ch") for ch in ("A", "B", "C")}, lnn, v["w"].to_numpy())


def map_tokens(df: pl.DataFrame, col: str, voc: pl.DataFrame, lnn: pl.DataFrame, pre: str) -> pl.DataFrame:
    """Per row: unique token ids found in the idf table (list[u32]) and W = sum idf over ALL its unique tokens."""
    tok = pl.lit(pre) + pl.col("tok") if pre else pl.col("tok")
    x = (df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r"), "country", tok=pl.col(col))
         .explode("tok", empty_as_null=True).drop_nulls("tok").with_columns(tok=tok).unique(["_r", "tok"])
         .join(voc, on=["country", "tok"], how="left").join(lnn, on="country", how="left")
         .sort("_r", "tok"))  # canonical order: without it the join order varies run to run -> W's float32 sum
    agg = x.group_by("_r", maintain_order=True).agg(t=pl.col("tid").drop_nulls(),  # and the list order did too
                                                    W=pl.col("w").fill_null(pl.col("lnN")).sum())
    return (pl.DataFrame({"_r": pl.int_range(df.height, dtype=pl.UInt32, eager=True)})
            .join(agg, on="_r", how="left", maintain_order="left")
            .select(pl.col("t").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))), pl.col("W").fill_null(0.0)))


def entities(split: str, n: int, voc: dict[str, pl.DataFrame], lnn: pl.DataFrame, b: tqdm | None = None,
             batch: int | None = None) -> pl.DataFrame:
    """prep() over the norm file in `batch`-row pieces, concatenated in file order. prep is row-wise (skeleton =
    per-token function, bigrams and token-id lists per row), so this equals one prep() over the whole file."""
    parts = []
    for rb in pq.ParquetFile(norm_path(split, n)).iter_batches(batch_size=batch or FCFG["prep_batch"], columns=ENT_COLS):
        parts.append(prep(pl.from_arrow(rb), voc, lnn))
        if b is not None:
            tick(b, rb.num_rows)
    return pl.concat(parts)


def prep(df: pl.DataFrame, voc: dict[str, pl.DataFrame], lnn: pl.DataFrame) -> pl.DataFrame:
    df = add_skeletons(df).with_columns(_bg=_bigrams(df["name_tokens"]))
    toks = {f: map_tokens(df, col, voc[ch], lnn, pre) for f, (col, ch, pre) in FIELDS.items()}
    return df.select(
        "entity_id", "country", core="core_name", legal="legal_form", aliases="aliases",
        addr=pl.col("addr_tokens").list.join(" "), num="addr_number", street="addr_street_core",
        has_addr="has_address", nonascii="is_nonascii_raw", marker=pl.col("business_name").str.contains(MARKER_RE),
        u_toks=U_TOKS,
    ).with_columns(*(s for f, t in toks.items() for s in (t["t"].alias(f"t_{f}"), t["W"].alias(f"W_{f}"))))


def _cp(x: list, y: list, scorer) -> np.ndarray:
    return cpdist(x, y, scorer=scorer, workers=-1, dtype=np.float32)


def idf_jaccard(ta: pl.Series, tb: pl.Series, wa: pl.Series, wb: pl.Series, w: np.ndarray) -> np.ndarray:
    """Row-aligned token-id lists -> sum w(shared) / sum w(union); NaN when either side has no tokens."""
    def ex(t: pl.Series) -> pl.DataFrame:
        return t.rename("t").to_frame().with_row_index("pid").explode("t", empty_as_null=True).drop_nulls()
    j = ex(ta).join(ex(tb), on=["pid", "t"], how="inner", maintain_order="left")  # fixed float summation order
    inter = np.bincount(j["pid"].to_numpy(), weights=w[j["t"].to_numpy()], minlength=ta.len())
    wa, wb = wa.to_numpy().astype(np.float64), wb.to_numpy().astype(np.float64)
    union = wa + wb - inter
    ok = (wa > 0) & (wb > 0)
    return np.where(ok, np.clip(inter / np.where(ok, union, 1.0), 0.0, 1.0), np.nan).astype(np.float32)


def best_alias(a: pl.DataFrame, b: pl.DataFrame) -> np.ndarray:
    """Max token_set over all (alias or core) x (alias or core) name variants; NaN when neither side has aliases."""
    out = np.full(a.height, np.nan, np.float32)
    idx = np.flatnonzero(((a["aliases"].list.len() > 0) | (b["aliases"].list.len() > 0)).to_numpy())
    if not len(idx):
        return out
    x = (pl.DataFrame({"pid": idx.astype(np.uint32), "ua": a["aliases"].gather(idx), "ca": a["core"].gather(idx),
                       "ub": b["aliases"].gather(idx), "cb": b["core"].gather(idx)})
         .select("pid", u=pl.concat_list("ua", "ca"), v=pl.concat_list("ub", "cb"))
         .explode("u", empty_as_null=True).explode("v", empty_as_null=True))
    best = (x.select("pid").with_columns(s=_cp(x["u"].to_list(), x["v"].to_list(), fuzz.token_set_ratio))
            .group_by("pid").agg(pl.col("s").max()))
    out[best["pid"].to_numpy()] = best["s"].to_numpy()
    return out


def code3(x: pl.Series, y: pl.Series) -> np.ndarray:
    ex, ey = (x == "").to_numpy(), (y == "").to_numpy()
    return np.select([ex & ey, ex | ey, (x == y).to_numpy()], [0, 1, 2], 3).astype(np.int8)


def s1_name_dup(s1: pl.DataFrame) -> pl.DataFrame:
    """entity_id -> count of OTHER S1 rows, same country, identical normalized core_name (self excluded)."""
    n = s1.select("entity_id", "country", "core").with_columns(
        dup=pl.len().over("country", "core").cast(pl.Int32) - 1)
    assert (n["dup"] >= 0).all(), "s1_name_dup: self-count underflow"
    return n.select("entity_id", "dup")


NUM_RE = r"\d+[A-Za-z]?"


def _num_toks(s: pl.Series) -> pl.Series:
    return s.str.extract_all(NUM_RE)


def _lev(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)


def num_rel(a_core: pl.Series, b_core: pl.Series, a_addr: pl.Series, b_addr: pl.Series) -> dict[str, np.ndarray]:
    """All \\d+[A-Za-z]? tokens from name+address both sides; align by min edit distance, classify best pair.
    # lean: row-by-row Python double loop over token pairs - fine for the 5% A/B, vectorize (e.g. rapidfuzz
    # cdist per row-batch or a numpy-only token-pair scan) before running on the full candidate set if kept."""
    xa = (a_core + " " + a_addr)
    xb = (b_core + " " + b_addr)
    ta, tb = _num_toks(xa).to_list(), _num_toks(xb).to_list()
    n = len(ta)
    cls = np.zeros(n, np.int8)  # 0 missing_both,1 missing_one_side,2 equal,3 zero_pad,4 prefix_trunc,
    # 5 suffix_trunc,6 transposed_digits,7 off_by_small,8 different
    absd = np.full(n, np.nan, np.float32)
    reld = np.full(n, np.nan, np.float32)
    for i in range(n):
        A, B = ta[i], tb[i]
        if not A and not B:
            cls[i] = 0
            continue
        if not A or not B:
            cls[i] = 1
            continue
        best_d, bp, bq = None, A[0], B[0]
        for p in A:
            for q in B:
                d = _lev(p, q)
                if best_d is None or d < best_d:
                    best_d, bp, bq = d, p, q
        if bp == bq:
            cls[i] = 2
        elif bp.lstrip("0") == bq.lstrip("0") and bp.lstrip("0") != "":
            cls[i] = 3
        elif bp.startswith(bq) or bq.startswith(bp):
            cls[i] = 4
        elif bp.endswith(bq) or bq.endswith(bp):
            cls[i] = 5
        elif sorted(bp) == sorted(bq) and bp != bq:
            cls[i] = 6
        else:
            try:
                d = abs(int("".join(ch for ch in bp if ch.isdigit())) - int("".join(ch for ch in bq if ch.isdigit())))
                cls[i] = 7 if 1 <= d <= 3 else 8
            except ValueError:
                cls[i] = 8
        try:
            na, nb = float("".join(ch for ch in bp if ch.isdigit()) or 0), float("".join(ch for ch in bq if ch.isdigit()) or 0)
            absd[i] = abs(na - nb)
            reld[i] = absd[i] / max(na, nb, 1.0)
        except ValueError:
            pass
    return {"num_rel_class": cls, "num_rel_absdiff": absd, "num_rel_reldiff": reld}


def idf_lookup_table(voc_a: pl.DataFrame) -> dict[str, dict[str, float]]:
    """{country: {tok: w}} over the channel-A vocab (the one unmatched_tokens reads). Same (country, tok) -> w
    mapping as a (country, tok)-tuple dict (last row wins on a repeat in both), without 3.2M key tuples and
    per-key country strings."""
    return {c: dict(zip(g["tok"].to_list(), g["w"].to_list()))
            for (c,), g in voc_a.select("country", "tok", "w").group_by(["country"], maintain_order=True)}


def unmatched_tokens(ua: pl.Series, ub: pl.Series, country: pl.Series, idf_lookup: dict) -> dict[str, np.ndarray]:
    """Tokens (name+addr, lowercased = U_TOKS per side, channel A/B matching = exact-token set difference) on
    either side not present on the other side. Counts per side, max log-IDF (idf_lookup[country][tok], global
    vocab, 0.0 for tokens absent from the df table) of the unmatched tokens, and best rapidfuzz.ratio between
    the two unmatched-token sets joined as strings."""
    ta_l, tb_l, ctry = ua.to_list(), ub.to_list(), country.to_list()
    unmatched_a = [sorted(set(x) - set(y)) for x, y in zip(ta_l, tb_l)]
    unmatched_b = [sorted(set(y) - set(x)) for x, y in zip(ta_l, tb_l)]
    cnt_a = np.array([len(r) for r in unmatched_a], np.int16)
    cnt_b = np.array([len(r) for r in unmatched_b], np.int16)
    max_idf_a = np.array([max((idf_lookup.get(co, {}).get(t, 0.0) for t in row), default=0.0)
                          for row, co in zip(unmatched_a, ctry)], np.float32)
    max_idf_b = np.array([max((idf_lookup.get(co, {}).get(t, 0.0) for t in row), default=0.0)
                          for row, co in zip(unmatched_b, ctry)], np.float32)
    sa = [" ".join(row) for row in unmatched_a]
    sb = [" ".join(row) for row in unmatched_b]
    best_sim = _cp(sa, sb, fuzz.ratio)
    return {"unmatched_cnt_a": cnt_a, "unmatched_cnt_b": cnt_b,
            "unmatched_max_idf_a": max_idf_a, "unmatched_max_idf_b": max_idf_b,
            "unmatched_char_sim": best_sim}


def name_tset_dict(a_core: pl.Series, b_core: pl.Series, dict_map: dict | None) -> np.ndarray:
    """name_tset recomputed after remapping tokens via token_dict.parquet (field='name'); -1 sentinel if the
    parquet doesn't exist (M5 dict not yet mined)."""
    n = a_core.len()
    if dict_map is None:
        return np.full(n, -1.0, np.float32)
    def remap(s: str) -> str:
        return " ".join(dict_map.get(t, t) for t in s.split())
    ra = [remap(x) for x in a_core.to_list()]
    rb = [remap(x) for x in b_core.to_list()]
    return _cp(ra, rb, fuzz.token_set_ratio)


def load_token_dict() -> dict | None:
    """{src_token: tgt_token} for field == 'name', or None if artifacts/interim/token_dict.parquet is absent."""
    p = path("interim_dir") / "token_dict.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p).filter(pl.col("field") == "name")
    return dict(zip(d["s"].to_list(), d["t"].to_list()))


def lookup(table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(sorted int64 id keys, row of each) for a record table. A chunk finds its rows by searchsorted + gather: no
    hash table over the full table (the per-chunk hash join against the 10.3M-row S2/S3 table cost ~2.3 GB)."""
    k = table.select(id_key("entity_id")).to_series().to_numpy()
    order = np.argsort(k, kind="stable")
    return k[order], order


def rows_for(ids: pl.Series, table: pl.DataFrame, lk: tuple[np.ndarray, np.ndarray]) -> pl.DataFrame:
    """table rows for `ids`, in ids order (the rows a left join on entity_id would return)."""
    keys = ids.to_frame().select(id_key(ids.name)).to_series().to_numpy()
    sk, order = lk
    pos = np.minimum(np.searchsorted(sk, keys), len(sk) - 1)
    assert (sk[pos] == keys).all(), "candidate id missing from norm cache"
    return table.gather(order[pos])


def pair_features(c: pl.DataFrame, s1: pl.DataFrame, rec: pl.DataFrame, w: np.ndarray,
                  idf_lookup: dict, dict_map: dict | None, lk1: tuple | None = None,
                  lkr: tuple | None = None) -> pl.DataFrame:
    a = rows_for(c["s1_id"], s1, lk1 or lookup(s1))
    b = rows_for(c["rec_id"], rec, lkr or lookup(rec))
    n1, n2 = a["core"].to_list(), b["core"].to_list()
    ad1, ad2 = a["addr"].to_list(), b["addr"].to_list()
    no_addr = ((a["addr"] == "") | (b["addr"] == "")).to_numpy()
    s_eq = (a["street"] == b["street"]).to_numpy().astype(np.float32)
    f = {
        "name_ratio": _cp(n1, n2, fuzz.ratio), "name_tsort": _cp(n1, n2, fuzz.token_sort_ratio),
        "name_tset": _cp(n1, n2, fuzz.token_set_ratio), "name_partial": _cp(n1, n2, fuzz.partial_ratio),
        "name_jw": _cp(n1, n2, JaroWinkler.normalized_similarity), "alias_best": best_alias(a, b),
        "name_len_diff": (a["core"].str.len_chars().cast(pl.Int32) - b["core"].str.len_chars().cast(pl.Int32))
        .abs().cast(pl.Int16).to_numpy(),
        "s1_nonascii": a["nonascii"].cast(pl.Int8).to_numpy(), "rec_nonascii": b["nonascii"].cast(pl.Int8).to_numpy(),
        "legal_code": code3(a["legal"], b["legal"]),
        "marker_code": (a["marker"].cast(pl.Int8) + b["marker"].cast(pl.Int8)).to_numpy(),
        "addr_tset": np.where(no_addr, np.nan, _cp(ad1, ad2, fuzz.token_set_ratio)).astype(np.float32),
        "num_code": code3(a["num"], b["num"]),
        "street_eq": np.where(((a["street"] == "") | (b["street"] == "")).to_numpy(), np.nan, s_eq).astype(np.float32),
        "has_addr_code": (a["has_addr"].cast(pl.Int8) + b["has_addr"].cast(pl.Int8)).to_numpy(),
        "rec_no_addr": (~b["has_addr"]).cast(pl.Int8).to_numpy(),
        "is_s3": c["rec_id"].str.starts_with("S3").cast(pl.Int8).to_numpy(),
    }
    for k in FIELDS:
        f[f"{k}_idfj"] = idf_jaccard(a[f"t_{k}"], b[f"t_{k}"], a[f"W_{k}"], b[f"W_{k}"], w)
    f["s1_name_dup"] = a["dup"].to_numpy()
    f.update(num_rel(a["core"], b["core"], a["addr"], b["addr"]))
    f.update(unmatched_tokens(a["u_toks"], b["u_toks"], a["country"], idf_lookup))
    f["name_tset_dict"] = name_tset_dict(a["core"], b["core"], dict_map)
    sib = [col for col in SIB_COLS if col in c.columns]
    return pl.concat([c.select(*ID_COLS, id_key("s1_id").alias("s1k"), id_key("rec_id").alias("reck"),
                               *RANK_COLS, "n_channels_hit", *sib),
                      pl.DataFrame(f)], how="horizontal_extend")


def say(msg: str) -> None:
    print(f"[features] {msg}", flush=True)


def bar(total: int, desc: str, unit: str = "row") -> tqdm:
    """ASCII-only bar on stdout (no cp1252 crash); tick() refreshes the process peak RSS in its postfix."""
    return tqdm(total=total, desc=desc, unit=unit, unit_scale=True, ascii=True, file=sys.stdout, ncols=110,
                mininterval=5)


def tick(b: tqdm, n: int) -> None:
    b.set_postfix_str(f"peak_rss={peak_rss_mb()}MB", refresh=False)
    b.update(n)


def _key_country(split: str, sources: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sorted int64 id keys of the given sources' entities and their country codes (for searchsorted lookups)."""
    df = pl.concat([pl.read_parquet(norm_path(split, n), columns=["entity_id", "country"]) for n in sources])
    df = df.select(k=id_key("entity_id"), c="country").sort("k")
    names = sorted(df["c"].unique().to_list())
    code = df["c"].replace_strict(names, list(range(len(names))), return_dtype=pl.Int8).to_numpy()
    return df["k"].to_numpy(), code, names


def _lookup(keys: np.ndarray, sorted_keys: np.ndarray, code: np.ndarray) -> np.ndarray:
    i = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    assert (sorted_keys[i] == keys).all(), "candidate id missing from the norm cache"
    return code[i]


def country_blocks(split: str, parts: list, b: tqdm | None = None) -> list[tuple[str, int, int]]:
    """(country, first row, end row) runs of the pass-1 rows. Asserts (1) every pair's S1 and record have the same
    country and (2) each country is ONE contiguous run. Pass 2 windows group by record and by S1, so with (1) and
    (2) a per-country window equals the whole-split window."""
    k1, c1, n1 = _key_country(split, (1,))
    kr, cr, nr = _key_country(split, (2, 3))
    assert n1 == nr, (n1, nr)
    codes = []
    for p in parts:
        k = pl.read_parquet(p, columns=KEY_COLS)
        a, r = _lookup(k["s1k"].to_numpy(), k1, c1), _lookup(k["reck"].to_numpy(), kr, cr)
        assert (a == r).all(), f"{p.name}: a candidate pair crosses countries"
        codes.append(a)
        if b is not None:
            tick(b, 1)
    code = np.concatenate(codes) if codes else np.empty(0, np.int8)
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]]) if code.size else np.empty(0, np.int64)
    ends = np.r_[starts[1:], code.size]
    runs = [(n1[int(code[s])], int(s), int(e)) for s, e in zip(starts, ends)]
    assert len({c for c, _, _ in runs}) == len(runs), f"a country is split over several runs: {[(c, s, e) for c, s, e in runs]}"
    return runs


def rows(parts: list, offs: np.ndarray, cols: list[str], lo: int, hi: int) -> pl.DataFrame:
    """Rows [lo, hi) of the pass-1 parts (in order), only the given columns."""
    frames = [pl.scan_parquet(p).select(cols).slice(int(max(lo, a) - a), int(min(hi, e) - max(lo, a)))
              for p, a, e in zip(parts, offs[:-1], offs[1:]) if max(lo, a) < min(hi, e)]
    return pl.concat(frames).collect()


def n_cand(keys: pl.DataFrame) -> pl.DataFrame:
    return keys.select(n_cand_rec=pl.len().over("reck").cast(pl.Int32), n_cand_s1=pl.len().over("s1k").cast(pl.Int32))


def name_hits95(x: pl.DataFrame) -> pl.DataFrame:
    """x(reck, name_tset) -> rec_name_hits95 = # of the record's candidate S1s with name_tset >= 95."""
    return x.select(rec_name_hits95=(pl.col("name_tset") >= 95).sum().over("reck").cast(pl.Int32))


def relative(keys: pl.DataFrame, v: pl.Series) -> pl.DataFrame:
    c, x, mr = v.name, pl.col("_v"), pl.col("_mr")
    # numpy-built columns carry NaN, not null: NaN survives fill_null and polars ranks it above every value
    return (keys.with_columns(_v=v.fill_nan(None).fill_null(0))
            .with_columns(_mr=x.max().over("reck"), _ms=x.max().over("s1k"))
            .with_columns(_m2=pl.when(x < mr).then(x).max().over("reck").fill_null(0),  # 0 = no other candidate
                          _tie=(x == mr).sum().over("reck"))
            .select((x - mr).alias(f"{c}_drec"), (x - pl.col("_ms")).alias(f"{c}_ds1"),
                    x.rank("min", descending=True).over("reck").cast(pl.Int32).alias(f"{c}_rkrec"),
                    x.rank("min", descending=True).over("s1k").cast(pl.Int32).alias(f"{c}_rks1"),
                    (x - pl.when((x == mr) & (pl.col("_tie") == 1)).then(pl.col("_m2")).otherwise(mr))
                    .alias(f"{c}_gaprec")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"))
    ap.add_argument("--smoke", type=int, default=0, help="first N candidate rows only (relative feats within them)")
    ap.add_argument("--selftest", action="store_true", help="batched/chunked/lookup/per-country == one pass (synthetic)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # never the cp1252 console crash
    if a.selftest:
        return selftest()
    assert a.split, "--split is required"
    log, out = StepLog(), out_dir(a.split, a.smoke)
    shutil.rmtree(out, ignore_errors=True)
    base_dir, rel_dir = out / "_base", out / "_rel"
    base_dir.mkdir(parents=True)
    rel_dir.mkdir()

    say(f"idf vocab: split={a.split}")
    voc, lnn, w = vocab(a.split)
    log("idf vocab", n_tokens=len(w))
    say("idf_lookup (channel A, per country) + token dict")
    idf_lookup = idf_lookup_table(voc["A"])
    dict_map = load_token_dict()
    log("idf_lookup + token dict", n_lookup=sum(map(len, idf_lookup.values())), dict_map=dict_map is not None)
    n_ent = {n: pq.ParquetFile(norm_path(a.split, n)).metadata.num_rows for n in (1, 2, 3)}
    say(f"record tables: S1 {n_ent[1]:,} + S2/S3 {n_ent[2] + n_ent[3]:,} entities, batches of {FCFG['prep_batch']:,}")
    with bar(sum(n_ent.values()), "record tables") as b:
        s1 = entities(a.split, 1, voc, lnn, b)
        dup = s1_name_dup(s1.select("entity_id", "country", "core"))
        s1 = s1.join(dup, on="entity_id", how="left", maintain_order="left")
        log("S1 entities", rows=s1.height)
        rec = pl.concat([entities(a.split, n, voc, lnn, b) for n in (2, 3)])
    del voc, dup
    lk1, lkr = lookup(s1), lookup(rec)
    log("S2+S3 entities", rows=rec.height)

    pf = pq.ParquetFile(path("interim_dir") / f"candidates_{a.split}.parquet")
    cand_cols = pf.schema_arrow.names
    sib_present = [col for col in SIB_COLS if col in cand_cols]
    total = min(a.smoke, pf.metadata.num_rows) if a.smoke else pf.metadata.num_rows
    say(f"pass1: {total:,} candidate pairs, chunks of {FCFG['chunk_rows']:,}")
    parts, sizes, done = [], [], 0
    with bar(total, "pass1") as b:
        for i, batch in enumerate(pf.iter_batches(batch_size=FCFG["chunk_rows"],
                                                  columns=[*ID_COLS, *RANK_COLS, "n_channels_hit", *sib_present])):
            c = pl.from_arrow(batch)
            if a.smoke:
                c = c.head(a.smoke - done)
            p = base_dir / f"part-{i:05d}.parquet"
            pair_features(c, s1, rec, w, idf_lookup, dict_map, lk1, lkr).write_parquet(p)
            parts.append(p)
            sizes.append(c.height)
            done += c.height
            tick(b, c.height)
            log(f"pass1 part {i}", rows=c.height, total=done)
            if a.smoke and done >= a.smoke:
                break
    del s1, rec, lk1, lkr, idf_lookup

    offs = np.r_[0, np.cumsum(sizes)]
    say(f"pass2: country blocks over {len(parts)} parts")
    with bar(len(parts), "country check", unit="part") as b:
        blocks = country_blocks(a.split, parts, b)
    say("pass2 blocks: " + ", ".join(f"{c} {e - s_:,}" for c, s_, e in blocks))
    groups = {g: [] for g in ("_n", "_tset95", *REL_COLS)}
    with bar(len(blocks) * len(groups), "pass2", unit="step") as b:
        for ci, (country, lo, hi) in enumerate(blocks):
            say(f"pass2 country={country} rows={(hi - lo) / 1e6:.1f}M")
            keys = rows(parts, offs, KEY_COLS, lo, hi)
            for g, fn in (("_n", lambda: n_cand(keys)),
                          ("_tset95", lambda: name_hits95(rows(parts, offs, ["reck", "name_tset"], lo, hi)))):
                f = rel_dir / f"{g}_{ci:02d}.parquet"
                fn().write_parquet(f)
                groups[g].append(f)
                tick(b, 1)
            for col in REL_COLS:
                f = rel_dir / f"{col}_{ci:02d}.parquet"
                relative(keys, rows(parts, offs, [col], lo, hi)[col]).write_parquet(f)
                groups[col].append(f)
                tick(b, 1)
                log(f"pass2 {country} {col}", rows=hi - lo)
            del keys

    say(f"assemble: {len(parts)} parts")
    off = 0
    with bar(len(parts), "assemble", unit="part") as b:
        for i, p in enumerate(parts):
            base = pl.read_parquet(p).drop(KEY_COLS)
            rel = [pl.scan_parquet(fs).slice(off, base.height).collect() for fs in groups.values()]
            df = pl.concat([base, *rel], how="horizontal_extend")
            df = df.with_columns(is_noaddr_ambiguous=((pl.col("rec_no_addr") == 1) & (pl.col("rec_name_hits95") >= 2))
                                 .cast(pl.Int8)).drop("rec_no_addr")
            assert "country" not in df.columns
            df.write_parquet(out / f"part-{i:05d}.parquet")
            off += base.height
            tick(b, 1)
    shutil.rmtree(base_dir)
    shutil.rmtree(rel_dir)
    log("assemble", rows=off, n_cols=df.width, parts=len(parts))

    sib_out = [c for c in SIB_COLS if c in df.columns]
    if CFG["gate"].get("sibling", {}).get("enabled", False):
        assert sib_out == SIB_COLS, f"gate.sibling.enabled but candidates_{a.split} is missing {set(SIB_COLS) - set(sib_out)}"
        # `df` is only the last part here: count over every written part
        n_sib = pl.scan_parquet(out / "part-*.parquet").select(pl.col("sib_hit").sum()).collect().item()
        assert n_sib > 0, f"gate.sibling.enabled but no sib_hit rows in {a.split} output"
    else:
        assert not sib_out, f"gate.sibling disabled but candidates_{a.split} still carries {sib_out}"
    # test only: train is rebuilt first, so when train runs the test dir is still the previous (stale) build
    other_dir = out_dir("train", a.smoke)
    if a.split == "test" and not a.smoke and other_dir.exists():
        other_cols = set(pl.scan_parquet(other_dir / "part-00000.parquet").collect_schema().names())
        assert set(df.columns) == other_cols, f"feature columns differ train vs test: {set(df.columns) ^ other_cols}"

    from .train import feature_cols  # local import: avoid a features<->train import cycle at module load
    feats = feature_cols(a.split if not a.smoke else f"{a.split}_smoke")
    print(f"model feature list ({len(feats)}): {feats}")  # schema-driven (feature_cols reads schema, not rows)

    log.dump(out / "features_timing.json", split=a.split, smoke=a.smoke, rows=off)


def selftest() -> None:
    """Each resequenced step against its one-pass form on synthetic data: batched prep (exact, and deterministic),
    lookup == left join, chunked pair_features, per-country pass-2 columns, and country_blocks' two assertions."""
    import tempfile
    from pathlib import Path
    rng = np.random.default_rng(0)
    words = ["ganesh", "traders", "sri", "shivam", "private", "eastern", "systems", "kumar", "summit", "rd", "st", "12", "12b"]

    def ents(n: int, pre: str, country: list[str]) -> pl.DataFrame:
        nm = [[str(x) for x in rng.choice(words, rng.integers(1, 4))] for _ in range(n)]
        ad = [[str(x) for x in rng.choice(words, rng.integers(0, 3))] for _ in range(n)]
        return pl.DataFrame({
            "entity_id": [f"{pre}-{i:09d}" for i in range(n)], "country": [country[i % len(country)] for i in range(n)],
            "business_name": [" ".join(x) + (" (India)" if i % 7 == 0 else "") for i, x in enumerate(nm)],
            "core_name": [" ".join(x) for x in nm], "legal_form": [["", "llc", "pvt ltd"][i % 3] for i in range(n)],
            "aliases": [[] if i % 5 else [" ".join(nm[(i + 1) % n])] for i in range(n)], "name_tokens": nm,
            "addr_tokens": ad, "addr_number": [str(i % 4) if i % 3 else "" for i in range(n)],
            "addr_street_core": [" ".join(x) for x in ad], "has_address": [bool(x) for x in ad],
            "is_nonascii_raw": [i % 11 == 0 for i in range(n)]})
    countries = ["India", "US"]
    toks = sorted({t for w_ in words for t in (w_, "n:" + w_[:3], "a:" + w_[:3])})
    voc = {ch: pl.DataFrame({"country": [c for c in countries for _ in toks], "tok": toks * len(countries)})
           .with_row_index("tid").with_columns(w=pl.Series(rng.random(len(toks) * len(countries)) * 3, dtype=pl.Float32))
           for ch in ("A", "B", "C")}
    lnn = pl.DataFrame({"country": countries, "lnN": [3.0, 3.2]}).with_columns(pl.col("lnN").cast(pl.Float32))
    w = voc["A"]["w"].to_numpy()
    tup = dict(zip(zip(voc["A"]["country"].to_list(), voc["A"]["tok"].to_list()), voc["A"]["w"].to_list()))
    idf_lookup = idf_lookup_table(voc["A"])
    assert {(c, t): v for c, d_ in idf_lookup.items() for t, v in d_.items()} == tup, "nested idf_lookup != tuple dict"
    raw = ents(97, "S1", countries)
    whole = prep(raw, voc, lnn)
    batched = pl.concat([prep(raw.slice(lo, 10), voc, lnn) for lo in range(0, raw.height, 10)])
    assert whole.equals(batched), "batched prep != whole prep (exact, incl. token-list order and W)"
    assert whole.equals(prep(raw, voc, lnn)), "prep is not deterministic"
    whole = whole.join(s1_name_dup(whole.select("entity_id", "country", "core")), on="entity_id", how="left",
                       maintain_order="left")
    rec = prep(ents(150, "S2", countries), voc, lnn)
    cand = pl.DataFrame({"s1_id": [whole["entity_id"][int(i)] for i in rng.integers(0, whole.height, 400)],
                         "rec_id": [rec["entity_id"][int(i)] for i in rng.integers(0, rec.height, 400)]})
    cand = cand.with_columns(*(pl.lit(None, pl.Float32).alias(c) for c in RANK_COLS), n_channels_hit=pl.lit(1, pl.Int8))
    for ids, tab in ((cand["s1_id"], whole), (cand["rec_id"], rec)):
        joined = ids.to_frame().join(tab, left_on=ids.name, right_on="entity_id", how="left", maintain_order="left")
        got = rows_for(ids, tab, lookup(tab))
        assert got.drop("entity_id").equals(joined.drop(ids.name)) and got["entity_id"].equals(ids.rename("entity_id")), \
            "lookup rows != left-join rows"
    try:
        rows_for(pl.Series("s1_id", ["S1-999999999"]), whole, lookup(whole))
        raise RuntimeError("rows_for did not fire on a missing id")
    except AssertionError as e:
        assert "missing" in str(e), e
    def unmatched_m52(a_name, b_name, a_addr, b_addr, country, lk):  # the M5-2 formulation, verbatim
        ta = (a_name.list.concat(a_addr)).list.eval(pl.element().str.to_lowercase()).list.unique()
        tb = (b_name.list.concat(b_addr)).list.eval(pl.element().str.to_lowercase()).list.unique()
        ta_l, tb_l, ctry = ta.to_list(), tb.to_list(), country.to_list()
        ua_ = [sorted(set(x) - set(y)) for x, y in zip(ta_l, tb_l)]
        ub_ = [sorted(set(y) - set(x)) for x, y in zip(ta_l, tb_l)]
        return {"unmatched_cnt_a": np.array([len(r) for r in ua_], np.int16),
                "unmatched_cnt_b": np.array([len(r) for r in ub_], np.int16),
                "unmatched_max_idf_a": np.array([max((lk.get((co, t), 0.0) for t in r), default=0.0)
                                                 for r, co in zip(ua_, ctry)], np.float32),
                "unmatched_max_idf_b": np.array([max((lk.get((co, t), 0.0) for t in r), default=0.0)
                                                 for r, co in zip(ub_, ctry)], np.float32),
                "unmatched_char_sim": _cp([" ".join(r) for r in ua_], [" ".join(r) for r in ub_], fuzz.ratio)}
    rb = ents(150, "S2", countries)
    ra, rb = raw[rng.integers(0, raw.height, 400)], rb[rng.integers(0, rb.height, 400)]
    ra = ra.with_columns(name_tokens=pl.when(pl.int_range(pl.len()) % 3 == 0)  # mixed case: lowercasing must matter
                         .then(pl.col("name_tokens").list.eval(pl.element().str.to_uppercase())).otherwise("name_tokens"))
    ref = unmatched_m52(ra["name_tokens"], rb["name_tokens"], ra["addr_tokens"], rb["addr_tokens"], ra["country"], tup)
    got = unmatched_tokens(ra.select(U_TOKS).to_series(), rb.select(U_TOKS).to_series(), ra["country"], idf_lookup)
    for k_, v_ in ref.items():
        assert np.array_equal(v_, got[k_], equal_nan=True) and v_.dtype == got[k_].dtype, f"unmatched_tokens {k_} != M5-2"
    assert (ref["unmatched_max_idf_a"] > 0).any() and (ref["unmatched_cnt_a"] > 0).any(), "unmatched test is vacuous"
    for dm in (None, {"sri": "shri"}):
        one = pair_features(cand, whole, rec, w, idf_lookup, dm)
        lk1, lkr = lookup(whole), lookup(rec)
        chunked = pl.concat([pair_features(cand.slice(lo, 64), whole, rec, w, idf_lookup, dm, lk1, lkr)
                             for lo in range(0, cand.height, 64)])
        assert one.equals(chunked), "chunked pass 1 != one pass"
    # per-country pass 2: three contiguous country blocks whose groups never cross blocks
    n = 3000
    block = np.repeat([0, 1, 2], [1100, 1200, 700])
    keys = pl.DataFrame({"s1k": block * 10_000 + rng.integers(0, 90, n), "reck": block * 10_000 + rng.integers(0, 400, n)})
    v = pl.Series("X_score", np.where(rng.random(n) < 0.1, np.nan, rng.random(n)).astype(np.float32))
    ts = pl.Series("name_tset", rng.integers(80, 101, n).astype(np.float32))
    cuts = [0, 1100, 2300, 3000]
    sl = list(zip(cuts[:-1], cuts[1:]))
    assert relative(keys, v).equals(pl.concat([relative(keys.slice(s_, e - s_), v.slice(s_, e - s_)) for s_, e in sl])), \
        "per-country relative != whole"
    assert n_cand(keys).equals(pl.concat([n_cand(keys.slice(s_, e - s_)) for s_, e in sl])), "per-country n_cand != whole"
    kt = keys.with_columns(ts)
    assert name_hits95(kt).equals(pl.concat([name_hits95(kt.slice(s_, e - s_)) for s_, e in sl])), \
        "per-country rec_name_hits95 != whole"
    # country_blocks on real-shaped files: passes on contiguous same-country pairs, fires on the two violations
    global norm_path
    real_norm = norm_path
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cc = ["France", "France", "India", "India", "US", "US"]
        pl.DataFrame({"entity_id": [f"S1-{i:09d}" for i in range(6)], "country": cc}).write_parquet(d / "n1.parquet")
        pl.DataFrame({"entity_id": [f"S2-{i:09d}" for i in range(6)], "country": cc}).write_parquet(d / "n2.parquet")
        pl.DataFrame({"entity_id": ["S3-000000000"], "country": ["US"]}).write_parquet(d / "n3.parquet")
        norm_path = lambda split, n_: d / f"n{n_}.parquet"  # noqa: E731
        try:
            def part(name: str, s1s: list[int], recs: list[str]) -> Path:
                pth = d / name
                pl.DataFrame({"s1k": [1_000_000_000 + i for i in s1s],
                              "reck": [int(r[1]) * 1_000_000_000 + int(r[3:]) for r in recs]}).write_parquet(pth)
                return pth
            ok = [part("a.parquet", [0, 1, 2], ["S2-0", "S2-1", "S2-2"]), part("b.parquet", [3, 4, 5], ["S2-3", "S3-0", "S2-5"])]
            assert country_blocks("x", ok) == [("France", 0, 2), ("India", 2, 4), ("US", 4, 6)]
            for bad, why in ((part("c.parquet", [0, 2], ["S2-0", "S2-0"]), "crosses"),
                             (part("e.parquet", [0, 2, 1], ["S2-0", "S2-2", "S2-1"]), "split over")):
                try:
                    country_blocks("x", [bad])
                    raise RuntimeError(f"country_blocks did not fire: {why}")
                except AssertionError as e:
                    assert why in str(e), e
        finally:
            norm_path = real_norm
    print(f"features self-test OK (batched prep {raw.height} rows / 10 exact + deterministic, lookup == left join, "
          f"unmatched_tokens (U_TOKS + per-country lookup) == M5-2 on 400 mixed-case pairs, "
          f"pass 1 {cand.height} pairs / 64 with and without dict_map, pass 2 (n_cand, rec_name_hits95, relative) "
          f"3 country blocks of {n} pairs, country_blocks asserts fire on crossing + split)")


if __name__ == "__main__":
    main()
