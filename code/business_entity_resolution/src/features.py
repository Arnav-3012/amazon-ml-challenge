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

Run from code/business_entity_resolution/:
  python -m src.features --split train --smoke 300000   # first 300k candidate rows -> artifacts/features/train_smoke
  python -m src.features --split train
  python -m src.features --split test
"""
import argparse
import shutil

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .block import RANK_COLS, _bigrams, norm_path
from .io import CFG, StepLog, path
from .phonetic import add_skeletons

ID_COLS = ["s1_id", "rec_id"]
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
ENT_COLS = ["entity_id", "country", "business_name", "core_name", "legal_form", "aliases", "name_tokens",
            "addr_tokens", "addr_number", "addr_street_core", "has_address", "is_nonascii_raw"]


def out_dir(split: str, smoke: int = 0):
    return path("features_dir") / f"{split}{'_smoke' if smoke else ''}"


def id_key(col: str) -> pl.Expr:
    """'S2-437567938' -> 2437567938 (injective: io.load_source asserts the 'S{n}-' prefix, n is one digit)."""
    return pl.col(col).str.replace_all(r"\D", "").cast(pl.Int64, strict=True)


def vocab(split: str) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, np.ndarray]:
    """Per channel (country, tok, tid, w) with w = ln(N_country / df); lnN per country; w by tid."""
    v = (pl.scan_parquet(path("interim_dir") / f"idf_{split}.parquet").filter(pl.col("ch").is_in(["A", "B", "C"]))
         .select("ch", "country", "tok",
                 w=(pl.col("n_country").cast(pl.Float64) / pl.sum_horizontal("n1", "n2", "n3")).log().cast(pl.Float32))
         .collect().with_row_index("tid"))
    lnn = (pl.scan_parquet(path("interim_dir") / f"idf_{split}.parquet").select("country", "n_country").unique()
           .collect().select("country", lnN=pl.col("n_country").cast(pl.Float64).log().cast(pl.Float32)))
    return ({ch: v.filter(pl.col("ch") == ch).drop("ch") for ch in ("A", "B", "C")}, lnn, v["w"].to_numpy())


def map_tokens(df: pl.DataFrame, col: str, voc: pl.DataFrame, lnn: pl.DataFrame, pre: str) -> pl.DataFrame:
    """Per row: unique token ids found in the idf table (list[u32]) and W = sum idf over ALL its unique tokens."""
    tok = pl.lit(pre) + pl.col("tok") if pre else pl.col("tok")
    x = (df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_r"), "country", tok=pl.col(col))
         .explode("tok", empty_as_null=True).drop_nulls("tok").with_columns(tok=tok).unique(["_r", "tok"])
         .join(voc, on=["country", "tok"], how="left").join(lnn, on="country", how="left"))
    agg = x.group_by("_r").agg(t=pl.col("tid").drop_nulls(), W=pl.col("w").fill_null(pl.col("lnN")).sum())
    return (pl.DataFrame({"_r": pl.int_range(df.height, dtype=pl.UInt32, eager=True)})
            .join(agg, on="_r", how="left", maintain_order="left")
            .select(pl.col("t").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))), pl.col("W").fill_null(0.0)))


def entities(split: str, n: int, voc: dict[str, pl.DataFrame], lnn: pl.DataFrame) -> pl.DataFrame:
    return prep(pl.read_parquet(norm_path(split, n), columns=ENT_COLS), voc, lnn)


def prep(df: pl.DataFrame, voc: dict[str, pl.DataFrame], lnn: pl.DataFrame) -> pl.DataFrame:
    df = add_skeletons(df).with_columns(_bg=_bigrams(df["name_tokens"]))
    toks = {f: map_tokens(df, col, voc[ch], lnn, pre) for f, (col, ch, pre) in FIELDS.items()}
    return df.select(
        "entity_id", core="core_name", legal="legal_form", aliases="aliases",
        addr=pl.col("addr_tokens").list.join(" "), num="addr_number", street="addr_street_core",
        has_addr="has_address", nonascii="is_nonascii_raw", marker=pl.col("business_name").str.contains(MARKER_RE),
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


def pair_features(c: pl.DataFrame, s1: pl.DataFrame, rec: pl.DataFrame, w: np.ndarray) -> pl.DataFrame:
    a = c.select("s1_id").join(s1, left_on="s1_id", right_on="entity_id", how="left", maintain_order="left")
    b = c.select("rec_id").join(rec, left_on="rec_id", right_on="entity_id", how="left", maintain_order="left")
    assert a["core"].null_count() == 0 and b["core"].null_count() == 0, "candidate id missing from norm cache"
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
        "is_s3": c["rec_id"].str.starts_with("S3").cast(pl.Int8).to_numpy(),
    }
    for k in FIELDS:
        f[f"{k}_idfj"] = idf_jaccard(a[f"t_{k}"], b[f"t_{k}"], a[f"W_{k}"], b[f"W_{k}"], w)
    return pl.concat([c.select(*ID_COLS, id_key("s1_id").alias("s1k"), id_key("rec_id").alias("reck"),
                               *RANK_COLS, "n_channels_hit"),
                      pl.DataFrame(f)], how="horizontal_extend")


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
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--smoke", type=int, default=0, help="first N candidate rows only (relative feats within them)")
    a = ap.parse_args()
    log, out = StepLog(), out_dir(a.split, a.smoke)
    shutil.rmtree(out, ignore_errors=True)
    base_dir, rel_dir = out / "_base", out / "_rel"
    base_dir.mkdir(parents=True)
    rel_dir.mkdir()

    voc, lnn, w = vocab(a.split)
    log("idf vocab", n_tokens=len(w))
    s1 = entities(a.split, 1, voc, lnn)
    log("S1 entities", rows=s1.height)
    rec = pl.concat([entities(a.split, n, voc, lnn) for n in (2, 3)])
    del voc
    log("S2+S3 entities", rows=rec.height)

    pf = pq.ParquetFile(path("interim_dir") / f"candidates_{a.split}.parquet")
    parts, done = [], 0
    for i, batch in enumerate(pf.iter_batches(batch_size=CFG["features"]["chunk_rows"],
                                              columns=[*ID_COLS, *RANK_COLS, "n_channels_hit"])):
        c = pl.from_arrow(batch)
        if a.smoke:
            c = c.head(a.smoke - done)
        p = base_dir / f"part-{i:05d}.parquet"
        pair_features(c, s1, rec, w).write_parquet(p)
        parts.append(p)
        done += c.height
        log(f"pass1 part {i}", rows=c.height, total=done)
        if a.smoke and done >= a.smoke:
            break
    del s1, rec

    def column(cols: list[str]) -> pl.DataFrame:
        return pl.concat([pl.scan_parquet(p).select(cols) for p in parts]).collect()
    keys = column(KEY_COLS)
    keys.select(n_cand_rec=pl.len().over("reck").cast(pl.Int32),
                n_cand_s1=pl.len().over("s1k").cast(pl.Int32)).write_parquet(rel_dir / "_n.parquet")
    rel_files = [rel_dir / "_n.parquet"]
    for col in REL_COLS:
        relative(keys, column([col])[col]).write_parquet(rel_dir / f"{col}.parquet")
        rel_files.append(rel_dir / f"{col}.parquet")
        log(f"pass2 {col}", rows=keys.height)
    del keys

    off = 0
    for i, p in enumerate(parts):
        base = pl.read_parquet(p).drop(KEY_COLS)
        rel = [pl.scan_parquet(f).slice(off, base.height).collect() for f in rel_files]
        df = pl.concat([base, *rel], how="horizontal_extend")
        assert "country" not in df.columns
        df.write_parquet(out / f"part-{i:05d}.parquet")
        off += base.height
    shutil.rmtree(base_dir, ignore_errors=True)  # exFAT "._" stubs vanish mid-walk
    shutil.rmtree(rel_dir, ignore_errors=True)
    log("assemble", rows=off, n_cols=df.width, parts=len(parts))
    log.dump(out / "features_timing.json", split=a.split, smoke=a.smoke, rows=off)


if __name__ == "__main__":
    main()
