"""TSV I/O for the challenge files. Paths come from configs/config.yaml, resolved against the repo root."""
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
import yaml

PKG_DIR = Path(__file__).resolve().parents[1]  # code/business_entity_resolution
ROOT = PKG_DIR.parents[1]  # repo root == submission zip root
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

with open(PKG_DIR / "configs" / "config.yaml") as _f:
    CFG = yaml.safe_load(_f)


def path(key: str) -> Path:
    return ROOT / CFG["paths"][key]


def _read_tsv(p: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    # QUOTE_NONE: one physical line = one record (every line of every file has exactly 4 tab fields);
    # a few hundred addresses carry CSV-style quotes, kept verbatim rather than risk rows merging.
    # dtype=str is pyarrow-backed under pandas 3; na_filter=False keeps empty fields as "".
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, usecols=usecols)


def load_source(split: str, n: int, cols: list[str] | None = None) -> pd.DataFrame:
    """Load dataset/{split}/{split}_source{n}.tsv; `cols` limits columns (entity_id always kept)."""
    cols = SOURCE_COLS if cols is None else ["entity_id", *[c for c in cols if c != "entity_id"]]
    df = _read_tsv(path(f"{split}_dir") / f"{split}_source{n}.tsv", usecols=cols)
    ids = df["entity_id"]
    assert ids.is_unique, f"{split} S{n}: duplicate entity_id"
    assert ids.str.startswith(f"S{n}-").all(), f"{split} S{n}: entity_id without S{n}- prefix"
    return df


def load_gt(validate: bool = True) -> dict[str, frozenset[str]]:
    """Train ground truth as {S1 id: frozenset of matched S2/S3 ids} (empty frozenset = singleton)."""
    gt = _read_tsv(path("train_dir") / "train_ground_truth.tsv")
    truth = {s1: frozenset(i for i in (x.strip() for x in m.split(",")) if i)
             for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    if validate:
        assert len(truth) == len(gt), "GT: duplicate source1_entity_id"
        s1 = load_source("train", 1, ["entity_id"])["entity_id"]
        assert set(s1) == truth.keys(), "GT S1 ids != train_source1 ids"
        matched = pd.Series([i for v in truth.values() for i in v], dtype=str)
        known = pd.concat([load_source("train", n, ["entity_id"])["entity_id"] for n in (2, 3)])
        bad = matched[~matched.isin(known)]
        assert bad.empty, f"GT: {len(bad)} matched ids not in train S2∪S3, e.g. {bad.head(3).tolist()}"
    return truth


def write_id_lists(p: Path, mapping: Mapping[str, Iterable[str]], id_col: str, list_col: str) -> None:
    """One row per key, comma-joined sorted unique ids ("" when none), tab-separated, no quoting."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{list_col}\n")
        f.writelines(f"{k}\t{','.join(sorted(set(v)))}\n" for k, v in mapping.items())
