"""Stage 1 - normalise every record of every source once, cache to interim/ as parquet.

    python -m src.prep --split train      (or test, or all)

Output: interim/norm_v{NORM_VERSION}_{split}_s{n}.parquet with the raw columns + normalised columns.
Re-runs are free (cache hit) until a rule in normalize.py changes (bump NORM_VERSION).
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .io import load_source, path
from .normalize import NORM_VERSION, normalize_record

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHUNK = 20_000
OUT_COLS = ["name_norm", "name_core", "name_legal", "name_parts", "name_skel",
            "addr_norm", "addr_nums", "addr_skel"]


def norm_path(split: str, n: int):
    return path("interim_dir") / f"norm_v{NORM_VERSION}_{split}_s{n}.parquet"


def _work(chunk: tuple[list[str], list[str]]) -> list[tuple]:
    names, addrs = chunk
    return [tuple(normalize_record(n, a).values()) for n, a in zip(names, addrs)]


def prep_one(split: str, n: int, pool: ProcessPoolExecutor) -> pd.DataFrame:
    out = norm_path(split, n)
    if out.exists():
        print(f"  [cache] {out.name}")
        return pd.read_parquet(out)
    t0 = time.time()
    df = load_source(split, n)
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    chunks = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
    parts, done = [], 0
    for res in pool.map(_work, chunks):
        # convert each chunk to arrow-backed strings right away: keeps peak RAM low on 16 GB
        parts.append(pd.DataFrame(res, columns=OUT_COLS).astype("string[pyarrow]"))
        done += len(res)
        if done % (CHUNK * 25) < CHUNK or done == len(df):
            print(f"    {split} S{n}: {done:,}/{len(df):,} ({time.time() - t0:.0f}s)", flush=True)
    norm = pd.concat(parts, ignore_index=True)
    norm.index = df.index
    df = pd.concat([df, norm], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)                                   # atomic: no half-written cache
    print(f"  wrote {out.name}: {len(df):,} rows in {time.time() - t0:.0f}s")
    return df


def load_norm(split: str, n: int) -> pd.DataFrame:
    p = norm_path(split, n)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m src.prep --split {split}` first")
    return pd.read_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = ap.parse_args()
    splits = ["train", "test"] if args.split == "all" else [args.split]
    print(f"normaliser v{NORM_VERSION}, {N_WORKERS} workers")
    with ProcessPoolExecutor(N_WORKERS) as pool:
        for split in splits:
            for n in (1, 2, 3):
                prep_one(split, n, pool)


if __name__ == "__main__":
    main()
