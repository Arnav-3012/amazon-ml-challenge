"""Workstream C feasibility (NOT shipped): does a multilingual sentence encoder retrieve v1 blocking misses?

Per country (train, labels known): 5,000 sampled missed true pairs (not in candidates_train_v1) + 2,000 hit pairs
(control), seed 42. Query = the S1; pool = the partners + N random same-country S2/S3 records (N = 50k, 500k; the
50k pool is a prefix of the 500k one). Recall@k = partner within the S1's top-k by cosine (FAISS inner product on
L2-normalised vectors). Text variants: "norm" = core_name + " | " + joined addr_tokens (spec); "raw" = business_name
+ " | " + business_address (keeps native script). Reports texts/s and peak RSS -> projected full-run embed time.
Run with the internal venv:  HF_HOME=<T7 tmp>/hf ~/p3venv/bin/python experiments/retrieval_probe.py [--variants norm raw]
-> experiments/retrieval_probe.json
"""
import argparse
import json
import resource
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[1]
I = REPO / "artifacts" / "interim"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED, N_MISS, N_HIT, POOLS, KS = 42, 5000, 2000, (50_000, 500_000), (10, 25, 50)


def id_key(c: str) -> pl.Expr:
    return pl.col(c).str.replace_all(r"\D", "").cast(pl.Int64)


def ents(n: int) -> pl.DataFrame:
    return pl.read_parquet(I / f"norm_train_s{n}.parquet", columns=["entity_id", "country", "business_name",
                                                                     "business_address", "core_name", "addr_tokens"])


def text(df: pl.DataFrame, v: str) -> list[str]:
    if v == "norm":
        return (df["core_name"] + " | " + df["addr_tokens"].list.join(" ")).to_list()
    return (df["business_name"] + " | " + df["business_address"]).to_list()


def rss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["norm"])
    ap.add_argument("--smoke", action="store_true", help="200 misses / 100 hits, pools 5k/20k: crash + rate test")
    a = ap.parse_args()
    global N_MISS, N_HIT, POOLS
    if a.smoke:
        N_MISS, N_HIT, POOLS = 200, 100, (5_000, 20_000)
    t0 = time.time()
    gt = pl.read_csv(REPO / "dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
    gt = (gt.select(s1_id="source1_entity_id", rec_id=pl.col("matched_entity_ids").str.split(","))
          .explode("rec_id").filter(pl.col("rec_id").is_not_null() & (pl.col("rec_id") != "")))
    hitset = (gt.lazy().join(pl.scan_parquet(I / "candidates_train_v1.parquet").select("s1_id", "rec_id"),
                             on=["s1_id", "rec_id"], how="semi").collect(engine="streaming").with_columns(hit=pl.lit(True)))
    lab = gt.join(hitset, on=["s1_id", "rec_id"], how="left").with_columns(pl.col("hit").fill_null(False))
    s1 = ents(1)
    lab = lab.join(s1.select(s1_id="entity_id", country="country"), on="s1_id")
    del hitset
    # records: only the sampled partners + a 1-in-8 hash sample per country (the 500k pool is drawn from it)
    pick = pl.concat([lab.filter((pl.col("country") == c) & (pl.col("hit") == h)).sample(nn, seed=SEED)
                      for c in ("India", "US") for h, nn in ((False, N_MISS), (True, N_HIT))])
    want = pick["rec_id"].unique().implode()
    rec = pl.concat([pl.scan_parquet(I / f"norm_train_s{n}.parquet").select(
        "entity_id", "country", "business_name", "business_address", "core_name", "addr_tokens")
        .filter(pl.col("entity_id").is_in(want) | (pl.col("entity_id").hash(SEED) % 8 == 0)).collect()
        for n in (2, 3)])
    # memory: cut every country's frames first, then drop the 10M-row source frames before loading the model
    work = {}
    for c in ["India", "US"]:
        pairs = pick.filter(pl.col("country") == c).with_columns(
            kind=pl.when(pl.col("hit")).then(pl.lit("hit")).otherwise(pl.lit("miss")))
        partners = pairs["rec_id"].unique()
        part = rec.filter(pl.col("entity_id").is_in(partners.implode()))
        pool = (rec.filter((pl.col("country") == c) & ~pl.col("entity_id").is_in(partners.implode()))
                .sample(max(POOLS), seed=SEED))
        work[c] = (pairs, pl.concat([part, pool]), s1.join(pairs.select(entity_id="s1_id").unique(), on="entity_id"),
                   part.height)  # partners first
    del rec, s1, lab, gt
    import gc
    gc.collect()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)
    res = {"model": MODEL, "device": dev, "by": []}
    rate_n, rate_s = 0, 0.0
    for c, (pairs, R, Q, n_part) in work.items():
        ridx = {e: i for i, e in enumerate(R["entity_id"].to_list())}
        qidx = {e: i for i, e in enumerate(Q["entity_id"].to_list())}
        for v in a.variants:
            ts = time.time()
            EQ = model.encode(text(Q, v), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
            ER = model.encode(text(R, v), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
            dt = time.time() - ts
            rate_n, rate_s = rate_n + Q.height + R.height, rate_s + dt
            for N in POOLS:
                sub = n_part + N
                idx = faiss.IndexFlatIP(ER.shape[1])
                idx.add(ER[:sub])
                _, nb = idx.search(EQ, max(KS))
                q = np.array([qidx[s] for s in pairs["s1_id"].to_list()])
                r = np.array([ridx[x] for x in pairs["rec_id"].to_list()])
                rank = np.array([(np.flatnonzero(nb[qi] == ri)[:1].tolist() or [10**9])[0] + 1 for qi, ri in zip(q, r)])
                for kind in ("miss", "hit"):
                    m = (pairs["kind"] == kind).to_numpy()
                    res["by"].append({"country": c, "variant": v, "pool": N, "kind": kind, "n": int(m.sum()),
                                      **{f"recall@{k}": float((rank[m] <= k).mean() * 100) for k in KS}})
                    print(res["by"][-1], flush=True)
            print(f"{c} {v}: {Q.height + R.height} texts in {dt:.0f}s ({(Q.height + R.height) / dt:.0f}/s), rss {rss_mb()} MB",
                  flush=True)
    n_all = sum(pl.scan_parquet(I / f"norm_{sp}_s{n}.parquet").select(pl.len()).collect().item()
                for sp in ("train", "test") for n in (1, 2, 3))
    rate = rate_n / rate_s
    res.update(texts_per_s=round(rate, 1), peak_rss_mb=rss_mb(), all_texts=n_all,
               projected_full_embed_h=round(n_all / rate / 3600, 2), total_s=round(time.time() - t0))
    (REPO / "experiments" / f"retrieval_probe{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by"}), flush=True)


if __name__ == "__main__":
    main()
