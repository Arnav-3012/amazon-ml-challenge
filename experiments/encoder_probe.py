"""krish-v2 Phase 3 feasibility (NOT shipped): can a small multilingual encoder recover v1 blocking misses?

Sample 5,000 train true pairs missed by candidates_train_v1 (+2,000 hit pairs as a control), seed 42. Texts are the RAW
"name, address" (the encoder's point is raw/native script), e5 "query: " prefix on both sides, L2-normalised.
  beats_cands  cos(record, true S1) > max cos(record, each of its current v1 candidate S1s)  (no candidates -> True)
  est_rank     1 + (# same-country distractor S1s scoring above the true S1) x (country S1 count / distractors sampled)
               from a pool of 45,000 random train S1s; recovered@10 = est_rank <= 10
Also: texts/s on MPS and peak RSS -> projected time to embed all train+test records and S1s.
Model: intfloat/multilingual-e5-small (MIT, ~118M params). -> experiments/encoder_probe.json
Run from repo root:  HF_HOME=<T7 path> .venv/bin/python experiments/encoder_probe.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src.block import norm_path  # noqa: E402
from src.io import load_gt_pairs, path  # noqa: E402

N_MISS, N_HIT, N_POOL, SEED = 5000, 2000, 45000, 42
MODEL = "intfloat/multilingual-e5-small"


def raw(split: str, n: int) -> pl.DataFrame:
    return pl.read_parquet(norm_path(split, n), columns=["entity_id", "business_name", "business_address", "country"])


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import torch
    t0 = time.time()
    I = path("interim_dir")
    gt = load_gt_pairs().rename({"match_id": "rec_id"})
    v1 = pl.read_parquet(I / "candidates_train_v1.parquet", columns=["s1_id", "rec_id"])
    lab = gt.join(v1.with_columns(hit=pl.lit(True)), on=["s1_id", "rec_id"], how="left").with_columns(
        pl.col("hit").fill_null(False))
    miss = lab.filter(~pl.col("hit")).sample(N_MISS, seed=SEED)
    hit = lab.filter(pl.col("hit")).sample(N_HIT, seed=SEED)
    pairs = pl.concat([miss, hit])
    s1 = raw("train", 1)
    rec = pl.concat([raw("train", 2), raw("train", 3)])
    cands = v1.join(pairs.select("rec_id").unique(), on="rec_id")
    pool = s1.sample(N_POOL, seed=SEED).select("entity_id")
    need_s1 = pl.concat([pairs.select(entity_id="s1_id"), cands.select(entity_id="s1_id"), pool]).unique()
    S = s1.join(need_s1, on="entity_id")
    R = rec.join(pairs.select(entity_id="rec_id").unique(), on="entity_id")
    txt = lambda df: ("query: " + df["business_name"] + ", " + df["business_address"]).to_list()
    print(f"texts: {S.height} S1 + {R.height} records", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(MODEL, device=dev)
    t1 = time.time()
    ES = m.encode(txt(S), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    ER = m.encode(txt(R), batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    enc_s = time.time() - t1
    rate = (S.height + R.height) / enc_s
    si = {e: i for i, e in enumerate(S["entity_id"].to_list())}
    ri = {e: i for i, e in enumerate(R["entity_id"].to_list())}
    s_country = S["country"].to_numpy()
    pool_idx = np.array([si[e] for e in pool["entity_id"].to_list()])
    n_country = dict(s1.group_by("country").len().iter_rows())
    pool_country = s_country[pool_idx]
    cand_by_rec = {r: l for r, l in cands.group_by("rec_id").agg("s1_id").iter_rows()}

    rows = []
    for s, r, h in pairs.select("s1_id", "rec_id", "hit").iter_rows():
        e = ER[ri[r]]
        ct = float(ES[si[s]] @ e)
        cs = [c for c in cand_by_rec.get(r, []) if c != s]
        cmax = float(max(ES[si[c]] @ e for c in cs)) if cs else -1.0
        c = s_country[si[s]]
        same = pool_idx[(pool_country == c) & (pool["entity_id"].to_numpy() != s)]
        above = int((ES[same] @ e > ct).sum())
        est = 1 + above * n_country[c] / max(len(same), 1)
        rows.append({"hit": h, "country": c, "cos_true": ct, "beats_cands": ct > cmax, "est_rank": est})
    df = pl.DataFrame(rows)
    summ = (df.group_by("hit", "country").agg(n=pl.len(), beats_cands_pct=pl.col("beats_cands").mean() * 100,
                                              recovered_at10_pct=(pl.col("est_rank") <= 10).mean() * 100,
                                              recovered_at50_pct=(pl.col("est_rank") <= 50).mean() * 100,
                                              median_est_rank=pl.col("est_rank").median(),
                                              mean_cos_true=pl.col("cos_true").mean()).sort("hit", "country"))
    allm = df.filter(~pl.col("hit"))
    n_texts_all = sum(pl.scan_parquet(norm_path(sp, n)).select(pl.len()).collect().item()
                      for sp in ("train", "test") for n in (1, 2, 3))
    res = {"model": MODEL, "device": dev, "texts": S.height + R.height, "encode_s": round(enc_s, 1),
           "texts_per_s": round(rate, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
           "all_texts_train_test": n_texts_all, "projected_full_embed_h": round(n_texts_all / rate / 3600, 2),
           "misses_recovered_at10_pct": float((allm["est_rank"] <= 10).mean() * 100),
           "misses_beat_cands_pct": float(allm["beats_cands"].mean() * 100),
           "by_segment": summ.to_dicts(), "total_s": round(time.time() - t0, 1)}
    Path(__file__).with_name("encoder_probe.json").write_text(json.dumps(res, indent=1))
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(summ)
    print(json.dumps({k: v for k, v in res.items() if k != "by_segment"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
