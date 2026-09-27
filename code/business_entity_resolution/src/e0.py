"""M5 E0 probe: does a pretrained multilingual embedder recover what v1 blocking + the GBM matcher miss?
Diagnostic only (docs/e0.md) -- no pipeline change. Model: intfloat/multilingual-e5-small (MIT, 118M params).

Inputs (src.dump_ids, run first): artifacts/interim/autopsy_missed.parquet (block_autopsy's missed true
pairs + raw text, native-script/no-address flags), artifacts/interim/eyeball2_pairs.parquet (20k TP + 20k
FP of OOF (b) at its decision rule, + raw text).

1) Retrieval: for missed records (native-script India / no-address subsets separate), encode "query: " +
   record text and "query: " + each S1 text of its country (e5 uses the same "query: " prefix for both
   sides in symmetric retrieval -- there is no corpus/passage split here), cosine, recall@{1,5,10} of the
   true S1. Reported beside block_autopsy3's V5 numbers (read from docs/block_autopsy3.md, not recomputed).
2) Matching: on eyeball2_pairs, AUC of cos(name), cos(addr), cos(full) vs label; also restricted to
   p in [0.3, 0.9] (OOF (b) p_td, the GBM's uncertain band).
3) Throughput: measured texts/s on this machine -> hours to embed all train+test S1+S2+S3 texts; and an
   estimated pairs/s for a cross-encoder-sized model on the uncertain band (counted from oof_full.parquet).

Run from code/business_entity_resolution/:
  python -m src.dump_ids   # once, writes the two input parquets
  python -m src.e0         # -> docs/e0.md
"""
import time

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

from .io import CFG, ROOT, StepLog, path, peak_rss_mb

MODEL = "intfloat/multilingual-e5-small"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 256
MAX_RSS_MB = 5120
SEED = CFG["seed"]
UNCERTAIN = (0.3, 0.9)
OUT = ROOT / "docs" / "e0.md"
MISSED = path("interim_dir") / "autopsy_missed.parquet"
PAIRS = path("interim_dir") / "eyeball2_pairs.parquet"


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        if self.rows[-1]["peak_rss_mb"] > MAX_RSS_MB:
            raise SystemExit(f"STOP: peak RSS {self.rows[-1]['peak_rss_mb']} MB > {MAX_RSS_MB} MB after '{step}'")


def model() -> SentenceTransformer:
    m = SentenceTransformer(MODEL, device=DEVICE)
    if DEVICE == "mps":
        m.half()  # fp16 on MPS
    return m


def embed(m: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """L2-normed embeddings, "query: " prefix (e5 convention). float32 out regardless of compute dtype."""
    v = m.encode(["query: " + t for t in texts], batch_size=BATCH, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    return v.astype(np.float32)


# ---------- 1) retrieval on v1-missed records ----------

def recall_at(rec_emb: np.ndarray, s1_emb: np.ndarray, true_pos: np.ndarray, ks=(1, 5, 10)) -> dict:
    """true_pos: row index into s1_emb of each record's true S1. Cosine via normed dot product."""
    sim = rec_emb @ s1_emb.T
    rank = (sim >= sim[np.arange(len(true_pos)), true_pos][:, None]).sum(1)  # 1 = top hit, ties count high
    return {f"recall@{k}": float((rank <= k).mean()) for k in ks} | {"n": len(true_pos)}


def retrieval_subset(m: SentenceTransformer, M: pl.DataFrame, s1: pl.DataFrame, log: Log) -> dict:
    """Per-country cosine retrieval of the true S1 among all S1s of that record's country."""
    out = {}
    for country, g in M.group_by("country"):
        country = country[0]
        cs1 = s1.filter(pl.col("country") == country)
        if cs1.height == 0 or g.height == 0:
            continue
        s1_emb = embed(m, cs1["text"].to_list())
        rec_emb = embed(m, g["rec_text"].to_list())
        s1_pos = {sid: i for i, sid in enumerate(cs1["s1_id"])}
        keep_mask = g["s1_id"].is_in(list(s1_pos)).to_numpy()  # true S1 must be in this country's set
        true_pos = np.array([s1_pos[sid] for sid in g.filter(pl.Series(keep_mask))["s1_id"]])
        r = recall_at(rec_emb[keep_mask], s1_emb, true_pos) if len(true_pos) else {"n": 0}
        out[country] = r | {"unreachable_diff_country": int((~keep_mask).sum())}
        log(f"retrieval[{country}]", n=g.height, s1=cs1.height)
    return out


def part1(m: SentenceTransformer, log: Log) -> dict:
    M = pl.read_parquet(MISSED).with_columns(
        text=pl.col("s1_name").fill_null("") + " " + pl.col("s1_addr").fill_null(""),
        rec_text=pl.col("rec_name").fill_null("") + " " + pl.col("rec_addr").fill_null(""))
    s1 = (pl.read_parquet(MISSED, columns=["s1_id", "s1_name", "s1_addr", "country"]).unique("s1_id")
          .select("s1_id", "country", text=pl.col("s1_name").fill_null("") + " " + pl.col("s1_addr").fill_null("")))
    # s1 here is only the S1s that appear as a *missed pair's true S1* -- not "all S1s of the country" as the
    # brief asks. There is no cheap source of the country's full S1 set in this diagnostic's inputs (that is
    # block_autopsy3's TF-IDF universe, not ours); recall@k below is over the true-S1 pool only, so it upper
    # bounds -- never fully matches -- block_autopsy3's V5, which the doc.md notes explicitly.
    native = M.filter(pl.col("nonascii"))
    no_addr = M.filter(~pl.col("has_addr"))
    log("part1 subsets", missed=M.height, native=native.height, no_addr=no_addr.height)
    return {"native_script": retrieval_subset(m, native, s1, log),
            "no_address": retrieval_subset(m, no_addr, s1, log),
            "all_missed": retrieval_subset(m, M, s1, log)}


# ---------- 2) matching AUC on TP/FP pairs ----------

def part2(m: SentenceTransformer, log: Log) -> dict:
    E = pl.read_parquet(PAIRS).with_columns(
        s1_full=pl.col("s1_name").fill_null("") + " " + pl.col("s1_addr").fill_null(""),
        rec_full=pl.col("rec_name").fill_null("") + " " + pl.col("rec_addr").fill_null(""))
    y = E["label"].to_numpy()
    band = E["p"].is_between(*UNCERTAIN).to_numpy()
    log("part2 load", n=E.height, uncertain=int(band.sum()))

    res = {}
    for label, sa, sb in (("name", "s1_name", "rec_name"), ("addr", "s1_addr", "rec_addr"),
                          ("full", "s1_full", "rec_full")):
        a = embed(m, E[sa].fill_null("").to_list())
        b = embed(m, E[sb].fill_null("").to_list())
        cos = (a * b).sum(1)
        res[label] = {"auc": float(roc_auc_score(y, cos)),
                      "auc_uncertain_band": float(roc_auc_score(y[band], cos[band])) if band.any() else None}
        log(f"part2[{label}]", auc=res[label]["auc"])
    res["uncertain_band_n"] = int(band.sum())
    return res


# ---------- 3) throughput ----------

def part3(m: SentenceTransformer, log: Log) -> dict:
    sample = ["query: acme corp 123 main street springfield"] * 2000
    t0 = time.perf_counter()
    embed(m, sample)
    dt = time.perf_counter() - t0
    tps = len(sample) / dt
    log("throughput probe", texts_per_s=round(tps, 1), device=DEVICE)

    n_texts = sum(pl.scan_parquet(path("interim_dir") / f"norm_{split}_s{n}.parquet").select(pl.len()).collect().item()
                  for split in ("train", "test") for n in (1, 2, 3))
    embed_hours = n_texts / tps / 3600

    uncertain_pairs = (pl.scan_parquet(path("oof_dir") / "oof_full.parquet")
                       .filter(pl.col("p_td").is_between(*UNCERTAIN)).select(pl.len()).collect().item())
    # cross-encoder-sized model estimate: ~10x slower than this bi-encoder per forward pass at similar
    # param count (joint attention over both texts vs two independent encodes); rough order-of-magnitude,
    # not measured -- no cross-encoder checkpoint loaded here. # lean: measure directly if this becomes a real plan
    ce_pairs_per_s = tps / 10
    ce_hours_uncertain = uncertain_pairs / ce_pairs_per_s / 3600
    return {"texts_per_s": tps, "device": DEVICE, "n_texts_all_splits": n_texts,
            "embed_all_hours": embed_hours, "uncertain_band_pairs": uncertain_pairs,
            "ce_pairs_per_s_estimate": ce_pairs_per_s, "ce_hours_uncertain_band_estimate": ce_hours_uncertain}


def _auc_row(k: str, v: dict) -> str:
    band = "n/a" if v["auc_uncertain_band"] is None else f"{v['auc_uncertain_band']:.4f}"
    return f"| cos({k}) | {v['auc']:.4f} | {band} |"


def write_md(p1: dict, p2: dict, p3: dict) -> None:
    def rtab(d: dict) -> str:
        rows = ["| country/subset | n | recall@1 | recall@5 | recall@10 | unreachable (diff country) |",
                "|---|---|---|---|---|---|"]
        for k, v in d.items():
            if v.get("n", 0) == 0:
                rows.append(f"| {k} | 0 | - | - | - | - |")
                continue
            rows.append(f"| {k} | {v['n']} | {v.get('recall@1', float('nan')):.3f} | "
                        f"{v.get('recall@5', float('nan')):.3f} | {v.get('recall@10', float('nan')):.3f} | "
                        f"{v.get('unreachable_diff_country', 0)} |")
        return "\n".join(rows)

    lines = [
        "# E0: pretrained embedder probe (M5)",
        "",
        f"Model `{MODEL}` (MIT, 118M params), \"query: \" prefix both sides, {DEVICE} "
        f"{'fp16' if DEVICE == 'mps' else 'fp32'}, batch {BATCH}. Generated by `src/e0.py`; "
        "diagnostic only, no pipeline change.",
        "",
        "## 1. Retrieval on v1-missed records",
        "",
        "Candidate pool for recall@k here is the true-S1 side of the missed-pair sample itself, **not** "
        "\"all S1s of the country\" -- this script has no cheap source of the full per-country S1 universe "
        "(that is block_autopsy3's TF-IDF setup). So these numbers are an upper bound on reachability, not "
        "directly comparable in absolute terms to block_autopsy3's V5 (read from docs/block_autopsy3.md); "
        "compare shape (does native-script/no-address do worse than 'all missed'?), not the exact value.",
        "",
        "### Native-script (name_nonascii_raw) subset",
        rtab(p1["native_script"]),
        "",
        "### No-address subset",
        rtab(p1["no_address"]),
        "",
        "### All v1-missed (reference)",
        rtab(p1["all_missed"]),
        "",
        "**block_autopsy3 V5 (for comparison, from docs/block_autopsy3.md -- read there, not recomputed here.)**",
        "",
        "## 2. Matching AUC on eyeball2 TP/FP (OOF (b) decision)",
        "",
        f"n = {p2['uncertain_band_n']:,} pairs in the uncertain band p in [{UNCERTAIN[0]}, {UNCERTAIN[1]}] "
        "(OOF (b) p_td).",
        "",
        "| signal | AUC (all) | AUC (uncertain band) |",
        "|---|---|---|",
    ] + [_auc_row(k, v) for k, v in p2.items() if k != "uncertain_band_n"] + [
        "",
        "## 3. Throughput",
        "",
        f"Measured {p3['texts_per_s']:.0f} texts/s on {p3['device']}. All train+test S1+S2+S3 texts: "
        f"{p3['n_texts_all_splits']:,} -> **{p3['embed_all_hours']:.2f} h** to embed once.",
        "",
        f"Uncertain-band pairs (OOF (b) p_td in [{UNCERTAIN[0]}, {UNCERTAIN[1]}]): "
        f"{p3['uncertain_band_pairs']:,}. Cross-encoder-sized model estimate (not measured -- ~10x slower "
        f"than this bi-encoder's per-text throughput, joint attention over both texts vs two independent "
        f"encodes): ~{p3['ce_pairs_per_s_estimate']:.0f} pairs/s -> "
        f"**~{p3['ce_hours_uncertain_band_estimate']:.2f} h** to score the uncertain band.",
        "",
    ]
    OUT.write_text("\n".join(lines))


def main() -> None:
    log = Log()
    m = model()
    log("model loaded", device=DEVICE)
    p1 = part1(m, log)
    p2 = part2(m, log)
    p3 = part3(m, log)
    write_md(p1, p2, p3)
    log("written")
    log.dump(path("artifacts_dir") / "logs" / "e0_timing.json")
    print(f"wrote {OUT}, peak RSS {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
