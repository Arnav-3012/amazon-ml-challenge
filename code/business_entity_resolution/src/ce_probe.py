"""M5 CE probe: does a FINE-TUNED cross-encoder add signal over the GBM stage-1 p_td? (e0.py was zero-shot only.)
Diagnostic only (docs/ce_probe.md) -- no pipeline change. Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (MIT, 33M
params), fine-tuned on this task's train pairs.

Sample: train S1s in cv_full's frozen fold assignment (s1_table(1.0) == oof_full.fold), folds 1-4 = train world,
fold 0 = eval world. A record that is a fold-0 candidate is excluded from train pairs (no leakage: its S2/S3 row
would otherwise appear as a negative for a fold-0 S1 while its positive lives in the weights via the other side).

1) Train (skipped with --eval-only, which reuses models/ce_probe/): 100k true pairs (folds 1-4) + 70k hard negatives
   (oof_full candidates, label 0, p_td >= 0.05) + 30k random candidate negatives, seeded. Text per side =
   "{core_name} {legal_form} | {business_address}", record first, S1 second. CrossEncoderTrainer, BCE, MPS, fp32,
   max_length=64, batch=64, lr=2e-5, 1 epoch.
2) Eval frame = every fold-0 oof row of the sampled fold-0 S1s (all of them; --smoke: 1%). Scopes are LABEL-FREE:
   rows with p_td in [lo, hi], nothing force-added (the previous version added every true pair of those S1s, so
   scope membership read the label). Per scope: LightGBM stack on [logit p_td, ce], inner GroupKFold(5) by s1k ->
   p_new inside the scope, p_td outside; macro F0.5 (record argmax, best global t each) over the SAME S1 set for
   p_td and p_new, overall and per country. Country is a report grouping only, never a feature.
3) Confident errors (diagnostic, label-selected on purpose, never part of a stacking scope): true pairs with
   p_td < 0.02 and false pairs with p_td > 0.98; ce distributions and whether the CE argmax over the record's
   fold-0 candidates is the true S1 (FN) / is not the wrongly-scored S1 (FP).
4) MPS throughput on up to 50k scope pairs: batch 64 fp32 vs batch 256 fp16, max_length 64 vs 48.
5) Test scope from oof/test_p.parquet (p = test-side stage-1 score): pairs per scope and country -> scoring hours.

--stack2 (no model, no re-scoring: reuses oof/ce_probe_scores.parquet) fixes the eval world and adds rank features:
  1) Baseline in cv_full's (b) world: every oof row with p_td (= S1 alive in world SEED+2000; dead S1s have null
     p_td and are out of scope, not scored as empty), record argmax over ALL folds' rows (a fold-0 S1 competes with
     fold 1-4 S1s for the same record), ties -> lowest s1_table code. Stop unless the all-fold recompute reproduces
     oof/cv_full.json b_test_density exactly and fold 0 lands within F0_TOL of it (the json stores no per-fold F0.5).
  2) Stacker on fold-0 rows with p_td in SCOPES[-1] (all CE-scored): logit(p_td), ce, ce_rank, ce_margin
     (ce - best OTHER scored ce of the record; = ce - 2nd best for the argmax row), ce_is_argmax, p_td_rank (over all
     the record's candidates, all folds), n_scored_cands. "Scored" = in-stack rows only: the confident-error
     records' extra CE rows were chosen by label, so they never enter a rank. Other folds keep p_td.
  3) ΔF0.5 overall / per country, and per p_td band (p_new applied inside that band only, same fitted stacker).
  4) Confident-FN rescue and the FPs p_new creates, with their F0.5 cost at p_new's t.
  Caveat: CE exists for fold-0 rows only, so ce_rank/margin see ~1/5 of a record's competitors; on test they'd see all.

--apply-test (reuses models/ce_probe/ and oof/ce_probe_scores.parquet, no retraining):
  1) Score every test pair (oof/test_p.parquet) with p in [0.0005, 0.9995], fp32, max_len 64, batch 64.
     Resumable via oof/ce_test_scores.parquet (chunked writes; already-scored (s1k, reck) pairs are skipped on rerun).
  2) Fit stack1 ([logit p_td, ce], LightGBM 100 rounds) on the fold-0 oof rows in the same p_td scope
     (oof/ce_probe_scores.parquet), apply to the test scope; write oof/test_p_ce.parquet = a copy of test_p with
     p replaced inside the scope (unchanged outside).
  3) Print pairs scored, pairs/s, ETA, and the OOF pairwise-F0.5 sanity delta of the stacker (expect ~+0.003;
     this is a pairwise proxy, not the record-argmax macro metric used elsewhere in this file).

Run from code/business_entity_resolution/:
  python -m src.ce_probe --eval-only --smoke   # 1% of fold-0 S1s, same path incl. docs/ce_probe.md, <3 min
  python -m src.ce_probe --eval-only           # full eval with the saved model -> docs/ce_probe.md
  python -m src.ce_probe                       # retrain models/ce_probe/ first, then eval
  python -m src.ce_probe --stack2 --smoke      # 1% of fold-0 scope S1s, same path incl. docs/ce_probe.md
  python -m src.ce_probe --stack2              # -> docs/ce_probe.md (overwritten)
  python -m src.ce_probe --apply-test --smoke  # 10k test pairs -> oof/test_p_ce.parquet
  python -m src.ce_probe --apply-test          # full test scope -> oof/test_p_ce.parquet
"""
import os

# Must run before numpy/torch/lightgbm load: this process holds 3 OpenMP runtimes (homebrew libomp via LightGBM,
# torch's and sklearn's bundled copies); their worker-thread barriers segfaulted mid fine-tune (2026-09-27 crash
# report, __kmp_fork_barrier). One thread = no OMP workers. Training is on MPS, the stack has 2 features.
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

from .cv_full import TS, dropped, s1_table
from .decide import f05_vec
from .block import norm_path
from .features import id_key
from .io import CFG, ROOT, StepLog, path, peak_rss_mb

SEED = CFG["seed"]
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_DIR = ROOT / "models" / "ce_probe"
SMOKE_MODEL_DIR = ROOT / "models" / "ce_probe_smoke"  # a --smoke retrain never clobbers the real model
MAX_LEN, BATCH, LR, EPOCHS = 64, 64, 2e-5, 1
N_TRUE, N_HARD, N_RAND = 100_000, 70_000, 30_000
HARD_P_MIN = 0.05
SCOPES = [(0.02, 0.98), (0.005, 0.995), (0.0005, 0.9995)]  # narrow -> wide; each is a subset of the next
FN_P_MAX, FP_P_MIN = 0.02, 0.98
SCORE_BATCH = 256  # eval scoring: fp32, max_len=MAX_LEN (the trained setting); batch only changes padding
BENCH_N = 50_000
BENCH_CFGS = [(64, "fp32", 64), (256, "fp16", 64), (64, "fp32", 48), (256, "fp16", 48)]  # (batch, dtype, max_len)
# --stack2
STACK_LO, STACK_HI = SCOPES[-1]
BANDS = [(STACK_LO, 0.02), (0.02, 0.3), (0.3, 0.9), (0.9, STACK_HI)]  # [lo, hi) except the last, which is closed
STACKS = {"stack1 [logit p_td, ce]": ["logit_p_td", "ce"],
          "stack2 (+ranks)": ["logit_p_td", "ce", "ce_rank", "ce_margin", "ce_is_argmax", "p_td_rank", "n_scored_cands"]}
STACK_ROUNDS = 200
EXACT_TOL, F0_TOL = 1e-6, 0.01


def text_of(df: pl.DataFrame) -> pl.DataFrame:
    """entity_id -> "{core_name} {legal_form} | {business_address}", legal_form/address blank-safe."""
    return df.select("entity_id", txt=(pl.col("core_name").fill_null("") + " " + pl.col("legal_form").fill_null("")
                                        + " | " + pl.col("business_address").fill_null("")).str.strip_chars())


def load_texts(split: str = "train") -> dict[str, pl.DataFrame]:
    cols = ["entity_id", "core_name", "legal_form", "business_address"]
    return {s: text_of(pl.read_parquet(norm_path(split, n), columns=cols)) for s, n in [("s1", 1), ("s2", 2), ("s3", 3)]}


def key_to_id(col: str) -> pl.Expr:
    """Inverse of features.id_key: 2437567938 -> 'S2-437567938' (one-digit source prefix, digits kept verbatim)."""
    s = pl.col(col).cast(pl.String)
    return pl.format("S{}-{}", s.str.head(1), s.str.slice(1))


def with_ids(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(key_to_id("s1k").alias("s1_id"), key_to_id("reck").alias("rec_id"))


def load_gt_keys() -> pl.DataFrame:
    from .io import load_gt_pairs
    return load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))


def oof_scan() -> pl.LazyFrame:
    return pl.scan_parquet(path("oof_dir") / "oof_full.parquet").select("s1k", "reck", "label", "fold", "p_td")


def build_train_pairs(smoke: bool) -> pl.DataFrame:
    """-> train_pairs[s1_id, rec_id, label] from folds 1-4 S1s; records that are fold-0 candidates excluded."""
    # oof.fold == s1_table fold for every S1 (checked 2026-09-27); oof.s1k is id_key(s1_id), NOT s1_table.code
    oof = oof_scan().collect()
    n_true, n_hard, n_rand = (200, 500, 300) if smoke else (N_TRUE, N_HARD, N_RAND)
    train_s1 = oof.filter(pl.col("fold") != 0)
    fold0_recs = oof.filter(pl.col("fold") == 0)["reck"].unique().implode()
    gt = load_gt_keys()
    true_pool = gt.filter(pl.col("s1k").is_in(train_s1["s1k"].unique().implode()) & ~pl.col("reck").is_in(fold0_recs))
    true_pairs = true_pool.sample(n=min(n_true, true_pool.height), seed=SEED).with_columns(label=pl.lit(1, pl.Int8))
    neg = train_s1.filter((pl.col("label") == 0) & ~pl.col("reck").is_in(fold0_recs))
    hard_pool = neg.filter(pl.col("p_td") >= HARD_P_MIN)
    hard_neg = hard_pool.sample(n=min(n_hard, hard_pool.height), seed=SEED + 1)
    rand_neg = neg.sample(n=min(n_rand, neg.height), seed=SEED + 2)
    train_pairs = (pl.concat([true_pairs, hard_neg.select(true_pairs.columns), rand_neg.select(true_pairs.columns)])
                   .unique(["s1k", "reck"], maintain_order=True))
    return with_ids(train_pairs).select("s1_id", "rec_id", "label")


def load_fold0(smoke: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """-> (f0[s1k, reck, label, p_td]: every fold-0 oof row of the sampled S1s, s1[s1k, country, ntrue])."""
    s1 = s1_table(1.0).filter(pl.col("fold") == 0).select("s1k", "country", "ntrue")
    if smoke:
        s1 = s1.sample(fraction=0.01, seed=SEED + 3)
    f0 = (oof_scan().filter(pl.col("fold") == 0).drop("fold")
          .join(s1.lazy().select("s1k"), on="s1k", how="semi").collect())
    return f0, s1


def scope_mask(pairs: pl.DataFrame, lo: float, hi: float) -> pl.Series:
    """Label-free scope: membership is a function of p_td alone (null p_td = never scored by stage 1 = out)."""
    assert "label" not in pairs.columns, "scope membership must not see the label"
    return pairs["p_td"].is_between(lo, hi).fill_null(False)


def make_texts(pairs: pl.DataFrame, texts: dict[str, pl.DataFrame]) -> list[tuple[str, str]]:
    s2s3 = pl.concat([texts["s2"], texts["s3"]])
    # maintain_order: callers align labels / scores to `pairs` by position
    p = pairs.select("s1_id", "rec_id").join(s2s3.rename({"entity_id": "rec_id", "txt": "rec_txt"}), on="rec_id",
                                             how="left", maintain_order="left").join(
        texts["s1"].rename({"entity_id": "s1_id", "txt": "s1_txt"}), on="s1_id", how="left", maintain_order="left")
    assert p["rec_txt"].null_count() == 0 and p["s1_txt"].null_count() == 0, "unmatched entity_id in text join"
    return list(zip(p["rec_txt"].to_list(), p["s1_txt"].to_list()))  # record first, S1 second


def device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_ce(model_dir, max_len: int, dtype: str):
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    return CrossEncoder(str(model_dir), max_length=max_len, device=device(),
                        model_kwargs={"dtype": {"fp32": torch.float32, "fp16": torch.float16}[dtype]})


def predict(model, pairs: list[tuple[str, str]], batch: int) -> np.ndarray:
    import torch
    s = model.predict(pairs, batch_size=batch, convert_to_numpy=True,
                      activation_fn=torch.nn.Sigmoid())  # raw logits by default (Identity()); we want p
    return np.asarray(s, dtype=np.float32)


def fit_ce(train_pairs: pl.DataFrame, texts: dict[str, pl.DataFrame], out, log: StepLog) -> None:
    # CrossEncoder.fit() is broken in sentence-transformers 6.1 (FitMixinLoss calls model(**tokens), v6 forward
    # takes `input`), so train through CrossEncoderTrainer. Same schedule as fit(): linear decay after warmup.
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    torch.manual_seed(SEED)
    rec_txt, s1_txt = zip(*make_texts(train_pairs, texts))
    ds = Dataset.from_dict({"rec": list(rec_txt), "s1": list(s1_txt),  # column order = (text_a, text_b)
                            "label": train_pairs["label"].cast(pl.Float32).to_list()})
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=MAX_LEN, device=device(),
                          model_kwargs={"dtype": torch.float32})
    n_steps = -(-len(ds) // BATCH) * EPOCHS
    args = CrossEncoderTrainingArguments(
        output_dir=str(out / "trainer"), num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
        learning_rate=LR, warmup_steps=max(1, round(0.1 * n_steps)), seed=SEED, data_seed=SEED,
        save_strategy="no", report_to="none", logging_steps=max(1, n_steps // 10),
        dataloader_pin_memory=False, disable_tqdm=True)  # pin_memory unsupported on MPS
    t0 = time.perf_counter()
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=BinaryCrossEntropyLoss(model)).train()
    dt = time.perf_counter() - t0
    log("fine-tune", pairs=len(ds) * EPOCHS, secs=round(dt, 1), pairs_per_s=round(len(ds) * EPOCHS / max(dt, 1e-6), 1),
        device=device(), peak_rss_mb=peak_rss_mb())
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))


def bench(model_dir, pairs: list[tuple[str, str]], log: StepLog) -> list[dict]:
    """pairs/s per BENCH_CFGS on the same pairs; max |dscore| vs fp32 at the same max_len and vs fp32 @ MAX_LEN."""
    rows, ref = [], {}
    for batch, dtype, max_len in BENCH_CFGS:
        m = load_ce(model_dir, max_len, dtype)
        predict(m, pairs[:2 * batch], batch)  # warm-up: MPS kernel compile stays out of the timing
        t0 = time.perf_counter()
        ref[(dtype, max_len)] = predict(m, pairs, batch)
        dt = time.perf_counter() - t0
        rows.append({"batch": batch, "dtype": dtype, "max_len": max_len, "pairs": len(pairs), "secs": round(dt, 1),
                     "pairs_per_s": round(len(pairs) / max(dt, 1e-6), 1)})
        del m
    for r in rows:
        s = ref[(r["dtype"], r["max_len"])]
        r["max_abs_d_vs_fp32_same_len"] = float(np.abs(s - ref[("fp32", r["max_len"])]).max())
        r["max_abs_d_vs_fp32_len64"] = float(np.abs(s - ref[("fp32", MAX_LEN)]).max())
        log("bench", **r)
    return rows


def stack(d: pl.DataFrame) -> np.ndarray:
    """LightGBM(100 rounds) on [logit(p_td), ce], inner GroupKFold(5) by s1k -> OOF p_new for the scope rows."""
    p_td = np.clip(d["p_td"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.log(p_td / (1 - p_td)), d["ce"].to_numpy()])
    y, groups = d["label"].to_numpy(), d["s1k"].to_numpy()
    p_new = np.full(len(y), np.nan)
    params = {"objective": "binary", "seed": SEED, "deterministic": True, "num_threads": 1, "verbose": -1}
    for tr, va in GroupKFold(5).split(X, y, groups):
        p_new[va] = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=100).predict(X[va])
    return p_new


def macro_f05(pairs: pl.DataFrame, p_col: str, s1: pl.DataFrame) -> dict:
    """Record argmax (best S1 per reck, ties -> lower s1k), sweep TS, best global t; macro F0.5 over EVERY S1 in
    `s1` (an S1 with no prediction and ntrue=0 scores 1, per f05_vec), overall and per country."""
    top = (pairs.filter(pl.col(p_col).is_not_null())
           .sort(["reck", p_col, "s1k"], descending=[False, True, False])
           .filter(pl.col("reck").is_first_distinct()))
    idx = s1.select("s1k", code=pl.int_range(pl.len(), dtype=pl.UInt32))
    top = top.join(idx, on="s1k", how="inner", maintain_order="left")
    code, p, y = top["code"].to_numpy(), top[p_col].to_numpy(), top["label"].to_numpy().astype(bool)
    n, ntrue = s1.height, s1["ntrue"].to_numpy()

    def per(t: float) -> np.ndarray:
        k = p >= t
        return f05_vec(np.bincount(code[k & y], minlength=n), np.bincount(code[k], minlength=n), ntrue)
    curve = np.array([per(t).mean() for t in TS])
    bi = int(np.argmax(curve))
    f, ctry = per(float(TS[bi])), s1["country"].to_numpy()
    return {"best_t": float(TS[bi]), "macro_f05": float(curve[bi]), "n_s1": n,
            "by_country": {c: float(f[ctry == c].mean()) for c in sorted(set(ctry))}}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def eval_scopes(f0: pl.DataFrame, s1: pl.DataFrame, f_td: dict, log: StepLog) -> list[dict]:
    p_td = f0["p_td"].to_numpy().astype(np.float64)  # null -> nan
    out = []
    for lo, hi in SCOPES:
        m = scope_mask(f0.drop("label"), lo, hi)
        d = f0.filter(m)
        assert d["ce"].null_count() == 0, "every in-scope pair must be CE-scored"
        p_final = p_td.copy()
        p_final[m.to_numpy()] = stack(d)
        framed = f0.with_columns(p_new=pl.Series(p_final).fill_nan(None))
        assert framed["p_new"].is_null().equals(framed["p_td"].is_null()), "p_new must cover exactly p_td's rows"
        f_new = macro_f05(framed, "p_new", s1)
        assert f_new["n_s1"] == f_td["n_s1"], "p_td and p_new must be scored on the same S1 set"
        y = d["label"].to_numpy()
        r = {"lo": lo, "hi": hi, "pairs": d.height, "s1_touched": d["s1k"].n_unique(),
             "pairs_per_s1": d.height / s1.height, "pairs_per_touched_s1": d.height / max(d["s1k"].n_unique(), 1),
             "pos_rate": float(y.mean()) if len(y) else float("nan"),
             "auc_ce": auc(y, d["ce"].to_numpy()), "auc_p_td": auc(y, d["p_td"].to_numpy()), "f_new": f_new}
        log("scope", **{k: v for k, v in r.items() if k != "f_new"}, best_t=f_new["best_t"],
            f05_p_new=f_new["macro_f05"], d_f05=f_new["macro_f05"] - f_td["macro_f05"])
        out.append(r)
    return out


def dist(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"n": len(x), "mean": float(x.mean()), **{f"p{k}": float(v) for k, v in zip([10, 25, 50, 75, 90], q)}}


def error_masks(f0: pl.DataFrame) -> tuple[pl.Expr, pl.Expr]:
    fn = (pl.col("label") == 1) & (pl.col("p_td") < FN_P_MAX)
    fp = (pl.col("label") == 0) & (pl.col("p_td") > FP_P_MIN)
    return fn, fp


def confident_errors(f0: pl.DataFrame) -> dict:
    fn_e, fp_e = error_masks(f0)
    fn, fp = f0.filter(fn_e), f0.filter(fp_e)
    recs = pl.concat([fn["reck"], fp["reck"]]).unique()
    cands = f0.filter(pl.col("reck").is_in(recs.implode()))
    assert cands["ce"].null_count() == 0, "every candidate of a confident-error record must be CE-scored"
    top = (cands.sort(["reck", "ce", "s1k"], descending=[False, True, False])
           .filter(pl.col("reck").is_first_distinct())
           .select("reck", ce_s1k="s1k", ce_top_label="label"))
    fn_j, fp_j = fn.join(top, on="reck"), fp.join(top, on="reck")
    has_true = cands.group_by("reck").agg(has_true=(pl.col("label") == 1).any())
    fp_t = fp_j.join(has_true, on="reck").filter("has_true")
    wide = f0.filter(pl.col("p_td").is_between(*SCOPES[-1]))
    mean = lambda s: float(s.mean()) if len(s) else float("nan")
    return {
        "fn_dist": dist(fn["ce"].to_numpy()), "fp_dist": dist(fp["ce"].to_numpy()),
        "ref_true_dist": dist(wide.filter(pl.col("label") == 1)["ce"].to_numpy()),
        "ref_false_dist": dist(wide.filter(pl.col("label") == 0)["ce"].to_numpy()),
        "fn_n": fn.height, "fn_recs": fn["reck"].n_unique(),
        "fn_ce_argmax_is_true": mean(fn_j["ce_s1k"] == fn_j["s1k"]),
        "fp_n": fp.height, "fp_recs": fp["reck"].n_unique(),
        "fp_ce_argmax_not_wrong_s1": mean(fp_j["ce_s1k"] != fp_j["s1k"]),
        "fp_with_true_cand": fp_t.height, "fp_ce_argmax_is_true": mean(fp_t["ce_top_label"] == 1),
        "cands_scored": cands.height,
    }


def test_scope(pps: dict[str, float]) -> list[dict]:
    """Label-free test counts per scope and country (country = report grouping only) -> hours at each pairs/s."""
    country = pl.scan_parquet(norm_path("test", 1)).select(s1_id="entity_id", country="country")
    n_s1 = country.group_by("country").agg(n_s1=pl.len())
    tp = pl.scan_parquet(path("oof_dir") / "test_p.parquet").select("s1_id", "p").join(country, on="s1_id", how="left")
    rows = []
    for lo, hi in SCOPES:
        g = (tp.filter(pl.col("p").is_between(lo, hi)).group_by("country").agg(pairs=pl.len())
             .join(n_s1, on="country", how="left").collect().sort("country"))
        for c, pairs, n in [("ALL", int(g["pairs"].sum()), int(g["n_s1"].sum())), *g.iter_rows()]:
            rows.append({"lo": lo, "hi": hi, "country": c, "pairs": pairs, "pairs_per_s1": pairs / max(n, 1),
                         **{f"hours_{k}": pairs / v / 3600 for k, v in pps.items()}})
    return rows


def f(x, nd: int = 4) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_md(ctx: dict, f_td: dict, scopes: list[dict], errs: dict, bench_rows: list[dict], pps: dict,
             test_rows: list[dict], log: StepLog) -> None:
    sc = lambda r: f"[{r['lo']}, {r['hi']}]"
    L = ["# CE probe: fine-tuned cross-encoder vs GBM p_td", "",
         f"Model: `{MODEL_NAME}` fine-tuned 1 epoch (max_len={MAX_LEN}, batch={BATCH}, lr={LR}), "
         f"{ctx['model_src']}.{' **--smoke: 1% of fold-0 S1s, plumbing check only.**' if ctx['smoke'] else ''}",
         f"Eval frame: {ctx['f0_rows']} fold-0 oof rows, {ctx['n_s1']} fold-0 S1s; CE-scored rows: {ctx['scored']} "
         f"(widest scope + all candidates of confident-error records).", "",
         "Scopes are label-free (p_td band only, no true pairs force-added). p_new = stacked score inside the scope, "
         "p_td outside; macro F0.5 over the same S1 set for both (asserted), record argmax, best global t each.", "",
         "## Scopes and stacked ΔF0.5",
         f"p_td baseline: macro F0.5 {f(f_td['macro_f05'])} @ t={f_td['best_t']} (n_s1={f_td['n_s1']}).", "",
         "| scope | pairs | pairs/S1 | pairs/touched S1 | pos rate | AUC ce | AUC p_td | t new | F0.5 p_new | ΔF0.5 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in scopes:
        fn_ = r["f_new"]
        L.append(f"| {sc(r)} | {r['pairs']} | {f(r['pairs_per_s1'], 3)} | {f(r['pairs_per_touched_s1'], 2)} | "
                 f"{f(r['pos_rate'])} | {f(r['auc_ce'])} | {f(r['auc_p_td'])} | {fn_['best_t']} | "
                 f"{f(fn_['macro_f05'])} | {fn_['macro_f05'] - f_td['macro_f05']:+.4f} |")
    L += ["", "### Per country (ΔF0.5 = p_new − p_td, same global t as the overall row)",
          "| country | F0.5 p_td | " + " | ".join(f"Δ {sc(r)}" for r in scopes) + " |",
          "|---|---|" + "---|" * len(scopes)]
    for c, v in f_td["by_country"].items():
        L.append(f"| {c} | {f(v)} | " + " | ".join(f"{r['f_new']['by_country'][c] - v:+.4f}" for r in scopes) + " |")
    L += ["", f"## Confident GBM errors (fold 0; FN: true, p_td < {FN_P_MAX}; FP: false, p_td > {FP_P_MIN})",
          "| group | n | mean | p10 | p25 | p50 | p75 | p90 |", "|---|---|---|---|---|---|---|---|"]
    for name, key in [("confident FN (true pairs)", "fn_dist"), ("confident FP (false pairs)", "fp_dist"),
                      (f"ref: true pairs in {list(SCOPES[-1])}", "ref_true_dist"),
                      (f"ref: false pairs in {list(SCOPES[-1])}", "ref_false_dist")]:
        d = errs[key]
        L.append(f"| {name} | {d['n']} | " + " | ".join(f(d.get(k, float('nan')), 3)
                                                     for k in ["mean", "p10", "p25", "p50", "p75", "p90"]) + " |")
    L += ["", f"- FN: {errs['fn_n']} pairs / {errs['fn_recs']} records; CE argmax over the record's fold-0 "
              f"candidates is the true S1 in **{f(100 * errs['fn_ce_argmax_is_true'], 1)}%**.",
          f"- FP: {errs['fp_n']} pairs / {errs['fp_recs']} records; CE argmax is NOT the wrongly-scored S1 in "
          f"**{f(100 * errs['fp_ce_argmax_not_wrong_s1'], 1)}%**; of the {errs['fp_with_true_cand']} FP records "
          f"that have a true candidate, CE argmax is the true S1 in {f(100 * errs['fp_ce_argmax_is_true'], 1)}%.",
          f"- Candidates scored for this section: {errs['cands_scored']}. Label-selected by design; none of these "
          "rows enter a stacking scope unless their p_td is in the band.", "",
          f"## MPS throughput ({bench_rows[0]['pairs']} scope pairs, warm-up excluded)",
          "| batch | dtype | max_len | secs | pairs/s | max abs Δ vs fp32 same len | max abs Δ vs fp32 len 64 |",
          "|---|---|---|---|---|---|---|"]
    for r in bench_rows:
        L.append(f"| {r['batch']} | {r['dtype']} | {r['max_len']} | {r['secs']} | {r['pairs_per_s']} | "
                 f"{r['max_abs_d_vs_fp32_same_len']:.2e} | {r['max_abs_d_vs_fp32_len64']:.2e} |")
    L += ["", "## Test scope (oof/test_p.parquet `p`, label-free) -> scoring hours",
          "Hours at: " + ", ".join(f"`{k}` = {v:.1f} pairs/s" for k, v in pps.items()) + ".", "",
          "| scope | country | pairs | pairs/S1 | " + " | ".join(f"hours @ {k}" for k in pps) + " |",
          "|---|---|---|---|" + "---|" * len(pps)]
    for r in test_rows:
        L.append(f"| [{r['lo']}, {r['hi']}] | {r['country']} | {r['pairs']} | {f(r['pairs_per_s1'], 2)} | "
                 + " | ".join(f(r[f'hours_{k}'], 2) for k in pps) + " |")
    L += ["", "## Steps", "```", *[json.dumps(r, default=str) for r in log.rows], "```"]
    (ROOT / "docs" / "ce_probe.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- --stack2
def load_world(s1: pl.DataFrame) -> pl.DataFrame:
    """Every oof row cv_full's (b) scorer saw (p_td not null) + its s1_table code (cv_full.score's tie-break key)."""
    return (oof_scan().filter(pl.col("p_td").is_not_null())
            .join(s1.lazy().select("s1k", "code"), on="s1k", how="inner").collect())


def s1_f05(pred: pl.DataFrame, s1: pl.DataFrame) -> np.ndarray:
    """Per-S1 F0.5 of the predicted pairs pred[code, y] (an S1 with no prediction and ntrue=0 scores 1)."""
    c, y, n = pred["code"].to_numpy(), pred["y"].to_numpy(), s1.height
    return f05_vec(np.bincount(c[y], minlength=n), np.bincount(c, minlength=n), s1["ntrue"].to_numpy())


def evaluate(w: pl.DataFrame, p_col: str, s1: pl.DataFrame, scope: np.ndarray) -> dict:
    """cv_full.score's rule on w[p_col]: record argmax (ties -> lowest code), best global t over TS for the S1s in
    `scope`. Also returns the predicted pairs at that t (pred[reck, code, y])."""
    top = (w.select("reck", "code", p=p_col, y=pl.col("label") == 1)
           .sort(["reck", "p", "code"], descending=[False, True, False])
           .filter(pl.col("reck").is_first_distinct()))
    c, p, y, n, ntrue = top["code"].to_numpy(), top["p"].to_numpy(), top["y"].to_numpy(), s1.height, s1["ntrue"].to_numpy()

    def per(t: float) -> np.ndarray:
        k = p >= t
        return f05_vec(np.bincount(c[k & y], minlength=n), np.bincount(c[k], minlength=n), ntrue)
    curve = np.array([per(t)[scope].mean() for t in TS])
    bi = int(np.argmax(curve))
    pred = top.filter(pl.Series(p >= TS[bi])).select("reck", "code", "y")
    f, ctry = s1_f05(pred, s1), s1["country"].to_numpy()
    return {"best_t": float(TS[bi]), "macro_f05": float(curve[bi]), "n_s1": int(scope.sum()), "pred": pred,
            "by_country": {c: float(f[scope & (ctry == c)].mean()) for c in sorted(set(ctry[scope]))}}


def stack_features(w: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    """Rows of every record with an in-stack row, + label-free stacker features. ce is kept on in-stack rows only
    (the scored set of a record); ranks are 1 = highest, "min" on ties, null where ce is null."""
    recs = w.filter("in_stack")["reck"].unique().implode()
    r = (w.filter(pl.col("reck").is_in(recs))
         .join(scores, on=["s1k", "reck"], how="left")
         .with_columns(ce=pl.when("in_stack").then("ce")))
    x = r.drop("label")  # nothing below may read the label
    ce, g = pl.col("ce"), "reck"
    p = pl.col("p_td").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    x = x.with_columns(logit_p_td=(p / (1 - p)).log(), _ro=ce.rank("ordinal", descending=True).over(g),
                       ce_rank=ce.rank("min", descending=True).over(g), n_scored_cands=ce.count().over(g),
                       p_td_rank=pl.col("p_td").rank("min", descending=True).over(g))
    x = x.with_columns(_t1=pl.when(pl.col("_ro") == 1).then(ce).max().over(g),
                       _t2=pl.when(pl.col("_ro") == 2).then(ce).max().over(g))
    x = x.with_columns(ce_margin=ce - pl.when(pl.col("_ro") == 1).then("_t2").otherwise("_t1"),
                       ce_is_argmax=(pl.col("ce_rank") == 1).cast(pl.Int8)).drop("_ro", "_t1", "_t2")
    return x.with_columns(label=r["label"]).filter("in_stack")


def fit_stack(d: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, dict]:
    """LightGBM on `feats`, inner GroupKFold(5) by s1k -> OOF p for d's rows; + gain importance of a full fit."""
    X = d.select(pl.col(feats).cast(pl.Float64)).to_numpy()  # null -> nan: LightGBM's missing value
    y, groups = d["label"].to_numpy(), d["s1k"].to_numpy()
    params = {"objective": "binary", "seed": SEED, "deterministic": True, "num_threads": 1, "verbose": -1}
    p = np.full(len(y), np.nan)
    for tr, va in GroupKFold(5).split(X, y, groups):
        p[va] = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), STACK_ROUNDS).predict(X[va])
    gain = lgb.train(params, lgb.Dataset(X, label=y), STACK_ROUNDS).feature_importance("gain")
    return p, dict(zip(feats, (gain / gain.sum()).round(4).tolist()))


def band_of(col: str) -> pl.Expr:
    e = pl.lit(None, pl.Int8)
    for b, (lo, hi) in reversed(list(enumerate(BANDS))):
        inside = pl.col(col).is_between(lo, hi, closed="both" if b == len(BANDS) - 1 else "left")
        e = pl.when(inside).then(pl.lit(b, pl.Int8)).otherwise(e)
    return e


def pred_diff(a: pl.DataFrame, b: pl.DataFrame, s1_scope: np.ndarray) -> pl.DataFrame:
    """Scope S1s' predicted pairs of a (p_td) vs b (p_new): [reck, code, y, in_a, in_b]."""
    keep = lambda t: t.filter(pl.Series(s1_scope[t["code"].to_numpy()]))
    return (keep(a).with_columns(in_a=pl.lit(True))
            .join(keep(b).with_columns(in_b=pl.lit(True)), on=["reck", "code", "y"], how="full", coalesce=True)
            .with_columns(pl.col("in_a", "in_b").fill_null(False)))


def rescue(w: pl.DataFrame, base: dict, new: dict, bands: pl.DataFrame, s1: pl.DataFrame, scope: np.ndarray) -> dict:
    """Confident FN (true, p_td < FN_P_MAX, scope S1) recovered by p_new; FPs p_new adds, and their F0.5 cost."""
    cfn = w.filter(pl.col("in_scope") & (pl.col("label") == 1) & (pl.col("p_td") < FN_P_MAX)).select("reck", "code", "p_td")
    d = pred_diff(base["pred"], new["pred"], scope).join(bands, on=["reck", "code"], how="left")
    got = d.filter(pl.col("in_b") & pl.col("y")).join(cfn, on=["reck", "code"], how="semi")
    cfn_recs = cfn["reck"].unique().implode()
    new_fp = d.filter(pl.col("in_b") & ~pl.col("in_a") & ~pl.col("y"))
    macro = lambda pred: float(s1_f05(pred, s1)[scope].mean())
    f_new = macro(new["pred"])
    return {"cfn_pairs": cfn.height, "cfn_recs": cfn["reck"].n_unique(),
            "cfn_stackable_pairs": int((cfn["p_td"] >= STACK_LO).sum()),
            "cfn_rescued_pairs": got.height, "cfn_rescued_recs": got["reck"].n_unique(),
            "cfn_recs_wrong_s1_new": d.filter(pl.col("in_b") & ~pl.col("y") & pl.col("reck").is_in(cfn_recs))["reck"].n_unique(),
            "cfn_recs_wrong_s1_td": d.filter(pl.col("in_a") & ~pl.col("y") & pl.col("reck").is_in(cfn_recs))["reck"].n_unique(),
            "new_fp": new_fp.height, "new_fp_cfn_recs": int(new_fp["reck"].is_in(cfn_recs).sum()),
            "new_fp_by_band": {str(k): v for k, v in new_fp.group_by("band").len().sort("band").iter_rows()},
            "fixed_fp": d.filter(pl.col("in_a") & ~pl.col("in_b") & ~pl.col("y")).height,
            "lost_tp": d.filter(pl.col("in_a") & ~pl.col("in_b") & pl.col("y")).height,
            "gained_tp": d.filter(pl.col("in_b") & ~pl.col("in_a") & pl.col("y")).height,
            # cost = F0.5(p_new's predictions) - F0.5(same minus the new FPs); gain of rescues likewise
            "new_fp_cost": f_new - macro(new["pred"].join(new_fp, on=["reck", "code"], how="anti")),
            "rescue_gain": f_new - macro(new["pred"].join(got, on=["reck", "code"], how="anti")),
            "t_new": new["best_t"]}


def band_flips(base: dict, new: dict, bands: pl.DataFrame, scope: np.ndarray) -> list[dict]:
    """Per p_td band of the predicted pair (null = outside the stack): decisions p_td -> p_new at each one's t."""
    d = pred_diff(base["pred"], new["pred"], scope).join(bands, on=["reck", "code"], how="left")
    agg = d.group_by("band").agg(
        gained_tp=(pl.col("in_b") & ~pl.col("in_a") & pl.col("y")).sum(),
        lost_tp=(pl.col("in_a") & ~pl.col("in_b") & pl.col("y")).sum(),
        new_fp=(pl.col("in_b") & ~pl.col("in_a") & ~pl.col("y")).sum(),
        fixed_fp=(pl.col("in_a") & ~pl.col("in_b") & ~pl.col("y")).sum()).sort("band", nulls_last=True)
    return agg.to_dicts()


def band_name(b) -> str:
    if b is None:
        return "outside stack"
    lo, hi = BANDS[b]
    return f"[{lo}, {hi}{']' if b == len(BANDS) - 1 else ')'}"


def write_md_stack2(ctx: dict, b_all: dict, b0: dict, stacks: dict, band_rows: list[dict], flips: list[dict],
                    resc: dict, log: StepLog) -> None:
    ctrys = list(b0["by_country"])
    d = lambda r: f"{r['macro_f05'] - b0['macro_f05']:+.4f}"
    dc = lambda r: " | ".join(f"{r['by_country'][c] - b0['by_country'][c]:+.4f}" for c in ctrys)
    L = ["# CE probe (--stack2): rank-aware stacker on the fine-tuned CE, scored in cv_full's (b) world", "",
         f"Reuses `oof/ce_probe_scores.parquet` (`{MODEL_NAME}` fine-tuned, see the docstring); no re-scoring, no "
         f"retraining.{' **--smoke: 1% of fold-0 scope S1s, plumbing check only.**' if ctx['smoke'] else ''}", "",
         "## 1. Baseline (fixed)",
         "The previous probe scored 0.7996: it kept the 19% of fold-0 S1s that are dead in world SEED+2000 (no "
         "predictions, ntrue > 0 -> F0.5 0) and took the record argmax over fold-0 rows only. Now: cv_full's rule "
         "over every oof row with p_td (all folds), scope = alive fold-0 S1s.", "",
         "| check | macro F0.5 | t | n_s1 |", "|---|---|---|---|",
         f"| oof/cv_full.json b_test_density (all folds) | {f(ctx['json_b'])} | {ctx['json_t']} | {ctx['json_n']} |",
         f"| recomputed here, all folds (must match to {EXACT_TOL}) | {f(b_all['macro_f05'], 6)} | {b_all['best_t']} | {b_all['n_s1']} |",
         f"| **fold 0, alive S1s** (within {F0_TOL} of json) | **{f(b0['macro_f05'])}** | {b0['best_t']} | {b0['n_s1']} |", "",
         "cv_full.json stores no per-fold F0.5, so fold 0 is checked against the all-fold value.", "",
         f"## 2-3. Stacked ΔF0.5, scope p_td in [{STACK_LO}, {STACK_HI}] (fold 0)",
         f"Stack rows: {ctx['stack_rows']} pairs, {ctx['stack_s1']} S1s, pos rate {f(ctx['pos_rate'])}. LightGBM "
         f"{STACK_ROUNDS} rounds, inner GroupKFold(5) by s1k; p_new inside the scope, p_td elsewhere (other folds too).", "",
         "| model | AUC (stack rows) | t | F0.5 | ΔF0.5 | " + " | ".join(f"Δ {c}" for c in ctrys) + " |",
         "|---|---|---|---|---|" + "---|" * len(ctrys),
         f"| p_td | {f(ctx['auc_p_td'])} | {b0['best_t']} | {f(b0['macro_f05'])} | — | " + " | ".join(
             f(b0['by_country'][c]) for c in ctrys) + " |"]
    for name, s in stacks.items():
        L.append(f"| {name} | {f(s['auc'])} | {s['ev']['best_t']} | {f(s['ev']['macro_f05'])} | {d(s['ev'])} | {dc(s['ev'])} |")
    L += ["", "Stack2 gain importance (full fit): " + ", ".join(f"{k} {v}" for k, v in stacks["stack2 (+ranks)"]["gain"].items()), "",
          "### Where the stack2 gain comes from: p_new applied inside ONE p_td band (p_td elsewhere), same fitted stacker",
          "| p_td band | pairs | pos rate | t | ΔF0.5 | " + " | ".join(f"Δ {c}" for c in ctrys) + " |",
          "|---|---|---|---|---|" + "---|" * len(ctrys)]
    for r in band_rows:
        L.append(f"| {band_name(r['band'])} | {r['pairs']} | {f(r['pos_rate'])} | {r['ev']['best_t']} | {d(r['ev'])} | {dc(r['ev'])} |")
    L += ["", "Band ablation Δs need not sum to the full-scope Δ (one global t each, shared record argmax).", "",
          "Decision flips p_td (its t) -> stack2 (its t), fold-0 scope S1s, by the predicted pair's p_td band:",
          "| band | gained TP | lost TP | new FP | fixed FP |", "|---|---|---|---|---|"]
    for r in flips:
        L.append(f"| {band_name(r['band'])} | {r['gained_tp']} | {r['lost_tp']} | {r['new_fp']} | {r['fixed_fp']} |")
    x = resc
    L += ["", f"## 4. Confident-FN rescue (true pair, p_td < {FN_P_MAX}, alive fold-0 S1) under stack2 @ t={x['t_new']}",
          f"- Confident FN: {x['cfn_pairs']} pairs / {x['cfn_recs']} records (old probe: 1039 pairs, dead S1s "
          f"included); {x['cfn_stackable_pairs']} pairs have p_td >= {STACK_LO} (the rest cannot change).",
          f"- Rescued (true pair now predicted): **{x['cfn_rescued_pairs']} pairs / {x['cfn_rescued_recs']} records**; "
          f"F0.5 gain of those rescues {x['rescue_gain']:+.5f}.",
          f"- Of the confident-FN records, assigned to a wrong S1: {x['cfn_recs_wrong_s1_td']} under p_td -> "
          f"{x['cfn_recs_wrong_s1_new']} under stack2.",
          f"- New FPs overall (predicted by stack2, not by p_td): **{x['new_fp']}** (by band: {x['new_fp_by_band']}; "
          f"{x['new_fp_cfn_recs']} on confident-FN records); F0.5 cost of the new FPs **{x['new_fp_cost']:+.5f}**.",
          f"- All flips: gained TP {x['gained_tp']}, lost TP {x['lost_tp']}, new FP {x['new_fp']}, fixed FP {x['fixed_fp']}.",
          "", "## Caveats",
          "- CE scores exist for fold-0 rows only: ce_rank / ce_margin / n_scored_cands see the record's fold-0 "
          "in-scope candidates (~1/5 of its competitors). On test every in-scope candidate would be scored, so the "
          "rank features would be computed over a harder set. p_td_rank uses all folds.",
          "- Confident-error records' extra CE rows (label-selected) are excluded from every rank.",
          "- Earlier sections of this doc (throughput, test scope hours) are in git history; rerun without --stack2 "
          "to regenerate them.", "",
          "## Steps", "```", *[json.dumps(r, default=str) for r in log.rows], "```"]
    (ROOT / "docs" / "ce_probe.md").write_text("\n".join(L) + "\n")


def stack2_main(smoke: bool) -> None:
    log = StepLog()
    s1 = s1_table(1.0)
    alive = ~dropped(s1, SEED + 2000)
    full = load_world(s1)
    log("load world", rows=full.height, peak_rss_mb=peak_rss_mb())

    # 1) baseline: reproduce cv_full.json (b) exactly, then fold 0
    cv_b = json.loads((path("oof_dir") / "cv_full.json").read_text())["b_test_density"]
    b_all = evaluate(full, "p_td", s1, alive)
    print(f"(b) all folds: json {cv_b['macro_f05']:.6f} @ t={cv_b['best_t']}  recomputed {b_all['macro_f05']:.6f} "
          f"@ t={b_all['best_t']}", flush=True)
    if abs(b_all["macro_f05"] - cv_b["macro_f05"]) > EXACT_TOL or b_all["best_t"] != cv_b["best_t"] or b_all["n_s1"] != cv_b["n_s1"]:
        raise SystemExit("STOP: recomputing cv_full's (b) from oof/oof_full.parquet does not reproduce oof/cv_full.json. "
                         "Either oof_full.parquet is not the run that wrote cv_full.json, or this scorer's rule/world "
                         "differs from cv_full.score (argmax set, tie-break by code, alive S1 scope). Fix before stacking.")
    scope = (s1["fold"].to_numpy() == 0) & alive
    if smoke:
        sub = np.zeros(s1.height, bool)
        sub[np.random.default_rng(SEED + 3).choice(np.flatnonzero(scope), round(0.01 * scope.sum()), replace=False)] = True
        scope = sub
    b0_full = evaluate(full, "p_td", s1, scope)
    full = full.with_columns(in_scope=pl.Series(scope[full["code"].to_numpy()]))
    # only records with a candidate among the scope S1s can move a scope S1's F0.5
    w = full.filter(pl.col("reck").is_in(full.filter("in_scope")["reck"].unique().implode()))
    del full
    b0 = evaluate(w, "p_td", s1, scope)
    assert abs(b0["macro_f05"] - b0_full["macro_f05"]) < 1e-12 and b0["best_t"] == b0_full["best_t"], "record restriction changed F0.5"
    print(f"fold 0 (b): {b0['macro_f05']:.6f} @ t={b0['best_t']} (n_s1={b0['n_s1']}) vs json all-fold "
          f"{cv_b['macro_f05']:.6f}: Δ {b0['macro_f05'] - cv_b['macro_f05']:+.5f}", flush=True)
    log("baseline", all_folds=b_all["macro_f05"], fold0=b0["macro_f05"], t=b0["best_t"], n_s1=b0["n_s1"], w_rows=w.height)
    if abs(b0["macro_f05"] - cv_b["macro_f05"]) > F0_TOL:
        msg = (f"fold-0 (b) {b0['macro_f05']:.4f} is > {F0_TOL} from the all-fold {cv_b['macro_f05']:.4f}: the "
               "all-fold number reproduces exactly, so this is a fold-0 effect (fold noise is ~0.002), not the world.")
        if not smoke:
            raise SystemExit("STOP: " + msg)
        print("smoke (1% sample, noisy): " + msg, flush=True)

    # 2) stacker features, label-free, in-stack = alive fold-0 scope S1 rows with p_td in the band
    scores = pl.read_parquet(path("oof_dir") / "ce_probe_scores.parquet", columns=["s1k", "reck", "ce"])
    assert scores.select(pl.struct("s1k", "reck").is_unique().all()).item(), "duplicate (s1k, reck) in CE scores"
    w = w.with_columns(in_stack=pl.col("in_scope") & pl.col("p_td").is_between(STACK_LO, STACK_HI))
    st = stack_features(w, scores).with_columns(band=band_of("p_td"))
    assert st["ce"].null_count() == 0, "every in-stack pair must be CE-scored (scores file from another frame?)"
    assert st["band"].null_count() == 0
    log("stack features", rows=st.height, s1=st["s1k"].n_unique(), peak_rss_mb=peak_rss_mb())

    y = st["label"].to_numpy()
    stacks, pcols = {}, {}
    for i, (name, feats) in enumerate(STACKS.items()):
        p, gain = fit_stack(st, feats)
        pcols[name] = f"p_s{i + 1}"
        st = st.with_columns(pl.Series(pcols[name], p))
        stacks[name] = {"auc": auc(y, p), "gain": gain}
        log(f"fit {name}", auc=stacks[name]["auc"])
    bands = st.select("reck", "code", "band")
    w = w.join(st.select("reck", "code", "band", *pcols.values()), on=["reck", "code"], how="left")
    for name, c in pcols.items():
        stacks[name]["ev"] = evaluate(w.with_columns(p_new=pl.coalesce(c, "p_td")), "p_new", s1, scope)
        log(f"score {name}", f05=stacks[name]["ev"]["macro_f05"], t=stacks[name]["ev"]["best_t"],
            d=stacks[name]["ev"]["macro_f05"] - b0["macro_f05"])

    # 3) band ablation with the stack2 scores
    s2 = pcols["stack2 (+ranks)"]
    band_rows = []
    for b in range(len(BANDS)):
        ev = evaluate(w.with_columns(p_new=pl.when(pl.col("band") == b).then(s2).otherwise("p_td")), "p_new", s1, scope)
        m = (st["band"] == b).to_numpy()
        band_rows.append({"band": b, "pairs": int(m.sum()), "pos_rate": float(y[m].mean()) if m.any() else float("nan"), "ev": ev})
        log(f"band {band_name(b)}", pairs=int(m.sum()), d=ev["macro_f05"] - b0["macro_f05"], t=ev["best_t"])
    new = stacks["stack2 (+ranks)"]["ev"]
    flips = band_flips(b0, new, bands, scope)

    # 4) confident-FN rescue + new FPs
    resc = rescue(w, b0, new, bands, s1, scope)
    log("rescue", **resc)
    ctx = {"smoke": smoke, "json_b": cv_b["macro_f05"], "json_t": cv_b["best_t"], "json_n": cv_b["n_s1"],
           "stack_rows": st.height, "stack_s1": st["s1k"].n_unique(), "pos_rate": float(y.mean()),
           "auc_p_td": auc(y, st["p_td"].to_numpy())}
    write_md_stack2(ctx, b_all, b0, stacks, band_rows, flips, resc, log)
    print(f"wrote docs/ce_probe.md (peak RSS {peak_rss_mb()} MB)")


# ---------------------------------------------------------------- --apply-test
TEST_SCOPE = (0.0005, 0.9995)  # SCOPES[-1]
TEST_CHUNK = 200_000


def load_test_scope(smoke: bool) -> pl.DataFrame:
    """-> test_p rows[s1_id, rec_id, p] with p in TEST_SCOPE (--smoke: first 10k of those)."""
    tp = pl.scan_parquet(path("oof_dir") / "test_p.parquet").filter(pl.col("p").is_between(*TEST_SCOPE)).collect()
    return tp.head(10_000) if smoke else tp


def score_test_chunked(scope: pl.DataFrame, texts: dict[str, pl.DataFrame], model_dir, log: StepLog) -> pl.DataFrame:
    """Resumable: appends new (s1k, reck, ce) rows to oof/ce_test_scores.parquet in TEST_CHUNK batches, skipping
    pairs already scored on rerun. Returns the full scores table (old + new) restricted to `scope`."""
    out_path = path("oof_dir") / "ce_test_scores.parquet"
    keyed = scope.with_columns(s1k=id_key("s1_id"), reck=id_key("rec_id"))
    done = (pl.read_parquet(out_path, columns=["s1k", "reck"]) if out_path.exists()
            else pl.DataFrame({"s1k": [], "reck": []}, schema={"s1k": pl.Int64, "reck": pl.Int64}))
    todo = keyed.join(done, on=["s1k", "reck"], how="anti")
    log("resume", already_scored=done.height, to_score=todo.height)
    if todo.height == 0:
        return pl.read_parquet(out_path).join(keyed.select("s1k", "reck"), on=["s1k", "reck"], how="semi")

    model = load_ce(model_dir, MAX_LEN, "fp32")
    t0 = time.perf_counter()
    for i in range(0, todo.height, TEST_CHUNK):
        chunk = todo[i:i + TEST_CHUNK]
        ce = predict(model, make_texts(chunk, texts), SCORE_BATCH)
        new_rows = chunk.select("s1k", "reck").with_columns(ce=pl.Series(ce))
        if out_path.exists():
            new_rows = pl.concat([pl.read_parquet(out_path), new_rows])
        new_rows.write_parquet(out_path)
        dt = time.perf_counter() - t0
        n_done = i + chunk.height
        log("score test chunk", pairs=n_done, of=todo.height, secs=round(dt, 1),
            pairs_per_s=round(n_done / max(dt, 1e-6), 1),
            eta_min=round((todo.height - n_done) / max(n_done / max(dt, 1e-6), 1e-6) / 60, 1))
    return pl.read_parquet(out_path).join(keyed.select("s1k", "reck"), on=["s1k", "reck"], how="semi")


def apply_test_main(smoke: bool) -> None:
    log = StepLog()
    texts = load_texts("test")
    log("load test texts", peak_rss_mb=peak_rss_mb())
    scope = load_test_scope(smoke)
    log("test scope", pairs=scope.height)

    scores = score_test_chunked(scope, texts, MODEL_DIR, log)
    d = scope.with_columns(s1k=id_key("s1_id"), reck=id_key("rec_id")).join(scores, on=["s1k", "reck"], how="left")
    assert d["ce"].null_count() == 0, "every scope pair must be CE-scored"
    log("scored", pairs=d.height, peak_rss_mb=peak_rss_mb())

    # fit stack1 on the current fold-0 oof rows in [logit p_td, ce] scope (oof/ce_probe_scores.parquet)
    fit_rows = pl.read_parquet(path("oof_dir") / "ce_probe_scores.parquet")
    fit_rows = fit_rows.filter(fit_rows["p_td"].is_between(*TEST_SCOPE))
    y = fit_rows["label"].to_numpy()

    def best_f05(p: np.ndarray) -> float:
        prec, rec, _ = precision_recall_curve(y, p)
        f05 = 1.25 * prec * rec / np.clip(0.25 * prec + rec, 1e-12, None)
        return float(np.nanmax(f05))
    f05_old = best_f05(fit_rows["p_td"].to_numpy())
    f05_new = best_f05(stack(fit_rows))
    log("oof stacker check (pairwise F0.5, sanity only, not the record-argmax macro metric)",
        f05_p_td=round(f05_old, 4), f05_p_new=round(f05_new, 4), delta=round(f05_new - f05_old, 4))

    p_td64 = np.clip(fit_rows["p_td"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    X_fit = np.column_stack([np.log(p_td64 / (1 - p_td64)), fit_rows["ce"].to_numpy()])
    params = {"objective": "binary", "seed": SEED, "deterministic": True, "num_threads": 1, "verbose": -1}
    stacker = lgb.train(params, lgb.Dataset(X_fit, label=fit_rows["label"].to_numpy()), num_boost_round=100)

    p_td_test = np.clip(d["p"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    X_test = np.column_stack([np.log(p_td_test / (1 - p_td_test)), d["ce"].to_numpy()])
    p_new_test = stacker.predict(X_test)
    d = d.with_columns(p_new=pl.Series(p_new_test.astype(np.float32)))
    log("applied to test", pairs=d.height, peak_rss_mb=peak_rss_mb())

    full = pl.scan_parquet(path("oof_dir") / "test_p.parquet").collect()
    out = (full.join(d.select("s1_id", "rec_id", "p_new"), on=["s1_id", "rec_id"], how="left")
           .with_columns(p=pl.coalesce("p_new", "p")).drop("p_new"))
    out_path = path("oof_dir") / "test_p_ce.parquet"
    out.write_parquet(out_path)
    log("wrote", path=str(out_path.relative_to(ROOT)), rows=out.height)
    print(f"wrote {out_path.relative_to(ROOT)} (peak RSS {peak_rss_mb()} MB)")


def main(smoke: bool, eval_only: bool) -> None:
    log = StepLog()
    model_dir = SMOKE_MODEL_DIR if (smoke and not eval_only) else MODEL_DIR
    texts = load_texts()
    log("load texts", peak_rss_mb=peak_rss_mb())
    if eval_only:
        assert (model_dir / "model.safetensors").exists(), f"--eval-only needs a saved model in {model_dir}"
    else:
        # CrossEncoderTrainer runs on HF Trainer, which needs accelerate; fail here, not after pair building.
        import importlib.util
        assert importlib.util.find_spec("accelerate"), "uv pip install -r requirements.txt  (from code/business_entity_resolution/)"
        train_pairs = build_train_pairs(smoke)
        log("train pairs", n=train_pairs.height, true=int((train_pairs["label"] == 1).sum()))
        fit_ce(train_pairs, texts, model_dir, log)

    f0, s1 = load_fold0(smoke)
    fn_e, fp_e = error_masks(f0)
    err_recs = f0.filter(fn_e | fp_e)["reck"].unique()
    wide = scope_mask(f0.drop("label"), *SCOPES[-1])
    need = (wide | f0["reck"].is_in(err_recs.implode())).to_numpy()
    log("fold-0 frame", rows=f0.height, n_s1=s1.height, wide_scope=int(wide.sum()), to_score=int(need.sum()),
        peak_rss_mb=peak_rss_mb())

    # throughput first: its numbers are useful even if the long scoring run is interrupted
    wide_rows = with_ids(f0.filter(wide).drop("label"))
    bench_rows = wide_rows.sample(n=min(2_000 if smoke else BENCH_N, wide_rows.height), seed=SEED + 4)
    bench_res = bench(model_dir, make_texts(bench_rows, texts), log)
    best = max(bench_res, key=lambda r: r["pairs_per_s"])
    pps = {"b64-fp32-L64": bench_res[0]["pairs_per_s"],
           f"best={best['batch']}-{best['dtype']}-L{best['max_len']}": best["pairs_per_s"]}

    to_score = with_ids(f0.filter(need))
    model = load_ce(model_dir, MAX_LEN, "fp32")
    t0 = time.perf_counter()
    ce = predict(model, make_texts(to_score, texts), SCORE_BATCH)
    dt = time.perf_counter() - t0
    log("score fold-0", pairs=len(ce), secs=round(dt, 1), pairs_per_s=round(len(ce) / max(dt, 1e-6), 1))
    ce_full = np.full(f0.height, np.nan, dtype=np.float32)
    ce_full[need] = ce
    f0 = f0.with_columns(ce=pl.Series(ce_full).fill_nan(None))
    scores_path = path("oof_dir") / ("ce_probe_scores_smoke.parquet" if smoke else "ce_probe_scores.parquet")
    to_score.with_columns(ce=pl.Series(ce)).select("s1_id", "rec_id", "s1k", "reck", "label", "p_td", "ce").write_parquet(scores_path)
    log("saved scores", path=str(scores_path.relative_to(ROOT)), rows=len(ce))

    f_td = macro_f05(f0, "p_td", s1)
    log("baseline", f05_p_td=f_td["macro_f05"], best_t=f_td["best_t"], n_s1=f_td["n_s1"])
    scopes = eval_scopes(f0, s1, f_td, log)
    errs = confident_errors(f0)
    log("confident errors", **{k: v for k, v in errs.items() if not k.endswith("_dist")})
    test_rows = test_scope(pps)
    log("test scope", peak_rss_mb=peak_rss_mb())
    ctx = {"smoke": smoke, "f0_rows": f0.height, "n_s1": s1.height, "scored": int(need.sum()),
           "model_src": f"reused saved {model_dir.relative_to(ROOT)}" if eval_only else f"trained -> {model_dir.relative_to(ROOT)}"}
    write_md(ctx, f_td, scopes, errs, bench_res, pps, test_rows, log)
    print(f"wrote docs/ce_probe.md, {scores_path.relative_to(ROOT)} (peak RSS {peak_rss_mb()} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% of fold-0 S1s, 2k bench pairs, <3 min")
    ap.add_argument("--eval-only", action="store_true", help="reuse the saved model, no retraining")
    ap.add_argument("--stack2", action="store_true", help="(b)-world baseline + rank-feature stacker on saved CE scores")
    ap.add_argument("--apply-test", action="store_true", help="score test pairs, fit stack1, write oof/test_p_ce.parquet")
    a = ap.parse_args()
    if a.apply_test:
        apply_test_main(a.smoke)
    elif a.stack2:
        stack2_main(a.smoke)
    else:
        main(a.smoke, a.eval_only)
