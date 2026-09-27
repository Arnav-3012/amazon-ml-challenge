"""M6 self-training probe: does pseudo-labeling an unseen-country's candidates and retraining on them recover
LOCO loss, as a proxy for France (unseen in train)? India stands in for France here because it's the LOCO
country we HAVE true labels for -- everything below runs on the SAME held-out labels only to grade the
pseudo-labels and the final score; the retrain step itself must never see them.

Setup: 20% S1 sample (s1_table(0.2), same seeded permutation as src.train's subset), fast LightGBM (lr 0.1,
<=1500 rounds, early stopping unchanged), fold 0 only, test-density (b) scorer (src.cv_full.score, context(dead_s1k): drop_frac of ALL train S1s) --
same machinery as src.ab_test, so this is directly comparable to that gate's numbers.

Round 1 (baseline): train on US-only rows of the sample's fold-0 training S1s -> score India (expect ~0.86,
between the M4-era pooled OOF 0.9654 and the M4-era India LOCO 0.8221; this sample/params differ from that
run so treat 0.86 as a ballpark, not a target).

Pseudo-labeling (round 1 model's p on India's fold-0-scope candidates):
  positive: p >= 0.98 AND this candidate is its record's argmax AND no OTHER candidate of the same S1
            (i.e. same record's competing S1) has name_tset >= 95 AND addr_tset >= 90 (the ambiguity guard --
            drop cases where a second near-duplicate S1 could just as well be the true match).
  negative: p <= 0.02, sampled 3:1 vs positives (MC["neg_ratio"], seeded).
Both labels are graded against the TRUE India labels (precision/recall) -- diagnostic only, never used to
pick which pseudo-labels enter training.

Round 2: retrain on US rows + pseudo-India rows (weight 0.5) -> score India.
Round 3: re-label India with the round-2 model (same rule), retrain US + new pseudo-India (weight 0.5) -> score India.

Gate: adopt self-training for France if round 1 -> round 2 gains >= +0.02 macro F0.5 on India.

Run from code/business_entity_resolution/:
  python -m src.selftrain_loco
"""
import json
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl

from .cv_full import Data, context, dead_s1k, dropped, s1_table, score, split_fold, weighted_sample_rows
from .io import CFG, StepLog, path

MC, SEED = CFG["matcher"], CFG["seed"]
FAST_LR, FAST_ROUNDS = 0.1, 1500
POS_P, NEG_P, AMB_NAME, AMB_ADDR = 0.98, 0.02, 95, 90
GATE_DELTA = 0.02
MAX_RSS_MB = 10_000


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        if self.rows[-1]["peak_rss_mb"] > MAX_RSS_MB:
            raise SystemExit(f"STOP: peak RSS {self.rows[-1]['peak_rss_mb']} MB > {MAX_RSS_MB} MB after '{step}'")


def country_of(s1: pl.DataFrame) -> np.ndarray:
    return s1["country"].to_numpy()


def fast_fit(X: np.ndarray, y: np.ndarray, feats: list[str], Xv: np.ndarray, yv: np.ndarray,
             w: np.ndarray | None = None) -> lgb.Booster:
    """Same knobs as src.train.fit but learning_rate overridden and rounds capped, for a quick LOCO probe."""
    from .train import params as base_params
    p = {**base_params(), "learning_rate": FAST_LR}
    dtr = lgb.Dataset(X, label=y, weight=w, feature_name=feats, params=p, free_raw_data=True).construct()
    dva = lgb.Dataset(Xv, label=yv, feature_name=feats, params=p, reference=dtr, free_raw_data=True).construct()
    return lgb.train(p, dtr, FAST_ROUNDS, valid_sets=[dva], valid_names=["heldout"],
                      callbacks=[lgb.early_stopping(MC["early_stopping"], verbose=False), lgb.log_evaluation(200)])


def pseudo_label(d: Data, rows: np.ndarray, X: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Among `rows` (India eval-scope candidate rows, sorted; X = their feature matrix, row-aligned), return
    sorted (pos_rows, neg_rows) per the spec's rule.
    pos: p >= POS_P, this row is its record's (reck) argmax p, AND no OTHER row of the SAME S1 (s1k) has
    name_tset >= AMB_NAME and addr_tset >= AMB_ADDR (the ambiguity guard is per-S1: a second near-duplicate
    candidate record for this S1 makes the match unsafe to trust blindly).
    neg: p <= NEG_P, sampled 3:1 vs #pos (MC['neg_ratio'], seeded)."""
    name_j, addr_j = d.feats.index("name_tset"), d.feats.index("addr_tset")
    t = pl.DataFrame({"row": rows, "s1k": d.meta["s1k"].to_numpy()[rows], "reck": d.meta["reck"].to_numpy()[rows],
                       "p": p[rows], "name_tset": X[:, name_j], "addr_tset": X[:, addr_j]})
    is_argmax = (t.sort(["reck", "p", "s1k"], descending=[False, True, False])  # ties -> lowest s1k, matches cv_full.score
                 .select("row", pl.col("row").is_first_distinct().over("reck").alias("argmax")))
    t = t.join(is_argmax, on="row")
    amb = (t.select("s1k", "row", amb_hit=((pl.col("name_tset") >= AMB_NAME) & (pl.col("addr_tset") >= AMB_ADDR)))
           .with_columns(other_amb=(pl.col("amb_hit").sum().over("s1k") - pl.col("amb_hit").cast(pl.Int32))))
    t = t.join(amb.select("row", "other_amb"), on="row")
    pos = t.filter((pl.col("p") >= POS_P) & pl.col("argmax") & (pl.col("other_amb") == 0))["row"].to_numpy()
    neg_pool = t.filter(pl.col("p") <= NEG_P)["row"].to_numpy()
    rng = np.random.default_rng(SEED + 7000)
    n_neg = min(len(neg_pool), MC["neg_ratio"] * len(pos))
    neg = rng.choice(neg_pool, n_neg, replace=False) if n_neg else neg_pool[:0]
    return np.sort(pos), np.sort(neg)


def grade(d: Data, rows: np.ndarray, universe: np.ndarray, label_true: int) -> dict:
    """precision/recall of `rows` (pseudo-labeled as `label_true`) against d.y (the TRUE labels); recall
    denominator = rows of `universe` (the India eval rows) that truly carry label_true."""
    n_true_total = int((d.y[universe] == label_true).sum())
    if len(rows) == 0:
        return {"n": 0, "precision": None, "recall": None, "n_true_total": n_true_total}
    correct = int((d.y[rows] == label_true).sum())
    return {"n": len(rows), "precision": correct / len(rows),
            "recall": correct / n_true_total if n_true_total else None, "n_true_total": n_true_total}


def main() -> None:
    log = Log()
    s1 = s1_table(0.2)
    d = Data(s1)
    ctry = country_of(s1)
    log("meta", rows=len(d.i), s1=s1.height, pos=int(d.y.sum()), n_feats=len(d.feats))

    drop, train_mask, inner_mask = split_fold(s1, 0)  # fold 0's world: reuse cv_full's dropout/early-stop split
    us_train = train_mask & (ctry == "US")
    us_inner = inner_mask & (ctry == "US")
    work = path("features_dir") / "_worlds_selftrain"
    # context(dead_s1k): drop_frac of ALL train S1s, recomputed over all rows -> test density at a 20% sample
    # (world() would drop only the sample's S1s, ~3.8% of all). split_fold's `drop` is the same draw.
    context("train", dead_s1k(SEED + 1000), work / "fold0", d.i, log)

    hard = d.meta["hard"].to_numpy()
    tr0, w0 = weighted_sample_rows(np.flatnonzero(us_train[d.code]), d.y, hard, d.in_v1, np.random.default_rng(SEED))
    va0 = np.flatnonzero(us_inner[d.code])
    X0, Xv0 = d.gather(tr0, work / "fold0"), d.gather(va0, work / "fold0")

    dropE = dropped(s1, SEED + 2000)  # test-density eval world, same draw as cv_full (b) / world_skew
    keepE = ~dropE[d.code]
    context("train", dead_s1k(SEED + 2000), work / "eval", d.i, log)
    india_rows = np.flatnonzero((ctry[d.code] == "India") & keepE)
    Xe_india = d.gather(india_rows, work / "eval")
    scope_india = (ctry == "India") & ~dropE
    log("data", us_train_rows=len(tr0), us_valid_rows=len(va0), india_eval_rows=len(india_rows))

    print(f"US-only train rows: {len(tr0)} (pos {int(d.y[tr0].sum())}), India eval rows: {len(india_rows)}", flush=True)

    def score_india(p_full: np.ndarray, tag: str) -> dict:
        r = score(s1, d, p_full, scope_india)
        print(f"[{tag}] India macro_f05={r['macro_f05']:.5f} best_t={r['best_t']:.2f} "
              f"n_s1={r['n_s1']} fp_pairs={r['fp_pairs']}", flush=True)
        log(f"score {tag}", macro_f05=r["macro_f05"], best_t=r["best_t"], n_s1=r["n_s1"])
        return r

    # ---- Round 1: US-only baseline ----
    bst1 = fast_fit(X0, d.y[tr0], d.feats, Xv0, d.y[va0], w0)
    p1 = np.full(len(d.i), np.nan, np.float32)
    p1[india_rows] = bst1.predict(Xe_india, num_iteration=bst1.best_iteration)
    r1 = score_india(p1, "round1_us_only")

    pos1, neg1 = pseudo_label(d, india_rows, Xe_india, p1)
    g_pos1, g_neg1 = grade(d, pos1, india_rows, 1), grade(d, neg1, india_rows, 0)
    print(f"pseudo-label round 1: positives {g_pos1}, negatives {g_neg1}", flush=True)
    log("pseudo round1", pos=g_pos1, neg=g_neg1)

    # ---- Round 2: US + pseudo-India (weight 0.5), re-score India ----
    def retrain_with_pseudo(pos: np.ndarray, neg: np.ndarray, tag: str) -> tuple[lgb.Booster, np.ndarray]:
        pseudo_rows = np.concatenate([pos, neg])
        # India eval rows are already in memory (Xe_india, aligned to sorted india_rows): index, don't re-gather
        Xp = Xe_india[np.searchsorted(india_rows, pseudo_rows)]
        yp = np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)]
        Xtr = np.concatenate([X0, Xp], axis=0)
        ytr = np.concatenate([d.y[tr0], yp])
        wtr = np.concatenate([w0, np.full(len(pseudo_rows), 0.5, np.float32)])
        bst = fast_fit(Xtr, ytr, d.feats, Xv0, d.y[va0], wtr)
        p = np.full(len(d.i), np.nan, np.float32)
        p[india_rows] = bst.predict(Xe_india, num_iteration=bst.best_iteration)
        return bst, p

    bst2, p2 = retrain_with_pseudo(pos1, neg1, "round2")
    r2 = score_india(p2, "round2_us_plus_pseudoIndia_r1")
    delta_r1_r2 = r2["macro_f05"] - r1["macro_f05"]
    gate_pass = delta_r1_r2 >= GATE_DELTA
    print(f"round1->round2 delta: {delta_r1_r2:+.5f} (gate {GATE_DELTA:+.5f}) -> "
          f"{'ADOPT' if gate_pass else 'DO NOT ADOPT'} for France", flush=True)

    # ---- Round 3: re-label with round-2 model, retrain again ----
    pos2, neg2 = pseudo_label(d, india_rows, Xe_india, p2)
    g_pos2, g_neg2 = grade(d, pos2, india_rows, 1), grade(d, neg2, india_rows, 0)
    print(f"pseudo-label round 2 (from round-2 model): positives {g_pos2}, negatives {g_neg2}", flush=True)
    log("pseudo round2", pos=g_pos2, neg=g_neg2)

    bst3, p3 = retrain_with_pseudo(pos2, neg2, "round3")
    r3 = score_india(p3, "round3_us_plus_pseudoIndia_r2")
    delta_r2_r3 = r3["macro_f05"] - r2["macro_f05"]
    print(f"round2->round3 delta: {delta_r2_r3:+.5f}", flush=True)

    shutil.rmtree(work, ignore_errors=True)

    report = {
        "sample_fraction": 0.2, "fast_lr": FAST_LR, "fast_rounds_cap": FAST_ROUNDS, "fold": 0,
        "gate_delta": GATE_DELTA, "adopt_for_france": gate_pass,
        "round1_us_only": r1, "round2_us_plus_pseudo": r2, "round3_relabeled": r3,
        "delta_round1_round2": delta_r1_r2, "delta_round2_round3": delta_r2_r3,
        "pseudo_round1": {"positives": g_pos1, "negatives": g_neg1},
        "pseudo_round2": {"positives": g_pos2, "negatives": g_neg2},
        "rule": {"pos_p": POS_P, "neg_p": NEG_P, "ambiguity_name_tset": AMB_NAME, "ambiguity_addr_tset": AMB_ADDR,
                 "neg_ratio": MC["neg_ratio"]},
    }
    (path("oof_dir") / "selftrain_loco.json").write_text(json.dumps(report, indent=1, default=lambda x: None))
    log.dump(path("oof_dir") / "selftrain_loco_timing.json")

    lines = ["# Self-training LOCO probe (India as France proxy)", "",
             f"20% S1 sample, fold 0, fast params (lr={FAST_LR}, <={FAST_ROUNDS} rounds), test-density scorer.", "",
             "## Scores (India, macro F0.5)",
             "| round | setup | macro_f05 | best_t | n_s1 |",
             "|---|---|---|---|---|",
             f"| 1 | US only | {r1['macro_f05']:.5f} | {r1['best_t']:.2f} | {r1['n_s1']} |",
             f"| 2 | US + pseudo-India (round 1, w=0.5) | {r2['macro_f05']:.5f} | {r2['best_t']:.2f} | {r2['n_s1']} |",
             f"| 3 | US + pseudo-India (round 2 relabel, w=0.5) | {r3['macro_f05']:.5f} | {r3['best_t']:.2f} | {r3['n_s1']} |",
             "", f"Round 1 -> 2 delta: **{delta_r1_r2:+.5f}** (gate {GATE_DELTA:+.5f}) -> "
             f"**{'ADOPT' if gate_pass else 'DO NOT ADOPT'}** for France.",
             f"Round 2 -> 3 delta: {delta_r2_r3:+.5f}.", "",
             "## Pseudo-label quality vs true India labels",
             "| round | class | n | precision | recall | n_true_total |",
             "|---|---|---|---|---|---|",
             f"| 1 | positive (p>={POS_P}, argmax, no ambiguous rival) | {g_pos1['n']} | "
             f"{g_pos1['precision'] if g_pos1['precision'] is None else round(g_pos1['precision'], 4)} | "
             f"{g_pos1['recall'] if g_pos1['recall'] is None else round(g_pos1['recall'], 4)} | {g_pos1['n_true_total']} |",
             f"| 1 | negative (p<={NEG_P}, 3:1 sampled) | {g_neg1['n']} | "
             f"{g_neg1['precision'] if g_neg1['precision'] is None else round(g_neg1['precision'], 4)} | "
             f"{g_neg1['recall'] if g_neg1['recall'] is None else round(g_neg1['recall'], 4)} | {g_neg1['n_true_total']} |",
             f"| 2 | positive (relabel, round-2 model) | {g_pos2['n']} | "
             f"{g_pos2['precision'] if g_pos2['precision'] is None else round(g_pos2['precision'], 4)} | "
             f"{g_pos2['recall'] if g_pos2['recall'] is None else round(g_pos2['recall'], 4)} | {g_pos2['n_true_total']} |",
             f"| 2 | negative (relabel, round-2 model) | {g_neg2['n']} | "
             f"{g_neg2['precision'] if g_neg2['precision'] is None else round(g_neg2['precision'], 4)} | "
             f"{g_neg2['recall'] if g_neg2['recall'] is None else round(g_neg2['recall'], 4)} | {g_neg2['n_true_total']} |",
             "", "## Rule", f"- positive: p >= {POS_P}, argmax of its record, and no other candidate of the same S1 "
             f"has name_tset >= {AMB_NAME} AND addr_tset >= {AMB_ADDR} (ambiguity guard)",
             f"- negative: p <= {NEG_P}, sampled {MC['neg_ratio']}:1 vs positives (seeded)",
             f"- pseudo-labeled rows enter training at weight 0.5, alongside full-weight US rows", "",
             "## Notes", "- India stands in for France (both LOCO-country proxies); grading uses TRUE India labels "
             "for diagnosis only -- never fed into training.",
             "- Precision/recall above are read straight off `docs/selftrain_loco.md`'s generating run "
             "(`oof/selftrain_loco.json` has the exact floats); rerun to refresh both.",
             "- Gate: adopt self-training for France if round 1 -> round 2 gains >= +0.02 macro F0.5 on India."]
    (path("oof_dir").parent / "docs" / "selftrain_loco.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
