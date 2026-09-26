"""A/B gate for the M5-3 feature additions (src.features: s1_name_dup/rec_name_hits95/is_noaddr_ambiguous,
num_rel_*, unmatched_*, name_tset_dict). 5% S1 sample (seed 42), 1 fold (fold 0), LightGBM, test-density
(b) macro-F0.5 -- same world/score machinery as src.cv_full, restricted to one fold.

Scored population: fold-0 S1s of the 5% sample that survive the test-density drop (seed+2000). Only fold-0
rows get a p (one model), so the argmax per record sees fold-0 competitors only -- identical for every
variant, so deltas are paired; the absolute level is not directly comparable to cv_full (b).
Noise floor: baseline refit with model seeds 43, 44 (same rows); std over {42, 43, 44}.
Keep bar = max(+0.002, 2 x std), delta vs the seed-42 baseline (variants use seed 42 too).
Reports baseline vs +a..+d vs +all, then +gi+gj for every pair that both clear the bar individually.
Worlds and feature matrices are built once (all features) and column-sliced per variant.

Run from code/business_entity_resolution:
  python -m src.ab_test
"""
import shutil
import time
from itertools import combinations

import numpy as np

from .cv_full import Data, drop_mask, s1_table, score, split_fold, weighted_sample_rows, world
from .io import CFG, StepLog, path
from .train import fit

MC, SEED = CFG["matcher"], CFG["seed"]
NOISE_SEEDS = [SEED + 1, SEED + 2]
SANITY = (0.94, 0.98)  # baseline must land near the known OOF (0.965) or the scorer is wrong again
MIN_DELTA = 0.002

GROUPS = {
    "a": ["s1_name_dup", "rec_name_hits95", "is_noaddr_ambiguous"],
    "b": ["num_rel_class", "num_rel_absdiff", "num_rel_reldiff"],
    "c": ["unmatched_cnt_a", "unmatched_cnt_b", "unmatched_max_idf_a", "unmatched_max_idf_b", "unmatched_char_sim"],
    "d": ["name_tset_dict"],
}


def main() -> None:
    log = StepLog()
    s1 = s1_table(0.05)
    d = Data(s1)
    all_feats = list(d.feats)
    log("A/B meta", rows=len(d.i), s1=s1.height, pos=int(d.y.sum()))

    # training side: fold-0 world, rows sampled once, shared by every variant
    hard = d.meta["hard"].to_numpy()
    drop, train, inner = split_fold(s1, 0)
    work = path("features_dir") / "_worlds_ab"
    world(d.meta, ~drop[d.code], work / "fold0", log)
    tr, w = weighted_sample_rows(np.flatnonzero(train[d.code]), d.y, hard, d.in_v1, np.random.default_rng(SEED))
    va = np.flatnonzero(inner[d.code])
    X, Xv = d.gather(tr, work / "fold0"), d.gather(va, work / "fold0")

    # eval side: test-density world; score only fold-0 S1s that survive the drop
    dropE = drop_mask(s1.height, SEED + 2000)
    keepE = ~dropE[d.code]
    world(d.meta, keepE, work / "eval", log)
    rows0 = np.flatnonzero((d.fold == 0) & keepE)
    X0 = d.gather(rows0, work / "eval")
    shutil.rmtree(work)
    scope = (s1["fold"].to_numpy() == 0) & ~dropE
    has_cand = np.bincount(d.code[rows0], minlength=s1.height) > 0
    single = s1["ntrue"].to_numpy() == 0
    print(f"scored S1: {int(scope.sum())}, with >=1 candidate: {int((scope & has_cand).sum())}, "
          f"singleton share: {single[scope].mean():.4f}", flush=True)
    log("A/B data", train_rows=len(tr), valid_rows=len(va), eval_rows=len(rows0), scored_s1=int(scope.sum()))

    def run_variant(name: str, feats: list[str], seed: int = SEED) -> dict:
        t0 = time.monotonic()
        cols = [all_feats.index(f) for f in feats]
        bst = fit(X[:, cols], d.y[tr], feats, MC["num_boost_round"], valid=(Xv[:, cols], d.y[va]), weight=w, seed=seed)
        p = np.full(len(d.i), np.nan, np.float32)
        p[rows0] = bst.predict(X0[:, cols], num_iteration=bst.best_iteration)
        r = score(s1, d, p, scope)
        out = {"variant": name, "n_feats": len(feats), "macro_f05": r["macro_f05"], "best_t": r["best_t"],
               "best_iter": bst.best_iteration, "secs": time.monotonic() - t0}
        log(f"variant {name}", **{k: v for k, v in out.items() if k != "variant"})
        return out

    base_feats = [f for f in all_feats if f not in sum(GROUPS.values(), [])]
    group_present = {}
    for g, cols in GROUPS.items():
        group_present[g] = [c for c in cols if c in all_feats]
        missing = set(cols) - set(group_present[g])
        if missing:
            print(f"WARNING group {g}: {missing} not in feature_cols (auto-discovery gap) -- skipped in +{g}")

    results = [run_variant("baseline", base_feats)]
    base = results[0]["macro_f05"]
    if not SANITY[0] <= base <= SANITY[1]:
        raise SystemExit(f"STOP: baseline macro_f05 {base:.5f} outside {SANITY} -- scorer/population still wrong")

    noise = [run_variant(f"baseline_s{s}", base_feats, seed=s) for s in NOISE_SEEDS]
    std = float(np.std([base] + [r["macro_f05"] for r in noise], ddof=1))
    bar = max(MIN_DELTA, 2 * std)
    print(f"noise floor: std {std:.5f} over seeds {[SEED, *NOISE_SEEDS]} -> keep bar {bar:+.5f}", flush=True)

    for g in GROUPS:
        results.append(run_variant(f"+{g}", base_feats + group_present[g]))
    results.append(run_variant("+all", all_feats))
    cleared = [g for g in GROUPS if next(r for r in results if r["variant"] == f"+{g}")["macro_f05"] - base >= bar]
    for g1, g2 in combinations(cleared, 2):
        results.append(run_variant(f"+{g1}+{g2}", base_feats + group_present[g1] + group_present[g2]))

    print(f"\n{'variant':14s} {'n_feats':>7s} {'macro_f05':>10s} {'delta':>9s} {'best_t':>6s} {'secs':>7s} "
          f"{f'keep(>={bar:+.4f})':>16s}")
    for r in results[:1] + noise + results[1:]:
        delta = r["macro_f05"] - base
        keep = "n/a" if r["variant"].startswith("baseline") else ("YES" if delta >= bar else "no")
        print(f"{r['variant']:14s} {r['n_feats']:7d} {r['macro_f05']:10.5f} {delta:+9.5f} {r['best_t']:6.2f} "
              f"{r['secs']:7.1f} {keep:>16s}")
    if len(cleared) < 2:
        print(f"(no pairwise unions run: {len(cleared)} group(s) cleared {bar:+.4f} individually)")

    log.dump(path("oof_dir") / "ab_test_timing.json")


if __name__ == "__main__":
    main()
