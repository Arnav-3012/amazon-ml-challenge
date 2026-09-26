"""M5-3 stage 2: a LightGBM stacked on stage-1 OOF p, seeing each pair's neighbourhood.

Rows/labels: the M5-2 test-density world (b) = rows of oof/oof_full.parquet with p_td not null (19% of S1s dropped,
seed+2000), so stage 2 trains at test density. Stage-1 p comes only from OOF (never in-fold); on test it is the
5-fold mean (oof/test_p.parquet, from predict --folds), which is smoother -> p-shift is reported, and a rank-only
variant (no absolute-p features, P_ABS) is trained next to the full one as the robustness check.
Features per (record r, S1 s), all recomputed from the rows present (so train (b) and test share one density):
  record: p rank, p - max p over r's other S1s (0 = none), top1 - top2 of r, #S1s of r
  S1:     p rank, sum p, #p > 0.5, #candidates, p / max p
  coref:  # other records of s with p > coref_p (top max_anchors by p) whose token_set to r >= coref_sim, for
          core name and for address (both non-empty)
  carry:  the n_carry strongest stage-1 features by summed fold gain. Record-side REC_COLS are excluded: their
          stored values are the no-drop world, not (b), so they would leak train density into stage 2.
Folds = the stage-1 S1 folds (column fold); early stopping on inner_valid_frac of the training S1s; training rows
sampled with cv_full.weighted_sample_rows (hardness = stage-1 p). Decision = decide.py rule via cv_full.score.
Keep rule: stage-2 OOF (b) >= stage-1 OOF (b) + keep_gain, else --predict refuses.
-> models/stage2_{variant}_fold_{k}.txt, oof/oof_stage2.parquet, oof/stage2.json;
   --predict: oof/test_p2.parquet + output/matching_results.tsv; the stage-1 matching_results/candidate_pairs are first
   copied once to output/*_stage1.tsv (an existing backup is kept).

Run from code/business_entity_resolution/ (after cv_full and predict --folds):
  python -m src.stage2 --smoke     # on the cv_full --smoke outputs (2% of S1s, 50 rounds): crash test
  python -m src.stage2             # both variants, OOF (b) vs stage 1
  python -m src.stage2 --predict [--variant rank_only]
"""
import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .block import norm_path
from .cv_full import REC_COLS, Log, drop_mask, s1_table, score, truth_keys, weighted_sample_rows
from .decide import with_top
from .features import _cp, id_key
from .io import CFG, path, write_candidates
from .train import fit, parts

S2, MC, FC, SEED = CFG["stage2"], CFG["matcher"], CFG["cv_full"], CFG["seed"]
CTX = ["p", "rec_rank", "rec_dmax", "rec_gap", "rec_n", "s1_rank", "s1_psum", "s1_n05", "s1_n", "s1_prel",
       "coref_name", "coref_addr"]
P_ABS = ["p", "rec_dmax", "rec_gap", "s1_psum", "s1_n05"]  # absolute-p features, dropped in the rank_only variant


def carry_cols(tag: str) -> list[str]:
    bs = [lgb.Booster(model_file=str(path("models_dir") / f"fold_{k}{tag}.txt")) for k in range(MC["n_folds"])]
    gain = np.sum([b.feature_importance("gain") for b in bs], axis=0)
    names = bs[0].feature_name()
    return [names[j] for j in np.argsort(-gain, kind="stable") if names[j] not in REC_COLS][: S2["n_carry"]]


def texts(split: str) -> pl.DataFrame:
    return (pl.concat([pl.read_parquet(norm_path(split, n), columns=["entity_id", "core_name", "addr_tokens"])
                       for n in (2, 3)])
            .select(r=id_key("entity_id"), core="core_name", addr=pl.col("addr_tokens").list.join(" ")))


def context(df: pl.DataFrame) -> pl.DataFrame:
    """df(s, r, p) -> + record- and S1-side context columns (row order kept)."""
    p = pl.col("p")
    rec = (df.group_by("r").agg(_t1=p.max(), _t2=p.top_k(2).min(), rec_n=pl.len())
           .with_columns(_t2=pl.when(pl.col("rec_n") > 1).then("_t2").otherwise(0.0)))
    return (df.join(rec, on="r", how="left", maintain_order="left")
            .with_columns(rec_rank=p.rank("min", descending=True).over("r"),
                          rec_dmax=p - pl.when(p == pl.col("_t1")).then("_t2").otherwise("_t1"),
                          rec_gap=pl.col("_t1") - pl.col("_t2"),
                          s1_rank=p.rank("min", descending=True).over("s"), s1_psum=p.sum().over("s"),
                          s1_n05=(p > 0.5).sum().over("s"), s1_n=pl.len().over("s"), s1_prel=p / p.max().over("s"))
            .drop("_t1", "_t2"))


def coref(df: pl.DataFrame, txt: pl.DataFrame, log: Log) -> pl.DataFrame:
    """df(s, r, p) -> + coref_name, coref_addr (see module doc). Anchor pairs are scored in pair_chunk slices."""
    anc = (df.filter(pl.col("p") > S2["coref_p"])
           .filter(pl.col("p").rank("ordinal", descending=True).over("s") <= S2["max_anchors"])
           .select("s", ra="r"))
    pairs = (df.select("s", "r").with_row_index("_j").with_columns(pl.col("_j").cast(pl.Int64))
             .join(anc, on="s").filter(pl.col("r") != pl.col("ra")))
    hit, out = S2["coref_sim"], [pl.DataFrame(schema={"_j": pl.Int64, "coref_name": pl.Int32, "coref_addr": pl.Int32})]
    for c in pairs.iter_slices(S2["pair_chunk"]):
        a = c.select("r").join(txt, on="r", how="left", maintain_order="left")
        b = c.select(r="ra").join(txt, on="r", how="left", maintain_order="left")
        assert a["core"].null_count() == 0 and b["core"].null_count() == 0, "record missing from norm cache"
        both = ((a["addr"] != "") & (b["addr"] != "")).to_numpy()
        out.append(pl.DataFrame({
            "_j": c["_j"],
            "coref_name": (_cp(a["core"].to_list(), b["core"].to_list(), fuzz.token_set_ratio) >= hit).astype(np.int32),
            "coref_addr": (both & (_cp(a["addr"].to_list(), b["addr"].to_list(), fuzz.token_set_ratio) >= hit))
            .astype(np.int32)}))
    log("coref", anchors=anc.height, pairs=pairs.height)
    agg = pl.concat(out).group_by("_j").agg(pl.col("coref_name", "coref_addr").sum())
    return (df.with_row_index("_j").with_columns(pl.col("_j").cast(pl.Int64))
            .join(agg, on="_j", how="left", maintain_order="left").drop("_j")
            .with_columns(pl.col("coref_name", "coref_addr").fill_null(0)))


def build(df: pl.DataFrame, carry: pl.DataFrame, split: str, log: Log) -> pl.DataFrame:
    df = coref(context(df), texts(split), log)
    log(f"features {split}", rows=df.height)
    return pl.concat([df, carry], how="horizontal")


def matrix(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def cv(X: np.ndarray, y: np.ndarray, p1: np.ndarray, fold: np.ndarray, code: np.ndarray, feats: list[str],
       name: str, rounds: int, log: Log) -> tuple[np.ndarray, list[int]]:
    """OOF p over the stage-1 folds; fold models -> models/stage2_{name}_fold_{k}.txt."""
    oof, iters, ones = np.full(len(y), np.nan, np.float32), [], np.ones(len(y), bool)
    for k in range(MC["n_folds"]):
        idx = np.flatnonzero(fold != k)
        trs = np.unique(code[idx])
        inner = np.zeros(code.max() + 1, bool)
        rng = np.random.default_rng(SEED + 5000 + k)
        inner[rng.choice(trs, round(FC["inner_valid_frac"] * len(trs)), replace=False)] = True
        va = idx[inner[code[idx]]]
        tr, w = weighted_sample_rows(idx[~inner[code[idx]]], y, p1, ones, np.random.default_rng(SEED + 6000 + k))
        bst = fit(X[tr], y[tr], feats, rounds, valid=(X[va], y[va]), weight=w)
        bst.save_model(path("models_dir") / f"stage2_{name}_fold_{k}.txt")
        te = np.flatnonzero(fold == k)
        oof[te] = bst.predict(X[te])
        iters.append(bst.best_iteration)
        log(f"{name} fold {k}", train_rows=len(tr), valid_rows=len(va), best_iter=bst.best_iteration)
    return oof, iters


def shift(p: np.ndarray, r: np.ndarray) -> dict:
    """p-histogram summary; the record-max mean is the one that moves the argmax/threshold decision."""
    rmax = pl.DataFrame({"r": r, "p": p}).group_by("r").agg(pl.col("p").max())["p"].to_numpy()
    return {**{f"q{q}": float(np.quantile(p, q)) for q in (0.5, 0.9, 0.99, 0.999)},
            "frac_gt05": float((p > 0.5).mean()), "frac_gt08": float((p > 0.8).mean()),
            "rec_max_mean": float(rmax.mean())}


def train(tag: str, log: Log) -> None:
    O = path("oof_dir")
    s1 = s1_table(0.02 if tag else 1.0)
    oof = (pl.read_parquet(O / f"oof_full{tag}.parquet").filter(pl.col("p_td").is_not_null())
           .join(s1.select(s="s1k", code="code"), left_on="s1k", right_on="s", how="left", maintain_order="left"))
    assert oof["code"].null_count() == 0, "OOF S1 missing from s1_table"
    car = carry_cols(tag)
    files = sorted(str(p) for p in Path(parts("train")).parent.glob("part-*.parquet"))
    cf = (pl.scan_parquet(files).with_row_index("_i").select("_i", *car)
          .filter(pl.col("_i").is_in(oof["_i"].implode())).collect())
    assert np.array_equal(cf["_i"].to_numpy(), oof["_i"].to_numpy()), "carry rows misaligned with OOF"
    df = build(oof.select(s="s1k", r="reck", p="p_td"), cf.drop("_i"), "train", log)
    y, p1 = oof["label"].to_numpy().astype(np.int8), oof["p_td"].to_numpy()
    fold, code, r = oof["fold"].to_numpy(), oof["code"].to_numpy(), oof["reck"].to_numpy()
    d = SimpleNamespace(code=code, y=y, meta=pl.DataFrame({"reck": r}))
    scope = ~drop_mask(s1.height, SEED + 2000)
    base = score(s1, d, p1, scope, exact=truth_keys(s1))
    ref = json.loads((O / f"cv_full{tag}.json").read_text())["b_test_density"]["macro_f05"]
    assert abs(base["macro_f05"] - ref) < 1e-9, f"stage-1 (b) rescored {base['macro_f05']} != cv_full.json {ref}"
    feats = {"full": CTX + car, "rank_only": [f for f in CTX + car if f not in P_ABS]}
    rounds = 50 if tag else MC["num_boost_round"]
    res, cols = {"stage1_b": {k: v for k, v in base.items() if k != "curve"}, "carry": car, "feats": feats}, {}
    for name, fs in feats.items():
        cols[name], iters = cv(matrix(df, fs), y, p1, fold, code, fs, f"{name}{tag}", rounds, log)
        s = score(s1, d, cols[name], scope)
        gain = s["macro_f05"] - base["macro_f05"]
        res[name] = {**s, "best_iters": iters, "gain": gain, "keep": bool(gain >= S2["keep_gain"])}
        print(name, {k: v for k, v in res[name].items() if k != "curve"}, flush=True)
        log(f"scored {name}", f05=round(s["macro_f05"], 5), gain=round(gain, 5))
    print("stage1_b", res["stage1_b"], flush=True)
    oof.select("_i", "p_td").with_columns(**{f"p2_{k}": pl.Series(v) for k, v in cols.items()}).write_parquet(
        O / f"oof_stage2{tag}.parquet")
    (O / f"stage2{tag}.json").write_text(json.dumps(res, indent=1))


def backup_stage1() -> None:
    """Keep the stage-1 submission: copy each output file to *_stage1.tsv once; an existing backup is never replaced."""
    for key in ("matching_results", "candidate_pairs"):
        src = path(key)
        dst = src.with_name(f"{src.stem}_stage1{src.suffix}")
        if not dst.exists():
            assert src.exists(), f"{src} missing: run predict --folds first so there is a stage-1 file to back up"
            shutil.copy2(src, dst)


def predict(variant: str, log: Log) -> None:
    O = path("oof_dir")
    res = json.loads((O / "stage2.json").read_text())
    v, feats = res[variant], res["feats"][variant]
    assert v["keep"], f"{variant}: gain {v['gain']:.5f} < keep_gain {S2['keep_gain']} -> ship stage 1 (predict --folds)"
    tp = pl.read_parquet(O / "test_p.parquet")
    cf = pl.concat([pl.read_parquet(f, columns=["s1_id", "rec_id", *res["carry"]])
                    for f in sorted((path("features_dir") / "test").glob("part-*.parquet"))])
    assert cf["s1_id"].equals(tp["s1_id"]) and cf["rec_id"].equals(tp["rec_id"]), "test_p rows != test features rows"
    df = build(tp.select(s=id_key("s1_id"), r=id_key("rec_id"), p="p"), cf.drop("s1_id", "rec_id"), "test", log)
    X = matrix(df, feats)
    bsts = [lgb.Booster(model_file=str(path("models_dir") / f"stage2_{variant}_fold_{k}.txt"))
            for k in range(MC["n_folds"])]
    p2 = np.mean([b.predict(X) for b in bsts], axis=0).astype(np.float32)
    scored = tp.select("s1_id", "rec_id").with_columns(p=pl.Series(p2))
    scored.write_parquet(O / "test_p2.parquet")

    o2 = pl.read_parquet(O / "oof_stage2.parquet", columns=["_i", "p_td", f"p2_{variant}"]).join(
        pl.read_parquet(O / "oof_full.parquet", columns=["_i", "reck"]), on="_i", how="left")
    r_tr, r_te = o2["reck"].to_numpy(), df["r"].to_numpy()
    res["p_shift"] = {"stage1": {"oof_b": shift(o2["p_td"].to_numpy(), r_tr), "test": shift(tp["p"].to_numpy(), r_te)},
                      variant: {"oof_b": shift(o2[f"p2_{variant}"].to_numpy(), r_tr), "test": shift(p2, r_te)}}
    print(json.dumps(res["p_shift"], indent=1), flush=True)
    (O / "stage2.json").write_text(json.dumps(res, indent=1))

    t = v["best_t"]
    cands = pl.scan_parquet(path("interim_dir") / "candidates_test.parquet").select("s1_id", "rec_id")
    matches = with_top(scored).filter(pl.col("top") & (pl.col("p") >= t)).select("s1_id", "rec_id")
    outside = matches.lazy().join(cands, on=["s1_id", "rec_id"], how="anti").select(pl.len()).collect().item()
    assert outside == 0, f"{outside} matches not in candidate set"
    assert matches["rec_id"].is_unique().all(), "a record matched to >1 S1"
    s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"])["entity_id"]
    backup_stage1()
    write_candidates(path("matching_results"), matches.lazy(), s1_ids, list_col="matched_entity_ids")
    n_with = matches["s1_id"].n_unique()
    log("decide + write", variant=variant, t=t, matches=matches.height, s1_with_match=n_with,
        empty_pct=round(100 * (1 - n_with / s1_ids.len()), 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--variant", default="full", choices=["full", "rank_only"])
    a = ap.parse_args()
    log = Log()
    if a.predict:
        assert not a.smoke, "--predict uses the full-run models"
        predict(a.variant, log)
    else:
        train("_smoke" if a.smoke else "", log)
    log.dump(path("oof_dir") / f"stage2_timing_{'predict' if a.predict else 'train'}{'_smoke' if a.smoke else ''}.json")


if __name__ == "__main__":
    main()
