"""krish-v2 decision layer on stage-1 OOF (b): calibrated expected-F0.5 set choice vs the tuned global threshold.

q = isotonic(p_td), cross-fitted by S1 fold (fold k's rows are calibrated on the other folds' rows).
Candidate members of S1 s = rows whose record's argmax is s (decide.py rule), sorted by q. For each prefix size k
of those members: E[F0.5](k) ~ 1.25 * sum_{i<=k} q_i / (0.25 * E[T] + k) with E[T] = sum of q over ALL rows of s,
and E[F0.5](0) = P(no true match among candidates) = prod(1 - q) over all rows of s.
Variants (macro F0.5 over the (b) scope, blocking misses included):
  thr        argmax + best global t on p (= cv_full.json b, asserted)
  ef_empty   thr's sets, but empty when E[F](0) > E[F](thr's set)          (strategy §5 step 9: empty option only)
  ef_prefix  the argmax_k E[F](k) set, k = 0..#members
  ef_prefix_s  ef_prefix with E[T] scaled by a (grid on the OOF, reported with its fold spread)
-> oof_dir/decide_ef{tag}.json. Apply to test with --apply (writes matching_results via path()).

Run from code/business_entity_resolution/:  python -m src.decide_ef [--smoke] [--apply]
"""
import argparse
import json

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from .block import norm_path
from .cv_full import SEED, drop_mask, s1_table, score
from .decide import f05_vec
from .features import id_key
from .io import CFG, StepLog, path, write_candidates
from .train import parts

MC = CFG["matcher"]


def calibrators(p: np.ndarray, y: np.ndarray, fold: np.ndarray) -> list[IsotonicRegression]:
    out = []
    for k in range(MC["n_folds"]):
        m = fold != k
        out.append(IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p[m], y[m]))
    return out


def choose(df: pl.DataFrame, t: float, et_scale: float = 1.0) -> dict[str, pl.DataFrame]:
    """df: code (S1 index), r (record key), p, q. Returns the chosen (code, r) rows per variant."""
    top = (df.sort(["r", "p", "code"], descending=[False, True, False]).filter(pl.col("r").is_first_distinct()))
    s1 = df.group_by("code").agg(ET=pl.col("q").sum() * et_scale,  # P0 = prod(1 - q) over all of s's rows
                                 P0=(1 - pl.col("q").clip(0, 1 - 1e-9)).log().sum().exp())
    mem = (top.sort(["code", "q", "p"], descending=[False, True, True]).join(s1, on="code", how="left")
           .with_columns(k=pl.col("q").cum_count().over("code"), S=pl.col("q").cum_sum().over("code"))
           .with_columns(EF=1.25 * pl.col("S") / (0.25 * pl.col("ET") + pl.col("k"))))
    thr = top.filter(pl.col("p") >= t)
    # E[F] of thr's set: members with p >= t are a prefix in q order (isotonic is monotone; ties broken the same way)
    ef_thr = (mem.filter(pl.col("p") >= t).group_by("code").agg(EFt=pl.col("EF").last(), P0=pl.col("P0").first()))
    empty_codes = ef_thr.filter(pl.col("P0") > pl.col("EFt"))["code"]
    best = mem.group_by("code").agg(kbest=pl.col("k").sort_by("EF").last(), EFb=pl.col("EF").max(), P0=pl.col("P0").first())
    best = best.with_columns(kbest=pl.when(pl.col("P0") >= pl.col("EFb")).then(0).otherwise(pl.col("kbest")))
    pre = mem.join(best.select("code", "kbest"), on="code").filter(pl.col("k") <= pl.col("kbest"))
    return {"thr": thr.select("code", "r"),
            "ef_empty": thr.filter(~pl.col("code").is_in(empty_codes.implode())).select("code", "r"),
            "ef_prefix": pre.select("code", "r")}


def thr2(df: pl.DataFrame, t: float, t2: float) -> pl.DataFrame:
    """argmax per record, then t2 for S1s that share their exact address with another S1 (adup), t otherwise."""
    top = df.sort(["r", "p", "code"], descending=[False, True, False]).filter(pl.col("r").is_first_distinct())
    return top.filter(pl.col("p") >= pl.when(pl.col("adup")).then(t2).otherwise(t)).select("code", "r")


def adup_col(split: str, rows: np.ndarray | None) -> np.ndarray:
    """s1_addr_dup > 0 per feature row of `split` (rows = feature-row indices to keep, None = all)."""
    v = (pl.scan_parquet(parts(split)).select(pl.col("s1_addr_dup") > 0).collect().to_series().to_numpy())
    return v if rows is None else v[rows]


def macro(sel: pl.DataFrame, lab: pl.DataFrame, s1: pl.DataFrame, scope: np.ndarray) -> dict:
    n = s1.height
    j = sel.join(lab, on=["code", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    tp = np.bincount(j["code"].to_numpy(), weights=j["y"].to_numpy(), minlength=n)
    npred = np.bincount(j["code"].to_numpy(), minlength=n)
    f = f05_vec(tp, npred, s1["ntrue"].to_numpy())
    ctry, single = s1["country"].to_numpy(), s1["ntrue"].to_numpy() == 0
    return {"macro_f05": float(f[scope].mean()), "f05_singleton": float(f[scope & single].mean()),
            "f05_nonsingleton": float(f[scope & ~single].mean()),
            **{f"f05_{c}": float(f[scope & (ctry == c)].mean()) for c in sorted(set(ctry[scope]))},
            "pred_per_s1": float(npred[scope].mean()), "empty_pct": float(100 * (npred[scope] == 0).mean()),
            "per_fold": [float(f[scope & (s1["fold"].to_numpy() == k)].mean()) for k in range(MC["n_folds"])]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--apply", choices=["ef_empty", "ef_prefix", "thr2_adup"], default=None)
    a = ap.parse_args()
    tag, log, O = "_smoke" if a.smoke else "", StepLog(), path("oof_dir")
    cv = json.loads((O / f"cv_full{tag}.json").read_text())
    s1 = s1_table(0.02 if a.smoke else 1.0)
    oof = (pl.read_parquet(O / f"oof_full{tag}.parquet", columns=["_i", "s1k", "reck", "label", "fold", "p_td"])
           .filter(pl.col("p_td").is_not_null())
           .join(s1.select(s1k="s1k", code="code"), on="s1k", how="left", maintain_order="left"))
    assert oof["code"].null_count() == 0
    p, y, fold = oof["p_td"].to_numpy(), oof["label"].to_numpy(), oof["fold"].to_numpy()
    cal = calibrators(p, y, fold)
    q = np.empty_like(p, dtype=np.float64)
    for k in range(MC["n_folds"]):
        m = fold == k
        q[m] = cal[k].predict(p[m])
    df = pl.DataFrame({"code": oof["code"], "r": oof["reck"], "p": p, "q": q,
                       "adup": adup_col("train" if not a.smoke else "train_smoke", oof["_i"].to_numpy())
                       if not a.smoke else np.zeros(len(p), bool)})
    lab = df.select("code", "r").with_columns(y=pl.Series(y.astype(np.float64)))
    log("calibrated", rows=df.height)
    scope = ~drop_mask(s1.height, SEED + 2000)
    t = cv["b_test_density"]["best_t"]
    res = {"t": t, "cv_full_b": cv["b_test_density"]["macro_f05"]}
    for sc in (0.8, 0.9, 1.0, 1.1, 1.25):
        sel = choose(df, t, sc)
        for name, s in sel.items():
            if name == "thr" and sc != 1.0:
                continue
            key = name if sc == 1.0 else f"{name}_ET{sc}"
            res[key] = macro(s, lab, s1, scope)
            log(key, f05=round(res[key]["macro_f05"], 5))
    best2 = None
    for t2 in np.round(np.arange(max(t - 0.25, 0.3), 0.99, 0.02), 2):  # exp2: separate t for address-shared S1s
        r2 = macro(thr2(df, t, float(t2)), lab, s1, scope)
        if best2 is None or r2["macro_f05"] > best2[1]["macro_f05"]:
            best2 = (float(t2), r2)
    res["thr2_adup"] = {**best2[1], "t2": best2[0], "adup_rows_pct": float(df["adup"].mean() * 100)}
    log("thr2_adup", t2=best2[0], f05=round(best2[1]["macro_f05"], 5))
    assert abs(res["thr"]["macro_f05"] - res["cv_full_b"]) < 1e-6, (res["thr"]["macro_f05"], res["cv_full_b"])
    for k, v in res.items():
        if isinstance(v, dict):
            v["gain_vs_thr"] = v["macro_f05"] - res["thr"]["macro_f05"]
    (O / f"decide_ef{tag}.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:22s} F0.5 {v['macro_f05']:.5f} gain {v['gain_vs_thr']:+.5f} single {v['f05_singleton']:.4f} "
                  f"empty% {v['empty_pct']:.2f} pred/S1 {v['pred_per_s1']:.3f}", flush=True)
    if a.apply:
        tp = pl.read_parquet(O / "test_p.parquet")
        ids = tp.select(pl.col("s1_id").unique()).with_row_index("code")
        tdf = tp.join(ids, on="s1_id", how="left", maintain_order="left").select("code", "s1_id", "rec_id", r=id_key("rec_id"), p="p")
        pa = tdf["p"].to_numpy()
        tq = np.mean([c.predict(pa) for c in cal], axis=0)  # test p is a 5-fold mean: average the fold calibrators
        if a.apply == "thr2_adup":
            tdf = tdf.with_columns(adup=pl.Series(adup_col("test", None)))  # test_p rows = test feature rows
            sel = thr2(tdf, t, res["thr2_adup"]["t2"])
        else:
            sel = choose(tdf.select("code", "r", "p").with_columns(q=pl.Series(tq)), t)[a.apply]
        m = sel.join(tdf.select("code", "r", "s1_id", "rec_id"), on=["code", "r"]).select("s1_id", "rec_id")
        s1_ids = pl.read_parquet(norm_path("test", 1), columns=["entity_id"])["entity_id"]
        write_candidates(path("matching_results"), m.lazy(), s1_ids, list_col="matched_entity_ids")
        log("apply", variant=a.apply, matches=m.height, s1_with=m["s1_id"].n_unique())
    log.dump(O / f"decide_ef_timing{tag}.json")


if __name__ == "__main__":
    main()
