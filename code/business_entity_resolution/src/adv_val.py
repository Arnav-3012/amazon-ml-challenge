"""Adversarial validation on test candidate pairs: can the matcher's features tell France from US+India?

Diagnostic only (docs/adv_val.md). Country is the TARGET here, never a feature; nothing feeds the pipeline.
Two-pass read keeps peak RSS low: pass 1 collects only (_i row index, is_france) for every test candidate,
samples n rows per side (seeded), pass 2 collects the feature matrix of the sampled rows only.
GroupKFold by s1_id: candidates of one S1 share relative features, so a row split would leak and inflate AUC.

  python -m src.adv_val            # -> oof/adv_val.json + markdown tables on stdout
"""
import argparse
import json
import resource
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .block import norm_path
from .io import path
from .train import SEED, feature_cols, fit, parts

ROUNDS, TOP, DROP = 300, 20, 5


def sample_index(n: int, france: str, rest: list[str]) -> pl.DataFrame:
    """(_i, is_france) of n seeded rows per side, in file order. s1_id is dropped here (60M strings ~1GB)."""
    country = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    seen = set(country["country"].unique())
    missing = {france, *rest} - seen
    assert not missing, f"countries {missing} not in test S1; available: {sorted(seen)}"
    idx = (pl.scan_parquet(parts("test")).select("s1_id").with_row_index("_i")
           .join(country.lazy(), on="s1_id", how="inner")
           .filter(pl.col("country").is_in([france, *rest]))
           .select("_i", is_france=(pl.col("country") == france).cast(pl.Int8))
           .collect(engine="streaming"))
    rng = np.random.default_rng(SEED)
    keep = []
    for v in (1, 0):
        side = np.flatnonzero(idx["is_france"].to_numpy() == v)
        print(f"is_france={v}: {len(side):,} candidate rows, sampling {min(n, len(side)):,}")
        keep.append(rng.choice(side, min(n, len(side)), replace=False))
    return idx[np.concatenate(keep)].sort("_i")  # streaming join does not keep file order


def load_x(idx: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (nulls -> NaN) and s1_id of the sampled rows, file order."""
    df = (pl.scan_parquet(parts("test")).with_row_index("_i").filter(pl.col("_i").is_in(idx["_i"].implode()))
          .select("_i", "s1_id", pl.col(feats).cast(pl.Float32)).collect(engine="streaming"))
    assert df["_i"].equals(idx["_i"]), "row index drifted between passes"
    return df.select(feats).to_numpy(), df["s1_id"].to_numpy()


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, feats: list[str]) -> tuple[float, list[float], np.ndarray]:
    """GroupKFold(5) OOF AUC, per-fold AUC, summed gain per feature."""
    oof, gain, folds = np.zeros(len(y)), np.zeros(len(feats)), []
    for tr, va in GroupKFold(5).split(X, y, groups):
        bst = fit(X[tr], y[tr], feats, ROUNDS)
        oof[va] = bst.predict(X[va])
        folds.append(float(roc_auc_score(y[va], oof[va])))
        gain += bst.feature_importance("gain")
    return float(roc_auc_score(y, oof)), folds, gain


def spread(x: np.ndarray) -> dict:
    q1, med, q3 = np.nanquantile(x, [0.25, 0.5, 0.75]) if np.isfinite(x).any() else (np.nan,) * 3
    return {"median": float(med), "iqr": float(q3 - q1), "null_frac": float(np.isnan(x).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000, help="rows per side")
    ap.add_argument("--france", default="France")
    ap.add_argument("--rest", default="US,India", help="comma-separated")
    a = ap.parse_args()

    feats = feature_cols("test")
    assert feats == feature_cols("train"), "test feature columns differ from the matcher's"
    idx = sample_index(a.n, a.france, a.rest.split(","))
    (X, groups), y = load_x(idx, feats), idx["is_france"].to_numpy()
    print(f"X {X.shape} ({X.nbytes / 2**20:.0f} MB), {len(feats)} features, {len(np.unique(groups)):,} S1")

    auc, folds, gain = cv_auc(X, y, groups, feats)
    order = np.argsort(-gain)
    top = [{"feature": feats[j], "gain_share": float(gain[j] / gain.sum()),
            "france": spread(X[y == 1, j]), "rest": spread(X[y == 0, j])} for j in order[:TOP]]

    keep = [j for j in range(len(feats)) if j not in set(order[:DROP])]
    auc_drop, folds_drop, _ = cv_auc(X[:, keep], y, groups, [feats[j] for j in keep])

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)
    out = {"n_per_side": a.n, "rows": len(y), "france_rows": int(y.sum()), "rounds": ROUNDS,
           "auc": auc, "fold_auc": folds, "top": top,
           "dropped": [feats[j] for j in order[:DROP]], "auc_drop": auc_drop, "fold_auc_drop": folds_drop,
           "peak_rss_gb": peak_gb}
    (path("oof_dir") / "adv_val.json").write_text(json.dumps(out, indent=2))

    print(f"\nAUC {auc:.4f} (folds {', '.join(f'{f:.4f}' for f in folds)})")
    print(f"AUC without top {DROP} {auc_drop:.4f} (dropped: {', '.join(out['dropped'])})\n")
    print("| # | feature | gain % | FR median | FR IQR | US+IN median | US+IN IQR | FR null | US+IN null |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, t in enumerate(top, 1):
        f, r = t["france"], t["rest"]
        print(f"| {k} | {t['feature']} | {100 * t['gain_share']:.1f} | {f['median']:.4g} | {f['iqr']:.4g} | "
              f"{r['median']:.4g} | {r['iqr']:.4g} | {f['null_frac']:.2f} | {r['null_frac']:.2f} |")
    print(f"\npeak RSS {peak_gb:.2f} GB")
    assert peak_gb <= 6, f"peak RSS {peak_gb:.2f} GB > 6 GB budget"


if __name__ == "__main__":
    main()
