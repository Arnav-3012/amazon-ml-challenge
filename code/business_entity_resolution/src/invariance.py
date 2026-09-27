"""M5 country-invariance probe: do post-hoc, ratio-built features cross US<->India better than the raw ones,
without a feature rebuild? Diagnostic only (docs/invariance.md); no pipeline change.

The "invariant" feature set = feature_cols(split) with 4 derived columns added and DROP_RAW dropped, all
built post-hoc from columns already in artifacts/features/{split} (features.py itself is untouched):
  nonlatin_s1 / nonlatin_rec : replaces s1_nonascii/rec_nonascii (accent-only Latin no longer flagged; needs
    the ORIGINAL business_name/business_address, since normalise's anyascii already transliterated the
    stored core_name -- read fresh from norm_{split}_s{1,2,3} and joined in by entity_id)
  {c}_idfj_norm  for c in name/nbg/nskel/addr/askel : {c}_idfj / ln(N_S1_country + 1) (idf_jaccard is already
    a ratio in [0,1]; this only rescales its implicit country-vocab-size dependence -- there is no absolute
    idf column stored to normalise, see features.py:111-120)
  {ch}_score_normrec / {ch}_score_norms1  for ch in A/B/C/X : {ch}_score / max({ch}_score) over the row's
    rec_id / s1_id group (0 when every candidate is null or 0, so a channel miss stays 0, not NaN)
Everything else (ratios, codes, ranks, *_drec/_ds1/_gaprec, which are already relative) passes through
unchanged; DROP_RAW (*_nonascii, the 4 raw *_score columns) is dropped only in the "invariant" variant.

Run from code/business_entity_resolution/:
  python -m src.invariance --smoke    # 5% S1 sample, both LOCO directions + in-domain, fast params
  python -m src.invariance            # 20% S1 sample (matcher.subset_fraction), full report
  python -m src.invariance --adv-val  # + adversarial validation (France vs US+India) on invariant test features
"""
import argparse
import shutil

import numpy as np
import polars as pl

from .adv_val import TOP, cv_auc, sample_index
from .block import CHANNELS, norm_path
from .cv_full import Data, s1_table, score, split_fold, weighted_sample_rows, world
from .decide import md
from .features import id_key
from .io import CFG, ROOT, StepLog, path, peak_rss_mb
from .train import SCORE_COLS, feature_cols, fit, parts

SEED = CFG["seed"]
IDFJ_COLS = [f"{k}_idfj" for k in ("name", "nbg", "nskel", "addr", "askel")]
DROP_RAW = ["s1_nonascii", "rec_nonascii", *SCORE_COLS]  # replaced by nonlatin_* and *_score_norm{rec,s1}
KEEP_BAR = 0.001  # in-domain invariant must not drop more than this vs baseline (task's "must not drop > 0.001")
REPORT = ROOT / "docs" / "invariance.md"

# \p{L} minus \p{Latin} (regex crate's \p{scx=Latin} isn't exposed by polars, so subtract the Latin block
# from "any letter" instead): a match means some alphabetic char is not Latin script. Accents (e, u-umlaut,
# i-dieresis) are already folded into the Latin block by Unicode, so accented Latin never matches.
NONLATIN_RE = r"[\p{L}&&[^\p{Latin}]]"


def script_flags(split: str) -> pl.DataFrame:
    """entity_id -> nonlatin (any alphabetic char outside the Latin Unicode script), on RAW business_name/address.
    Vectorised (polars regex, Rust engine): the old per-character unicodedata.name() loop took 138s over
    12.5M S1+S2+S3 rows; this is a single regex pass."""
    ids = pl.concat([pl.scan_parquet(norm_path(split, n)).select("entity_id", "business_name", "business_address")
                     for n in (1, 2, 3)])
    return ids.select("entity_id", nonlatin=(pl.col("business_name") + " " + pl.col("business_address"))
                      .str.contains(NONLATIN_RE)).collect(engine="streaming")


def derive(df: pl.DataFrame) -> pl.DataFrame:
    """+ {c}_idfj_norm, {ch}_score_norm{rec,s1}. df already carries nonlatin_s1/rec, n_country_s1, s1_id, rec_id,
    the raw *_idfj and *_score columns (candidates_*'s RANK_COLS name); score groups use s1_id/rec_id, the
    stable string ids -- s1k/reck are only in `meta`, not in the stored feature parts."""
    df = df.with_columns(*(
        (pl.col(c) / (pl.col("n_country_s1").cast(pl.Float64) + 1).log()).cast(pl.Float32).alias(f"{c}_norm")
        for c in IDFJ_COLS))
    for ch in CHANNELS:
        x = pl.col(f"{ch}_score").fill_null(0.0)
        df = df.with_columns(
            (x / pl.max_horizontal(x.max().over("rec_id"), 1e-9)).alias(f"{ch}_score_normrec"),
            (x / pl.max_horizontal(x.max().over("s1_id"), 1e-9)).alias(f"{ch}_score_norms1"))
    return df


def invariant_feature_cols(base: list[str]) -> list[str]:
    added = [f"{c}_norm" for c in IDFJ_COLS] + [f"{ch}_score_norm{s}" for ch in CHANNELS for s in ("rec", "s1")]
    return [c for c in base if c not in DROP_RAW] + ["nonlatin_s1", "nonlatin_rec"] + added


class InvData(Data):
    """cv_full.Data, but gather() adds the derived invariant columns per chunk before slicing rows. All
    per-row lookups needed for the derivation (nonlatin flags, per-S1-country N) are resolved once in
    __init__ against self.meta (already s1k/reck-keyed, file order), so gather() only re-joins by row index."""

    def __init__(self, s1: pl.DataFrame, nonlatin: pl.DataFrame):
        super().__init__(s1)
        self.feats = invariant_feature_cols(self.feats)
        self._wide_cache: dict[str, pl.DataFrame] = {}  # part file -> derived wide frame, built once per part
        nl = nonlatin.with_columns(k=id_key("entity_id"))
        n_by_ctry = dict(s1.group_by("country").len().iter_rows())
        s1_n = s1.with_columns(n_country_s1=pl.col("country").replace_strict(n_by_ctry, return_dtype=pl.Int64))
        self.meta = (self.meta
                     .join(nl.select(s1k="k", nonlatin_s1="nonlatin"), on="s1k", how="left")
                     .join(nl.select(reck="k", nonlatin_rec="nonlatin"), on="reck", how="left")
                     .join(s1_n.select("s1k", "n_country_s1"), on="s1k", how="left")
                     .with_columns(pl.col("nonlatin_s1", "nonlatin_rec").fill_null(False).cast(pl.Int8)))

    def _wide(self, f: str, a: int, b: int) -> pl.DataFrame:
        """derive()'s output for part file `f` (meta rows [a, b)), cached: loco_pair/indomain_pair call
        gather() several times per variant (train/valid/eval row sets), each touching every part, so without
        this cache derive() (a full read_parquet + the *_norm join/groupby chain) reruns per call, not per part."""
        cached = self._wide_cache.get(f)
        if cached is None:
            stored = pl.read_parquet(f)
            m = self.meta.slice(a, b - a).select("nonlatin_s1", "nonlatin_rec", "n_country_s1")
            cached = derive(pl.concat([stored, m], how="horizontal")).select(pl.col(self.feats).cast(pl.Float32))
            self._wide_cache[f] = cached
        return cached

    def gather(self, pos: np.ndarray, wdir) -> np.ndarray:
        """Same contract as Data.gather: per part file, the derived invariant columns (cached across calls
        on this instance), then select self.feats. REC_COLS (world-swappable) are overridden after, as in Data."""
        from .cv_full import REC_COLS
        X = np.empty((len(pos), len(self.feats)), np.float32)
        for f, a, b, rows in self.chunks():
            u, v = np.searchsorted(pos, [a, b])
            if u < v:
                sel = rows[pos[u:v] - a]
                X[u:v] = self._wide(f, a, b).to_numpy()[sel]
        if wdir is not None:
            for j, c in enumerate(self.feats):
                if c in REC_COLS:
                    X[:, j] = np.load(wdir / f"{c}.npy", mmap_mode="r")[pos]
        return X


def loco_pair(name: str, train_countries: list[str], eval_country: str, s1: pl.DataFrame,
              data_by_variant: dict[str, Data], rounds: int, log: StepLog) -> list[dict]:
    """One fold-0-style world/split (no early stopping: fixed `rounds`), train on train_countries' rows,
    test-density score on eval_country only. Returns one row per variant (each has its own Data/feats)."""
    ctry = s1["country"].to_numpy()
    train_mask = np.isin(ctry, train_countries)
    drop, _, _ = split_fold(s1, 0)  # reuse the fold-0 dropout world for a realistic test-density eval
    eval_mask = (ctry == eval_country) & ~drop
    out = []
    for vname, d in data_by_variant.items():
        work = path("features_dir") / f"_worlds_inv_{name}_{vname}"
        world(d.meta, ~drop[d.code], work, log)
        tr, w = weighted_sample_rows(np.flatnonzero(train_mask[d.code]), d.y, d.meta["hard"].to_numpy(),
                                     d.in_v1, np.random.default_rng(SEED))
        rows_eval = np.flatnonzero(eval_mask[d.code])
        bst = fit(d.gather(tr, work), d.y[tr], d.feats, rounds, weight=w)
        p = np.full(len(d.i), np.nan, np.float32)
        p[rows_eval] = bst.predict(d.gather(rows_eval, work))
        r = score(s1, d, p, eval_mask)
        shutil.rmtree(work, ignore_errors=True)
        out.append({"setup": name, "variant": vname, "n_feats": len(d.feats), "train_s1": int(train_mask.sum()),
                    "eval_s1": int(eval_mask.sum()), "macro_f05": r["macro_f05"], "best_t": r["best_t"]})
        log(f"{name} {vname}", **{k: v for k, v in out[-1].items() if k not in ("setup", "variant")})
    return out


def indomain_pair(s1: pl.DataFrame, data_by_variant: dict[str, Data], rounds: int, log: StepLog) -> list[dict]:
    """Fold-0 GroupKFold split (as cv_full.run does for one fold), test-density score, all countries together."""
    drop, train, inner = split_fold(s1, 0)
    scope = (s1["fold"].to_numpy() == 0) & ~drop
    out = []
    for vname, d in data_by_variant.items():
        work = path("features_dir") / f"_worlds_inv_indomain_{vname}"
        world(d.meta, ~drop[d.code], work, log)
        tr, w = weighted_sample_rows(np.flatnonzero(train[d.code]), d.y, d.meta["hard"].to_numpy(),
                                     d.in_v1, np.random.default_rng(SEED))
        va = np.flatnonzero(inner[d.code])
        rows_s1 = np.flatnonzero((d.fold == 0) & ~drop[d.code])
        bst = fit(d.gather(tr, work), d.y[tr], d.feats, rounds, valid=(d.gather(va, work), d.y[va]), weight=w)
        p = np.full(len(d.i), np.nan, np.float32)
        p[rows_s1] = bst.predict(d.gather(rows_s1, work), num_iteration=bst.best_iteration)
        r = score(s1, d, p, scope)
        shutil.rmtree(work, ignore_errors=True)
        out.append({"setup": "in-domain", "variant": vname, "n_feats": len(d.feats),
                    "macro_f05": r["macro_f05"], "best_t": r["best_t"], "best_iter": bst.best_iteration})
        log(f"in-domain {vname}", **{k: v for k, v in out[-1].items() if k not in ("setup", "variant")})
    return out


def run_adv_val(log: StepLog) -> dict:
    """adv_val's sampling + CV machinery, on test candidate rows widened with the same derive() used in
    training (test's *_score/*_idfj/business_name/address columns give it everything derive() needs)."""
    idx = sample_index(300_000, "France", ["US", "India"])
    raw_feats = [c for c in feature_cols("test") if c not in DROP_RAW]
    nonlatin = script_flags("test")
    df = (pl.scan_parquet(parts("test")).with_row_index("_i").filter(pl.col("_i").is_in(idx["_i"].implode()))
          .collect(engine="streaming"))
    assert df["_i"].equals(idx["_i"]), "row index drifted between passes"
    s1_ctry = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    n_by_ctry = dict(s1_ctry.group_by("country").len().iter_rows())
    df = (df.join(s1_ctry.with_columns(n_country_s1=pl.col("country").replace_strict(n_by_ctry, return_dtype=pl.Int64))
              .select("s1_id", "n_country_s1"), on="s1_id", how="left")
          .join(nonlatin.rename({"entity_id": "s1_id", "nonlatin": "nonlatin_s1"}), on="s1_id", how="left")
          .join(nonlatin.rename({"entity_id": "rec_id", "nonlatin": "nonlatin_rec"}), on="rec_id", how="left")
          .with_columns(pl.col("nonlatin_s1", "nonlatin_rec").fill_null(False).cast(pl.Int8)))
    df = derive(df)
    feats = invariant_feature_cols(raw_feats)
    X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
    groups = df["s1_id"].to_numpy()
    y = idx["is_france"].to_numpy()
    auc, folds, gain = cv_auc(X, y, groups, feats)
    order = np.argsort(-gain)[:TOP]
    top = [{"feature": feats[j], "gain_share": float(gain[j] / gain.sum())} for j in order]
    log("adv_val (invariant feats)", auc=auc)
    return {"auc": auc, "fold_auc": folds, "top": top}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="5% S1 sample, 50 rounds")
    ap.add_argument("--adv-val", action="store_true", help="also run adversarial validation on invariant test feats")
    a = ap.parse_args()
    log = StepLog()
    frac = 0.05 if a.smoke else CFG["matcher"]["subset_fraction"]
    rounds = 50 if a.smoke else 300

    s1 = s1_table(frac)
    log("meta", s1=s1.height, countries=sorted(s1["country"].unique().to_list()))
    nonlatin = script_flags("train")  # covers S1+S2+S3 in one table, keyed by entity_id
    data_variants = {"baseline": Data(s1), "invariant": InvData(s1, nonlatin)}
    log("data", **{v: len(d.feats) for v, d in data_variants.items()})

    rows = []
    rows += loco_pair("US->India", ["US"], "India", s1, data_variants, rounds, log)
    rows += loco_pair("India->US", ["India"], "US", s1, data_variants, rounds, log)
    rows += indomain_pair(s1, data_variants, rounds, log)

    by = {(r["setup"], r["variant"]): r["macro_f05"] for r in rows}
    delta_indomain = by[("in-domain", "invariant")] - by[("in-domain", "baseline")]
    gate = "OK" if delta_indomain >= -KEEP_BAR else f"FAIL (< -{KEEP_BAR})"

    adv = run_adv_val(log) if a.adv_val else None

    L = ["# Country-invariance probe (M5)",
         "Generated by `src/invariance.py`. Post-hoc columns only (features.py untouched): `nonlatin_s1/rec` "
         "replaces `*_nonascii` (Unicode script check on the raw business_name/address, accents excluded); "
         "`*_idfj_norm` = `*_idfj / ln(N_S1_country + 1)`; `{A,B,C,X}_score_norm{rec,s1}` = the channel score "
         "divided by its max over the row's record / S1 group (0 when the channel never hit). `invariant` drops "
         f"the raw `{', '.join(DROP_RAW)}` columns. LOCO uses cv_full's fold-0 world (test-density dropout), "
         f"fixed {rounds} rounds (no early stopping across countries), {'5%' if a.smoke else 'subset_fraction'} "
         "S1 sample.", "",
         "## 1. LOCO + in-domain (test-density macro F0.5)", "",
         *md([{"setup": r["setup"], "variant": r["variant"], "n_feats": r["n_feats"], "eval_s1": r.get("eval_s1", "-"),
              "macro_f05": round(r["macro_f05"], 4), "best_t": r["best_t"]} for r in rows]), "",
         f"In-domain delta (invariant - baseline): {delta_indomain:+.4f} -- gate (must not drop > {KEEP_BAR}): "
         f"**{gate}**.", ""]
    if adv is not None:
        L += ["## 2. Adversarial validation (France vs US+India), invariant features", "",
              f"AUC {adv['auc']:.4f} (folds {', '.join(f'{x:.4f}' for x in adv['fold_auc'])})", "",
              *md([{"feature": t["feature"], "gain_share": round(t["gain_share"], 4)} for t in adv["top"]]), ""]
    L += [f"Peak RSS: {peak_rss_mb()} MB.", ""]
    REPORT.write_text("\n".join(L))
    log("report", path=str(REPORT))
    log.dump(path("oof_dir") / "invariance_timing.json")


if __name__ == "__main__":
    main()
