"""Floor tests: can S1-side signals break core-name ties (H1)? Is the empty-address tie problem the same size on test
(H2)? Does our metric match the organisers' (H3)? Read-only; no pipeline change.

H1  World, populations A/B (+ TP sanity), tie sets, loss weight `w` and confident predictions = src.tiebreak
    (populations + sec1, unchanged). Scope = records with 2 <= tie size <= TIE_CAP and the true S1 in the tie set.
    Per (record r, tie S1 s):
      f1 char-3gram cosine of lowercased raw names (mechanisms.vec, l2; legal form / punctuation kept)
      f2 normaliser legal_form(r) == legal_form(s) (both "" counts as equal)
      f3 lname(r) == lname(s), lname = lowercased, whitespace-collapsed raw name (tiebreak.LNAME)
      f4 rank of s in the tie set by train_source1 row order; f5 same by entity_id digits (ordinal, 1 = first)
      f6 # records kept with p >= P_CONF to s (tiebreak's conf = OOF argmax), r itself excluded
      f7 # unique whitespace tokens of s's lowercased raw name also in r's (legal/punctuation tokens kept)
    Per feature and direction (max; also min for f4-f6): % true S1 unique argmax, % tied at the max, accuracy when
    untied, loss pts recoverable = sum w of records whose unique argmax is the true S1. Floor row = random pick.
    Ranker: LGBMRanker (lambdarank) on f1-f7, GroupKFold(5) by record; OOF top-1 accuracy (score ties = wrong);
    margin = top-1 minus top-2 score; threshold = lowest margin keeping precision >= PREC on A+B FN/miss (picked
    on the same OOF scores, so mildly optimistic); resolved-correct loss pts at that margin per population.
H2  Label-free, both splits alike: records = S2 + S3 normalised rows; empty = normaliser has_address False (raw
    blank share also shown). Pool = that split's full S1 file; tie key = tiebreak.core_key (token_dict + FILLER);
    tie size = pool S1s with the record's country and key. Country is only a report grouping here.
H3  Greps organiser-provided text (Documentation_template.md, non-TSV files under dataset/) for metric lines and
    quotes src/metric.py with line numbers; GT facts that bear on the questions (records with > 1 true S1).

Run from code/business_entity_resolution/:
  python -m src.floor_tests --smoke   # 1% S1 sample, 1 partition, first 200k records/file for H2, same report path
  python -m src.floor_tests           # 20% sample -> docs/floor_tests.md
Aborts if peak RSS > 12 GB (tiebreak.Log guard).
"""
import argparse
import re

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import src.mechanisms as M
import src.tiebreak as TB
from .decide import md
from .features import id_key, load_token_dict
from .io import PKG_DIR, ROOT, load_gt_pairs, load_source, path, peak_rss_mb
from .mechanisms import NORM, P_CONF, SEED, load, rowdot, vec
from .tiebreak import BINS, BREAKS, GROUPS, LNAME, TIE_CAP, Log, core_key

REPORT = ROOT / "docs" / "floor_tests.md"
METRIC_PY = PKG_DIR / "src" / "metric.py"
NP, PREC, SMOKE_ROWS = 8, 0.9, 200_000
FEATS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
DIRS = {"f4": (1, -1), "f5": (1, -1), "f6": (1, -1)}  # both directions; others max only
FN_GROUPS = ["A FN/miss", "B FN/miss"]
RAWL = pl.col("business_name").str.to_lowercase()
TOKS = RAWL.str.extract_all(r"\S+").list.unique()

# tiebreak.populations() drops world()'s S1 table; keep its (s1k, code) map instead of calling world() twice.
_S1: dict = {}


def _world_keep_s1(log):
    w = M.world(log)
    _S1["code"] = w[1].select("s1k", "code")
    return w


TB.world = _world_keep_s1


def s1_order(split: str) -> pl.DataFrame:
    return (pl.from_pandas(load_source(split, 1, ["entity_id"])).with_row_index("s1_row")
            .select(s1k=id_key("entity_id"), s1_row=pl.col("s1_row").cast(pl.Int64),
                    s1_num=pl.col("entity_id").str.replace(r"^S\d-", "").cast(pl.Int64)))


def order_groups(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(pl.col("population").replace_strict(GROUPS, list(range(len(GROUPS)))), maintain_order=True)


# ---------------------------------------------------------------- H1
def h1_features(log: Log) -> tuple[pl.DataFrame, pl.DataFrame]:
    X, pool, conf = TB.populations(log)
    _, X = TB.sec1(X, pool)
    scope = X.filter(pl.col("n_tie").is_between(2, TIE_CAP) & pl.col("hit"))
    meta = scope.select("rid", population="grp", w=pl.col("w").cast(pl.Float64), n_tie="n_tie")
    tie = (scope.select("rid", "reck", "code", "rc", "ckey")
           .join(pool.select(tcode="code", rc="country", ckey="ckey"), on=["rc", "ckey"])
           .join(_S1.pop("code").rename({"code": "tcode"}), on="tcode")
           .select("rid", "reck", "tcode", "s1k", is_true=pl.col("tcode") == pl.col("code")))
    assert tie.group_by("rid").agg(pl.col("is_true").sum())["is_true"].eq(1).all(), "true S1 not once per tie set"
    del X
    s_att = load(NORM[0], tie["s1k"].unique(), "s1k", s_raw=RAWL, s_lname=LNAME, s_legal=pl.col("legal_form"),
                 s_tok=TOKS).join(s1_order("train"), on="s1k")
    r_att = load(NORM[1:], scope["reck"].unique(), "reck", r_raw=RAWL, r_lname=LNAME, r_legal=pl.col("legal_form"),
                 r_tok=TOKS)
    n_conf = conf.group_by("tcode").agg(n_conf=pl.len())
    self_c = conf.select("tcode", reck="nb", self_c=pl.lit(1, pl.Int64))  # kept = argmax: <= 1 row per record
    log("h1 ties", scope=scope.height, tie_rows=tie.height, conf=conf.height)
    parts = []
    for j in range(NP):
        t = (tie.filter(pl.col("rid") % NP == j).join(s_att, on="s1k").join(r_att, on="reck")
             .join(n_conf, on="tcode", how="left").join(self_c, on=["tcode", "reck"], how="left"))
        ids = pl.concat([t.select(s="r_raw"), t.select(s="s_raw")]).unique().with_row_index("i")
        t = t.join(ids.rename({"s": "r_raw", "i": "ia"}), on="r_raw").join(ids.rename({"s": "s_raw", "i": "ib"}),
                                                                           on="s_raw")
        A = vec(ids["s"], 3)
        cos = rowdot(A, t["ia"].to_numpy(), A, t["ib"].to_numpy()) if t.height else []
        parts.append(t.select(
            "rid", "tcode", "is_true",
            f1=pl.Series(cos, dtype=pl.Float32),
            f2=(pl.col("r_legal") == pl.col("s_legal")).cast(pl.Int8),
            f3=(pl.col("r_lname") == pl.col("s_lname")).cast(pl.Int8),
            f4=pl.col("s1_row").rank("ordinal").over("rid").cast(pl.Int32),
            f5=pl.col("s1_num").rank("ordinal").over("rid").cast(pl.Int32),
            f6=(pl.col("n_conf").fill_null(0) - pl.col("self_c").fill_null(0)).cast(pl.Int32),
            f7=pl.col("s_tok").list.set_intersection(pl.col("r_tok")).list.len().cast(pl.Int32)))
        log(f"h1 part {j}", rows=t.height)
    F = pl.concat(parts).sort("rid", "tcode")
    assert F.height == tie.height, (F.height, tie.height)
    assert (F["f6"] >= 0).all()
    return F, meta


def per_feature(F: pl.DataFrame, meta: pl.DataFrame) -> list[dict]:
    rid_pts = lambda ok: pl.col("w").filter(ok).sum()
    out = []
    rnd = meta.group_by("population").agg(n=pl.len(), ok=100 * (1 / pl.col("n_tie")).mean(),
                                          pts_ok=(pl.col("w") / pl.col("n_tie")).sum(), pts=pl.col("w").sum())
    for r in rnd.iter_rows(named=True):
        out.append({"population": r["population"], "feature": "random pick", "dir": "-", "n": r["n"],
                    "% true unique argmax": r["ok"], "% tied": 0.0, "acc | untied %": r["ok"],
                    "loss pts recoverable": r["pts_ok"], "loss pts in scope": r["pts"]})
    for f in FEATS:
        for sign in DIRS.get(f, (1,)):
            s = (pl.col(f).cast(pl.Float64) * sign).round(6)
            R = (F.group_by("rid").agg(nmax=(s == s.max()).sum(), true_at=((s == s.max()) & pl.col("is_true")).any())
                 .join(meta, on="rid").with_columns(ok=(pl.col("nmax") == 1) & pl.col("true_at"),
                                                    untied=pl.col("nmax") == 1))
            for r in R.group_by("population").agg(
                    n=pl.len(), ok=100 * pl.col("ok").mean(), tied=100 * (~pl.col("untied")).mean(),
                    acc=100 * pl.col("ok").sum() / pl.col("untied").sum(), pts_ok=rid_pts(pl.col("ok")),
                    pts=pl.col("w").sum()).iter_rows(named=True):
                out.append({"population": r["population"], "feature": f, "dir": "max" if sign > 0 else "min",
                            "n": r["n"], "% true unique argmax": r["ok"], "% tied": r["tied"],
                            "acc | untied %": r["acc"] if r["acc"] is not None else float("nan"),
                            "loss pts recoverable": float(r["pts_ok"]), "loss pts in scope": float(r["pts"])})
    return order_groups(pl.DataFrame(out)).to_dicts()


def ranker(F: pl.DataFrame, meta: pl.DataFrame, log: Log) -> tuple[list[dict], list[dict], float]:
    Xf, y, rid = F.select(FEATS).to_numpy().astype(np.float32), F["is_true"].to_numpy().astype(int), F["rid"].to_numpy()
    oof, gain = np.zeros(len(y)), np.zeros(len(FEATS))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(Xf, y, rid)):
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, random_state=SEED, deterministic=True, force_row_wise=True,
                           importance_type="gain", verbose=-1)
        m.fit(Xf[tr], y[tr], group=np.unique(rid[tr], return_counts=True)[1])  # F sorted by rid, tr ascending
        oof[va] = m.predict(Xf[va])
        gain += m.feature_importances_
        log(f"ranker fold {k}", train=len(tr))
    top = (F.select("rid", "is_true").with_columns(s=pl.Series(oof)).sort(["rid", "s"], descending=[False, True])
           .group_by("rid", maintain_order=True).agg(s1=pl.col("s").first(), s2=pl.col("s").get(1),
                                                     true1=pl.col("is_true").first())
           .with_columns(margin=pl.col("s1") - pl.col("s2"))
           .with_columns(ok=pl.col("true1") & (pl.col("margin") > 0)).join(meta, on="rid"))
    c = top.filter(pl.col("population").is_in(FN_GROUPS) & (pl.col("margin") > 0)).sort("margin", descending=True)
    prec = np.cumsum(c["ok"].to_numpy()) / np.arange(1, c.height + 1)
    good = np.flatnonzero(prec >= PREC)
    thr = float(c["margin"][int(good[-1])]) if len(good) else float("inf")
    res = pl.col("margin") >= thr
    rows = [{"population": r["population"], "n": r["n"], "top-1 acc %": r["acc"],
             "loss pts top-1 correct": float(r["pts_top1"]), "resolved %": r["cov"],
             "precision | resolved %": r["prec"] if r["prec"] is not None else float("nan"),
             "resolved wrong (n)": r["wrong"], "loss pts recoverable @P>=0.9": float(r["pts_res"]),
             "loss pts in scope": float(r["pts"])}
            for r in top.group_by("population").agg(
                n=pl.len(), acc=100 * pl.col("ok").mean(), pts_top1=pl.col("w").filter(pl.col("ok")).sum(),
                cov=100 * res.mean(), prec=100 * (res & pl.col("ok")).sum() / res.sum(),
                wrong=(res & ~pl.col("ok")).sum(), pts_res=pl.col("w").filter(res & pl.col("ok")).sum(),
                pts=pl.col("w").sum()).iter_rows(named=True)]
    imp = [{"feature": f, "mean gain share %": float(100 * g / gain.sum())} for f, g in zip(FEATS, gain)]
    return order_groups(pl.DataFrame(rows)).to_dicts(), imp, thr


def h1(log: Log) -> list[str]:
    F, meta = h1_features(log)
    feat_rows = per_feature(F, meta)
    log("h1 per-feature")
    rk_rows, imp, thr = ranker(F, meta, log)
    counts = meta.group_by("population").agg(n=pl.len(), tie_rows=pl.col("n_tie").sum(),
                                             median_tie=pl.col("n_tie").median(), pts=pl.col("w").sum())
    return ["## H1. S1-side tie-breakers", "",
            f"Scope: records with 2 <= tie size <= {TIE_CAP} and the true S1 in the tie set (tiebreak section 1). "
            f"f6 neighbours = records kept with p >= {P_CONF}, the record excluded. Loss pts recoverable = sum of "
            "w over records whose unique argmax is the true S1 (an upper bound: it ignores FPs from wrong picks).",
            "", *md(order_groups(counts).to_dicts()), "", "### Per feature", "", *md(feat_rows), "",
            "### LightGBM ranker (f1-f7, GroupKFold 5 by record, OOF)", "",
            f"Margin threshold = {thr:.6f}: lowest top-1 minus top-2 margin with precision >= {PREC} on A+B FN/miss "
            "(chosen on the same OOF scores). Resolved wrong = records that would get a wrong S1 (new FP).", "",
            *md(rk_rows), "", *md(imp), ""]


# ---------------------------------------------------------------- H2
def h2(smoke: bool, log: Log) -> list[str]:
    dm = load_token_dict() or {}
    norm = lambda split, n: path("interim_dir") / f"norm_{split}_s{n}.parquet"
    share, dist, touch = [], [], []
    for split in ("train", "test"):
        pool = (pl.scan_parquet(norm(split, 1)).select("country", ckey=core_key("name_tokens", dm))
                .collect(engine="streaming"))
        sizes = pool.group_by("country", "ckey").agg(n_tie=pl.len())
        scans = [pl.scan_parquet(norm(split, n)) for n in (2, 3)]
        recs = pl.concat([s.head(SMOKE_ROWS) for s in scans] if smoke else scans)
        empty = ~pl.col("has_address")
        share += (recs.group_by("country").agg(n=pl.len(), raw=100 * (pl.col("business_address").str.strip_chars() == "").mean(),
                                               empty=100 * empty.mean())
                  .collect(engine="streaming").with_columns(split=pl.lit(split)).to_dicts())
        E = (recs.filter(empty).select("country", ckey=core_key("name_tokens", dm)).collect(engine="streaming")
             .join(sizes, on=["country", "ckey"], how="left").with_columns(pl.col("n_tie").fill_null(0))
             .with_columns(bin=pl.col("n_tie").cut(BREAKS, labels=BINS).cast(pl.String)))
        for (c,), d in sorted(E.partition_by("country", as_dict=True).items()):
            b = dict(d.group_by("bin").len().iter_rows())
            dist.append({"split": split, "country": c, "empty records": d.height,
                         **{f"% tie {k}": 100 * b.get(k, 0) / d.height for k in BINS},
                         "% tie >= 2": 100 * float((d["n_tie"] >= 2).mean())})
        hit = sizes.filter(pl.col("n_tie") >= 2).join(E.select("country", "ckey").unique(), on=["country", "ckey"],
                                                      how="semi")
        multi = sizes.filter(pl.col("n_tie") >= 2).group_by("country").agg(m=pl.col("n_tie").sum())
        for r in (pool.group_by("country").agg(n=pl.len()).join(multi, on="country", how="left")
                  .join(hit.group_by("country").agg(h=pl.col("n_tie").sum()), on="country", how="left")
                  .fill_null(0).sort("country").iter_rows(named=True)):
            touch.append({"split": split, "country": r["country"], "S1s": r["n"],
                          "% S1 in key groups >= 2": 100 * r["m"] / r["n"],
                          "% S1 in a tie set >= 2 of an empty record": 100 * r["h"] / r["n"]})
        log(f"h2 {split}", pool=pool.height, empty=E.height)
    share = pl.DataFrame(share).sort("split", "country").select(
        "split", "country", records="n", **{"% raw address blank": "raw", "% has_address False": "empty"}).to_dicts()
    return ["## H2. Empty-address ties, train vs test (label-free)", "",
            "Records = S2 + S3 normalised rows" + (f" (SMOKE: first {SMOKE_ROWS:,} per file)" if smoke else "")
            + ". Empty = normaliser has_address False. Pool = the split's full S1 file; tie key = tiebreak.core_key; "
              "country only groups the report.", "", *md(share), "", "### Tie-size distribution of empty-address records",
            "", *md(dist), "", "### S1s touched", "", *md(touch), ""]


# ---------------------------------------------------------------- H3
def h3(log: Log) -> list[str]:
    pat = re.compile(r"f_?0\.5|f-?beta|macro|precision|recall|singleton|metric|evaluat|scor", re.I)
    org = [ROOT / "Documentation_template.md", *sorted(p for p in path("train_dir").parent.rglob("*")
                                                       if p.is_file() and p.suffix != ".tsv" and p.name != ".DS_Store")]
    quotes = [f"- `{p.relative_to(ROOT)}:{i}`: {line.strip()}" for p in org if p.exists()
              for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1) if pat.search(line)]
    src = METRIC_PY.read_text().splitlines()
    ln = lambda s: next(i for i, line in enumerate(src, 1) if s in line)
    a, b = ln("def f05"), ln("if __name__") - 3  # f05 + macro_f05 bodies
    gt = load_gt_pairs()
    n_s1 = load_source("train", 1, ["entity_id"]).shape[0]
    multi = gt.group_by("match_id").len().filter(pl.col("len") > 1).height
    n_rec = gt["match_id"].n_unique()
    log("h3")
    rel = METRIC_PY.relative_to(ROOT)
    return ["## H3. Metric definition: organisers vs ours", "",
            "### Organiser-provided text found in the repo (all lines matching metric terms)", "",
            *(quotes or ["(none)"]), "",
            "No organiser evaluation script and no problem-statement text exist in the repo or `dataset/` (only the "
            "TSVs). The lines above name the metric (macro F_0.5) but define none of the four rules asked about. "
            "`docs/context.md` lines 20-29 describe them, but that file is our own paraphrase, not a verbatim quote.", "",
            f"### Ours: `{rel}:{a}-{b}`", "", "```python", *[f"{i:>3} {src[i - 1]}" for i in range(a, b + 1)], "```", "",
            "### The four questions", "",
            *md([
                {"question": "S1 with no true matches and no prediction", "organiser text": "not in repo",
                 "ours": f"1.0 (`{rel}:{ln('if not pred or not true')}-{ln('return float(not pred and not true)')}`)"},
                {"question": "are those S1s averaged in", "organiser text": "not in repo ('macro' only)",
                 "ours": f"yes: mean over ALL truth keys, missing key = empty (`{rel}:{ln('keys = list(truth)')}-"
                         f"{ln('return float(scores.mean())')}`)"},
                {"question": "predicted S2/S3 id not in truth", "organiser text": "not in repo",
                 "ours": f"FP: in |pred|, lowers precision; tp = 0 gives 0 (`{rel}:{ln('tp = len(pred & true)')}-"
                         f"{ln('p, r = tp / len(pred)')}`). Ids absent from test are not rejected; duplicates "
                         "collapse (sets)"},
                {"question": "may a record map to > 1 S1", "organiser text": "not in repo",
                 "ours": f"metric: yes, each S1 scored independently. GT: {multi:,} of {n_rec:,} matched records have "
                         "> 1 true S1. Pipeline (mechanisms.world kept argmax): no, one S1 per record"},
            ]), "",
            f"Train: {n_s1:,} S1s, {gt['s1_id'].n_unique():,} with >= 1 true match.", "",
            "### Differences (every one is unverifiable until the organiser text is available)", "",
            "1. Singleton = 1.0 and singletons averaged in: ours does both; the organiser text is not in the repo.",
            "2. Duplicate ids in one list: ours collapses them (sets); an organiser scorer that counts list entries "
            "would count a duplicate twice in |pred|.",
            "3. Unknown ids (not in test S2/S3): ours counts them as FPs; the organisers may instead reject the file.",
            "4. A record in > 1 S1 list: the metric allows it and the GT has it; our argmax decision rule cannot "
            "output it, which caps recall on those S1s. This is a pipeline limit, not a metric difference.", ""]


def main() -> None:
    global NP
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% S1 sample, 1 partition, 200k records/file in H2")
    smoke = ap.parse_args().smoke
    if smoke:
        M.FRAC, NP = 0.01, 1
    log = Log()
    lines = h1(log) + h2(smoke, log) + h3(log)
    head = ["# Floor tests: S1-side tie-breakers, empty-address ties on test, metric definition"
            + (" [SMOKE]" if smoke else ""), "",
            f"Generated by `src/floor_tests.py` (definitions in its docstring). World and populations = src.tiebreak "
            f"(cv_full (b), t = {M.T}, sample {M.FRAC:.0%} of S1s, seed {SEED}). Loss pts = sample macro F0.5 points.",
            ""]
    REPORT.write_text("\n".join(head + lines + [f"Peak RSS: {peak_rss_mb()} MB.", ""]))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / f"floor_tests{'_smoke' if smoke else ''}_timing.json")


if __name__ == "__main__":
    main()
