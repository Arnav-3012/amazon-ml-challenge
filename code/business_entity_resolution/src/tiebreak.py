"""M5 tiebreak: when a record's core name ties with several same-country S1s, do the S1s' OTHER confidently
matched records (their "copies") pick the true one? Read-only; no pipeline change, no features built.

World / 20%-S1 sample / pair status (TP/FN/miss) / loss weight `w` = src.mechanisms (cv_full (b) world,
t = 0.75, macro F0.5 loss points of that sample). addr_empty predicate = mechanisms.tag_exprs(); core_identical =
src.addr_empty's first grammar class (dict-remapped name_tokens of record == of true S1). token_dict =
src.features.load_token_dict().
Populations (pairs of the 20% sample; a record true for 2 S1s appears twice):
  A  addr_empty & status in (FN, miss);   B  core_identical & record address non-empty & status in (FN, miss);
  A TP / B TP  the same predicates with status TP (sanity), seeded subsample of <= N_TP each.
Core key = name_tokens -> token_dict -> FILLER tokens dropped -> joined. FILLER = the extra words W of
docs/addr_empty.md section 2 enriched >= 5x in addr_empty vs all true records. Pool = ALL world S1s (not the 20%
sample); tie set of a record = pool S1s with the record's own country and an equal core key.
1) Tie-set size distribution per population; % where the true S1 is in the tie set.
2) Scope: records with 2 <= tie size <= TIE_CAP. For each tie S1 s: its records predicted with p >= P_CONF (kept
   argmax, in-world), the record itself excluded, split into same source as the record / other source.
   sim(r, s) = max over those records of (a) exact equality of lname = raw business_name lowercased, whitespace
   collapsed (punctuation, typos kept), (b) char-3gram cosine of lname (mechanisms.vec, l2). S1s with no such
   record score 0. Per record: no signal (max == 0), tied (max shared by >= 2 S1s; cosines rounded to 1e-6),
   untied (unique max); correct = untied and the argmax is the true S1.
3) Report only: row order of train S2/S3 files vs true S1 (adjacent-row same-S1 share vs a seeded shuffle of the
   S1 column; Spearman of row index vs true S1 row index in train_source1) and Spearman of entity_id digits.
   Records with > 1 true S1 keep the first by S1 row.

Run from code/business_entity_resolution/:
  python -m src.tiebreak --smoke   # 1%% S1 sample (mechanisms.FRAC), same path incl. docs/tiebreak.md (< 3 min)
  python -m src.tiebreak           # 20%% sample -> docs/tiebreak.md
Aborts if peak RSS > MAX_RSS_MB (12 GB).
"""
import argparse

import numpy as np
import polars as pl

import src.mechanisms as M
from .decide import md
from .features import id_key, load_token_dict
from .io import ROOT, StepLog, load_gt_pairs, load_source, path, peak_rss_mb
from .mechanisms import NORM, P_CONF, SEED, load, rowdot, sample_pairs, tag_exprs, vec, world

MAX_RSS_MB = 12_000
M.MAX_RSS_MB = MAX_RSS_MB  # mechanisms.guard (rowdot) reads its module global at call time
REPORT = ROOT / "docs" / "tiebreak.md"
FILLER = ["enterprises", "trading", "ventures", "industries", "associates"]
TIE_CAP, N_TP = 1000, 50_000
NP = 8  # tie-row partitions for section 2; --smoke sets 1
GROUPS = ["A FN/miss", "B FN/miss", "A TP", "B TP"]
BREAKS, BINS = [0, 1, 2, 5, 20, 100, TIE_CAP], ["0", "1", "2", "3-5", "6-20", "21-100", "101-1000", "1001+"]
E, NULL = pl.element(), pl.lit(None, dtype=pl.String)
LNAME = pl.col("business_name").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def guard(step: str) -> None:
    if (rss := peak_rss_mb()) > MAX_RSS_MB:
        raise SystemExit(f"ABORT src.tiebreak: peak RSS {rss} MB > {MAX_RSS_MB} MB at '{step}'. Raise NP.")


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        guard(step)


def remapped(col: str, dm: dict) -> pl.Expr:
    return pl.col(col).list.eval(E.replace(dm)) if dm else pl.col(col)


def core_key(col: str, dm: dict) -> pl.Expr:
    return (remapped(col, dm).list.eval(pl.when(E.is_in(FILLER)).then(NULL).otherwise(E))
            .list.drop_nulls().list.join(" "))


def source(c: str) -> pl.Expr:  # id_key keeps the "S{n}-" digit first
    return pl.col(c).cast(pl.String).str.head(1)


def populations(log: Log):
    """X: one row per population pair (rid, grp, reck, true code, w, rc, key, lname); pool; kept."""
    _, s1, _, kept, in_cand, cnt, f, L = world(log)
    gt = load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))
    P, _ = sample_pairs(s1, kept, in_cand, gt, cnt, f, L, log)
    del in_cand, gt
    P = P.filter(pl.col("status") != "FP")
    dm = load_token_dict() or {}
    s_raw = load(NORM[0], P["s1k"].unique(), "s1k", s_a=pl.col("business_address"),
                 s_core=remapped("name_tokens", dm).list.join(" "))
    r_raw = load(NORM[1:], P["reck"].unique(), "reck", r_a=pl.col("business_address"), rc=pl.col("country"),
                 r_core=remapped("name_tokens", dm).list.join(" "), ckey=core_key("name_tokens", dm), lname=LNAME)
    X = P.join(s_raw, on="s1k").join(r_raw, on="reck")
    assert X.height == P.height, (X.height, P.height)
    tp = pl.col("status") == "TP"
    pop = (pl.when(tag_exprs()["addr_empty"]).then(pl.lit("A"))
           .when((pl.col("r_core") == pl.col("s_core")) & (pl.col("r_a").str.strip_chars() != "")).then(pl.lit("B")))
    X = (X.with_columns(grp=pl.concat_str(pop, pl.when(tp).then(pl.lit("TP")).otherwise(pl.lit("FN/miss")),
                                          separator=" "))
         .drop_nulls("grp").select("grp", "status", "reck", "code", "w", "rc", "ckey", "lname").sort("grp", "code", "reck"))
    X = pl.concat([d.sample(N_TP, seed=SEED) if g.endswith("TP") and d.height > N_TP else d
                   for (g,), d in X.partition_by("grp", as_dict=True, maintain_order=True).items()])
    X = X.with_row_index("rid")
    pool = load(NORM[0], s1["s1k"], "s1k", ckey=core_key("name_tokens", dm)).join(
        s1.select("s1k", "code", "country"), on="s1k").select("code", "country", "ckey")
    log("populations", **{g: int((X["grp"] == g).sum()) for g in GROUPS}, pool=pool.height, dict_size=len(dm))
    # confident = world()'s kept argmax filtered on OOF p_td only; `label` is dropped here and never read
    conf = kept.filter(pl.col("p_td") >= P_CONF).select(tcode="code", nb="reck", p="p_td")
    assert conf.columns == ["tcode", "nb", "p"] and (conf["p"] >= P_CONF).all(), conf.columns
    return X, pool, conf


def sec1(X: pl.DataFrame, pool: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    sizes = pool.group_by("country", "ckey").agg(n_tie=pl.len()).rename({"country": "rc"})
    X = (X.join(sizes, on=["rc", "ckey"], how="left").with_columns(pl.col("n_tie").fill_null(0))
         .join(pool.select("code", s_ckey="ckey", s_ctry="country"), on="code")
         .with_columns(hit=(pl.col("s_ckey") == pl.col("ckey")) & (pl.col("s_ctry") == pl.col("rc")),
                       bin=pl.col("n_tie").cut(BREAKS, labels=BINS).cast(pl.String)))
    rows = []
    for g in GROUPS:
        d = X.filter(pl.col("grp") == g)
        if not d.height:
            continue
        a = d.group_by("bin").agg(n=pl.len(), pts=pl.col("w").sum(), hit=100 * pl.col("hit").mean())
        a = {r["bin"]: r for r in a.iter_rows(named=True)}
        for b in [*BINS, "all"]:
            r = {"n": d.height, "pts": d["w"].sum(), "hit": 100 * d["hit"].mean()} if b == "all" else a.get(b)
            if r:
                rows.append({"population": g, "tie size": b, "n": r["n"], "% of pop": 100 * r["n"] / d.height,
                             "loss pts": float(r["pts"]), "% true S1 in tie set": float(r["hit"])})
    return (["## 1. Tie sets (same-country world S1s with the record's core key)", "",
             f"Core key = token_dict-remapped name_tokens minus FILLER {FILLER}. Pool = all "
             f"{pool.height:,} world S1s.", "", *(md(rows) if rows else ["(empty)"]), ""], X)


def sec2(X: pl.DataFrame, pool: pl.DataFrame, conf: pl.DataFrame, log: Log) -> list[str]:
    scope = X.filter(pl.col("n_tie").is_between(2, TIE_CAP))
    tie = (scope.select("rid", "reck", "code", "rc", "ckey", "lname")
           .join(pool.select(tcode="code", rc="country", ckey="ckey"), on=["rc", "ckey"])
           .select("rid", "reck", "lname", "tcode", is_true=pl.col("tcode") == pl.col("code")))
    conf = conf.join(tie.select("tcode").unique(), on="tcode", how="semi")
    conf = conf.join(load(NORM[1:], conf["nb"].unique(), "nb", nb_name=LNAME), on="nb")
    log("sec2 ties", scope=scope.height, tie_rows=tie.height, conf=conf.height)
    out = []
    for j in range(NP):
        t = tie.filter(pl.col("rid") % NP == j)
        pr = t.join(conf, on="tcode").filter(pl.col("nb") != pl.col("reck"))
        assert (pr["nb"] != pr["reck"]).all() and "label" not in pr.columns  # r excluded; no labels in sim
        pr = (pr.with_columns(same=source("nb") == source("reck"), a=(pl.col("lname") == pl.col("nb_name")).cast(pl.Float32)))
        ids = pl.concat([pr.select(s="lname"), pr.select(s="nb_name")]).unique().with_row_index("i")
        pr = pr.join(ids.rename({"s": "lname", "i": "ia"}), on="lname").join(ids.rename({"s": "nb_name", "i": "ib"}),
                                                                           on="nb_name")
        if pr.height:
            A = vec(ids["s"], 3)
            pr = pr.with_columns(b=pl.Series(rowdot(A, pr["ia"].to_numpy(), A, pr["ib"].to_numpy())))
        else:
            pr = pr.with_columns(b=pl.lit(0.0, pl.Float32))
        agg = pr.group_by("rid", "tcode", "same").agg(pl.max("a"), pl.max("b"))
        for mode, flag in (("same source", True), ("other source", False)):
            v = (t.select("rid", "tcode", "is_true")
                 .join(agg.filter(pl.col("same") == flag).drop("same"), on=["rid", "tcode"], how="left")
                 .with_columns(pl.col("a", "b").fill_null(0.0)))
            for sim in ("a", "b"):
                s = pl.col(sim).round(6)
                out.append(v.group_by("rid").agg(mx=s.max(), nmax=(s == s.max()).sum(),
                                                 true_at=((s == s.max()) & pl.col("is_true")).any())
                           .with_columns(mode=pl.lit(mode), sim=pl.lit({"a": "(a) exact", "b": "(b) cos3"}[sim])))
        log(f"sec2 part {j}", pairs=pr.height)
    R = pl.concat(out).join(scope.select("rid", "grp", "w"), on="rid")
    cat = pl.when(pl.col("mx") <= 0).then(pl.lit("none")).when(pl.col("nmax") > 1).then(pl.lit("tied")).otherwise(
        pl.lit("untied"))
    R = R.with_columns(cat=cat).with_columns(ok=(pl.col("cat") == "untied") & pl.col("true_at"))
    rows = [{"population": r["grp"], "source": r["mode"], "sim": r["sim"], "n in scope": r["n"],
             "% true unique argmax": r["ok"], "% tied": r["tied"], "% no signal": r["none"],
             "acc | untied %": r["acc"] if r["acc"] is not None else float("nan"),
             "loss pts untied-correct": float(r["pts_ok"]), "loss pts in scope": float(r["pts"])}
            for r in R.group_by("grp", "mode", "sim").agg(
                n=pl.len(), ok=100 * pl.col("ok").mean(), tied=100 * (pl.col("cat") == "tied").mean(),
                none=100 * (pl.col("cat") == "none").mean(),
                acc=100 * pl.col("ok").sum() / (pl.col("cat") == "untied").sum(),
                pts_ok=pl.col("w").filter(pl.col("ok")).sum(), pts=pl.col("w").sum())
            .sort(pl.col("grp").replace_strict(GROUPS, list(range(len(GROUPS)))), "mode", "sim").iter_rows(named=True)]
    return ["## 2. Copy signal: other confident records of each tie S1", "",
            f"Scope = records with 2 <= tie size <= {TIE_CAP} ({scope.height:,} of {X.height:,}; the rest are in "
            f"section 1's bins). Neighbours = records kept with p >= {P_CONF} to the tie S1, record itself excluded "
            f"({conf.height:,} neighbour rows over {tie.height:,} tie rows).", "",
            *(md(rows) if rows else ["(empty)"]), ""]


def sec3(log: Log) -> list[str]:
    digits = lambda c: pl.col(c).str.replace(r"^S\d-", "").cast(pl.Int64)
    s1o = pl.from_pandas(load_source("train", 1, ["entity_id"])).with_row_index("s1_row").rename({"entity_id": "s1_id"})
    gt = load_gt_pairs().join(s1o, on="s1_id")
    n_multi = gt.group_by("match_id").len().filter(pl.col("len") > 1).height
    gt1 = gt.sort("s1_row").unique("match_id", keep="first").rename({"match_id": "entity_id"})
    rng, rows = np.random.default_rng(SEED), []
    for n in (2, 3):
        r = (pl.from_pandas(load_source("train", n, ["entity_id"])).with_row_index("row")
             .join(gt1, on="entity_id", how="left").sort("row"))
        s = r["s1_row"]
        adj = lambda x: float(((x == x.shift(1)) & x.is_not_null() & x.shift(1).is_not_null()).sum()
                              / (x.is_not_null() & x.shift(1).is_not_null()).sum())
        m = r.drop_nulls("s1_row")
        sp = lambda a, b: float(m.select(pl.corr(a, b, method="spearman")).item())
        rows.append({"file": f"train_source{n}", "rows": r.height, "with true S1": m.height,
                     "adjacent same-S1 %": 100 * adj(s), "shuffled %": 100 * adj(s.gather(rng.permutation(len(s)))),
                     "Spearman row vs S1 row": sp("row", "s1_row"),
                     "Spearman id digits": float(m.select(pl.corr(digits("entity_id"), digits("s1_id"),
                                                                  method="spearman")).item())})
        log(f"sec3 S{n}")
    return ["## 3. Row order and id leakage (report only; no features built from these)", "",
            f"All train GT pairs (not just the world). {n_multi:,} records have > 1 true S1 (first by S1 row kept). "
            "Adjacent share = consecutive rows both with a true S1 that share it; shuffled = same with the S1 column "
            "permuted (seed).", "", *md(rows), ""]


def main() -> None:
    global NP
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% S1 sample, 1 partition, same path incl. report")
    smoke = ap.parse_args().smoke
    if smoke:
        M.FRAC, NP = 0.01, 1
    log = Log()
    X, pool, conf = populations(log)
    s1_lines, X = sec1(X, pool)
    log("sec1")
    s2_lines = sec2(X, pool, conf, log)
    del pool, conf
    s3_lines = sec3(log)
    head = ["# Tie-breaking core-name ties with copy signal" + (" [SMOKE: 1% sample]" if smoke else ""), "",
            f"Generated by `src/tiebreak.py` (definitions in its docstring). World: cv_full (b) via src.mechanisms, "
            f"t = {M.T}, sample {M.FRAC:.0%} of S1s (seed {SEED}). Populations: A = addr_empty, B = core_identical "
            f"with record address; TP rows are a <= {N_TP:,} sanity subsample. Loss pts = sample macro F0.5 points.",
            ""]
    REPORT.write_text("\n".join(head + s1_lines + s2_lines + s3_lines + [f"Peak RSS: {peak_rss_mb()} MB.", ""]))
    log("report", path=str(REPORT))
    log.dump(path("artifacts_dir") / "logs" / f"tiebreak{'_smoke' if smoke else ''}_timing.json")


if __name__ == "__main__":
    main()
