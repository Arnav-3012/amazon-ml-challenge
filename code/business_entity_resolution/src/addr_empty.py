"""M5 addr_empty: the top-loss tag (32.1% of D1, docs/mechanisms.md) is the record's address being empty while
its true S1's is not. Read-only; no pipeline change.

World / 20%-S1 sample / pair status (TP/FP/FN/miss) / loss weight `w` = src.mechanisms (cv_full (b) world,
t = 0.75, macro F0.5 loss points on that sample, np default_rng(config seed)) -- reused via import. `addr_empty`
predicate, `pseudo_brand`'s cosine idea and DOMAIN_RE are src.mechanisms' own tag_exprs()/constants, reused
rather than redefined. core_name / legal_form / token lists = src.normalise; token_dict = src.features.
load_token_dict() (src.mine_dict output), applied to name_tokens to get "core tokens after token_dict".

Population: true pairs (TP/FN/miss, unfiltered -- section 1's baseline) and FP pairs whose RECORD address is
empty (section 2's "FP-predicted S1 contains W" stat only).

1) Grammar: each true record's dict-remapped core tokens R vs its true S1's S, exactly one class (first match,
   in this order): core_identical (R==S); s1_prefix_k, k=1..3 (R == S[:k] and len(R)==k); s1_prefix_k_plus_W
   (R[:k]==S[:k], k=1..3, plus >=1 trailing token(s) none of which occur anywhere in S); reorder (same multiset,
   different order); subset (token set of R proper subset of S's, not a prefix case); superset (S's proper
   subset of R's, not a prefix+W case); pseudo (no shared token and char-4gram cosine(core text) < 0.3,
   mechanisms.pseudo_brand's rule); native (record raw name/address has a non-ASCII char, i.e. is_nonascii_raw);
   domain (record raw name matches mechanisms.DOMAIN_RE); other.
2) Extra words W = tokens present in a true record's core but absent from its true S1's core (any class, not
   just s1_prefix_k_plus_W). Top 30 W by count among addr_empty true records: count, rate among addr_empty true
   records, rate among all true records (same "is an extra word of this record" test), and how often the
   record's FP-predicted S1 (its kept-argmax wrong prediction, if it has one) contains W in its own core.
3) Ambiguity: per addr_empty true record, # same-country S1s in the sample whose dict-remapped core starts with
   the record's first core token, and separately with its first two tokens. Buckets 1 / 2-5 / 6-20 / 21+: n and
   loss pts, for each prefix length.
4) Rule test (no label used in the score; W mined on a disjoint 50% split of sample S1s from the 50% scored):
   candidates for a record = same-country sample S1s whose core starts with the record's first core token.
   score(S1) = (largest k, 0..len(R), with R[:k] == S1_core[:k]) - lambda * (# of the record's W, from the
   OTHER split's mining, present anywhere in the S1's core), lambda in {0, 0.5, 1}. Top-1 = argmax score;
   ties broken arbitrarily for the reported tie rate, then by lowest S1 code for the "untied" and example rows.
   Reported for FN and miss addr_empty records in the scored split only.
5) 20 raw examples each for FN and miss (addr_empty, scored split, first 20 by reck): record business_name |
   true S1 business_name | top competing S1 business_name (lambda=1 rule, ties -> lowest S1 code).

Run from code/business_entity_resolution/:
  python -m src.addr_empty --smoke   # 1%% S1 sample (mechanisms.FRAC path), same output incl. docs/addr_empty.md (< 3 min)
  python -m src.addr_empty           # 20%% sample -> docs/addr_empty.md
Aborts if peak RSS > MAX_RSS_MB (10 GB).
"""
import argparse
import re
from collections import Counter

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer

from .decide import md
from .features import id_key, load_token_dict
from .io import ROOT, StepLog, load_gt_pairs, path, peak_rss_mb
from .mechanisms import DOMAIN_RE, NORM, SEED, T, load, sample_pairs, side, tag_exprs, world

MAX_RSS_MB = 10_000  # 24 GB machine
REPORT = ROOT / "docs" / "addr_empty.md"
KMAX = 3
TOPW = 30
NEX = 20
JBINS = [(1, 1), (2, 5), (6, 20), (21, None)]
LAMBDAS = (0.0, 0.5, 1.0)
DOMAIN_PY = re.compile(DOMAIN_RE)


def guard(step: str) -> None:
    if (rss := peak_rss_mb()) > MAX_RSS_MB:
        raise SystemExit(f"ABORT src.addr_empty: peak RSS {rss} MB > {MAX_RSS_MB} MB at '{step}'.")


class Log(StepLog):
    def __call__(self, step: str, **kw) -> None:
        super().__call__(step, **kw)
        guard(step)


def remap(toks: list[str], dm: dict) -> list[str]:
    return [dm.get(t, t) for t in toks] if dm else list(toks)


def cos4(a: list[str], b: list[str], hv: HashingVectorizer) -> np.ndarray:
    A, B = hv.transform([" ".join(x) for x in a]), hv.transform([" ".join(x) for x in b])
    na = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
    nb = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel())
    return np.asarray(A.multiply(B).sum(1)).ravel() / (na * nb + 1e-9)


def classify_row(r: list[str], s: list[str]) -> tuple[str, list[str]]:
    if r == s:
        return "core_identical", []
    rn = len(r)
    best_k = 0
    for k in range(1, min(KMAX, rn) + 1):
        if r[:k] == s[:k]:
            best_k = k
        else:
            break
    if best_k:
        if rn == best_k:
            return f"s1_prefix_{best_k}", []
        extra = r[best_k:]
        if not (set(extra) & set(s)):
            return f"s1_prefix_{best_k}_plus_W", extra
        # longest matching prefix has colliding trailing tokens -> not a prefix(+W) case; fall through
    if rn > 1 and sorted(r) == sorted(s):
        return "reorder", []
    rs, ss = set(r), set(s)
    if rs and rs < ss:
        return "subset", []
    if ss and ss < rs:
        return "superset", []
    if not (rs & ss):
        return "_pseudo_check", []
    return "_other_check", []


def classify(R: list[list[str]], S: list[list[str]], nonascii: list[bool], names: list[str],
             hv: HashingVectorizer) -> tuple[list[str], list[list[str]]]:
    cls, extra = [], []
    pending = []
    for i, (r, s) in enumerate(zip(R, S)):
        c, w = classify_row(r, s)
        cls.append(c)
        extra.append(w)
        if c == "_pseudo_check":
            pending.append(i)
    if pending:
        cos = cos4([R[i] for i in pending], [S[i] for i in pending], hv)
        for j, i in enumerate(pending):
            cls[i] = "pseudo" if cos[j] < 0.3 else "_other_check"
    for i, c in enumerate(cls):
        if c != "_other_check":
            continue
        if nonascii[i]:
            cls[i] = "native"
        elif DOMAIN_PY.search(names[i] or ""):
            cls[i] = "domain"
        else:
            cls[i] = "other"
    return cls, extra


def extra_words(r: list[str], s: list[str]) -> set[str]:
    return set(r) - set(s)


def leading_match(r: list[str], s: list[str]) -> int:
    k = 0
    for a, b in zip(r, s):
        if a != b:
            break
        k += 1
    return k


def bucket(n: int) -> str:
    for lo, hi in JBINS:
        if hi is None:
            if n >= lo:
                return f"{lo}+"
        elif lo <= n <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return "0"


def build_population(log: Log):
    """oof/s1/code/kept + tagged true-pair frame TRUE (status != FP) and FP_AE (status==FP, addr_empty),
    both carrying r_/s_ raw+normalised fields and dict-remapped core token lists."""
    oof, s1, code, kept, in_cand, cnt, f, L = world(log)
    gt = load_gt_pairs().select(s1k=id_key("s1_id"), reck=id_key("match_id"))
    gtw = gt.join(code, on="s1k", how="semi").unique()
    P, samp = sample_pairs(s1, kept, in_cand, gt, cnt, f, L, log)
    del in_cand
    s_raw = load(NORM[0], P["s1k"].unique(), "s1k", **side("s_"))
    r_raw = load(NORM[1:], P["reck"].unique(), "reck", **side("r_"), r_nonascii=pl.col("is_nonascii_raw"))
    X = P.join(s_raw, on="s1k").join(r_raw, on="reck").join(s1.select("code", ctry="country"), on="code")
    X = X.with_columns(ae=tag_exprs()["addr_empty"])
    log("joined", pairs=X.height)
    dm = load_token_dict() or {}
    R = [remap(t, dm) for t in X["r_nt"].to_list()]
    S = [remap(t, dm) for t in X["s_nt"].to_list()]
    X = X.with_columns(R=pl.Series(R, dtype=pl.List(pl.String)), S=pl.Series(S, dtype=pl.List(pl.String)))
    log("remapped", dict_size=len(dm))
    TRUE = X.filter(pl.col("status") != "FP")
    FP_AE = X.filter((pl.col("status") == "FP") & pl.col("ae"))
    return s1, samp, dm, TRUE, FP_AE, X.height


def sec1(TRUE: pl.DataFrame, log: Log) -> tuple[list[str], pl.DataFrame]:
    hv = HashingVectorizer(analyzer="char", ngram_range=(4, 4), n_features=2 ** 18, alternate_sign=False,
                           dtype=np.float32)
    R, S = TRUE["R"].to_list(), TRUE["S"].to_list()
    cls, extra = classify(R, S, TRUE["r_nonascii"].to_list(), TRUE["r_n"].to_list(), hv)
    TRUE = TRUE.with_columns(cls=pl.Series(cls, dtype=pl.String),
                             extra=pl.Series(extra, dtype=pl.List(pl.String)))
    log("classified", n=TRUE.height)
    order = (["core_identical"] + [f"s1_prefix_{k}" for k in range(1, KMAX + 1)]
             + [f"s1_prefix_{k}_plus_W" for k in range(1, KMAX + 1)]
             + ["subset", "superset", "reorder", "pseudo", "native", "domain", "other"])
    n = TRUE.height
    rows = []
    for c in order:
        m = TRUE.filter(pl.col("cls") == c)
        if m.height == 0:
            rows.append({"class": c, "n": 0, "%": 0.0, "n TP": 0, "n FN": 0, "n miss": 0, "loss pts": 0.0})
            continue
        st = m["status"].value_counts().to_dicts()
        stc = {r["status"]: r["count"] for r in st}
        rows.append({"class": c, "n": m.height, "%": 100 * m.height / n, "n TP": stc.get("TP", 0),
                     "n FN": stc.get("FN", 0), "n miss": stc.get("miss", 0), "loss pts": float(m["w"].sum())})
    return (["## 1. Grammar of true records vs their true S1", "",
             f"n true pairs (TP/FN/miss) = {n:,}. Exactly one class per record (first-match order above).", "",
             *md(rows), ""], TRUE)


def sec2(TRUE: pl.DataFrame, FP_AE: pl.DataFrame, log: Log) -> list[str]:
    ae = TRUE.filter(pl.col("ae"))
    n_ae, n_all = ae.height, TRUE.height
    extra_ae = [extra_words(r, s) for r, s in zip(ae["R"].to_list(), ae["S"].to_list())]
    extra_all = [extra_words(r, s) for r, s in zip(TRUE["R"].to_list(), TRUE["S"].to_list())]
    cnt_ae = Counter(w for e in extra_ae for w in e)
    top = [w for w, _ in cnt_ae.most_common(TOPW)]
    cnt_all = Counter(w for e in extra_all for w in e if w in set(top))
    # FP-predicted S1 per record (addr_empty records only, at most one kept-argmax row per reck)
    fp_map = dict(zip(FP_AE["reck"].to_list(), FP_AE["S"].to_list()))
    ae_reck = ae["reck"].to_list()
    fp_hit = Counter()
    fp_n = Counter()
    for reck, e in zip(ae_reck, extra_ae):
        s1core = fp_map.get(reck)
        if s1core is None:
            continue
        s1set = set(s1core)
        for w in e:
            if w in set(top):
                fp_n[w] += 1
                if w in s1set:
                    fp_hit[w] += 1
    rows = [{"W": w, "count (addr_empty)": cnt_ae[w], "rate addr_empty %": 100 * cnt_ae[w] / n_ae,
             "rate all true %": 100 * cnt_all.get(w, 0) / n_all,
             "FP-S1 contains W %": 100 * fp_hit[w] / fp_n[w] if fp_n[w] else float("nan"),
             "n FP records w/ W": fp_n[w]} for w in top]
    log("sec2", n_ae=n_ae, n_fp_ae=FP_AE.height, top_w=len(top))
    return ["## 2. Top extra words W (record core, not in true S1's core)", "",
            f"n addr_empty true records = {n_ae:,} (of {n_all:,} true records). FP population (record address "
            f"empty, kept-argmax wrong S1) = {FP_AE.height:,} pairs; 'FP-S1 contains W' is over the subset of "
            "addr_empty true records whose record also has such an FP row.", "", *md(rows), ""]


def sample_s1_pool(s1: pl.DataFrame, samp: np.ndarray, dm: dict, with_name: bool = False) -> pl.DataFrame:
    """code, country, dict-remapped core tokens S (and business_name if with_name) for every sampled S1."""
    samp_s1 = s1.filter(pl.Series(samp)).select("code", "country", "s1k")
    cols = {"s_nt": pl.col("name_tokens"), **({"s_n": pl.col("business_name")} if with_name else {})}
    s_raw = load(NORM[0], samp_s1["s1k"], "s1k", **cols)
    samp_s1 = samp_s1.join(s_raw, on="s1k")
    return samp_s1.with_columns(S=pl.Series([remap(t, dm) for t in samp_s1["s_nt"].to_list()],
                                            dtype=pl.List(pl.String)))


def sec3(s1: pl.DataFrame, samp: np.ndarray, ae: pl.DataFrame, dm: dict, log: Log) -> list[str]:
    samp_s1 = sample_s1_pool(s1, samp, dm)
    by_country: dict[str, list[list[str]]] = {}
    for c, s in zip(samp_s1["country"].to_list(), samp_s1["S"].to_list()):
        by_country.setdefault(c, []).append(s)
    log("sec3 s1 pool", n_s1=samp_s1.height, countries=len(by_country))

    out_rows = {1: Counter(), 2: Counter()}
    loss_rows = {1: Counter(), 2: Counter()}
    for ctry, r, w in zip(ae["ctry"].to_list(), ae["R"].to_list(), ae["w"].to_list()):
        pool = by_country.get(ctry, [])
        for plen in (1, 2):
            if len(r) < plen:
                continue
            pre = r[:plen]
            n_match = sum(1 for s in pool if len(s) >= plen and s[:plen] == pre)
            b = bucket(n_match)
            out_rows[plen][b] += 1
            loss_rows[plen][b] += w
    log("sec3 counted", n_ae=ae.height)
    lines = ["## 3. Ambiguity: same-country S1s sharing the record's leading core token(s)", "",
             f"n addr_empty true records = {ae.height:,}.", ""]
    order = [f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}") for lo, hi in JBINS]
    for plen, label in ((1, "first core token"), (2, "first two core tokens")):
        rows = [{"# matching S1s": b, "n": out_rows[plen][b], "loss pts": loss_rows[plen][b]} for b in order]
        lines += [f"### Prefix length {plen} ({label})", "", *md(rows), ""]
    return lines


def rule_test(recs: pl.DataFrame, pool_by_country: dict, w_by_lambda: dict, log: Log, label: str) -> tuple[list[str], dict]:
    """recs: reck, ctry, R, S(true), status, w, name. pool_by_country[ctry] = list of (code, S1 core tokens)."""
    results = {lam: {"FN": [], "miss": []} for lam in LAMBDAS}
    examples = {}  # (status) -> list of (reck, name, true_name, top_name)
    for row in recs.iter_rows(named=True):
        ctry, r, s_true, status = row["ctry"], row["R"], row["S"], row["status"]
        pool = pool_by_country.get(ctry, [])
        cands = [(code, s) for code, s in pool if len(s) and s[0] == r[0]] if r else []
        if not cands:
            continue
        Wset = w_by_lambda.get(row["reck"], set())
        scores = []
        for code, s in cands:
            base = leading_match(r, s)
            pen = len(Wset & set(s))
            scores.append((code, s, base, pen))
        true_in_cands = any(s == s_true for _, s, _, _ in scores)
        for lam in LAMBDAS:
            sc = [(code, s, base - lam * pen) for code, s, base, pen in scores]
            best = max(x[2] for x in sc)
            top = [x for x in sc if x[2] == best]
            tied = len(top) > 1
            top_sorted = sorted(top, key=lambda x: x[0])
            picked = top_sorted[0]
            correct = picked[1] == s_true
            results[lam][status].append((correct, tied))
            if lam == 1.0 and status in ("FN", "miss") and len(examples.setdefault(status, [])) < NEX:
                examples[status].append((row["reck"], row["r_n"], row["true_name"], picked, ctry))
    return [], (results, examples)


def sec4_5(TRUE: pl.DataFrame, samp_s1_by_country: dict, ae: pl.DataFrame, dm: dict, s1code_name: dict,
           log: Log) -> list[str]:
    # 50/50 split of sampled S1s by s1k hash (deterministic, seeded)
    codes = ae["code"].unique().to_list()
    rng = np.random.default_rng(SEED + 7)
    codes_arr = np.array(codes)
    perm = rng.permutation(len(codes_arr))
    half = len(codes_arr) // 2
    split_a = set(codes_arr[perm[:half]].tolist())
    ae2 = ae.with_columns(split=pl.col("code").is_in(split_a))
    log("sec4 split", n_s1=len(codes), split_a=len(split_a))

    def mine_w(df: pl.DataFrame) -> Counter:
        ex = [extra_words(r, s) for r, s in zip(df["R"].to_list(), df["S"].to_list())]
        return Counter(w for e in ex for w in e)

    w_a = mine_w(ae2.filter(pl.col("split")))
    w_b = mine_w(ae2.filter(~pl.col("split")))
    top_a = set(w for w, _ in w_a.most_common(TOPW))
    top_b = set(w for w, _ in w_b.most_common(TOPW))

    def score_split(scored_df: pl.DataFrame, w_from_other: set[str], name: str) -> tuple[pl.DataFrame, dict]:
        w_by_rec = {reck: w_from_other for reck in scored_df["reck"].to_list()}
        recs = scored_df.filter(pl.col("status").is_in(["FN", "miss"])).join(
            TRUE.select("reck", true_name="s_n"), on="reck")
        _, (results, examples) = rule_test(recs, samp_s1_by_country, w_by_rec, log, name)
        return recs, (results, examples)

    _, (results_b, examples_b) = score_split(ae2.filter(~pl.col("split")), top_a, "B scored w/ W(A)")
    log("sec4 scored B")
    _, (results_a, examples_a) = score_split(ae2.filter(pl.col("split")), top_b, "A scored w/ W(B)")
    log("sec4 scored A")

    def combine(key: str) -> dict[str, list]:
        return {st: results_a[key][st] + results_b[key][st] for st in ("FN", "miss")}

    lines = ["## 4. Rule test: leading-token match minus extra-word penalty", "",
             "Candidates = same-country sample S1s sharing the record's first core token. W mined on one 50% "
             "split of addr_empty true records' S1s, scored on the other (both directions pooled below).", ""]
    rows = []
    for lam in LAMBDAS:
        c = combine(lam)
        for st in ("FN", "miss"):
            vals = c[st]
            n = len(vals)
            if n == 0:
                rows.append({"lambda": lam, "status": st, "n": 0, "top-1 acc %": float("nan"),
                             "% tied": float("nan"), "top-1 acc | untied %": float("nan")})
                continue
            correct = sum(v[0] for v in vals)
            tied = sum(v[1] for v in vals)
            untied = [v for v in vals if not v[1]]
            u_correct = sum(v[0] for v in untied)
            rows.append({"lambda": lam, "status": st, "n": n, "top-1 acc %": 100 * correct / n,
                         "% tied": 100 * tied / n,
                         "top-1 acc | untied %": 100 * u_correct / len(untied) if untied else float("nan")})
    lines += [*md(rows), ""]

    ex_all = {"FN": examples_a.get("FN", []) + examples_b.get("FN", []),
              "miss": examples_a.get("miss", []) + examples_b.get("miss", [])}
    lines += ["## 5. Raw examples (addr_empty, lambda=1 top competitor)", ""]
    for st in ("FN", "miss"):
        lines += [f"### {st}", "", "| record name | true S1 name | top competing S1 name |", "|---|---|---|"]
        for reck, rname, tname, picked, ctry in ex_all[st][:NEX]:
            comp_name = s1code_name.get(picked[0], "?")
            lines.append(f"| {rname} | {tname} | {comp_name} |")
        lines.append("")
    return lines


def main() -> None:
    import src.mechanisms as M
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1%% S1 sample, same path incl. docs/addr_empty.md")
    smoke = ap.parse_args().smoke
    if smoke:
        M.FRAC = 0.01
    log = Log()
    s1, samp, dm, TRUE, FP_AE, npairs = build_population(log)
    sec1_lines, TRUE = sec1(TRUE, log)
    ae = TRUE.filter(pl.col("ae"))
    sec2_lines = sec2(TRUE, FP_AE, log)
    sec3_lines = sec3(s1, samp, ae, dm, log)

    samp_s1 = sample_s1_pool(s1, samp, dm, with_name=True)
    pool_by_country: dict[str, list] = {}
    s1code_name = dict(zip(samp_s1["code"].to_list(), samp_s1["s_n"].to_list()))
    for c, code, s in zip(samp_s1["country"].to_list(), samp_s1["code"].to_list(), samp_s1["S"].to_list()):
        pool_by_country.setdefault(c, []).append((code, s))
    log("pool built", n_s1=samp_s1.height)

    sec45_lines = sec4_5(TRUE, pool_by_country, ae, dm, s1code_name, log)

    n = TRUE.height
    head = ["# addr_empty: grammar, ambiguity and a lexical rule for the top loss tag"
            + (" [SMOKE: 1% sample]" if smoke else ""), "",
            f"Generated by `src/addr_empty.py` (definitions in its docstring). World: cv_full (b) via "
            f"src.mechanisms, t = {T}. True pairs (TP/FN/miss) in the sample: {n:,}. FP pairs with record "
            f"address empty: {FP_AE.height:,}.", ""]
    REPORT.write_text("\n".join(head + sec1_lines + sec2_lines + sec3_lines + sec45_lines
                                + [f"Peak RSS: {peak_rss_mb()} MB.", ""]))
    log("report", path=str(REPORT))
    log_path = path("artifacts_dir") / "logs" / f"addr_empty{'_smoke' if smoke else ''}_timing.json"
    log.dump(log_path)


if __name__ == "__main__":
    main()
