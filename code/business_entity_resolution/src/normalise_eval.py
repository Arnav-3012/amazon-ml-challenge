"""M3a step 3: evaluate normaliser v1 on train -> docs/normalise.md.

Works from the src.normalise caches (they keep the raw columns), so 'before' and every ablation are
recomputed from them without re-reading TSVs. Sections: recall proxy, precision guard, per-rule ablation,
40 before/after examples, runtime/memory.
Cost: one all-rules-off pass over train (12.5M rows, no anyascii, pure polars) + len(RULES)+2 ablation
passes over S1 (2.2M) and a <=400k S2/S3 subset. Run from code/business_entity_resolution/:
  python -m src.normalise_eval
"""
import json
import time

import polars as pl

from .io import ROOT, load_gt_pairs, path
from .normalise import ADDR_RULES, RULES, normalise, peak_rss_mb

SEED = 42
N_ABL_PAIRS = 200_000
N_ABL_RANDOM = 200_000
OUT = ROOT / "docs" / "normalise.md"
RAW = ["entity_id", "business_name", "business_address", "country"]
ALL_OFF = frozenset(RULES)
CACHE_COLS = [*RAW, "core_name", "legal_form", "addr_number", "addr_street_core", "is_nonascii_raw"]
JCOLS = {"name": ("n1", "n2"), "name_legal": ("l1", "l2"), "addr": ("a1", "a2")}


def cache(split: str, n: int, cols: list[str] | None = CACHE_COLS) -> pl.DataFrame:
    return pl.read_parquet(path("interim_dir") / f"norm_{split}_s{n}.parquet", columns=cols)


def keys(df: pl.DataFrame, raw_name: bool = False) -> pl.DataFrame:
    """Equality keys: name (core_name, or raw lowercased), name_legal, addr = number|street ('' if no number)."""
    name = pl.col("business_name").str.to_lowercase().str.strip_chars() if raw_name else pl.col("core_name")
    return df.select("entity_id", "country", "is_nonascii_raw", name=name,
                     name_legal=pl.col("core_name") + "|" + pl.col("legal_form"),
                     addr=pl.when(pl.col("addr_number") != "")
                     .then(pl.col("addr_number") + "|" + pl.col("addr_street_core")).otherwise(pl.lit("")))


def pair_join(pairs: pl.DataFrame, k1: pl.DataFrame, km: pl.DataFrame) -> pl.DataFrame:
    j = (pairs.join(k1.select("entity_id", "country", n1="name", l1="name_legal", a1="addr"),
                    left_on="s1_id", right_on="entity_id")
         .join(km.select("entity_id", "is_nonascii_raw", n2="name", l2="name_legal", a2="addr"),
               left_on="match_id", right_on="entity_id"))
    eq = {k: (pl.col(x) != "") & (pl.col(x) == pl.col(y)) for k, (x, y) in JCOLS.items()}
    return j.with_columns(name_eq=eq["name"], addr_eq=eq["addr"], either=eq["name"] | eq["addr"])


def recall_by_group(j: pl.DataFrame) -> pl.DataFrame:
    j = j.with_columns(grp=pl.when("is_nonascii_raw").then(pl.lit("non-ASCII")).otherwise(pl.lit("ASCII")))
    return (pl.concat([j, j.with_columns(grp=pl.lit("all"))])
            .group_by("country", "grp")
            .agg(pairs=pl.len(), **{m: pl.col(f"{m}_eq" if m != "either" else m).mean() * 100
                                    for m in ("name", "addr", "either")})
            .sort("country", "grp"))


def s1_collision(k1: pl.DataFrame, key: str) -> pl.DataFrame:
    """% of S1 records (per country, plus pooled) whose non-empty key is shared with another S1 record."""
    d = k1.with_columns(dup=(pl.col(key) != "") & (pl.len().over("country", key) > 1))
    per = d.group_by("country").agg(pct=pl.col("dup").mean() * 100)
    return pl.concat([per, pl.DataFrame({"country": ["pooled"], "pct": [d["dup"].mean() * 100]})])


def nonpair_ppm(k1: pl.DataFrame, km: pl.DataFrame, j: pl.DataFrame, key: str) -> pl.DataFrame:
    """Exact same-country NON-pair key-collision rate (per million pairs), from group counts, no sampling:
    (sum_k c1(k)*cm(k) - true pairs with equal key) / (N1*Nm - true pairs)."""
    x, y = JCOLS[key]
    cnt = lambda k, c: (k.filter(pl.col(key) != "").group_by("country", key)  # noqa: E731
                        .agg(pl.len().cast(pl.Int64).alias(c)))
    same = (cnt(k1, "c1").join(cnt(km, "c2"), on=["country", key])
            .group_by("country").agg(same=(pl.col("c1") * pl.col("c2")).sum()))
    teq = (j.filter((pl.col(x) != "") & (pl.col(x) == pl.col(y)))
           .group_by("country").agg(teq=pl.len().cast(pl.Int64)))
    size = lambda k, c: k.group_by("country").agg(pl.len().cast(pl.Int64).alias(c))  # noqa: E731
    d = (size(k1, "N1").join(size(km, "N2"), on="country")
         .join(j.group_by("country").agg(t=pl.len().cast(pl.Int64)), on="country", how="left")
         .join(same, on="country", how="left").join(teq, on="country", how="left").fill_null(0)
         .with_columns(num=pl.col("same") - pl.col("teq"), den=pl.col("N1") * pl.col("N2") - pl.col("t")))
    pooled = d.select(country=pl.lit("pooled"), num=pl.col("num").sum(), den=pl.col("den").sum())
    return pl.concat([d.select("country", "num", "den"), pooled]).select(
        "country", ppm=pl.col("num") / pl.col("den") * 1e6)


def by_country(df: pl.DataFrame, col: str) -> dict[str, float]:
    return dict(zip(df["country"], df[col]))


def ablation(s1: pl.DataFrame, m: pl.DataFrame, pairs: pl.DataFrame) -> list[dict]:
    samp = pairs.sample(N_ABL_PAIRS, seed=SEED)
    rand_ids = m["entity_id"].sample(N_ABL_RANDOM, seed=SEED)
    sub = m.filter(pl.col("entity_id").is_in(samp["match_id"]) | pl.col("entity_id").is_in(rand_ids)).select(RAW)
    s1raw = s1.select(RAW)
    pairs_rand = pairs.filter(pl.col("match_id").is_in(rand_ids))
    runs = [("all rules on", frozenset()), *[(r, frozenset({r})) for r in RULES], ("all rules off", ALL_OFF)]
    rows = []
    for label, off in runs:
        t0 = time.perf_counter()
        k1, ks = keys(normalise(s1raw, off)), keys(normalise(sub, off))
        js = pair_join(samp, k1, ks)
        kr = ks.filter(pl.col("entity_id").is_in(rand_ids))
        jr = pair_join(pairs_rand, k1, kr)
        pooled = lambda df, col: by_country(df, col)["pooled"]  # noqa: E731
        rows.append({"label": label, "rec_name": js["name_eq"].mean() * 100, "rec_addr": js["addr_eq"].mean() * 100,
                     "col_name": pooled(s1_collision(k1, "name"), "pct"),
                     "col_addr": pooled(s1_collision(k1, "addr"), "pct"),
                     "ppm_name": pooled(nonpair_ppm(k1, kr, jr, "name"), "ppm"),
                     "ppm_addr": pooled(nonpair_ppm(k1, kr, jr, "addr"), "ppm")})
        print(f"ablation {label}: {time.perf_counter() - t0:.1f}s", flush=True)
    return rows


def examples() -> pl.DataFrame:
    """40 random records: quotas per (split, country, non-ASCII); 10 non-ASCII. France is test-only."""
    quotas = [("train", "US", False, 10), ("train", "US", True, 3), ("train", "India", False, 9),
              ("train", "India", True, 5), ("test", "France", False, 11), ("test", "France", True, 2)]
    cols = ["country", "business_name", "core_name", "legal_form", "aliases", "business_address",
            "addr_number", "addr_street_core", "admin_region", "city"]
    out = []
    for split, c, nonascii, k in quotas:
        lf = pl.concat([pl.scan_parquet(path("interim_dir") / f"norm_{split}_s{n}.parquet")
                        .with_columns(src=pl.lit(f"{split} S{n}")) for n in (1, 2, 3)])
        hit = (pl.col("country") == c) & (pl.col("is_nonascii_raw") == nonascii) & (pl.col("entity_id").hash(SEED) % 997 == 0)
        out.append(lf.filter(hit).select("src", *cols).collect().sample(k, seed=SEED))
    return pl.concat(out)


def cell(v) -> str:
    s = ", ".join(v) if isinstance(v, list) else str(v)
    return s.replace("|", "\\|")


def main() -> None:
    t_start = time.perf_counter()
    pairs = load_gt_pairs()
    s1 = cache("train", 1)
    m = pl.concat([cache("train", 2), cache("train", 3)])
    ka1, kam = keys(s1), keys(m)
    kb1 = keys(normalise(s1.select(RAW), ALL_OFF), raw_name=True)
    kbm = keys(normalise(m.select(RAW), ALL_OFF), raw_name=True)
    ja, jb = pair_join(pairs, ka1, kam), pair_join(pairs, kb1, kbm)
    print(f"joins done {time.perf_counter() - t_start:.0f}s", flush=True)

    L = ["# Normaliser v1 evaluation (M3a)",
         "Generated by `src/normalise_eval.py` from `artifacts/interim/norm_*.parquet`, seed 42. Train only "
         "(labels); France appears only in the examples (test).",
         "",
         "- **before** = name: raw lowercased + stripped; address: lowercase + punct strip + component parse "
         "with every rule off. **after** = all rules on.",
         "- name_eq = exact core_name equality (non-empty). addr_eq = exact (addr_number, addr_street_core) "
         "equality (number non-empty). either = name_eq OR addr_eq.",
         f"- Rule order: {' → '.join(RULES)} (lowercase, punct/whitespace strip and the component parse "
         "always run; digit_fix runs after punct strip because token boundaries need it).",
         "", "## 1. Recall proxy (all train true pairs; non-ASCII = the matched S2/S3 record)",
         "| country | match script | pairs | name_eq % before → after | addr_eq % before → after | either % before → after |",
         "|---|---|---|---|---|---|"]
    t = recall_by_group(ja).join(recall_by_group(jb), on=["country", "grp"], suffix="_b")
    for r in t.iter_rows(named=True):
        L.append(f"| {r['country']} | {r['grp']} | {r['pairs']:,} | {r['name_b']:.2f} → {r['name']:.2f} | "
                 f"{r['addr_b']:.2f} → {r['addr']:.2f} | {r['either_b']:.2f} → {r['either']:.2f} |")

    L += ["", "## 2. Precision guard (S1 is de-duplicated: a jump = a rule merging distinct businesses)",
          "S1 collision = % of S1 records sharing their key with another S1 record in the same country. "
          "Non-pair ppm = exact collisions per million same-country S1×(S2∪S3) non-pairs (all pairs, from "
          "group counts). core+legal = core_name|legal_form (legal form kept as its own column).",
          "| country | S1 collision name % before → after | after, core+legal % | non-pair ppm name before → after "
          "| after, core+legal ppm | S1 collision addr % before → after | non-pair ppm addr before → after |",
          "|---|---|---|---|---|---|---|"]
    cb_n, ca_n, ca_l = (by_country(s1_collision(k, c), "pct") for k, c in ((kb1, "name"), (ka1, "name"), (ka1, "name_legal")))
    cb_a, ca_a = (by_country(s1_collision(k, "addr"), "pct") for k in (kb1, ka1))
    pb_n, pa_n = (by_country(nonpair_ppm(k1, km, j, "name"), "ppm") for k1, km, j in ((kb1, kbm, jb), (ka1, kam, ja)))
    pa_l = by_country(nonpair_ppm(ka1, kam, ja, "name_legal"), "ppm")
    pb_a, pa_a = (by_country(nonpair_ppm(k1, km, j, "addr"), "ppm") for k1, km, j in ((kb1, kbm, jb), (ka1, kam, ja)))
    for c in sorted(ca_n, key=lambda x: (x == "pooled", x)):
        L.append(f"| {c} | {cb_n[c]:.2f} → {ca_n[c]:.2f} | {ca_l[c]:.2f} | {pb_n[c]:.3f} → {pa_n[c]:.3f} | "
                 f"{pa_l[c]:.3f} | {cb_a[c]:.2f} → {ca_a[c]:.2f} | {pb_a[c]:.3f} → {pa_a[c]:.3f} |")
    del ja, jb, kb1, kbm, kam

    rows = ablation(s1, m, pairs)
    on = rows[0]
    L += ["", f"## 3. Per-rule ablation ({N_ABL_PAIRS:,} sampled pairs for recall; full S1 for S1 collision; "
          f"S1 × {N_ABL_RANDOM:,} random S2∪S3 records for non-pair ppm; pooled over countries)",
          "Δ = (all rules on) − (this rule off), percentage points. Name rules are judged on name metrics, "
          "address rules on address metrics. ⚠ = precision cost (Δ S1 collision) > recall gain (Δ recall), "
          "or the rule lowers recall.",
          "| rule off | recall name % | recall addr % | S1 coll. name % | S1 coll. addr % | ppm name | ppm addr "
          "| Δ recall | Δ S1 coll. | flag |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = "addr" if r["label"] in ADDR_RULES else "name"
        gain, cost = on[f"rec_{f}"] - r[f"rec_{f}"], on[f"col_{f}"] - r[f"col_{f}"]
        flag = "⚠" if r["label"] in RULES and (cost > gain or gain < 0) else ""
        delta = f"{gain:+.3f} | {cost:+.3f}" if r["label"] in RULES else "– | –"
        L.append(f"| {r['label']} | {r['rec_name']:.2f} | {r['rec_addr']:.2f} | {r['col_name']:.3f} | "
                 f"{r['col_addr']:.3f} | {r['ppm_name']:.3f} | {r['ppm_addr']:.3f} | {delta} | {flag} |")

    ex = examples()
    L += ["", "## 4. 40 random before/after examples (10 non-ASCII; France from test)",
          "| src | country | raw name | core_name | legal | aliases | raw address | no. | street core | admin | city |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(cell(r[c]) for c in ex.columns) + " |" for r in ex.iter_rows(named=True)]

    timing = json.loads((path("interim_dir") / "norm_timing.json").read_text())
    L += ["", "## 5. Runtime + memory (`python -m src.normalise`, all 6 source files)",
          "peak RSS = process high-water mark after that file (cumulative, monotone).",
          "| file | rows | load s | normalise s | total s | peak RSS MB |", "|---|---|---|---|---|---|"]
    L += [f"| {t['file']} | {t['rows']:,} | {t['load_s']} | {t['normalise_s']} | {t['total_s']} | {t['peak_rss_mb']} |"
          for t in timing]
    L += [f"- Total: {sum(t['total_s'] for t in timing):.0f}s. This eval: "
          f"{time.perf_counter() - t_start:.0f}s, peak RSS {peak_rss_mb()} MB."]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
