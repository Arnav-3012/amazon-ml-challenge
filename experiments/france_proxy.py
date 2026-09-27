"""B1 France proxy (no test labels): French-style noise on a labelled US/India holdout, scored with B (and F).

Holdout H = 20,000 fold-0 S1s (seed 42), standard world (no S1 dropout). Noise is applied to the RAW text of the
H S1s and of their true records, then re-normalised (src.normalise) and re-featurised (src.features functions):
  legal     name-final legal suffix -> a French form (S1 random; each record: same 60% / other 25% / none 15%)
  street    street types -> rue/route/avenue/allee/boulevard/chemin (records abbreviate half: R./rte/av./bd/ch.)
  cedex     20% of records get ", Cedex 0N"
  accents   30% of records get e -> é in their name (anyascii should undo it)
  et        " & " / " and " -> " et " in half the records
  shared    8.4% of H S1s (pairs, same country) take another H S1's address, and so do their true records, which lifts
            exact-address sharing to about France's 13.4%
Rows = every candidate row of every record that touches H (so record-side features are complete). S1-side relative
features (*_ds1, *_rks1, n_cand_s1) of competitor S1s outside H come from the stored features (partial otherwise).
FIXED: the candidate set and its blocking scores/ranks (a matcher-robustness proxy, not a blocking proxy).
Scored with the fold models of each row's S1 (OOF-valid), then stage 2 (context + coref from the noised texts).
Sanity: with noise off, rebuilt features must equal the stored ones (asserted on the pair columns).
Run from repo root:  .venv/bin/python experiments/france_proxy.py [--models models_krish] [--oof oof_krish] [--n 20000]
-> experiments/france_proxy_<models>.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.block import RANK_COLS, norm_path  # noqa: E402
from src.cv_full import SEED, s1_table  # noqa: E402
from src.decide import f05_vec  # noqa: E402
from src.features import (ENT_COLS, REL_COLS, id_key, load_token_dict, pair_features, prep, relative,  # noqa: E402
                          s1_addr_dup, s1_name_dup, vocab)
from src.io import load_gt_pairs, path  # noqa: E402
from src.normalise import normalise  # noqa: E402
from src.stage2 import context, coref  # noqa: E402
from src.train import parts  # noqa: E402

FR_LEGAL = ["SARL", "SAS", "SA", "EURL", "SNC", "SASU"]
LEGAL_RE = re.compile(r"(?i)[\s,]+(private limited|pvt\.? ?ltd\.?|limited|ltd\.?|llc|l\.l\.c\.|inc\.?|incorporated|"
                      r"corp\.?|corporation|llp|pllc|co\.)\s*$")
STREET = {"street": ("rue", "R."), "st": ("rue", "R."), "road": ("route", "rte"), "rd": ("route", "rte"),
          "avenue": ("avenue", "av."), "ave": ("avenue", "av."), "lane": ("allée", "all."), "ln": ("allée", "all."),
          "boulevard": ("boulevard", "bd"), "blvd": ("boulevard", "bd"), "drive": ("chemin", "ch."),
          "dr": ("chemin", "ch.")}
STREET_RE = re.compile(r"(?i)\b(" + "|".join(STREET) + r")\b\.?")
PAIR_CHECK = ["name_tset", "addr_tset", "name_idfj", "addr_idfj", "num_code", "legal_code"]


OPS = ("legal", "street", "cedex", "accents", "et", "shared")


def frenchify(names: list[str], addrs: list[str], rng: np.random.Generator, record: bool,
              s1_legal: list[str | None], ops: set[str] = set(OPS)) -> tuple[list[str], list[str]]:
    on, oa = [], []
    for nm, ad, lf in zip(names, addrs, s1_legal):
        m = LEGAL_RE.search(nm) if "legal" in ops else None
        if m:
            base = nm[:m.start()]
            if not record:
                nm = f"{base} {lf}"
            else:
                u = rng.random()
                nm = f"{base} {lf}" if u < 0.6 else f"{base} {rng.choice([x for x in FR_LEGAL if x != lf])}" if u < 0.85 else base
        if "et" in ops and record and rng.random() < 0.5:
            nm = re.sub(r"\s+(&|and)\s+", " et ", nm)
        if "accents" in ops and record and rng.random() < 0.3:
            nm = " ".join(w.replace("e", "é", 1) for w in nm.split())
        short = record and rng.random() < 0.5
        if "street" in ops:
            ad = STREET_RE.sub(lambda x: STREET[x.group(1).lower()][1 if short else 0], ad)
        if "cedex" in ops and record and ad and rng.random() < 0.2:
            ad = f"{ad}, Cedex {rng.integers(1, 20):02d}"
        on.append(nm)
        oa.append(ad)
    return on, oa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models_krish")
    ap.add_argument("--oof", default="oof_krish")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    a = ap.parse_args()
    ops = set(a.ops)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    s1 = s1_table(1.0)
    H = s1.filter(pl.col("fold") == 0).sample(a.n, seed=SEED)
    gt = load_gt_pairs()
    true_rec = gt.join(H.select("s1_id"), on="s1_id").rename({"match_id": "rec_id"})
    cand = pl.read_parquet(path("interim_dir") / "candidates_train.parquet",
                           columns=["s1_id", "rec_id", *RANK_COLS, "n_channels_hit"])
    touch = cand.join(H.select("s1_id"), on="s1_id", how="semi").select("rec_id").unique()
    rows = cand.join(touch, on="rec_id", how="semi")
    del cand
    print(f"H {H.height} S1, {true_rec.height} true pairs, {touch.height} touching records, {rows.height} rows", flush=True)

    # stored features for these rows (sanity + S1-side columns of competitor S1s)
    stored = (pl.scan_parquet(parts("train")).join(rows.lazy().select("s1_id", "rec_id"), on=["s1_id", "rec_id"],
                                                   how="semi").collect(engine="streaming"))
    stored = rows.select("s1_id", "rec_id").join(stored, on=["s1_id", "rec_id"], how="left", maintain_order="left")
    assert stored["name_tset"].null_count() == 0, "stored feature rows missing"
    voc, lnn, w = vocab("train")
    idf_lookup = dict(zip(zip(voc["A"]["country"].to_list(), voc["A"]["tok"].to_list()), voc["A"]["w"].to_list()))
    dict_map = load_token_dict()
    raw_cols = ["entity_id", "business_name", "business_address", "country"]
    EC = ENT_COLS + ["business_address"]
    s1_raw = pl.read_parquet(norm_path("train", 1), columns=EC)
    rec_ids = rows["rec_id"].unique().implode()
    rec_raw = pl.concat([pl.read_parquet(norm_path("train", n), columns=EC).filter(pl.col("entity_id").is_in(rec_ids))
                         for n in (2, 3)])
    res = {"models": a.models, "n_s1": H.height, "rows": rows.height, "ops": sorted(ops)}
    for noise in (False, True):
        S, R = s1_raw, rec_raw
        if noise:
            hs = s1_raw.join(H.select(entity_id="s1_id"), on="entity_id", how="semi")
            lf = {e: str(rng.choice(FR_LEGAL)) for e in hs["entity_id"]}
            # shared address: pair up 8.4% of H S1s within country; the second takes the first's address
            sh = {}
            for c in ("India", "US"):
                ids = hs.filter(pl.col("country") == c)["entity_id"].to_list()
                k = int(round(0.084 * len(ids))) // 2 * 2 if "shared" in ops else 0
                pick = rng.choice(ids, k, replace=False)
                addr = dict(zip(hs["entity_id"], hs["business_address"]))
                sh.update({b: addr[a_] for a_, b in zip(pick[0::2], pick[1::2])})
            tr = true_rec.filter(pl.col("rec_id").is_in(rec_ids))
            rec_s1 = dict(zip(tr["rec_id"], tr["s1_id"]))
            hs = hs.with_columns(business_address=pl.Series([sh.get(e, x) for e, x in zip(hs["entity_id"], hs["business_address"])]))
            n1, a1 = frenchify(hs["business_name"].to_list(), hs["business_address"].to_list(), rng, False,
                               [lf[e] for e in hs["entity_id"]], ops)
            hs = hs.with_columns(business_name=pl.Series(n1), business_address=pl.Series(a1))
            hr = R.filter(pl.col("entity_id").is_in(tr["rec_id"].implode()))
            hr = hr.with_columns(business_address=pl.Series([sh.get(rec_s1[e], x) if rec_s1[e] in sh else x
                                                             for e, x in zip(hr["entity_id"], hr["business_address"])]))
            n2, a2 = frenchify(hr["business_name"].to_list(), hr["business_address"].to_list(), rng, True,
                               [lf[rec_s1[e]] for e in hr["entity_id"]], ops)
            hr = hr.with_columns(business_name=pl.Series(n2), business_address=pl.Series(a2))
            renorm = lambda df: normalise(df.select(raw_cols), frozenset()).select(EC)
            S = pl.concat([S.join(hs.select("entity_id"), on="entity_id", how="anti"), renorm(hs)])
            R = pl.concat([R.join(hr.select("entity_id"), on="entity_id", how="anti"), renorm(hr)])
            res["noise"] = {"shared_pairs": len(sh), "s1_noised": hs.height, "rec_noised": hr.height}
        Se = prep(S, voc, lnn)
        Se = Se.join(s1_name_dup(Se.select("entity_id", "country", "core")), on="entity_id", how="left")
        Se = Se.join(s1_addr_dup(Se.select("entity_id", "country", "addr")), on="entity_id", how="left")
        Re = prep(R, voc, lnn)
        pf = pl.concat([pair_features(rows.slice(o, 1_000_000), Se, Re, w, idf_lookup, dict_map)
                        for o in range(0, rows.height, 1_000_000)])
        keys = pf.select("s1k", "reck")
        rel = [keys.select(n_cand_rec=pl.len().over("reck").cast(pl.Int32), n_cand_s1=pl.len().over("s1k").cast(pl.Int32)),
               pf.select(rec_name_hits95=(pl.col("name_tset") >= 95).sum().over("reck").cast(pl.Int32),
                         rec_addr_hits95=(pl.col("addr_tset") >= 95).sum().over("reck").cast(pl.Int32))]
        rel += [relative(keys, pf[c]) for c in REL_COLS]
        df = pl.concat([pf.drop("s1k", "reck"), *rel], how="horizontal").with_columns(
            is_noaddr_ambiguous=((pl.col("rec_no_addr") == 1) & (pl.col("rec_name_hits95") >= 2)).cast(pl.Int8)).drop("rec_no_addr")
        s1side = [c for c in df.columns if c.endswith("_ds1") or c.endswith("_rks1")] + ["n_cand_s1"]
        df = (df.with_columns(_inH=pl.col("s1_id").is_in(H["s1_id"].implode()),
                              **{f"_st_{c}": stored[c].cast(df.schema[c]) for c in s1side})
              .with_columns([pl.when("_inH").then(pl.col(c)).otherwise(pl.col(f"_st_{c}")).alias(c) for c in s1side])
              .drop("_inH", *[f"_st_{c}" for c in s1side]))
        if not noise:
            for c in PAIR_CHECK:
                x, y = df[c].cast(pl.Float64).fill_nan(-9).fill_null(-9), stored[c].cast(pl.Float64).fill_nan(-9).fill_null(-9)
                assert (x - y).abs().max() < 1e-4, f"noise-off rebuild differs from stored features: {c}"
        cvj = json.loads((REPO / a.oof / "cv_full.json").read_text())
        feats = cvj["features"]
        X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        fold = df.select("s1_id").join(s1.select("s1_id", "fold"), on="s1_id", how="left", maintain_order="left")["fold"].to_numpy()
        p1 = np.zeros(len(X), np.float32)
        for k in range(5):
            m = fold == k
            if m.any():
                p1[m] = lgb.Booster(model_file=str(REPO / a.models / f"fold_{k}.txt")).predict(X[m])
        # stage 2
        s2j = json.loads((REPO / a.oof / "stage2.json").read_text())
        base = df.select(s=id_key("s1_id"), r=id_key("rec_id")).with_columns(p=pl.Series(p1))
        txt = pl.concat([S.select(r=id_key("entity_id"), core="core_name", addr=pl.col("addr_tokens").list.join(" ")),
                         R.select(r=id_key("entity_id"), core="core_name", addr=pl.col("addr_tokens").list.join(" "))]).unique("r")
        ctx = coref(context(base), txt, lambda *x, **y: None)
        f2 = s2j["feats"]["full"]
        X2 = pl.concat([ctx, df.select(s2j["carry"])], how="horizontal").select(pl.col(f2).cast(pl.Float32)).to_numpy()
        p2 = p1.copy()
        keep = p1 >= s2j.get("p_floor", 0.0)
        for k in range(5):
            m = (fold == k) & keep
            if m.any():
                p2[m] = lgb.Booster(model_file=str(REPO / a.models / f"stage2_full_fold_{k}.txt")).predict(X2[m])
        # decisions + F0.5 over H
        lab = df.select("s1_id", "rec_id").join(gt.rename({"match_id": "rec_id"}).with_columns(y=pl.lit(1)),
                                                on=["s1_id", "rec_id"], how="left")["y"].fill_null(0).to_numpy()
        hcode = {s: i for i, s in enumerate(H["s1_id"].to_list())}
        out = {}
        for name, p, t in (("stage1", p1, cvj["b_test_density"]["best_t"]), ("B", p2, s2j["full"]["best_t"])):
            d = pl.DataFrame({"s1_id": df["s1_id"], "rec_id": df["rec_id"], "p": p, "y": lab})
            top = d.sort(["rec_id", "p", "s1_id"], descending=[False, True, False]).filter(pl.col("rec_id").is_first_distinct())
            pr = top.filter((pl.col("p") >= t) & pl.col("s1_id").is_in(H["s1_id"].implode()))
            code = np.array([hcode[s] for s in pr["s1_id"]], dtype=np.int64)
            tp = np.bincount(code, weights=pr["y"].to_numpy(), minlength=H.height)
            npred = np.bincount(code, minlength=H.height)
            f = f05_vec(tp, npred, H["ntrue"].to_numpy())
            ctry = H["country"].to_numpy()
            out[name] = {"f05": float(f.mean()), **{f"f05_{c}": float(f[ctry == c].mean()) for c in ("India", "US")},
                         "empty_pct": float(100 * (npred == 0).mean()), "pred_per_s1": float(npred.mean()), "t": t}
        res["noise" if noise else "clean"] = {**res.get("noise", {}), **out} if noise else out
        print(("NOISE " if noise else "CLEAN ") + json.dumps(out), flush=True)
    res["drop_B"] = res["noise"]["B"]["f05"] - res["clean"]["B"]["f05"]
    res["drop_stage1"] = res["noise"]["stage1"]["f05"] - res["clean"]["stage1"]["f05"]
    res["total_s"] = round(time.time() - t0)
    (REPO / "experiments" / f"france_proxy_{a.models}_{'-'.join(sorted(ops))}_n{a.n}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("clean", "noise")}), flush=True)


if __name__ == "__main__":
    main()
