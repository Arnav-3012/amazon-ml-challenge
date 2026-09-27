"""France-only threshold probe on test p: per country, predicted matches/S1 and % S1 with zero predictions under
per-record argmax + p >= t (decide.with_top rule) at the global t (oof/cv_full.json b_test_density.best_t), then the
France t on a 0.60-0.99 grid whose matches/S1 equals the non-France (US+India) rate at the global t.

Reads oof/test_p.parquet, oof/cv_full.json and test_source1 country only; writes nothing.
Run from code/business_entity_resolution/:  python -m src.france_t
"""
import json

import numpy as np
import polars as pl

from .decide import with_top
from .io import ROOT
from .lb_probe import s1_country

TARGET = "France"
GRID = np.round(np.arange(0.60, 0.9901, 0.01), 2)
CTX = 3  # grid rows printed either side of the match


def main():
    t_glob = json.loads((ROOT / "oof" / "cv_full.json").read_text())["b_test_density"]["best_t"]
    s1_c = s1_country()
    ctry = pl.DataFrame({"s1_id": list(s1_c), "country": list(s1_c.values())})
    n_c = dict(ctry.group_by("country").len().iter_rows())

    top = (with_top(pl.read_parquet(ROOT / "oof" / "test_p.parquet"))
           .filter("top").select("s1_id", "p").join(ctry, on="s1_id", how="left"))
    assert top["country"].null_count() == 0, "test_p s1_id missing from test_source1"

    def rate(countries: list[str], t: float) -> tuple[float, float]:
        k = top.filter(pl.col("country").is_in(countries) & (pl.col("p") >= t))
        n = sum(n_c[c] for c in countries)
        return k.height / n, 100 * (1 - k["s1_id"].n_unique() / n)

    print(f"global t = {t_glob:.2f} (cv_full.json b_test_density)")
    print(f"{'country':<12}{'n_S1':>10}{'matches/S1':>12}{'%zero':>8}")
    for c in sorted(n_c, key=n_c.get, reverse=True):
        m, z = rate([c], t_glob)
        print(f"{c:<12}{n_c[c]:>10,}{m:>12.4f}{z:>8.2f}")

    ref = [c for c in n_c if c != TARGET]
    r_ref, z_ref = rate(ref, t_glob)
    print(f"\nreference {'+'.join(ref)} @ t={t_glob:.2f}: matches/S1 = {r_ref:.4f}, %zero = {z_ref:.2f}")

    rows = [(t, *rate([TARGET], t)) for t in GRID]
    i = int(np.argmin([abs(m - r_ref) for _, m, _ in rows]))
    print(f"\n{TARGET} grid (* = closest to reference):")
    print(f"{'t':>6}{'matches/S1':>12}{'diff':>9}{'%zero':>8}")
    for t, m, z in rows[max(0, i - CTX): i + CTX + 1]:
        print(f"{t:>6.2f}{m:>12.4f}{m - r_ref:>+9.4f}{z:>8.2f}{'  *' if t == rows[i][0] else ''}")
    if i in (0, len(rows) - 1):
        print(f"WARNING: match at grid edge t={rows[i][0]:.2f}; true crossing may lie outside 0.60-0.99")


if __name__ == "__main__":
    main()
