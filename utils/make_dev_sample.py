"""Build a small, real, labeled slice of the dataset for development (not part of the pipeline).

Run from the repo root:   python utils/make_dev_sample.py
Writes dataset_sample/{train,test}/ with the same file names as dataset/, plus dataset_sample.zip.

Train slice: N_S1 S1 entities (stratified by country, incl. singletons), ALL of their true matches,
plus random S2/S3 records from the same countries as distractors/lookalikes. The GT is restricted
to the sampled S1 rows, so every matched id in it exists in the sampled S2/S3.
Test slice: France only (unseen in train) - random S1/S2/S3 rows, no labels, for normaliser rules.
"""
import csv
import random
import shutil
from pathlib import Path

import pandas as pd

SEED = 42
N_S1_PER_COUNTRY = 3000          # train S1 entities per country
DISTRACTOR_RATIO = 1.0           # extra random S2/S3 rows per matched row
N_FR_S1, N_FR_S23 = 3000, 8000   # test France rows (S1, and each of S2/S3)

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "dataset", ROOT / "dataset_sample"


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE)


def write(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(df.columns) + "\n")
        for row in df.itertuples(index=False):
            f.write("\t".join(row) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    if DST.exists():
        shutil.rmtree(DST)

    # ---- train ----
    s1 = read(SRC / "train" / "train_source1.tsv")
    gt = read(SRC / "train" / "train_ground_truth.tsv")
    picked = []
    for _, grp in s1.groupby("country", sort=True):
        ids = grp["entity_id"].tolist()
        picked += rng.sample(ids, min(N_S1_PER_COUNTRY, len(ids)))
    picked_set = set(picked)
    s1_s = s1[s1["entity_id"].isin(picked_set)]
    gt_s = gt[gt["source1_entity_id"].isin(picked_set)]
    matched = {i.strip() for m in gt_s["matched_entity_ids"] for i in m.split(",") if i.strip()}
    countries = set(s1_s["country"])
    for n in (2, 3):
        df = read(SRC / "train" / f"train_source{n}.tsv")
        keep = df["entity_id"].isin(matched)
        n_extra = int(keep.sum() * DISTRACTOR_RATIO)
        pool = df[~keep & df["country"].isin(countries)]
        extra = pool.sample(n=min(n_extra, len(pool)), random_state=SEED)
        out = pd.concat([df[keep], extra]).sort_values("entity_id")
        write(out, DST / "train" / f"train_source{n}.tsv")
        print(f"train S{n}: {int(keep.sum())} matched + {len(extra)} random = {len(out)}")
    write(s1_s.sort_values("entity_id"), DST / "train" / "train_source1.tsv")
    write(gt_s.sort_values("source1_entity_id"), DST / "train" / "train_ground_truth.tsv")
    n_single = int((gt_s["matched_entity_ids"] == "").sum())
    print(f"train S1: {len(s1_s)} ({n_single} singletons) | GT rows: {len(gt_s)}")

    # ---- test: France only ----
    for n, k in ((1, N_FR_S1), (2, N_FR_S23), (3, N_FR_S23)):
        df = read(SRC / "test" / f"test_source{n}.tsv")
        fr = df[df["country"] == "France"]
        out = fr.sample(n=min(k, len(fr)), random_state=SEED).sort_values("entity_id")
        write(out, DST / "test" / f"test_source{n}.tsv")
        print(f"test S{n} France: {len(out)}")

    zip_path = shutil.make_archive(str(ROOT / "dataset_sample"), "zip", root_dir=ROOT,
                                   base_dir="dataset_sample")
    print(f"wrote {zip_path} ({Path(zip_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
