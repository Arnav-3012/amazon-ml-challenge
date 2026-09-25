"""M1 baseline: predict empty for every S1 entity.

Train: all-empty macro F0.5 must equal the singleton rate; prints basic data stats.
Test: writes all-empty matching_results.tsv and candidate_pairs.tsv (validator smoke test).
"""
import numpy as np

from .io import load_gt, load_source, path, write_id_lists
from .metric import macro_f05


def main() -> None:
    truth = load_gt()
    score, br = macro_f05({}, truth)
    singleton_rate = br["n_singleton"] / len(truth)
    print(f"train all-empty macro F0.5 = {score:.6f} | singleton rate = {singleton_rate:.6f} "
          f"({br['n_singleton']} / {len(truth)})")
    assert abs(score - singleton_rate) < 1e-9, "metric/singleton mismatch"

    sizes = np.fromiter(map(len, truth.values()), int, len(truth))
    hist = np.bincount(np.minimum(sizes, 5), minlength=6)
    print("match cardinality:", " ".join(f"{k}:{c}" for k, c in zip(["0", "1", "2", "3", "4", "5+"], hist)))
    n_all = int(sizes.sum())
    n_s2 = sum(i.startswith("S2-") for v in truth.values() for i in v)
    print(f"matched ids: {n_all} | S2 share {n_s2 / n_all:.4f} | S3 share {1 - n_s2 / n_all:.4f}")

    for split in ("train", "test"):
        for n in (1, 2, 3):
            vc = load_source(split, n, ["country"])["country"].value_counts()
            print(f"{split} S{n} rows={int(vc.sum())} | " + " ".join(f"{c!r}:{k}" for c, k in vc.items()))

    empty = dict.fromkeys(load_source("test", 1, ["entity_id"])["entity_id"], ())
    write_id_lists(path("matching_results"), empty, "source1_entity_id", "matched_entity_ids")
    write_id_lists(path("candidate_pairs"), empty, "source1_entity_id", "candidate_entity_ids")
    print(f"wrote {len(empty)} all-empty rows → {path('matching_results')}, {path('candidate_pairs')}")


if __name__ == "__main__":
    main()
