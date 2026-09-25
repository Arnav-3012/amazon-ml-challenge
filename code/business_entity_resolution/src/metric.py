"""Exact local replica of the leaderboard metric: macro F0.5 over Source 1 entities."""
import numpy as np

EMPTY = frozenset()


def f05(pred: set, true: set) -> float:
    """Per-entity F0.5. Singleton rule: empty/empty = 1, any side empty otherwise = 0."""
    if not pred or not true:
        return float(not pred and not true)
    tp = len(pred & true)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict) -> tuple[float, dict]:
    """Mean F0.5 over ALL truth keys (missing pred key = empty set), plus singleton breakdown."""
    keys = list(truth)
    scores = np.fromiter((f05(pred.get(k, EMPTY), truth[k]) for k in keys), float, len(keys))
    single = np.fromiter((not truth[k] for k in keys), bool, len(keys))
    part = lambda m: float(scores[m].mean()) if m.any() else float("nan")
    return float(scores.mean()), {
        "n_singleton": int(single.sum()), "f05_singleton": part(single),
        "n_nonsingleton": int((~single).sum()), "f05_nonsingleton": part(~single),
    }


if __name__ == "__main__":
    a, b, c = "S2-00047", "S2-00193", "S3-00812"
    assert abs(f05({a, b, c}, {a, c}) - 0.714) < 1e-3  # PS worked example
    assert f05(set(), set()) == 1.0
    assert f05({a}, set()) == 0.0
    assert f05(set(), {a}) == 0.0
    assert f05({a}, {b}) == 0.0
    score, br = macro_f05({"x": {a, b, c}}, {"x": frozenset({a, c}), "y": EMPTY})  # y missing → empty → 1
    assert abs(score - (0.714 + 1) / 2) < 1e-3 and br["n_singleton"] == 1 and br["f05_singleton"] == 1.0
    print("metric self-test: PASS")
