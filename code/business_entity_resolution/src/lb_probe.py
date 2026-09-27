"""LB probe: per-country blanking of Submit #1 to back out per-country F0.5 from the leaderboard.

For each probe country c, writes output/probe_blank_<c>/ with
  matching_results.tsv  = sub1 byte-for-byte, except matched_entity_ids emptied for every S1 of country c
  candidate_pairs.tsv   = sub1 copied unchanged
then runs utils/validate_submission.py on the folder and prints n_c, w_c = n_c/N and
F_c = 0.056 + (LB_SUB1 - LB_probe)/w_c (LB_probe is filled in by hand after submitting).

Reads/writes TSVs only, no model. Run from code/business_entity_resolution/:  python -m src.lb_probe
"""
import shutil
import subprocess
import sys
from collections import Counter

from .io import ROOT

OUT = ROOT / "output"
SUB1_MATCH = OUT / "matching_results_sub1.tsv"
SUB1_CAND = OUT / "candidate_pairs_sub1.tsv"
TEST_DIR = ROOT / "dataset" / "test"
VALIDATOR = ROOT / "utils" / "validate_submission.py"
LB_SUB1 = 0.964872
PROBES = ["France", "India"]  # asserted against the actual values printed below


def s1_country() -> dict[str, str]:
    # stdlib split, same semantics as sep="\t", dtype=str, keep_default_na=False
    with open(TEST_DIR / "test_source1.tsv", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_id, i_c = header.index("entity_id"), header.index("country")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    return {r[i_id]: r[i_c] for r in rows}


def write_probe(country: str, s1_c: dict[str, str]) -> tuple:
    d = OUT / f"probe_blank_{country.lower()}"
    d.mkdir(exist_ok=True)
    n_blanked = n_nonempty_blanked = 0
    with open(SUB1_MATCH, "rb") as src, open(d / "matching_results.tsv", "wb") as dst:
        dst.write(src.readline())  # header unchanged
        for line in src:
            s1, rest = line.split(b"\t", 1)
            if s1_c[s1.decode()] == country:
                eol = rest[len(rest.rstrip(b"\r\n")):]
                n_nonempty_blanked += bool(rest.strip())
                line = s1 + b"\t" + eol
                n_blanked += 1
            dst.write(line)
    shutil.copyfile(SUB1_CAND, d / "candidate_pairs.tsv")
    return d, n_blanked, n_nonempty_blanked


def validate(d) -> bool:
    r = subprocess.run([sys.executable, str(VALIDATOR),
                        "--matching", str(d / "matching_results.tsv"),
                        "--candidate", str(d / "candidate_pairs.tsv"),
                        "--test-dir", str(TEST_DIR)], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:], sep="")
    return r.returncode == 0


def main():
    s1_c = s1_country()
    counts = Counter(s1_c.values())
    N = len(s1_c)
    print("country values in test_source1:", {repr(k): v for k, v in counts.most_common()})
    missing = [c for c in PROBES if c not in counts]
    assert not missing, f"probe countries not in test_source1: {missing}"

    for c in PROBES:
        d, n_b, n_ne = write_probe(c, s1_c)
        ok = validate(d)
        print(f"[{c}] {d}  rows blanked={n_b:,} (had matches: {n_ne:,})  validator: {'PASS' if ok else 'FAIL'}")

    print(f"\nN = {N:,} test S1 entities")
    print(f"{'country':<16}{'n_c':>10}{'w_c':>10}   F_c formula")
    for c, n in counts.most_common():
        w = n / N
        print(f"{c!r:<16}{n:>10,}{w:>10.6f}   F_c = 0.056 + ({LB_SUB1} - LB_probe) / {w:.6f}")


if __name__ == "__main__":
    main()
