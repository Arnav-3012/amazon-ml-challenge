"""Diagnostic (read-only, no pipeline change): for true train pairs, how many mined ops does it take to turn
the S2/S3 raw name/address into the S1 raw name/address, exactly?

Scope note: noise_ops.py's ~67 named operators are pair-diff DETECTORS (they classify what changed between
a pair, e.g. "token order change (same multiset)", "legal-form swap") -- they don't all carry a rewrite rule.
Only a subset is mechanically applicable as a S2/S3 -> S1 string edit; this script's op library is exactly
that subset, reusing normalise.py's own primitives (not reimplemented):
  legal_form, street_type, ordinal, digit_fix, dedupe_adjacent, honorifics, id_tag, domain_strip, acronym_dots
plus the 807 mined src.mine_dict token_dict replacements (per field: name/addr).
This measures composability for THAT applicable subset -- not all ~67 detected noise types. Token reorder,
token add/drop, char typos, script/accent changes etc. have no single-token rewrite here and are out of scope;
pairs needing them will show up as unreachable, which is itself the diagnostic signal.

Method: greedy set-cover per pair per field. At each step, try every remaining candidate op on the current
string; keep the one that gets closest to (or reaches) exact equality with the S1 string (case/whitespace-
normalized); stop at equality or after 5 ops (bail -> "5+/unreachable").

Run from code/business_entity_resolution/:  python -m src.composition_depth
Writes only to the scratchpad dir (SCRATCH env var, else /tmp) -- nothing under docs/ or artifacts/.
"""
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from .io import CFG, load_gt_pairs, load_source, path
from .normalise import DIGIT_MAP, LEX, ORDINALS, _words_re

N_PAIRS = 500
SEED = CFG["seed"]
MAX_OPS = 5
SCRATCH = Path(os.environ.get("SCRATCH", "/tmp")) / "composition_depth"

WS_RE = re.compile(r"\s+")


def norm_eq(s: str) -> str:
    """Case/whitespace-normalized string for the exact-equality check (not a rewrite op)."""
    return WS_RE.sub(" ", s.strip().lower())


# ---- token_dict ----
def load_token_dict() -> dict[str, dict[str, str]]:
    d = pl.read_parquet(path("interim_dir") / "token_dict.parquet")
    return {f: dict(d.filter(pl.col("field") == f).select("s", "t").iter_rows()) for f in ("name", "addr")}


def token_replace_ops(s: str, tmap: dict[str, str]) -> list[tuple[str, str]]:
    """(op_name, result) for every token_dict mapping whose source token appears in s."""
    toks = s.split()
    out = []
    for i, t in enumerate(toks):
        if t in tmap:
            new = toks.copy()
            new[i] = tmap[t]
            out.append((f"token_dict:{t}->{tmap[t]}", " ".join(new)))
    return out


# ---- normalise.py-backed ops (single-shot string versions of the same primitives) ----
def op_legal_form(s: str, lex) -> str | None:
    if not lex.legal:
        return None
    r = _words_re(lex.legal)
    core = re.sub(r, "", s)
    core = WS_RE.sub(" ", core).strip()
    return core if core and core != s else None


def op_street_type(s: str, lex) -> str | None:
    if not lex.street:
        return None
    toks = s.split()
    new = [lex.street.get(t, t) for t in toks]
    out = " ".join(new)
    return out if out != s else None


def op_ordinal(s: str) -> str | None:
    toks = s.split()
    new = [ORDINALS.get(t, t) for t in toks]
    out = " ".join(new)
    return out if out != s else None


def op_digit_fix(s: str) -> str | None:
    toks = s.split()
    new = []
    changed = False
    for t in toks:
        if re.fullmatch(r"[a-z01568]+", t) and re.search(r"[01568]", t) and re.search(r"[a-z].*[a-z]", t) \
                and not re.fullmatch(r"\d+(?:st|nd|rd|th)", t):
            nt = "".join(DIGIT_MAP.get(c, c) for c in t)
            if nt != t:
                changed = True
            new.append(nt)
        else:
            new.append(t)
    return " ".join(new) if changed else None


def op_dedupe_adjacent(s: str) -> str | None:
    toks = s.split()
    new = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1] or len(t) == 1]
    out = " ".join(new)
    return out if out != s else None


def op_honorifics(s: str) -> str | None:
    HON = ["m s", "smt", "shri", "sri", "mr", "dr", "the"]
    for h in HON:
        if s == h or s.startswith(h + " "):
            rest = s[len(h):].strip()
            if rest:
                return rest
    return None


def op_id_tag(s: str) -> str | None:
    out = re.sub(r"\(\s*id\s*:?\s*\d+\s*\)|#\s?\d{3,}", " ", s)
    out = WS_RE.sub(" ", out).strip()
    return out if out != s else None


def op_domain_strip(s: str) -> str | None:
    out = re.sub(r"\s*\|\s*(?:www\.)?\S+\.(?:com|net|org|in)\b", " ", s)
    out = re.sub(r"\bwww\.|\.(?:com|net|org|in)\b", " ", out)
    out = WS_RE.sub(" ", out).strip()
    return out if out != s else None


def op_acronym_dots(s: str) -> str | None:
    out = re.sub(r"\b([a-z])\.", r"\1", s)
    return out if out != s else None


NAME_OPS = [op_id_tag, op_acronym_dots, op_domain_strip, op_digit_fix, op_dedupe_adjacent, op_honorifics]
NAME_LEX_OPS = [op_legal_form]  # needs lex
ADDR_LEX_OPS = [op_street_type]  # needs lex
ADDR_OPS = [op_ordinal]


def candidates(s: str, field: str, lex, tmap: dict[str, str]) -> list[tuple[str, str]]:
    out = token_replace_ops(s, tmap)
    fns = (NAME_OPS if field == "name" else ADDR_OPS)
    lex_fns = (NAME_LEX_OPS if field == "name" else ADDR_LEX_OPS)
    for fn in fns:
        r = fn(s)
        if r is not None:
            out.append((fn.__name__, r))
    for fn in lex_fns:
        r = fn(s, lex)
        if r is not None:
            out.append((fn.__name__, r))
    return out


def greedy_cover(src: str, target: str, field: str, lex, tmap: dict[str, str]) -> int | None:
    """Min ops (greedy, not exhaustive) to turn src into target exactly (case/whitespace-normalized).
    Returns ops used, or None if not reached within MAX_OPS."""
    s = norm_eq(src)
    tgt = norm_eq(target)
    if s == tgt:
        return 0
    for step in range(1, MAX_OPS + 1):
        cands = candidates(s, field, lex, tmap)
        if not cands:
            return None
        best = min(cands, key=lambda c: Levenshtein.distance(norm_eq(c[1]), tgt))
        s = norm_eq(best[1])
        if s == tgt:
            return step
    return None


def sample_pairs() -> list[tuple]:
    pairs = load_gt_pairs().sample(N_PAIRS, seed=SEED)
    s1 = load_source("train", 1)
    s1 = s1[s1["entity_id"].isin(set(pairs["s1_id"]))]
    want = set(pairs["match_id"])
    m = [df[df["entity_id"].isin(want)] for df in (load_source("train", 2), load_source("train", 3))]
    rec = {i: (nm, ad, co) for df in (s1, *m)
           for i, nm, ad, co in zip(df["entity_id"], df["business_name"], df["business_address"], df["country"])}
    return [(rec[a], rec[b]) for a, b in zip(pairs["s1_id"], pairs["match_id"])]


def main() -> None:
    tmap = load_token_dict()
    rows = []
    for (a_name, a_addr, c), (b_name, b_addr, _) in sample_pairs():
        lex = LEX.get(c)
        if lex is None:  # unseen-country guard, never filter/skip on it -- just no street/region lex to use
            from .normalise import DEFAULT_LEX
            lex = DEFAULT_LEX
        n_ops = greedy_cover(b_name, a_name, "name", lex, tmap["name"])
        a_ops = greedy_cover(b_addr, a_addr, "addr", lex, tmap["addr"])
        rows.append({"country": c, "name_ops": n_ops, "addr_ops": a_ops})

    df = pl.DataFrame(rows)
    edges = [0, 1, 2, 3, 4, 5, 6]  # last bucket = "5+/unreachable" (None -> 6 by fillna below)
    labels = ["0", "1", "2", "3", "4", "5+/unreachable"]

    def hist(sub: pl.DataFrame, col: str) -> dict[str, int]:
        vals = sub[col].fill_null(6).to_numpy()
        counts, _ = np.histogram(vals, bins=[0, 1, 2, 3, 4, 5, 7])
        return dict(zip(labels, counts.tolist()))

    L = ["# Composition depth (src/composition_depth.py) -- diagnostic, no pipeline change", "",
         f"{N_PAIRS} sampled train true pairs (seed {SEED}), both countries present in sample: "
         f"{sorted(df['country'].unique().to_list())}.", "",
         "**Op library scope**: only ops with a real rewrite rule are used -- legal_form, street_type, "
         "ordinal, digit_fix, dedupe_adjacent, honorifics, id_tag, domain_strip, acronym_dots (reused from "
         "src/normalise.py) + 807 src.mine_dict token_dict replacements. This is NOT all ~67 noise_ops "
         "detector types -- token reorder, token add/drop, char typos, script/accent changes etc. have no "
         "single-token rewrite here and will show as unreachable. Cap 5 ops/pair; greedy (not exhaustive), "
         "so counts are an upper bound on true min ops.", "",
         "## ops-needed histogram, name field", "", *[f"- {k}: {v}" for k, v in hist(df, 'name_ops').items()], "",
         "## ops-needed histogram, address field", "", *[f"- {k}: {v}" for k, v in hist(df, 'addr_ops').items()], ""]
    for c in sorted(df["country"].unique().to_list()):
        sub = df.filter(pl.col("country") == c)
        L += [f"## by country: {c} (n={sub.height})", "",
              "**name**: " + ", ".join(f"{k}={v}" for k, v in hist(sub, 'name_ops').items()), "",
              "**addr**: " + ", ".join(f"{k}={v}" for k, v in hist(sub, 'addr_ops').items()), ""]

    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / "composition_depth.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
