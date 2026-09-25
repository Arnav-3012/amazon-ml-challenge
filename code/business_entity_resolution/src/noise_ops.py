"""M3a step 1: mine the noise operators from 200k sampled train true pairs (S1 vs matched S2/S3 record).

Writes docs/noise_ops.md: operator | field | country | freq% of pairs | 3 examples, then the token tables
the normaliser lexicons (src/normalise.py) are checked against. Numbers + examples only.
Cost: 200k pairs x ~40 regex/Counter checks in Python (~100us/pair, ~30s); full 7.6M pairs is not needed.
Run from code/business_entity_resolution/: python -m src.noise_ops
"""
import re
from collections import Counter, defaultdict

from anyascii import anyascii
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio, token_set_ratio

from .io import ROOT, load_gt_pairs, load_source
from .normalise import ALIAS_KW, LEX, ORDINALS, Lex

SEED = 42
N_PAIRS = 200_000
TOP = 50
OUT = ROOT / "docs" / "noise_ops.md"

TOK = re.compile(r"[a-z0-9]+")
LATIN_ACC = re.compile(r"[À-ɏ]")
ALIAS = re.compile(rf"^(?P<alias>.+?)\s+(?P<kw>{ALIAS_KW})\s+(?P<core>.+)$")
HON_CAND = ("smt", "shri", "sri", "shree", "mr", "mrs", "ms", "dr", "the", "m")  # m = m/s
DOTTED = re.compile(r"\b[a-z]\.[a-z]\.")
BRACKET = re.compile(r"[()\[\]{}]")
HYPHEN = re.compile(r"\w-\w")
PHONE = re.compile(r"\d[\d\s-]{6,}\s*$")
DOMAIN = re.compile(r"\.com\b|\bwww\.|com$")
MARKER = re.compile(r"\((?:france|india)\)")
NULL_COMP = re.compile(r"^<?\s*(?:null|n/?a|none|nil)\s*>?$")
NUM_PREFIX = re.compile(r"#\s*\d|\b(?:h\.?\s?no|house\s?no|door\s?no|d\.?\s?no|no|ndeg)\b\.?\s*\d")
NUMRX = re.compile(r"^(?:#\s*|(?:no|ndeg)\.?\s*)?(\d+)([a-z]?)\s+[a-z]")
LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside)\b")
UNIT = re.compile(r"\b(?:unit|apt|apartment|suite|fl|floor)\b")
POBOX = re.compile(r"\b(?:p\.?\s?o\.?\s?box|pmb)\b")
ORD = re.compile(r"^\d+(?:st|nd|rd|th)$")


def fold(s: str) -> str:
    return (s if s.isascii() else anyascii(s)).lower()


def nonlatin(s: str) -> bool:
    return not s.isascii() and any(c.isalpha() and ord(c) > 0x24F for c in s)


def short(s: str, n: int = 60) -> str:
    s = s.replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


class Miner:
    def __init__(self) -> None:
        self.n = Counter()                  # country -> pairs
        self.hits = Counter()               # (field, op, country) -> pairs
        self.ex = defaultdict(list)         # (field, op, country) -> <=3 examples
        self.tab = defaultdict(Counter)     # (table, country) -> Counter

    def record(self, field: str, ops: set[str], c: str, a: str, b: str) -> None:
        for op in ops:
            k = (field, op, c)
            self.hits[k] += 1
            if len(self.ex[k]) < 3:
                self.ex[k].append(f"{short(a)} → {short(b)}")

    def common(self, field: str, a: str, b: str, ta: list[str], tb: list[str], c: str) -> tuple[set, Counter, Counter]:
        """Operators shared by name and address: case, accents, script, order, dup, token add/drop, subs."""
        ops = set()
        if b.isupper() and not a.isupper():
            ops.add("case: all-upper")
        elif b.islower() and not a.islower():
            ops.add("case: all-lower")
        if not b.isascii():
            if LATIN_ACC.search(b) and not LATIN_ACC.search(a):
                ops.add("accent injection (Latin diacritics)")
            if nonlatin(b):
                ops.add("script change (non-Latin match, S1 Latin)")
        ca, cb = Counter(ta), Counter(tb)
        added, dropped = cb - ca, ca - cb
        if ta != tb and ca == cb:
            ops.add("token order change (same multiset)")
        elif list(dict.fromkeys(t for t in ta if t in cb)) != list(dict.fromkeys(t for t in tb if t in ca)):
            ops.add("token order change (shared tokens)")
        dup = lambda t: any(t[i] == t[i + 1] for i in range(len(t) - 1)) or any(  # noqa: E731
            t[i:i + 2] == t[i + 2:i + 4] for i in range(len(t) - 3))
        if dup(tb) and not dup(ta):
            ops.add("adjacent token duplication (uni/bigram)")
        if added:
            ops.add("token(s) added")
            self.tab[f"{field}: tokens added", c].update(added)
        if dropped:
            ops.add("token(s) dropped")
            self.tab[f"{field}: tokens dropped", c].update(dropped)
        for x in added:
            for y in dropped:
                if len(x) == len(y):
                    diff = [(p, q) for p, q in zip(y, x) if p != q]
                    if all(q.isdigit() and p.isalpha() for p, q in diff):
                        ops.add("digit↔letter substitution")
                        self.tab[f"{field}: digit substitutions (letter→digit)", c].update(f"{p}→{q}" for p, q in diff)
                        break
            else:
                if len(x) >= 4 and x.isalpha() and any(len(y) >= 4 and Levenshtein.distance(x, y) <= 2 for y in dropped):
                    ops.add("char typo (<=2 edits, token-level)")
        if ta == tb:
            ops.add("no token change (case/accent/punct only)")
        return ops, added, dropped

    def name(self, a: str, b: str, c: str, lex: Lex) -> None:
        fa, fb = fold(a), fold(b)
        ta, tb = TOK.findall(fa), TOK.findall(fb)
        ops, added, dropped = self.common("name", a, b, ta, tb, c)
        la = {lex.legal[t] for t in ta if t in lex.legal}
        lb = {lex.legal[t] for t in tb if t in lex.legal}
        if la and not lb:
            ops.add("legal-form drop")
        elif lb and not la:
            ops.add("legal-form add")
        elif la != lb:
            ops.add("legal-form swap")
        elif la and {t for t in ta if t in lex.legal} != {t for t in tb if t in lex.legal}:
            ops.add("legal-form spelling change (same canonical)")
            self.tab["legal spelling pairs (S1→match)", c].update(
                f"{y}→{x}" for y in dropped if y in lex.legal for x in added if lex.legal.get(x) == lex.legal[y])
        for y in dropped:  # unmapped look-alikes of a dropped legal word = lexicon gaps
            if y in lex.legal:
                self.tab["legal-form variant candidates (unmapped, ratio>=60)", c].update(
                    f"{y}→{x}" for x in added if x not in lex.legal and ratio(x, y) >= 60)
        if DOTTED.search(b.lower()) and not DOTTED.search(a.lower()):
            ops.add("dotted acronym (L.L.C.)")
        if BRACKET.search(b) and not BRACKET.search(a):
            ops.add("bracket wrapping")
        if HYPHEN.search(b) and not HYPHEN.search(a):
            ops.add("hyphen join")
        if "&" in a and "&" not in b and ("+" in b or " and " in f" {fb} "):
            ops.add("& → + / and")
        m = ALIAS.match(fb)
        if m and not ALIAS.match(fa):
            kw = m["kw"]
            ops.add(f"alias prefix: {kw}")
            right = token_set_ratio(m["core"], fa) >= token_set_ratio(m["alias"], fa)
            self.tab["alias keyword: S1 name on right side", c][(kw, right)] += 1
        if tb and tb[0] in HON_CAND and (not ta or ta[0] != tb[0]):
            ops.add(f"honorific add: {tb[0]}")
        if ta and ta[0] in HON_CAND:
            self.tab["S1 names starting with honorific candidate", c][ta[0]] += 1
        if PHONE.search(b) and not re.search(r"\d\s*$", a):
            ops.add("trailing digit run (phone-like, >=7)")
        if DOMAIN.search(fb) and "com" not in fa:
            ops.add("domain form (x.com)")
        if MARKER.search(fb) and not MARKER.search(fa):
            ops.add("country marker add ((india)/(france))")
        elif MARKER.search(fa) and not MARKER.search(fb):
            ops.add("country marker drop")
        if token_set_ratio(fa, fb) < 40:
            ops.add("name replaced (token_set_ratio<40)")
        self.record("name", ops, c, a, b)

    def address(self, a: str, b: str, c: str, lex: Lex) -> None:
        if not b.strip():
            self.record("address", {"empty address"}, c, a, b)
            return
        fa, fb = fold(a), fold(b)
        ta, tb = TOK.findall(fa), TOK.findall(fb)
        ops, added, dropped = self.common("address", a, b, ta, tb, c)
        comps = lambda f: [x.strip() for x in f.split(",") if x.strip()]  # noqa: E731
        raw_a, raw_b = comps(fa), comps(fb)
        if any(NULL_COMP.match(x) for x in raw_b):
            ops.add("null token component (null/<null>/n/a)")
        if NUM_PREFIX.search(fb) and not NUM_PREFIX.search(fa):
            ops.add("number prefix add (#/no/ndeg/h.no/door no)")
        elif NUM_PREFIX.search(fa) and not NUM_PREFIX.search(fb):
            ops.add("number prefix drop")
        na = next((m for x in raw_a if (m := NUMRX.match(x))), None)
        nb = next((m for x in raw_b if (m := NUMRX.match(x))), None)
        if na and not nb:
            ops.add("house number: dropped")
        elif na and nb and na.group(1, 2) != nb.group(1, 2):
            x, y = na[1], nb[1]
            if x.lstrip("0") == y.lstrip("0"):
                ops.add("house number: zero-padded" if y != x else "house number: letter suffix change")
            elif len(y) < len(x) and (x.startswith(y) or x.endswith(y)):
                ops.add("house number: digit(s) dropped")
            else:
                ops.add("house number: other change")
        for y in dropped:
            for x in added:
                if lex.street.get(x, x) == lex.street.get(y, y):
                    ops.add("street-type abbr/expand")
                    self.tab["street-type pairs (S1→match)", c][f"{y}→{x}"] += 1
                if ORD.match(y) and ORDINALS.get(x) == y or ORD.match(x) and ORDINALS.get(y) == x:
                    ops.add("ordinal digit↔word")
        norm = lambda xs: [re.sub(r"[^a-z0-9]+", " ", x).strip() for x in xs]  # noqa: E731
        ca_, cb_ = norm(raw_a), norm(raw_b)
        sa = {lex.region[x]: x for x in ca_ if x in lex.region}
        sb = {lex.region[x]: x for x in cb_ if x in lex.region}
        for canon, form in sa.items():
            if canon in sb and sb[canon] != form:
                ops.add("state/region form change (full↔abbr)")
                self.tab["state/region form pairs (S1→match)", c][f"{form}→{sb[canon]}"] += 1
            elif canon not in sb:
                ops.add("state/region missing or unrecognised in match")
                full = max((k for k, v in lex.region.items() if v == canon), key=len)
                self.tab["state/region variant candidates (unmapped, ratio>=50)", c].update(
                    f"{canon}→{x}" for x in cb_ if x not in ca_ and not re.search(r"\d", x) and ratio(x, full) >= 50)
        if LANDMARK.search(fb) and not LANDMARK.search(fa):
            ops.add("landmark phrase add (near/opp/behind)")
        elif LANDMARK.search(fa) and not LANDMARK.search(fb):
            ops.add("landmark phrase drop")
        if ca_ != cb_ and sorted(ca_) == sorted(cb_):
            ops.add("component reorder (same components)")
        elif [x for x in ca_ if x in cb_] != [x for x in cb_ if x in ca_]:
            ops.add("component reorder (shared components)")
        if set(ca_) - set(cb_):
            ops.add("component(s) dropped")
        if set(cb_) - set(ca_):
            ops.add("component(s) added")
        if UNIT.search(fa) and not UNIT.search(fb):
            ops.add("unit/apt/floor dropped")
        if POBOX.search(fb) and not POBOX.search(fa):
            ops.add("po box / pmb added")
        self.record("address", ops, c, a, b)


def sample_pairs() -> list[tuple]:
    pairs = load_gt_pairs().sample(N_PAIRS, seed=SEED)
    s1 = load_source("train", 1)
    s1 = s1[s1["entity_id"].isin(set(pairs["s1_id"]))]
    want = set(pairs["match_id"])
    m = [df[df["entity_id"].isin(want)] for df in (load_source("train", 2), load_source("train", 3))]
    rec = {i: (nm, ad, co) for df in (s1, *m)
           for i, nm, ad, co in zip(df["entity_id"], df["business_name"], df["business_address"], df["country"])}
    return [(rec[a], rec[b]) for a, b in zip(pairs["s1_id"], pairs["match_id"])]


def write(mn: Miner) -> None:
    countries = sorted(mn.n)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Noise operators (M3a)\nGenerated by `src/noise_ops.py`, seed 42. "
                f"{N_PAIRS:,} sampled train true pairs (S1 → matched S2/S3 record): "
                + ", ".join(f"{c} {mn.n[c]:,}" for c in countries) + ".\n"
                "freq% = share of that country's sampled pairs where the operator fires (S1 → match "
                "direction). Operators overlap. Numbers + examples only.\n\n")
        f.write("## Operators\n| operator | field | country | freq% | examples (S1 → match) |\n|---|---|---|---|---|\n")
        for field, op, c in sorted(mn.hits, key=lambda k: (k[0] != "name", k[1], k[2])):
            k = (field, op, c)
            f.write(f"| {op} | {field} | {c} | {100 * mn.hits[k] / mn.n[c]:.2f} | {'<br>'.join(mn.ex[k])} |\n")
        f.write("\n## Token tables (counts over sampled pairs)\n")
        for table, c in sorted(mn.tab):
            items = mn.tab[table, c].most_common(TOP)
            if table.startswith("alias"):
                items = [(f"{kw}: right={right}", n) for (kw, right), n in items]
            f.write(f"\n**{table} — {c}** (top {TOP}): {items}\n")


def main() -> None:
    mn = Miner()
    for (a_name, a_addr, c), (b_name, b_addr, _) in sample_pairs():
        mn.n[c] += 1
        lex = LEX[c]
        mn.name(a_name, b_name, c, lex)
        mn.address(a_addr, b_addr, c, lex)
    write(mn)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
