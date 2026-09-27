"""Text normalisation for business names and addresses.

Design (see docs/breakdown.md Revision 3):
- ONE rulebook applied to every record regardless of the `country` label (country is an open set;
  rules are never routed by label). French legal forms never occur in US names, so a union of
  US/India/France rules is harmless; where a token means different things in different places
  ("st" = street / saint) both long forms fold to the same short form on BOTH sides, so the
  mapping is symmetric and never needs to know the country.
- Canonical space = lowercase ASCII (anyascii transliterates Devanagari/Tamil/... and folds accents).
- Names are split into: parts (DBA / formerly / nee / aka), legal-form tokens, and a core.
- A consonant "skeleton" key bridges phonetic transliteration ("sivm istrn" ~ "shivam eastern").
"""
from __future__ import annotations

import re

from anyascii import anyascii

NORM_VERSION = 1  # bump when any rule changes -> invalidates interim caches

# ---------------------------------------------------------------- shared pieces
_FRANCE_MARK = re.compile(r"\(\s*fr[a-z]{3,5}\s*\)")        # "(France)", "(Frence)" injected noise
_AMP = re.compile(r"\s*[&+]\s*")
_APOS = re.compile(r"['`´]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ALNUM_SPLIT = re.compile(r"(?<=[a-z])(?=[0-9]{2,})|(?<=[0-9])(?=[a-z]{3,})")  # "shop12" "12thfloor"-ish

# multi-word phrases folded BEFORE tokenising (applied on the space-normalised string)
_PHRASES_COMMON = [
    ("doing business as", "dba"), ("also known as", "aka"), ("formerly known as", "formerly"),
    ("trading as", "dba"), ("private limited", "pvt ltd"), ("public limited", "public ltd"),
    ("limited liability partnership", "llp"), ("limited liability company", "llc"),
    ("professional limited liability company", "pllc"),
]
_PHRASES_ADDR = [
    ("uttar pradesh", "up"), ("madhya pradesh", "mp"), ("andhra pradesh", "ap"),
    ("himachal pradesh", "hp"), ("arunachal pradesh", "ar"), ("tamil nadu", "tn"),
    ("west bengal", "wb"), ("jammu and kashmir", "jk"), ("new delhi", "delhi"),
    ("new york", "ny"), ("new jersey", "nj"), ("new mexico", "nm"), ("new hampshire", "nh"),
    ("north carolina", "nc"), ("south carolina", "sc"), ("north dakota", "nd"),
    ("south dakota", "sd"), ("west virginia", "wv"), ("rhode island", "ri"),
    ("district of columbia", "dc"), ("po box", "pobox"), ("p o box", "pobox"),
    ("hauts de france", "hdf"), ("nouvelle aquitaine", "naq"), ("pays de la loire", "pdl"),
    ("loire atlantique", "loireatl"), ("pas de calais", "pdc"),
]

# ---------------------------------------------------------------- name rules
_NAME_MAP = {
    # India legal forms incl. phonetic/OCR/transliteration noise seen in EDA D11
    "private": "pvt", "pvt": "pvt", "praivet": "pvt", "piraivet": "pvt", "praibhet": "pvt",
    "praivrr": "pvt", "pra": "pvt", "prvt": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "limitet": "ltd", "limirrd": "ltd", "li": "ltd", "lmited": "ltd",
    "elelpi": "llp",
    # US
    "incorporated": "inc", "corporation": "corp", "company": "co", "companies": "co",
    # France
    "compagnie": "cie", "etablissements": "ets", "etablissement": "ets", "societe": "ste",
    # generic
    "and": "and", "et": "and", "brothers": "bros", "international": "intl", "intl": "intl",
    "centre": "center", "technologies": "tech", "technology": "tech",
}
LEGAL = frozenset({
    "pvt", "ltd", "llp", "public", "opc",                          # India
    "llc", "inc", "corp", "co", "pllc", "lp", "llp", "pc", "pa", "plc", "dds", "md",  # US
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "ei", "cie", "selarl", "scp", "ets",  # FR
    "ste", "gmbh",
})
_HONORIFIC = frozenset({"the", "smt", "shri", "sri", "shree", "mr", "mrs", "ms", "dr", "com", "www", "m"})
_PART_WORDS = frozenset({"dba", "formerly", "nee", "aka", "fka"})
_PART_SPLIT = re.compile(r"\b(?:dba|formerly|nee|aka|fka)\b")

# ---------------------------------------------------------------- address rules
_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
# NB: US-state and Indian-state codes collide ("tn" = Tennessee / Tamil Nadu, "ar", "ap"...).
# Harmless: candidates never cross a country group, and the collision is symmetric.
_IN_STATES = {
    "maharashtra": "mh", "mharastr": "mh", "maharastra": "mh", "delhi": "dl", "dilli": "dl",
    "karnataka": "ka", "gujarat": "gj", "gujrat": "gj", "telangana": "tg", "kerala": "kl",
    "rajasthan": "rj", "haryana": "hr", "bihar": "br", "punjab": "pb", "odisha": "od",
    "orissa": "od", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "goa": "ga", "bengal": "wb", "prdes": "pradesh", "pradesh": "pradesh",
}
_ADDR_MAP = {
    "street": "st", "str": "st", "saint": "st", "sainte": "ste", "suite": "ste",
    "road": "rd", "drive": "dr", "avenue": "ave", "av": "ave", "lane": "ln",
    "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl", "circle": "cir",
    "parkway": "pkwy", "highway": "hwy", "apartment": "apt", "floor": "fl", "flr": "fl",
    "building": "bldg", "sector": "sec", "opposite": "opp", "nr": "near",
    "rue": "rue", "r": "rue", "route": "rte", "allee": "allee", "all": "allee",
    "chemin": "chemin", "impasse": "imp", "square": "sq", "mount": "mt", "fort": "ft",
    "north": "n", "south": "s", "east": "e", "west": "w",
    **_US_STATES, **_IN_STATES,
}
_ADDR_DROP = frozenset({"no", "ndeg", "number", "num", "null", "nan", "none", "na"})

# ---------------------------------------------------------------- skeleton
_SK_DIGRAPH = re.compile(r"(?<=[bcdfgjklmnpqrstvxz])h")       # kh gh sh th dh bh ph -> drop h
_SK_MAP = str.maketrans({"c": "k", "q": "k", "z": "s", "w": "v", "m": "n", "x": "k"})
_SK_VOWEL = re.compile(r"[aeiouy]")
_SK_REPEAT = re.compile(r"(.)\1+")


def skeleton_token(t: str) -> str:
    if not t or t.isdigit():
        return t
    s = t.replace("ph", "f")
    s = _SK_DIGRAPH.sub("", s).translate(_SK_MAP)
    s = _SK_REPEAT.sub(r"\1", _SK_VOWEL.sub("", s))
    return s or t[:1]


# ---------------------------------------------------------------- core helpers
def _join_initials(tokens: list[str]) -> list[str]:
    """'e u r l' -> 'eurl', 'l l c' -> 'llc': join runs of >=2 single letters."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def _basic(s: str) -> str:
    s = anyascii(s).lower()
    s = _AMP.sub(" and ", s)
    s = _APOS.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    return _ALNUM_SPLIT.sub(" ", s).strip()


def _apply_phrases(s: str, phrases) -> str:
    s = f" {s} "
    for a, b in phrases:
        if a in s:
            s = s.replace(f" {a} ", f" {b} ")
    return s.strip()


def _map_tokens(tokens, mapping):
    out = []
    for t in tokens:
        m = mapping.get(t)
        if m is None:
            out.append(t.lstrip("0") or "0" if t.isdigit() else t)
        else:
            out.extend(m.split())
    return out


def normalize_name(raw: str) -> dict:
    """Returns name_norm (all tokens), name_core (identity tokens), name_legal, name_parts, name_skel."""
    s = _FRANCE_MARK.sub(" ", anyascii(raw).lower())
    s = _apply_phrases(" ".join(_basic(s).split()), _PHRASES_COMMON)
    toks = _map_tokens(_join_initials(s.split()), _NAME_MAP)
    norm = " ".join(toks)
    parts = [p.split() for p in _PART_SPLIT.split(norm)]
    parts = [p for p in parts if p] or [toks]
    legal = sorted({t for t in toks if t in LEGAL})
    core_parts = [[t for t in p if t not in LEGAL and t not in _HONORIFIC] for p in parts]
    core_parts = [c for c in core_parts if c]
    if not core_parts:                                   # name made only of legal/honorific words
        core_parts = [[t for t in toks if t not in _PART_WORDS] or toks]
    core = " ".join(t for c in core_parts for t in c)
    return {
        "name_norm": norm,
        "name_core": core,
        "name_legal": " ".join(legal),
        "name_parts": " | ".join(" ".join(c) for c in core_parts),
        "name_skel": " ".join(skeleton_token(t) for t in core.split()),
    }


def normalize_address(raw: str) -> dict:
    s = _apply_phrases(" ".join(_basic(raw).split()), _PHRASES_ADDR)
    toks = [t for t in _map_tokens(_join_initials(s.split()), _ADDR_MAP) if t not in _ADDR_DROP]
    nums = sorted({t for t in toks if any(ch.isdigit() for ch in t)})
    return {
        "addr_norm": " ".join(toks),
        "addr_nums": " ".join(nums),
        "addr_skel": " ".join(skeleton_token(t) for t in toks if not t.isdigit()),
    }


def normalize_record(name: str, address: str) -> dict:
    return {**normalize_name(name), **normalize_address(address)}


if __name__ == "__main__":
    for n, a in [
        ("Shivam Eastern Industries Limited", "Plot 4, Near SBI ATM, Andheri West, Mumbai, Maharashtra"),
        ("शिवम ईस्टर्न इंडस्ट्रीज लिमिटेड", "प्लॉट 4, अंधेरी, मुंबई, महाराष्ट्र"),
        ("Novisolx née Dermatology Green Associates Care", "1601 142nd Street, Lubbock, Texas"),
        ("Asso Etablissement (Frànce) SARL", "N° 14 R. DE BOULOGNE, TOURCOING, Hauts-de-France"),
        ("Team Ecole E.U.R.L.", "029 H Rue Eugene Jacquet, Lille"),
        ("Barnes, Rútherford + Konkle Salon ((Inc))", "138 Madison Lane, Le Roy, WV"),
        ("Ps Finance Private Limited Limited", ""),
        ("M/S Raj Agro Pvt. Ltd.", "null"),
    ]:
        print(n, "|", a)
        for k, v in normalize_record(n, a).items():
            print(f"    {k:11s} {v}")
