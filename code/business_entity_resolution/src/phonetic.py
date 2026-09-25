"""M3b phonetic skeleton: a lossy consonant key that makes transliteration/OCR variants collide.

skeleton(): sh->s, ph->f, w->v, c/q->k, z->j, m/n before a consonant -> n, drop vowels except the first
char (a leading vowel becomes the marker "a", so istrn ~ eastern), collapse repeats. A non-initial y counts
as a vowel (sistms ~ systems, keyr ~ care): in transliterated English it stands for i/e. Tokens with a digit
are returned unchanged (collapsing "1100" -> "10" would merge house numbers).
`add_skeletons(df)` maps name_tokens / addr_tokens -> name_skel_tokens / addr_skel_tokens through the
distinct-token vocabulary (Python runs once per distinct token, not per row).

  python -m src.phonetic   # unit asserts + a few examples
"""
import re

import polars as pl

VOWELS = frozenset("aeiou")
_SUBS = (("sh", "s"), ("ph", "f"), ("w", "v"), ("c", "k"), ("q", "k"), ("z", "j"))
_NASAL = re.compile(r"[mn](?=[b-df-hj-np-tvxz])")  # consonants after the subs; y excluded
_REPEAT = re.compile(r"(.)\1+")
_DROP_VOWELS = str.maketrans("", "", "aeiouy")


def skeleton(tok: str) -> str:
    if any(ch.isdigit() for ch in tok):
        return tok
    for a, b in _SUBS:
        tok = tok.replace(a, b)
    tok = _NASAL.sub("n", tok)
    head = "a" if tok[:1] in VOWELS else tok[:1]
    return _REPEAT.sub(r"\1", head + tok[1:].translate(_DROP_VOWELS))


def add_skeletons(df: pl.DataFrame) -> pl.DataFrame:
    vocab = pl.concat([df[c].explode(empty_as_null=True) for c in ("name_tokens", "addr_tokens")]).drop_nulls().unique()
    skel = pl.Series([skeleton(t) for t in vocab.to_list()], dtype=pl.String)

    def m(c: str) -> pl.Expr:
        return pl.col(c).list.eval(pl.element().replace_strict(vocab, skel))  # order kept (bigrams need it)
    return df.with_columns(name_skel_tokens=m("name_tokens"), addr_skel_tokens=m("addr_tokens"))


def _selftest() -> None:
    for a, b in [("praivet", "private"), ("sivm", "shivam"), ("istrn", "eastern")]:
        assert skeleton(a) == skeleton(b), (a, b, skeleton(a), skeleton(b))
    assert skeleton("1100") == "1100" and skeleton("bottuguda") == "btgd"
    assert skeleton("universal") != skeleton("international")
    assert skeleton("sistms") == skeleton("systems") and skeleton("keyr") == skeleton("care")


if __name__ == "__main__":
    _selftest()
    for t in ["praivet", "private", "sivm", "shivam", "istrn", "eastern", "sistms", "systems", "keyr", "care",
              "limitet", "limited", "chandra", "kumar", "photo", "foto", "ganesh", "ganesa"]:
        print(f"{t:>10} -> {skeleton(t)}")
    print("phonetic self-test OK")
