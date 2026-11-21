"""Text helpers: tokenising, splitting pages into passages and sentences."""

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    may more most not of on or our so such than that the their them then there these they this to was we were
    what when where which while who why will with would you your about also after all any before between both
    each other over same some only own very just
    """.split()  # noqa: SIM905
)


def tokens(text: str) -> list[str]:
    """Lower-case word tokens without stopwords."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping citation markers with their sentence."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def passages(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Split a page into overlapping passages of about `size` words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if len(chunk) < overlap and out:
            break
        out.append(" ".join(chunk))
    return out
