"""Checks every citation in an answer against the passages it points to."""

import re
from dataclasses import dataclass

from .text import sentences, tokens

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class CitationCheck:
    sentence: str
    cited: list[int]
    support: float  # share of the sentence's key words found in the cited sources
    status: str  # "supported", "weak", "invalid" or "uncited"


def cited_ids(sentence: str) -> list[int]:
    ids = []
    for group in CITE.findall(sentence):
        ids.extend(int(x) for x in group.split(","))
    return sorted(set(ids))


def support_score(claim: str, evidence: str) -> float:
    """Share of the claim's content words (minus citation markers) that appear in the evidence."""
    words = set(tokens(CITE.sub(" ", claim)))
    if not words:
        return 1.0
    found = set(tokens(evidence))
    return len(words & found) / len(words)


def check(answer: str, sources: dict[int, str], threshold: float = 0.5) -> list[CitationCheck]:
    """Grade each sentence of the answer.

    `sources` maps a citation number to the text the model was given for that source.
    """
    results = []
    for s in sentences(answer):
        ids = cited_ids(s)
        if not ids:
            results.append(CitationCheck(s, [], 0.0, "uncited"))
            continue
        if any(i not in sources for i in ids):
            results.append(CitationCheck(s, ids, 0.0, "invalid"))
            continue
        score = support_score(s, " ".join(sources[i] for i in ids))
        results.append(CitationCheck(s, ids, score, "supported" if score >= threshold else "weak"))
    return results


def strip_invalid(answer: str, valid: set[int]) -> str:
    """Remove citation numbers that do not match any source, so readers never follow a dead link."""

    def fix(m: re.Match) -> str:
        ids = [x.strip() for x in m.group(2).split(",") if int(x) in valid]
        return f"{m.group(1)}[{', '.join(ids)}]" if ids else ""

    return re.sub(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]", fix, answer)
