"""BM25 ranking of passages against the question. No external dependencies."""

import math
from collections import Counter
from dataclasses import dataclass

from .text import tokens


@dataclass(frozen=True)
class Passage:
    source_id: int
    text: str


def bm25(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of each document for the query."""
    q = tokens(query)
    doc_tokens = [tokens(d) for d in docs]
    n = len(docs)
    if not n or not q:
        return [0.0] * n
    avg_len = sum(len(d) for d in doc_tokens) / n or 1.0
    df = Counter(t for d in doc_tokens for t in set(d))
    scores = []
    for d in doc_tokens:
        tf = Counter(d)
        score = 0.0
        for term in dict.fromkeys(q):  # fixed order keeps scores identical run to run
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(score)
    return scores


def top_passages(question: str, candidates: list[Passage], k: int = 8, per_source: int = 3) -> list[Passage]:
    """Best passages for the question, with at most `per_source` from any one page."""
    scores = bm25(question, [p.text for p in candidates])
    ranked = sorted(zip(scores, range(len(candidates)), strict=True), key=lambda s: (-s[0], s[1]))
    picked, used = [], Counter()
    for score, i in ranked:
        p = candidates[i]
        if score <= 0 or used[p.source_id] >= per_source:
            continue
        picked.append(p)
        used[p.source_id] += 1
        if len(picked) == k:
            break
    return picked
