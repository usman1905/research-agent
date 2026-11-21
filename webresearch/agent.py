"""The research loop: plan searches, read pages, pick evidence, answer with citations, check the citations."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from . import web
from .citations import CitationCheck, check, strip_invalid
from .llm import LLM, parse_json_list
from .rank import Passage, top_passages
from .text import passages

PLAN_PROMPT = """You plan web searches for a research assistant.
Today is {today}. Return a JSON list of 2 to 4 short search queries that together cover the user's question.
Return only the JSON list."""

ANSWER_PROMPT = """You are a careful research assistant. Answer the question using ONLY the numbered sources.
Rules:
- Put a citation like [1] or [2, 3] at the end of every sentence that states a fact.
- Only cite a source if it directly supports the sentence. Never invent sources or numbers.
- If the sources do not answer the question, say so plainly.
- Be concise: a short direct answer first, then the key details. Use plain English."""


@dataclass
class Source:
    id: int
    title: str
    url: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    question: str
    answer: str
    queries: list[str]
    sources: list[Source]
    checks: list[CitationCheck]

    @property
    def supported_share(self) -> float:
        cited = [c for c in self.checks if c.status != "uncited"]
        return sum(c.status == "supported" for c in cited) / len(cited) if cited else 0.0


Step = Callable[[str], None]


class ResearchAgent:
    def __init__(self, llm: LLM, search=None, read_all=None, max_pages: int = 6, max_passages: int = 8):
        self.llm = llm
        self.search = search or web.search
        self.read_all = read_all or web.read_all
        self.max_pages, self.max_passages = max_pages, max_passages

    def plan(self, question: str) -> list[str]:
        reply = self.llm.chat(PLAN_PROMPT.format(today=date.today().isoformat()), question, temperature=0.3)
        queries = parse_json_list(reply)[:4]
        return queries or [question]

    def run(self, question: str, step: Step = lambda _: None) -> ResearchResult:
        question = question.strip()
        if not question:
            raise ValueError("Ask a question first.")

        step("Planning searches")
        queries = self.plan(question)

        step(f"Searching the web for {len(queries)} queries")
        hits = []
        for q in queries:
            try:
                hits.extend(self.search(q))
            except Exception:  # one failed query should not stop the research
                continue
        hits = web.dedupe(hits, self.max_pages)
        if not hits:
            message = "I couldn't find any web results for this question. Try rephrasing it."
            return ResearchResult(question, message, queries, [], [])

        step(f"Reading {len(hits)} pages")
        pages = self.read_all(hits)

        step("Picking the most relevant passages")
        candidates = [Passage(i, chunk) for i, page in enumerate(pages) for chunk in passages(page.text)]
        best = top_passages(" ".join([question, *queries]), candidates, k=self.max_passages)
        if not best:
            return ResearchResult(question, "The pages I found didn't contain usable text for this question.", queries, [], [])

        # Number sources in the order they are first used, so citations read [1], [2], [3].
        order: dict[int, int] = {}
        for p in best:
            order.setdefault(p.source_id, len(order) + 1)
        sources = {n: Source(n, pages[i].title or web.domain(pages[i].url), pages[i].url) for i, n in order.items()}
        for p in best:
            sources[order[p.source_id]].evidence.append(p.text)

        step("Writing a cited answer")
        context = "\n\n".join(f"[{s.id}] {s.title} ({s.url})\n" + "\n...\n".join(s.evidence) for s in sources.values())
        answer = self.llm.chat(ANSWER_PROMPT, f"Question: {question}\n\nSources:\n{context}")

        step("Checking every citation")
        answer = strip_invalid(answer, set(sources))
        checks = check(answer, {n: " ".join(s.evidence) for n, s in sources.items()})
        return ResearchResult(question, answer, queries, list(sources.values()), checks)
