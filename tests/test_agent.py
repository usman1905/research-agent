import pytest

from webresearch.agent import ResearchAgent
from webresearch.llm import LLMError, parse_json_list
from webresearch.web import Page, SearchHit, dedupe


class FakeLLM:
    def __init__(self, plan, answer):
        self.plan_reply, self.answer_reply, self.prompts = plan, answer, []

    def chat(self, system, user, temperature=0.2):
        self.prompts.append(user)
        return self.plan_reply if "plan web searches" in system else self.answer_reply


PAGES = {
    "https://a.com/steel": "316L stainless steel contains molybdenum, which helps it resist corrosion and tarnish.",
    "https://b.org/guide": "304 steel is cheaper but can tarnish faster when exposed to sweat and salt water.",
    "https://c.net/cats": "Cats sleep for most of the day.",
}


def fake_search(query):
    return [SearchHit(url.split("/")[2], url, "") for url in PAGES]


def fake_read_all(hits):
    return [Page(h.title, h.url, PAGES[h.url]) for h in hits]


def make_agent(answer):
    llm = FakeLLM('```json\n["316L vs 304 steel", "steel jewellery tarnish"]\n```', answer)
    return ResearchAgent(llm, search=fake_search, read_all=fake_read_all), llm


def test_full_research_run_with_checked_citations():
    answer = "316L resists corrosion and tarnish [{a}]. 304 steel is cheaper and can tarnish [{b}]. Made up claim [7]."
    agent, llm = make_agent(answer)
    steps = []
    ids = {"https://a.com/steel": 0, "https://b.org/guide": 0}
    probe = agent.run("Is 316L better than 304 for jewellery?")
    for src in probe.sources:
        ids[src.url] = src.id
    llm.answer_reply = llm.answer_reply.format(a=ids["https://a.com/steel"], b=ids["https://b.org/guide"])
    result = agent.run("Is 316L better than 304 for jewellery?", step=steps.append)

    assert result.queries == ["316L vs 304 steel", "steel jewellery tarnish"]
    assert {s.url for s in result.sources} == {"https://a.com/steel", "https://b.org/guide"}
    assert [s.id for s in result.sources] == [1, 2]
    assert "Cats" not in llm.prompts[-1], "irrelevant pages are not sent to the model"
    assert "[7]" not in result.answer, "citations to unknown sources are removed"
    assert [c.status for c in result.checks][:2] == ["supported", "supported"]
    assert result.supported_share == 1.0
    assert steps[0] == "Planning searches" and steps[-1] == "Checking every citation"


def test_plan_falls_back_to_the_question():
    llm = FakeLLM("no json here", "")
    assert ResearchAgent(llm).plan("my question") == ["my question"]


def test_empty_question_is_rejected():
    agent, _ = make_agent("")
    with pytest.raises(ValueError):
        agent.run("   ")


def test_no_results_gives_a_clear_message():
    agent = ResearchAgent(FakeLLM('["q"]', ""), search=lambda q: [], read_all=fake_read_all)
    assert "couldn't find" in agent.run("anything").answer


def test_failed_search_query_is_skipped():
    calls = []

    def flaky(q):
        calls.append(q)
        if len(calls) == 1:
            raise RuntimeError("rate limited")
        return fake_search(q)

    agent = ResearchAgent(FakeLLM('["a", "b"]', "316L resists corrosion [1]."), search=flaky, read_all=fake_read_all)
    assert agent.run("steel").sources


def test_dedupe_limits_sites_and_skips_files():
    hits = [
        SearchHit("", u, "")
        for u in [
            "https://x.com/1",
            "https://www.x.com/2",
            "https://x.com/3",
            "https://x.com/1#top",
            "https://y.com/a.pdf",
            "https://z.com/",
        ]
    ]
    assert [h.url for h in dedupe(hits, 10)] == ["https://x.com/1", "https://www.x.com/2", "https://z.com/"]


def test_parse_json_list_and_missing_key(monkeypatch):
    assert parse_json_list('Sure: ["a", " b ", ""]') == ["a", "b"]
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from webresearch.llm import LLM

    with pytest.raises(LLMError):
        LLM.from_env({})
