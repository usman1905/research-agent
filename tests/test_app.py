"""Drives the Streamlit app with a fake model and fake web, so the whole page is tested without network access."""

from unittest import mock

from streamlit.testing.v1 import AppTest

from webresearch.web import Page, SearchHit

PAGE = Page("Steel guide", "https://a.com/steel", "316L stainless steel resists corrosion and tarnish better than 304 steel.")


def fake_chat(self, system, user, temperature=0.2):
    if "plan web searches" in system:
        return '["316L vs 304"]'
    return "316L stainless steel resists corrosion and tarnish [1]."


def run_app(env):
    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch("webresearch.llm.LLM.chat", fake_chat),
        mock.patch("webresearch.web.search", lambda q: [SearchHit(PAGE.title, PAGE.url, "")]),
        mock.patch("webresearch.web.read_all", lambda hits: [PAGE]),
    ):
        at = AppTest.from_file("../app.py", default_timeout=30).run()
        if at.text_area:
            at.text_area(key="question").input("Is 316L better than 304?")
            at.button[-1].click().run()
        return at


def test_app_answers_with_linked_citations_and_sources():
    at = run_app({"LLM_API_KEY": "test-key"})
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "[[1]](https://a.com/steel)" in page
    assert "Steel guide" in page
    assert any("1 of 1 cited sentences" in c.value for c in at.caption)


def test_app_explains_a_missing_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert at.warning and "API key" in at.warning[0].value
