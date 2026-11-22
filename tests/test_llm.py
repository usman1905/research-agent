import httpx
import pytest

from webresearch import llm as llm_mod
from webresearch.llm import LLM, LLMError


def _ok(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_falls_back_when_main_model_is_busy(monkeypatch):
    calls = []

    def fake_post(self, model, *a):
        calls.append(model)
        return httpx.Response(503, text="busy") if model == "main" else _ok("hi")

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    out = LLM("k", model="main", fallbacks=("backup",), retries=1).chat("s", "u")
    assert out == "hi"
    assert calls == ["main", "main", "backup"]


def test_bad_key_fails_fast(monkeypatch):
    monkeypatch.setattr(LLM, "_post", lambda self, *a: httpx.Response(401))
    with pytest.raises(LLMError, match="rejected"):
        LLM("k", fallbacks=("b",)).chat("s", "u")


def test_gemini_gets_default_fallbacks():
    llm = LLM.from_env({"LLM_API_KEY": "k", "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai"})
    assert llm.fallbacks == llm_mod.GEMINI_FALLBACKS


def test_timeout_moves_to_next_model(monkeypatch):
    calls = []

    def fake_post(self, model, *a):
        calls.append(model)
        return None if model == "slow" else _ok("done")

    monkeypatch.setattr(LLM, "_post", fake_post)
    assert LLM("k", model="slow", fallbacks=("fast",)).chat("s", "u") == "done"
    assert calls == ["slow", "fast"]


def test_search_falls_back_to_wikipedia(monkeypatch):
    import ddgs

    from webresearch import web

    class Blocked:
        def text(self, *a, **k):
            raise RuntimeError("blocked")

    monkeypatch.setattr(ddgs, "DDGS", Blocked)
    monkeypatch.setattr(web, "wikipedia_search", lambda q, n: [web.SearchHit("RAG", "https://en.wikipedia.org/wiki/RAG", "")])
    assert web.search("rag")[0].url.endswith("/RAG")
