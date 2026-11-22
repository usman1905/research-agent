"""Minimal client for any OpenAI-compatible chat API (Groq, Gemini, OpenAI, OpenRouter, local servers)."""

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
# Stable aliases tried when the main Gemini model is overloaded.
GEMINI_FALLBACKS = ("gemini-flash-latest", "gemini-flash-lite-latest")
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLM:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 60.0
    fallbacks: tuple[str, ...] = field(default_factory=tuple)
    retries: int = 2

    @classmethod
    def from_env(cls, secrets: dict | None = None) -> "LLM":
        """Reads LLM_API_KEY, LLM_BASE_URL and LLM_MODEL from Streamlit secrets or the environment."""
        get = lambda k, d=None: (secrets or {}).get(k) or os.environ.get(k) or d  # noqa: E731
        key = get("LLM_API_KEY") or get("GROQ_API_KEY")
        if not key:
            raise LLMError("No API key found. Set LLM_API_KEY (a free Groq key works).")
        base_url = get("LLM_BASE_URL", DEFAULT_BASE_URL)
        raw = get("LLM_FALLBACK_MODELS", "")
        fallbacks = tuple(m.strip() for m in str(raw).split(",") if m.strip())
        if not fallbacks and "generativelanguage.googleapis.com" in base_url:
            fallbacks = GEMINI_FALLBACKS
        return cls(key, base_url, get("LLM_MODEL", DEFAULT_MODEL), fallbacks=fallbacks)

    def _post(self, model: str, system: str, user: str, temperature: float) -> httpx.Response:
        try:
            return httpx.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach the language model: {e}") from e

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Ask the model; retries busy models with backoff, then falls back to other models."""
        models = [self.model, *(m for m in self.fallbacks if m != self.model)]
        last = None
        for model in models:
            for attempt in range(self.retries + 1):
                res = self._post(model, system, user, temperature)
                if res.status_code < 400:
                    return res.json()["choices"][0]["message"]["content"].strip()
                if res.status_code in (401, 403):
                    raise LLMError("The API key was rejected. Check LLM_API_KEY.")
                last = res
                if res.status_code not in RETRYABLE:
                    break  # e.g. unknown model: try the next one
                if attempt < self.retries:
                    time.sleep(1.5 * 2**attempt)
        if last is not None and last.status_code == 429:
            raise LLMError("The model's rate limit was hit. Wait a minute and try again.")
        if last is not None and last.status_code in RETRYABLE:
            raise LLMError("The language model is busy right now. Try again in a minute.")
        code = last.status_code if last is not None else "?"
        text = last.text[:200] if last is not None else ""
        raise LLMError(f"The model returned an error ({code}): {text}")


def parse_json_list(text: str) -> list[str]:
    """Pull a JSON list of strings out of a model reply, tolerating code fences and extra prose."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in data if str(x).strip()]
