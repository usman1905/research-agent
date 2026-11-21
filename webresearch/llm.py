"""Minimal client for any OpenAI-compatible chat API (Groq, Gemini, OpenAI, OpenRouter, local servers)."""

import json
import os
import re
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMError(RuntimeError):
    pass


@dataclass
class LLM:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 60.0

    @classmethod
    def from_env(cls, secrets: dict | None = None) -> "LLM":
        """Reads LLM_API_KEY, LLM_BASE_URL and LLM_MODEL from Streamlit secrets or the environment."""
        get = lambda k, d=None: (secrets or {}).get(k) or os.environ.get(k) or d  # noqa: E731
        key = get("LLM_API_KEY") or get("GROQ_API_KEY")
        if not key:
            raise LLMError("No API key found. Set LLM_API_KEY (a free Groq key works).")
        return cls(key, get("LLM_BASE_URL", DEFAULT_BASE_URL), get("LLM_MODEL", DEFAULT_MODEL))

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        try:
            res = httpx.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach the language model: {e}") from e
        if res.status_code == 401:
            raise LLMError("The API key was rejected. Check LLM_API_KEY.")
        if res.status_code == 429:
            raise LLMError("The model's rate limit was hit. Wait a minute and try again.")
        if res.status_code >= 400:
            raise LLMError(f"The model returned an error ({res.status_code}): {res.text[:200]}")
        return res.json()["choices"][0]["message"]["content"].strip()


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
