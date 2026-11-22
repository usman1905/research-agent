"""Web search (DuckDuckGo, no API key) and page reading."""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; WebResearchAI/1.0; +https://github.com/usman1905/research-agent)"
SKIP_EXT = (".pdf", ".zip", ".mp4", ".mp3", ".jpg", ".png", ".gif")


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


@dataclass
class Page:
    title: str
    url: str
    text: str


# Tried in order. Cloud hosts are often blocked by one engine but not another.
BACKENDS = ("auto", "brave,mojeek,yahoo,startpage")


def search(query: str, max_results: int = 6) -> list[SearchHit]:
    """Search the web, trying several engines, then Wikipedia as a last resort."""
    from ddgs import DDGS  # imported lazily so tests and the UI load without network libraries

    for backend in BACKENDS:
        try:
            hits = DDGS().text(query, max_results=max_results, backend=backend) or []
        except Exception:  # an engine refusing us should not end the search
            continue
        out = [SearchHit(h.get("title", ""), h.get("href", ""), h.get("body", "")) for h in hits if h.get("href")]
        if out:
            return out
    return wikipedia_search(query, max_results)


def wikipedia_search(query: str, max_results: int = 6) -> list[SearchHit]:
    """Wikipedia's free search API, which works from cloud servers."""
    try:
        res = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": max_results, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        res.raise_for_status()
        items = res.json().get("query", {}).get("search", [])
    except (httpx.HTTPError, ValueError):
        return []
    strip = lambda t: re.sub(r"<[^>]+>", "", t)  # noqa: E731
    return [
        SearchHit(i["title"], "https://en.wikipedia.org/wiki/" + quote(i["title"].replace(" ", "_")), strip(i.get("snippet", "")))
        for i in items
    ]


def domain(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def dedupe(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """Unique URLs, at most two per site, skipping files that are not web pages."""
    seen, per_site, out = set(), {}, []
    for h in hits:
        key = h.url.split("#")[0].rstrip("/")
        site = domain(h.url)
        if key in seen or per_site.get(site, 0) >= 2 or key.lower().endswith(SKIP_EXT):
            continue
        seen.add(key)
        per_site[site] = per_site.get(site, 0) + 1
        out.append(h)
        if len(out) == limit:
            break
    return out


def read(hit: SearchHit, timeout: float = 12.0, max_chars: int = 20000) -> Page | None:
    """Download a page and keep its main text. Falls back to the search snippet if extraction fails."""
    import trafilatura

    try:
        res = httpx.get(hit.url, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)
        res.raise_for_status()
        text = trafilatura.extract(res.text, include_comments=False, include_tables=False) or ""
    except (httpx.HTTPError, ValueError):
        text = ""
    text = text.strip() or hit.snippet
    return Page(hit.title, hit.url, text[:max_chars]) if text else None


def read_all(hits: list[SearchHit], workers: int = 6) -> list[Page]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p for p in pool.map(read, hits) if p]
