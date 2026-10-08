"""Web search tool used to complement chatbot answers.

Provider order: Tavily (best for LLM use) -> Wikipedia (free, no key).
Results share one shape: {title, url, content, source}
"""

from __future__ import annotations

import re

from . import config
from .credibility import domain_of
from .http_utils import APIError, request_json


def _tavily(query: str, news_only: bool, max_results: int) -> list[dict]:
    key = config.get_key(config.TAVILY_API_KEY)
    if not key:
        raise APIError("Tavily", "missing_key")
    body = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "topic": "news" if news_only else "general",
        "include_answer": False,
    }
    data = request_json("Tavily", "POST", "https://api.tavily.com/search",
                        json_body=body, headers={"Authorization": f"Bearer {key}"})
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:800],
            "source": domain_of(r.get("url", "")),
            "published_at": r.get("published_date"),
        }
        for r in data.get("results", [])
        if r.get("url")
    ]


def _wikipedia(query: str, news_only: bool, max_results: int) -> list[dict]:
    if news_only:  # an encyclopedia is not a news source
        return []
    data = request_json("Wikipedia", "GET", "https://en.wikipedia.org/w/api.php",
                        params={"action": "query", "list": "search", "srsearch": query,
                                "format": "json", "srlimit": max_results})
    results = []
    for hit in data.get("query", {}).get("search", []):
        title = hit.get("title", "")
        snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
        results.append({
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "content": snippet,
            "source": "wikipedia.org",
            "published_at": None,
        })
    return results


PROVIDERS = [("Tavily", _tavily), ("Wikipedia", _wikipedia)]


def search_web(query: str, news_only: bool = False, max_results: int = 5) -> dict:
    """Never raises. Returns {results, provider, errors}."""
    errors: list[str] = []
    for name, provider in PROVIDERS:
        try:
            results = provider(query, news_only, max_results)
        except APIError as exc:
            if exc.kind != "missing_key":
                errors.append(exc.user_message())
            continue
        if results:
            return {"results": results, "provider": name, "errors": errors}
    return {"results": [], "provider": "", "errors": errors}
