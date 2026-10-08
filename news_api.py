"""Real-time news retrieval with a provider chain and caching.

Provider order: NewsAPI -> GNews -> demo sample data (only if no keys at all).
Each provider returns articles in one common shape:
    {title, description, url, source, published_at, provider}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .http_utils import APIError, request_json

# Our category id -> provider category name
NEWSAPI_CATEGORIES = {
    "technology": "technology", "finance": "business", "sports": "sports",
    "health": "health", "science": "science", "entertainment": "entertainment",
    "general": "general", "world": "general",
}
GNEWS_CATEGORIES = {
    "technology": "technology", "finance": "business", "sports": "sports",
    "health": "health", "science": "science", "entertainment": "entertainment",
    "general": "general", "world": "world",
}

_cache: dict[tuple, tuple[float, list[dict]]] = {}
_SAMPLE_FILE = Path(__file__).parent / "data" / "sample_news.json"


class NewsResult(dict):
    """articles: list, provider: str, errors: list[str], demo: bool"""


# --- Providers -------------------------------------------------------------
def _newsapi(category: str, query: str | None) -> list[dict]:
    key = config.get_key(config.NEWSAPI_KEY)
    if not key:
        raise APIError("NewsAPI", "missing_key")
    headers = {"X-Api-Key": key}
    if query:
        data = request_json("NewsAPI", "GET", "https://newsapi.org/v2/everything",
                            params={"q": query, "language": "en", "sortBy": "publishedAt",
                                    "pageSize": 30}, headers=headers)
    else:
        data = request_json("NewsAPI", "GET", "https://newsapi.org/v2/top-headlines",
                            params={"category": NEWSAPI_CATEGORIES.get(category, "general"),
                                    "country": "us", "pageSize": 30}, headers=headers)
    if data.get("status") != "ok":
        code = data.get("code", "")
        kind = "auth" if "apiKey" in code else "rate_limit" if "rateLimited" in code else "bad_response"
        raise APIError("NewsAPI", kind, code)
    return [
        {
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "url": a.get("url") or "",
            "source": (a.get("source") or {}).get("name") or "",
            "published_at": a.get("publishedAt"),
            "provider": "NewsAPI",
        }
        for a in data.get("articles", [])
    ]


def _gnews(category: str, query: str | None) -> list[dict]:
    key = config.get_key(config.GNEWS_API_KEY)
    if not key:
        raise APIError("GNews", "missing_key")
    base = {"lang": "en", "max": 10, "apikey": key}
    if query:
        data = request_json("GNews", "GET", "https://gnews.io/api/v4/search",
                            params={**base, "q": query, "sortby": "publishedAt"})
    else:
        data = request_json("GNews", "GET", "https://gnews.io/api/v4/top-headlines",
                            params={**base, "country": "us",
                                    "category": GNEWS_CATEGORIES.get(category, "general")})
    if "articles" not in data:
        raise APIError("GNews", "bad_response", str(data.get("errors", ""))[:80])
    return [
        {
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "url": a.get("url") or "",
            "source": (a.get("source") or {}).get("name") or "",
            "published_at": a.get("publishedAt"),
            "provider": "GNews",
        }
        for a in data["articles"]
    ]


def _sample(category: str, query: str | None) -> list[dict]:
    """Bundled demo articles so the app can be shown without any API keys."""
    try:
        data = json.loads(_SAMPLE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    articles = data.get(category) or data.get("general", [])
    if query:
        words = [w for w in query.lower().split() if len(w) > 3]
        everything = [a for items in data.values() if isinstance(items, list) for a in items]
        matched = [a for a in everything
                   if any(w in (a["title"] + " " + a["description"]).lower() for w in words)]
        articles = matched or articles
    return [{**a, "provider": "Demo data"} for a in articles]


PROVIDERS = [("NewsAPI", _newsapi), ("GNews", _gnews)]


# --- Public entry point ----------------------------------------------------
def fetch_news(category: str = "general", query: str | None = None,
               use_cache: bool = True) -> NewsResult:
    """Try each provider in order and return the first non-empty result.

    Never raises: failures are collected in `errors` so the workflow can
    decide on a fallback (web search, or an honest "nothing found").
    """
    query = (query or "").strip() or None
    cache_key = (category, (query or "").lower())
    if use_cache and cache_key in _cache:
        stamp, articles = _cache[cache_key]
        if time.time() - stamp < config.NEWS_CACHE_TTL_SECONDS:
            return NewsResult(articles=articles, provider=articles[0]["provider"] if articles else "",
                              errors=[], demo=False, cached=True)

    errors: list[str] = []
    missing = 0
    for name, provider in PROVIDERS:
        try:
            articles = provider(category, query)
        except APIError as exc:
            if exc.kind == "missing_key":
                missing += 1
            else:
                errors.append(exc.user_message())
            continue
        if articles:
            _cache[cache_key] = (time.time(), articles)
            return NewsResult(articles=articles, provider=name, errors=errors,
                              demo=False, cached=False)
        errors.append(f"{name} found no articles")

    if missing == len(PROVIDERS):  # no news keys at all -> demo mode
        return NewsResult(articles=_sample(category, query), provider="Demo data",
                          errors=[], demo=True, cached=False)
    return NewsResult(articles=[], provider="", errors=errors, demo=False, cached=False)


def clear_cache() -> None:
    _cache.clear()
