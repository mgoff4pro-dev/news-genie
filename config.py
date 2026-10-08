"""Central configuration: API keys, model settings, and app constants.

Keys are looked up in this order:
  1. Values typed into the Streamlit sidebar (set via `set_overrides`)
  2. Environment variables (including a local .env file)
Missing keys never crash the app - each feature falls back gracefully.
"""

from __future__ import annotations

import os

try:  # .env support is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# --- Key names -------------------------------------------------------------
OPENAI_API_KEY = "OPENAI_API_KEY"
NEWSAPI_KEY = "NEWSAPI_KEY"      # https://newsapi.org (free developer tier)
GNEWS_API_KEY = "GNEWS_API_KEY"  # https://gnews.io (free tier) - backup news source
TAVILY_API_KEY = "TAVILY_API_KEY"  # https://tavily.com (free tier) - web search

ALL_KEYS = [OPENAI_API_KEY, NEWSAPI_KEY, GNEWS_API_KEY, TAVILY_API_KEY]

_overrides: dict[str, str] = {}


def set_overrides(values: dict[str, str]) -> None:
    """Store keys entered in the UI. Blank values are ignored."""
    for name, value in values.items():
        if value and value.strip():
            _overrides[name] = value.strip()
        else:
            _overrides.pop(name, None)


def get_key(name: str) -> str | None:
    """Return a key if it is configured, otherwise None (never raises)."""
    value = _overrides.get(name) or os.getenv(name, "")
    value = value.strip()
    # Treat obvious placeholders as missing so users get a clear message.
    if not value or value.lower().startswith(("your_", "sk-your", "xxx")):
        return None
    return value


def key_status() -> dict[str, bool]:
    return {name: get_key(name) is not None for name in ALL_KEYS}


# --- Model and behavior settings ------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT_SECONDS = 30

HTTP_TIMEOUT_SECONDS = 10
HTTP_MAX_RETRIES = 2          # retries after the first attempt
NEWS_CACHE_TTL_SECONDS = 600  # reuse identical news requests for 10 minutes
MAX_ARTICLES = 8
MAX_QUERY_CHARS = 500
HISTORY_TURNS_FOR_CONTEXT = 6  # how many past messages the LLM sees

# UI category label -> internal category id
CATEGORIES = {
    "Technology": "technology",
    "Finance": "finance",
    "Sports": "sports",
    "World": "world",
    "Health": "health",
    "Science": "science",
    "Entertainment": "entertainment",
    "General": "general",
}
