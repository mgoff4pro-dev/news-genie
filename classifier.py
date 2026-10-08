"""Query differentiation: decide whether a message is a news request,
a question needing a live web lookup, or a general question.

Two layers:
  1. LLM classifier (understands phrasing and follow-ups)
  2. Rule-based classifier (used when the LLM is unavailable or returns bad JSON)
"""

from __future__ import annotations

import json
import re

from . import llm

INTENTS = ("news", "web", "general")
CATEGORY_IDS = ("technology", "finance", "sports", "world", "health", "science",
                "entertainment", "general")

CATEGORY_KEYWORDS = {
    "technology": ["tech", "ai", "artificial intelligence", "software", "apple", "google",
                   "microsoft", "openai", "chip", "smartphone", "startup", "cybersecurity",
                   "gadget", "technology"],
    "finance": ["stock", "market", "economy", "inflation", "interest rate", "fed", "crypto",
                "bitcoin", "earnings", "dow jones", "nasdaq", "s&p", "finance", "business",
                "investing"],
    "sports": ["sport", "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
               "baseball", "hockey", "score", "game", "match", "playoff", "tennis", "golf"],
    "health": ["health", "medical", "disease", "vaccine", "fda", "hospital", "covid"],
    "science": ["science", "space", "nasa", "climate", "research", "study finds"],
    "entertainment": ["movie", "film", "music", "celebrity", "tv show", "album", "oscars"],
    "world": ["war", "election", "president", "prime minister", "united nations", "global"],
}

NEWS_PATTERNS = [
    r"\bnews\b", r"\bheadlines?\b", r"\bbreaking\b", r"\blatest\b", r"\bupdates?\b",
    r"what'?s (happening|going on|new)", r"\btoday'?s\b", r"\bthis (week|morning)\b",
    r"\brecent(ly)?\b", r"\btop stories\b", r"\bscores?\b",
]
WEB_PATTERNS = [
    r"\bcurrent(ly)?\b", r"\bright now\b", r"\bprice of\b", r"\bhow much (is|does)\b",
    r"\bwho (is|won|leads)\b.*\b(now|current|latest)\b", r"\bweather\b",
    r"\bwhen (is|does)\b", r"\b20(2[5-9]|3\d)\b", r"\blook up\b", r"\bsearch (for|the web)\b",
]
FOLLOW_UP_PATTERNS = [
    r"^(tell me )?more\b", r"\bmore (about|on) (that|this|it)\b", r"^what about\b",
    r"^and\b", r"^why\b.*\b(that|this|it)\b", r"\bexplain (that|this|it)\b",
    r"^(the )?(first|second|third|last) (one|story|article)\b",
]

_SYSTEM = """You route messages for NewsGenie, a news and information assistant.
Classify the user's latest message. Reply with JSON only:
{"intent": "news" | "web" | "general",
 "category": one of technology, finance, sports, world, health, science, entertainment, general,
 "topic": short search phrase, or "" if the user wants general headlines,
 "reason": a few words}

- "news": wants news, headlines, recent events, or updates on a topic.
- "web": a factual question whose answer may change over time or needs a lookup
  (prices, current officeholders, schedules, recent product versions), but is not asking for news.
- "general": anything answerable from general knowledge (explanations, how-to, advice,
  definitions, writing help, small talk).
Use the conversation to resolve follow-ups like "tell me more about that"."""


def rule_based(query: str, last_intent: str | None = None,
               last_topic: str | None = None) -> dict:
    text = " " + query.lower().strip() + " "

    if last_intent and any(re.search(p, text.strip()) for p in FOLLOW_UP_PATTERNS):
        return {"intent": last_intent, "category": None, "topic": last_topic or query,
                "reason": "follow-up to previous question", "method": "rules"}

    category = None
    best = 0
    for cat, words in CATEGORY_KEYWORDS.items():
        hits = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"s?\b", text))
        if hits > best:
            category, best = cat, hits

    if any(re.search(p, text) for p in NEWS_PATTERNS):
        intent = "news"
    elif any(re.search(p, text) for p in WEB_PATTERNS):
        intent = "web"
    else:
        intent = "general"

    topic = _extract_topic(query) if intent == "news" else query.strip()
    return {"intent": intent, "category": category, "topic": topic,
            "reason": "keyword rules", "method": "rules"}


def _extract_topic(query: str) -> str:
    """Strip filler so 'show me the latest news about Tesla' -> 'Tesla'."""
    text = query.lower()
    text = re.sub(r"\b(show|give|get|tell|find)( me)?\b", " ", text)
    text = re.sub(r"\b(what'?s|what is|whats|any|the|latest|recent|top|breaking|today'?s|"
                  r"news|headlines?|updates?|stories|on|about|in|for|happening|going|new|"
                  r"this|week|morning|please|me|some|of)\b", " ", text)
    text = re.sub(r"[^\w\s&.'-]", " ", text)
    topic = re.sub(r"\s+", " ", text).strip()
    # A bare category word means "headlines for that category", not a search.
    if topic in {"tech", "technology", "finance", "business", "sports", "sport", "world",
                 "health", "science", "entertainment", "general", ""}:
        return ""
    return topic


def classify(query: str, history: list[dict] | None = None,
             last_intent: str | None = None, last_topic: str | None = None) -> dict:
    """Return {intent, category, topic, reason, method}."""
    if llm.available():
        text, _error = llm.ask(_SYSTEM, query, history=history, json_mode=True)
        if text:
            try:
                data = json.loads(text)
                intent = data.get("intent")
                if intent in INTENTS:
                    category = data.get("category")
                    return {
                        "intent": intent,
                        "category": category if category in CATEGORY_IDS else None,
                        "topic": (data.get("topic") or "").strip(),
                        "reason": data.get("reason", ""),
                        "method": "llm",
                    }
            except (ValueError, AttributeError):
                pass  # fall through to rules
    return rule_based(query, last_intent, last_topic)
