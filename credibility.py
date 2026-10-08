"""Misinformation filtering: source reputation, clickbait detection, de-duplication.

This is a transparent, rule-based layer. Every article gets a credibility
score (0-100) and a label, so users can see *why* something was ranked
higher or lower. It is a heuristic, not a fact-checker.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.parse import urlparse

# Established outlets with editorial standards and public corrections policies.
TRUSTED_DOMAINS = {
    # wire services and public broadcasters
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
    # national and international outlets
    "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com", "ft.com",
    "economist.com", "bloomberg.com", "latimes.com", "usatoday.com", "axios.com",
    "politico.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com", "cnn.com",
    "aljazeera.com", "theatlantic.com", "time.com",
    # business and finance
    "cnbc.com", "marketwatch.com", "barrons.com", "fortune.com", "businessinsider.com",
    "finance.yahoo.com",
    # technology
    "techcrunch.com", "theverge.com", "arstechnica.com", "wired.com", "engadget.com",
    "zdnet.com", "technologyreview.com", "cnet.com",
    # sports
    "espn.com", "theathletic.com", "cbssports.com", "si.com", "nfl.com", "nba.com",
    "mlb.com", "nhl.com",
    # science and health
    "nature.com", "science.org", "scientificamerican.com", "statnews.com",
    "newscientist.com", "who.int", "cdc.gov", "nih.gov",
}

# Satire or sources widely documented as unreliable. Kept short on purpose.
UNRELIABLE_DOMAINS = {
    "theonion.com", "babylonbee.com", "infowars.com", "naturalnews.com",
    "beforeitsnews.com", "worldnewsdailyreport.com",
}

CLICKBAIT_PATTERNS = [
    r"you won'?t believe",
    r"shocking",
    r"this one (weird )?trick",
    r"doctors hate",
    r"what happens next",
    r"goes viral",
    r"\bmust see\b",
    r"!!+",
    r"\b(miracle|secret) (cure|method)\b",
]
_CLICKBAIT_RE = re.compile("|".join(CLICKBAIT_PATTERNS), re.IGNORECASE)


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _matches(domain: str, domains: set[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in domains)


def score_article(article: dict) -> dict:
    """Return a copy of the article with `credibility` (0-100), `credibility_label`
    and `credibility_reasons` added."""
    domain = domain_of(article.get("url", ""))
    score = 50
    reasons: list[str] = []

    if _matches(domain, UNRELIABLE_DOMAINS):
        score -= 45
        reasons.append("source is satire or known for unreliable content")
    elif _matches(domain, TRUSTED_DOMAINS):
        score += 35
        reasons.append("established news outlet")
    else:
        reasons.append("source not on the trusted list")

    title = article.get("title", "") or ""
    if _CLICKBAIT_RE.search(title):
        score -= 15
        reasons.append("sensational headline")
    if title.isupper() and len(title) > 15:
        score -= 10
        reasons.append("all-caps headline")
    if not (article.get("description") or "").strip():
        score -= 5
        reasons.append("no summary provided")
    if article.get("published_at"):
        score += 5
    else:
        reasons.append("no publish date")

    score = max(0, min(100, score))
    label = "High" if score >= 75 else "Medium" if score >= 45 else "Low"
    return {**article, "credibility": score, "credibility_label": label,
            "credibility_reasons": reasons}


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s+[-|]\s+[^-|]+$", "", title)  # drop " - Source Name" suffix
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def deduplicate(articles: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate headlines (the same story syndicated by many sites).
    Keeps the first occurrence, so sort by credibility before calling."""
    kept: list[dict] = []
    seen: list[str] = []
    for article in articles:
        norm = _normalize_title(article.get("title", ""))
        if not norm:
            continue
        if any(SequenceMatcher(None, norm, s).ratio() >= threshold for s in seen):
            continue
        seen.append(norm)
        kept.append(article)
    return kept


def _published_sort_key(article: dict) -> float:
    raw = article.get("published_at") or ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def filter_and_rank(articles: list[dict], trusted_only: bool = False,
                    limit: int = 8) -> list[dict]:
    """Full pipeline: drop junk, score, optionally keep trusted only, rank, dedupe."""
    cleaned = [
        a for a in articles
        if a.get("title") and a.get("url")
        and "[removed]" not in a.get("title", "").lower()
    ]
    scored = [score_article(a) for a in cleaned]
    scored = [a for a in scored if a["credibility"] >= 20]  # always drop the worst
    if trusted_only:
        scored = [a for a in scored if a["credibility_label"] == "High"]
    scored.sort(key=lambda a: (a["credibility"], _published_sort_key(a)), reverse=True)
    return deduplicate(scored)[:limit]


def time_ago(published_at: str | None) -> str:
    if not published_at:
        return ""
    try:
        when = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    minutes = int((datetime.now(timezone.utc) - when).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 1440:
        return f"{minutes // 60} hr ago"
    return f"{minutes // 1440} days ago"
