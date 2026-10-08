"""LangGraph workflow nodes and routing functions.

Each node takes the current state and returns only the fields it changes.
Nodes never raise: every failure becomes an entry in `errors` and the
router picks the next-best path (see graph.py for the full diagram).
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, TypedDict

from . import classifier, config, credibility, llm, news_api, web_search


class NewsGenieState(TypedDict, total=False):
    # --- input for this turn
    query: str
    mode: str                 # "chat" (typed message) or "headlines" (sidebar button)
    selected_category: str    # category chosen in the sidebar
    trusted_only: bool
    # --- filled in by the workflow
    intent: str               # news | web | general | invalid
    category: str
    topic: str
    classified_by: str        # llm | rules | button
    articles: list[dict]
    search_results: list[dict]
    provider: str
    answer: str
    sources: list[dict]
    used_fallback: bool
    demo: bool
    errors: list[str]
    trace: list[str]          # which nodes ran, for transparency and debugging
    # --- conversation memory (kept across turns by the checkpointer)
    history: Annotated[list[dict], operator.add]
    last_intent: str
    last_topic: str


def _add(state: NewsGenieState, field: str, *items) -> list:
    return list(state.get(field) or []) + [i for i in items if i]


# --- Nodes -----------------------------------------------------------------
def start(state: NewsGenieState) -> dict:
    """Validate input and reset per-turn fields."""
    query = (state.get("query") or "").strip()
    reset = {
        "intent": "", "category": "", "topic": "", "classified_by": "",
        "articles": [], "search_results": [], "provider": "", "answer": "",
        "sources": [], "used_fallback": False, "demo": False, "errors": [],
        "trace": ["start"],
    }
    if state.get("mode") == "headlines":
        return {**reset, "query": query}
    if not query:
        return {**reset, "intent": "invalid",
                "answer": "Please type a question, or pick a category and press **Get headlines**."}
    if len(query) > config.MAX_QUERY_CHARS:
        reset["errors"] = [f"Your message was shortened to {config.MAX_QUERY_CHARS} characters."]
        query = query[: config.MAX_QUERY_CHARS]
    return {**reset, "query": query}


def classify(state: NewsGenieState) -> dict:
    selected = state.get("selected_category") or "general"
    if state.get("mode") == "headlines":
        return {"intent": "news", "category": selected, "topic": "",
                "classified_by": "button", "trace": _add(state, "trace", "classify")}

    result = classifier.classify(
        state["query"],
        history=state.get("history"),
        last_intent=state.get("last_intent"),
        last_topic=state.get("last_topic"),
    )
    category = result.get("category")
    if not category or category == "general":
        category = selected  # sidebar choice is the default context
    return {
        "intent": result["intent"],
        "category": category,
        "topic": result.get("topic") or "",
        "classified_by": result.get("method", "rules"),
        "trace": _add(state, "trace", f"classify:{result['intent']}"),
    }


def fetch_news(state: NewsGenieState) -> dict:
    result = news_api.fetch_news(state.get("category", "general"), state.get("topic") or None)
    raw = result["articles"]
    trusted_only = bool(state.get("trusted_only"))
    articles = credibility.filter_and_rank(raw, trusted_only=trusted_only,
                                           limit=config.MAX_ARTICLES)
    errors = list(result["errors"])
    if raw and not articles and trusted_only:
        errors.append("No articles from trusted sources matched; try turning off "
                      "'Trusted sources only'.")
    return {
        "articles": articles,
        "provider": result["provider"] + (" (cached)" if result.get("cached") else ""),
        "demo": result["demo"],
        "errors": _add(state, "errors", *errors),
        "trace": _add(state, "trace", f"fetch_news:{len(articles)}"),
    }


def search(state: NewsGenieState) -> dict:
    """Web search. Used for 'web' questions and as the fallback when the
    news APIs return nothing."""
    news_only = state.get("intent") == "news"
    query = state.get("topic") or state.get("query") or state.get("category", "")
    if news_only and not state.get("topic"):
        query = f"latest {state.get('category', '')} news".replace("general ", "")
    result = web_search.search_web(query, news_only=news_only)

    results = result["results"]
    if news_only:  # apply the same misinformation filter as the news API path
        scored = [credibility.score_article(r) for r in results]
        results = [r for r in scored if r["credibility"] >= 20]
        results.sort(key=lambda r: r["credibility"], reverse=True)

    return {
        "search_results": results,
        "provider": result["provider"] or state.get("provider", ""),
        "used_fallback": state.get("intent") == "news",
        "errors": _add(state, "errors", *result["errors"]),
        "trace": _add(state, "trace", f"web_search:{len(results)}"),
    }


_NEWS_SYSTEM = """You are NewsGenie, a careful news assistant. Today is {today}.
Write a short briefing using ONLY the numbered articles provided.
Format:
- One or two sentences summarizing the big picture.
- Then 3-5 bullet points, each ending with its source number like [2].
Do not add facts that are not in the articles. If an article has Low credibility,
say so briefly. Keep it under 180 words."""


def summarize_news(state: NewsGenieState) -> dict:
    articles = state.get("articles") or []
    listing = "\n".join(
        f"[{i}] {a['title']} ({a.get('source') or 'unknown source'}, "
        f"credibility: {a.get('credibility_label', '?')}) - {a.get('description', '')}"
        for i, a in enumerate(articles, 1)
    )
    subject = state.get("topic") or f"{state.get('category', 'general')} headlines"
    request = f"User asked: {state.get('query') or subject}\nTopic: {subject}\n\nArticles:\n{listing}"

    text, error = llm.ask(_NEWS_SYSTEM.format(today=date.today().isoformat()), request)
    if not text:
        text = _plain_news_list(articles, subject)
    if state.get("demo"):
        text = ("*Demo mode: these are sample articles, not real news. Add a NewsAPI or "
                "GNews key for live headlines.*\n\n" + text)
    return {
        "answer": text,
        "sources": [{"title": a["title"], "url": a["url"]} for a in articles],
        "errors": _add(state, "errors", error if llm.available() else None),
        "trace": _add(state, "trace", "summarize_news" + ("" if text and not error else ":plain")),
    }


def _plain_news_list(articles: list[dict], subject: str) -> str:
    lines = [f"**Top stories: {subject}**", ""]
    for i, a in enumerate(articles[:5], 1):
        lines.append(f"{i}. {a['title']} ({a.get('source') or 'unknown'})")
    return "\n".join(lines)


_SEARCH_SYSTEM = """You are NewsGenie. Today is {today}.
Answer the user's question using ONLY the numbered search results.
Cite sources inline like [1]. If the results do not answer the question,
say so plainly instead of guessing. Keep it under 200 words."""


def answer_from_search(state: NewsGenieState) -> dict:
    results = state.get("search_results") or []
    listing = "\n".join(f"[{i}] {r['title']} ({r['source']}): {r['content']}"
                        for i, r in enumerate(results, 1))
    question = state.get("query") or state.get("topic", "")
    text, error = llm.ask(_SEARCH_SYSTEM.format(today=date.today().isoformat()),
                          f"Question: {question}\n\nSearch results:\n{listing}",
                          history=state.get("history"))
    if not text:
        lines = ["Here is what I found on the web:", ""]
        for i, r in enumerate(results[:4], 1):
            snippet = r["content"][:220].rstrip() + ("..." if len(r["content"]) > 220 else "")
            lines.append(f"{i}. **{r['title']}**: {snippet}")
        text = "\n".join(lines)
    if state.get("used_fallback"):
        text = ("*The news service had nothing for this, so these results come from a "
                "web search.*\n\n" + text)
    return {
        "answer": text,
        "sources": [{"title": r["title"], "url": r["url"]} for r in results],
        "errors": _add(state, "errors", error if llm.available() else None),
        "trace": _add(state, "trace", "answer_from_search"),
    }


_GENERAL_SYSTEM = """You are NewsGenie, a friendly, accurate assistant for news and general
questions. Today is {today}. Answer clearly and concisely. If the question depends on
very recent events, say your knowledge may be out of date and suggest the user ask
for the latest news on it. Never make up sources or quotes."""


def general_answer(state: NewsGenieState) -> dict:
    intent = state.get("intent")
    note = ""
    if intent in ("news", "web"):
        # We only get here when live sources found nothing.
        note = ("*I couldn't find live results for this right now, so this answer comes "
                "from general knowledge and may be out of date.*\n\n")

    question = state.get("query") or f"What is the latest {state.get('category', '')} news?"
    text, error = llm.ask(_GENERAL_SYSTEM.format(today=date.today().isoformat()),
                          question, history=state.get("history"))
    sources: list[dict] = []
    if not text:
        # No language model: give the most useful thing we can without one.
        lookup = web_search.search_web(question) if intent == "general" else {"results": []}
        if lookup["results"]:
            top = lookup["results"][0]
            text = (f"Here is a quick reference I found: **{top['title']}**: {top['content']}\n\n"
                    "*Add an OpenAI key for full conversational answers.*")
            sources = [{"title": top["title"], "url": top["url"]}]
        elif intent in ("news", "web"):
            text = ("I couldn't find any live results for that. Try different keywords, "
                    "another category, or check back shortly.")
            note = ""
        else:
            text = ("I can't answer general questions right now because the AI model is "
                    f"unavailable ({error}). News headlines still work: pick a category "
                    "and press **Get headlines**.")
    return {
        "answer": note + text,
        "sources": sources,
        "errors": _add(state, "errors", error if llm.available() else None),
        "trace": _add(state, "trace", "general_answer"),
    }


def finalize(state: NewsGenieState) -> dict:
    """Save the turn to conversation memory."""
    user_text = state.get("query") or f"[Get {state.get('selected_category', '')} headlines]"
    update: dict = {
        "history": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": state.get("answer", "")},
        ],
        "trace": _add(state, "trace", "finalize"),
    }
    if state.get("intent") in classifier.INTENTS:
        update["last_intent"] = state["intent"]
        update["last_topic"] = state.get("topic") or state.get("query", "")
    return update


# --- Routers (conditional edges) -------------------------------------------
def route_after_start(state: NewsGenieState) -> str:
    return "finalize" if state.get("intent") == "invalid" else "classify"


def route_by_intent(state: NewsGenieState) -> str:
    return {"news": "fetch_news", "web": "search"}.get(state.get("intent"), "general_answer")


def route_after_news(state: NewsGenieState) -> str:
    return "summarize_news" if state.get("articles") else "search"


def route_after_search(state: NewsGenieState) -> str:
    return "answer_from_search" if state.get("search_results") else "general_answer"
