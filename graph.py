"""The LangGraph workflow that processes every user request.

    start ──(empty input)──────────────────────────────────────► finalize
      │
    classify ──news──► fetch_news ──articles found──► summarize_news ──► finalize
      │                    │
      │                    └──nothing found / API failed──┐
      │                                                    ▼
      ├──web──────────────────────────────────────────► search ──results──► answer_from_search ──► finalize
      │                                                    │
      │                                                    └──no results──┐
      │                                                                   ▼
      └──general────────────────────────────────────────────────────► general_answer ──► finalize

A MemorySaver checkpointer stores state per `thread_id`, so each browser
session keeps its own conversation history and follow-up context.
"""

from __future__ import annotations

import time
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .nodes import NewsGenieState


def build_graph():
    g = StateGraph(NewsGenieState)

    g.add_node("start", nodes.start)
    g.add_node("classify", nodes.classify)
    g.add_node("fetch_news", nodes.fetch_news)
    g.add_node("search", nodes.search)
    g.add_node("summarize_news", nodes.summarize_news)
    g.add_node("answer_from_search", nodes.answer_from_search)
    g.add_node("general_answer", nodes.general_answer)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "start")
    g.add_conditional_edges("start", nodes.route_after_start,
                            {"classify": "classify", "finalize": "finalize"})
    g.add_conditional_edges("classify", nodes.route_by_intent,
                            {"fetch_news": "fetch_news", "search": "search",
                             "general_answer": "general_answer"})
    g.add_conditional_edges("fetch_news", nodes.route_after_news,
                            {"summarize_news": "summarize_news", "search": "search"})
    g.add_conditional_edges("search", nodes.route_after_search,
                            {"answer_from_search": "answer_from_search",
                             "general_answer": "general_answer"})
    for node in ("summarize_news", "answer_from_search", "general_answer"):
        g.add_edge(node, "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def new_thread_id() -> str:
    return uuid.uuid4().hex


def run(query: str, thread_id: str, *, mode: str = "chat",
        category: str = "general", trusted_only: bool = False) -> dict:
    """Run one turn through the workflow. Always returns a result dict,
    even if something unexpected breaks inside the graph."""
    started = time.perf_counter()
    try:
        state = get_graph().invoke(
            {"query": query, "mode": mode, "selected_category": category,
             "trusted_only": trusted_only},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:  # last-resort safety net
        state = {"answer": "Sorry, something went wrong while handling that. Please try again.",
                 "errors": [f"Unexpected error: {type(exc).__name__}"], "intent": "error",
                 "trace": ["error"]}
    state["elapsed"] = round(time.perf_counter() - started, 2)
    return state
