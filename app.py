"""NewsGenie - Streamlit interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st

from newsgenie import config, credibility, graph, news_api

st.set_page_config(page_title="NewsGenie", page_icon="🗞️", layout="wide")

# Streamlit Cloud: keys can live in st.secrets instead of a .env file.
try:
    config.set_overrides({k: st.secrets[k] for k in config.ALL_KEYS if k in st.secrets})
except Exception:  # no secrets file is normal when running locally
    pass

st.markdown(
    """
    <style>
    .ng-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px;
              padding: 12px 14px; margin-bottom: 10px;}
    .ng-card a {font-weight: 600; text-decoration: none;}
    .ng-meta {font-size: .82rem; opacity: .75; margin-top: 4px;}
    .ng-badge {font-size: .72rem; padding: 2px 8px; border-radius: 999px; margin-left: 6px;
               font-weight: 600;}
    .ng-High {background: #d8f3dc; color: #1b4332;}
    .ng-Medium {background: #fff3bf; color: #5c4400;}
    .ng-Low {background: #ffe3e3; color: #7d1d1d;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state ---------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = graph.new_thread_id()
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, articles, sources, meta}]

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("🗞️ NewsGenie")
    st.caption("Reliable news and quick answers in one place.")

    category_label = st.selectbox("News category", list(config.CATEGORIES), index=0)
    category = config.CATEGORIES[category_label]
    trusted_only = st.toggle("Trusted sources only", value=False,
                             help="Show only articles from established outlets.")
    get_headlines = st.button(f"Get {category_label} headlines", type="primary",
                              use_container_width=True)

    st.divider()
    with st.expander("API keys", expanded=not any(config.key_status().values())):
        st.caption("Keys stay in this browser session only. You can also put them in a "
                   ".env file (see README).")
        entered = {
            config.OPENAI_API_KEY: st.text_input("OpenAI", type="password"),
            config.NEWSAPI_KEY: st.text_input("NewsAPI", type="password"),
            config.GNEWS_API_KEY: st.text_input("GNews (backup)", type="password"),
            config.TAVILY_API_KEY: st.text_input("Tavily (web search)", type="password"),
        }
        config.set_overrides({k: v for k, v in entered.items() if v})

    status = config.key_status()
    labels = {config.OPENAI_API_KEY: "AI model", config.NEWSAPI_KEY: "NewsAPI",
              config.GNEWS_API_KEY: "GNews", config.TAVILY_API_KEY: "Web search"}
    st.markdown("**Service status**")
    for key, label in labels.items():
        st.markdown(f"{'🟢' if status[key] else '⚪'} {label}"
                    + ("" if status[key] else " - not set"))
    if not (status[config.NEWSAPI_KEY] or status[config.GNEWS_API_KEY]):
        st.info("No news key set, so headlines use **demo sample data**.")
    if not status[config.OPENAI_API_KEY]:
        st.info("No OpenAI key: answers use simple fallbacks instead of the AI model.")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("New chat", use_container_width=True):
        st.session_state.thread_id = graph.new_thread_id()
        st.session_state.messages = []
        st.rerun()
    if col2.button("Refresh news", use_container_width=True,
                   help="Clear cached headlines and fetch fresh ones"):
        news_api.clear_cache()
        st.toast("News cache cleared")


# --- Rendering helpers -----------------------------------------------------
def render_articles(articles: list[dict]) -> None:
    for a in articles:
        label = a.get("credibility_label", "Medium")
        reasons = "; ".join(a.get("credibility_reasons", []))
        meta = " · ".join(x for x in [a.get("source"), credibility.time_ago(a.get("published_at"))] if x)
        desc = html.escape((a.get("description") or "")[:240])
        st.markdown(
            f"""<div class="ng-card">
            <a href="{html.escape(a['url'])}" target="_blank">{html.escape(a['title'])}</a>
            <span class="ng-badge ng-{label}" title="{html.escape(reasons)}">{label} credibility</span>
            <div>{desc}</div>
            <div class="ng-meta">{html.escape(meta)}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="🧞" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("articles"):
            with st.expander(f"📰 {len(msg['articles'])} articles", expanded=True):
                render_articles(msg["articles"])
        elif msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"{i}. [{s['title']}]({s['url']})")
        for warning in msg.get("errors", []):
            st.caption(f"⚠️ {warning}")
        if msg.get("meta"):
            st.caption(msg["meta"])


def handle(query: str, mode: str) -> None:
    user_text = query if mode == "chat" else f"Get {category_label} headlines"
    st.session_state.messages.append({"role": "user", "content": user_text})
    render_message(st.session_state.messages[-1])

    with st.spinner("Working on it..."):
        result = graph.run(query, st.session_state.thread_id, mode=mode,
                           category=category, trusted_only=trusted_only)

    intent_names = {"news": "News request", "web": "Web lookup", "general": "General question",
                    "invalid": "Input check", "error": "Error"}
    meta_bits = [intent_names.get(result.get("intent", ""), "")]
    if result.get("provider"):
        meta_bits.append(f"via {result['provider']}")
    meta_bits.append(f"{result.get('elapsed', 0)}s")
    message = {
        "role": "assistant",
        "content": result.get("answer") or "I couldn't produce an answer.",
        "articles": result.get("articles") or [],
        "sources": result.get("sources") or [],
        "errors": result.get("errors") or [],
        "meta": " · ".join(b for b in meta_bits if b),
    }
    st.session_state.messages.append(message)
    render_message(message)


# --- Main area -------------------------------------------------------------
st.header("Ask NewsGenie")
st.caption("Ask anything, or ask for news, for example *\"latest AI news\"*, "
           "*\"how did the markets do today?\"*, or *\"explain how inflation works\"*.")

if not st.session_state.messages:
    st.markdown("**Try one:**")
    examples = ["What's the latest in technology?", "Any news on interest rates?",
                "Explain what a stock index is"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex

for msg in st.session_state.messages:
    render_message(msg)

typed = st.chat_input("Ask a question or request news...")
pending = st.session_state.pop("pending", None)

if get_headlines:
    handle("", mode="headlines")
elif typed or pending:
    handle(typed or pending, mode="chat")
