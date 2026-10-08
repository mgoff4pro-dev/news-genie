# NewsGenie: AI-Powered Information and News Assistant

NewsGenie is a chatbot for both kinds of question. Ask for the news and it returns real-time headlines that are checked for credibility. Ask anything else and it gives a quick, accurate answer. It runs a **LangGraph** workflow behind a **Streamlit** interface.

## Features

| Requirement | How NewsGenie does it |
|---|---|
| Handles conversations | An LLM classifier sorts each message as **news**, **web lookup**, or **general**, and falls back to keyword rules if the LLM is unavailable. Follow-ups like "tell me more about that" keep the earlier context. |
| Integrates APIs | **NewsAPI** for live news, with **GNews** as backup. **Tavily** for web search, with **Wikipedia** as a free backup. **OpenAI** for answers and summaries. |
| Manages workflow | A LangGraph `StateGraph` with 8 nodes and conditional routing. A `MemorySaver` checkpointer keeps each session's conversation. |
| Intuitive UI | Streamlit app with a category picker, a "Trusted sources only" filter, a one-click headlines button, chat history, service status lights, and a "New chat" button. |
| Filters misinformation | Each article gets a credibility score: trusted-outlet list, satire/unreliable list, clickbait detection, and duplicate removal. The score and its reasons appear on each article card. |
| Error handling | Missing keys, failed calls, rate limits, timeouts, and empty results are all handled. The app never crashes and always tells the user what happened. |

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys (optional; see "Demo mode" below)
cp .env.example .env            # Windows: copy .env.example .env
# then open .env and paste your keys

# 4. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501.

### Getting API keys (all have free tiers except OpenAI)
- **NewsAPI**: https://newsapi.org/register (free developer plan)
- **GNews** (optional backup): https://gnews.io
- **Tavily** (web search): https://tavily.com
- **OpenAI**: https://platform.openai.com/api-keys (pay-as-you-go; `gpt-4o-mini` costs very little)

You can also paste keys into the **API keys** section of the sidebar while the app is running.

### Demo mode
With no keys at all, the app still runs:
- Headlines use bundled **sample articles**, clearly labeled as demo data.
- General questions try a free Wikipedia lookup.
- The sidebar shows which services are on or off.

This is useful for testing the interface. For your final screenshots, add at least a NewsAPI key and an OpenAI key.

## Project structure

```
newsgenie/
├── app.py                  # Streamlit interface
├── generate_samples.py     # Writes sample outputs (tech/finance/sports) for the report
├── requirements.txt
├── .env.example
├── newsgenie/
│   ├── config.py           # API keys, settings, categories
│   ├── graph.py            # LangGraph workflow (nodes + conditional edges + memory)
│   ├── nodes.py            # Workflow steps and routing functions
│   ├── classifier.py       # Query differentiation (LLM + rule-based fallback)
│   ├── news_api.py         # NewsAPI -> GNews -> demo data, with 10-min cache
│   ├── web_search.py       # Tavily -> Wikipedia
│   ├── credibility.py      # Misinformation filter: scoring, clickbait, de-duplication
│   ├── llm.py              # OpenAI wrapper that never crashes the app
│   ├── http_utils.py       # Timeouts, retries with backoff, error types
│   └── data/sample_news.json
└── tests/test_newsgenie.py # 25 tests for routing, fallbacks, and error handling
```

## How a request flows

```
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
```

1. **start**: validates input (empty or too long) and resets the per-turn fields.
2. **classify**: decides the intent, category, and search topic. The headlines button skips straight to news.
3. **fetch_news**: calls the news provider chain, then filters and ranks the articles by credibility.
4. **search**: web search, used for "web" questions **and** as the fallback when the news APIs come back empty.
5. **summarize_news / answer_from_search / general_answer**: the LLM writes the answer and cites only the sources it was given.
6. **finalize**: saves the turn to conversation memory.

## Fallbacks and error handling

| Situation | What happens |
|---|---|
| No OpenAI key | Classification uses keyword rules. News shows a plain headline list. General questions try Wikipedia. |
| No news keys | Demo sample articles, clearly labeled. |
| NewsAPI fails (timeout, 5xx, rate limit) | Retries with backoff (1s, 2s), then GNews, then web search. |
| Invalid API key (401/403) | Not retried. The user sees a clear message such as "NewsAPI rejected the API key". |
| No news found | Web search for news on that topic. If that finds nothing too, an honest "nothing found" with suggestions. |
| LLM returns bad JSON | The classifier falls back to keyword rules. |
| LLM call fails | A fallback answer is built from the raw articles or search results. |
| Unexpected crash in the graph | Caught in `graph.run()`, and the user gets a friendly message. |
| Same request repeated | Served from a 10-minute cache, which is faster and saves API quota. "Refresh news" clears it. |

## Tests

```bash
python -m pytest -q
```
All external APIs are mocked, so the tests need no keys.

## Generating sample outputs for the report

```bash
python generate_samples.py
```
This creates `sample_outputs.md` with technology, finance, and sports headlines plus example general, web, and topic queries. Each one shows the route it took through the graph.
