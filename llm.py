"""Thin wrapper around the chat model (OpenAI via LangChain).

`ask()` never raises. It returns (text, error). When the key is missing or
the call fails, text is None and the caller uses its own fallback.
"""

from __future__ import annotations

from . import config

_model_cache: dict[str, object] = {}


def available() -> bool:
    return config.get_key(config.OPENAI_API_KEY) is not None


def _get_model():
    key = config.get_key(config.OPENAI_API_KEY)
    if not key:
        return None
    if key not in _model_cache:
        from langchain_openai import ChatOpenAI  # imported lazily

        _model_cache.clear()
        _model_cache[key] = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=2,
            api_key=key,
        )
    return _model_cache[key]


def ask(system: str, user: str, history: list[dict] | None = None,
        json_mode: bool = False) -> tuple[str | None, str | None]:
    """Send a prompt with optional chat history ([{role, content}, ...])."""
    try:
        model = _get_model()
    except ImportError:
        return None, "langchain-openai is not installed"
    if model is None:
        return None, "OpenAI key is not set"

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=system)]
    for turn in (history or [])[-config.HISTORY_TURNS_FOR_CONTEXT:]:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        messages.append(cls(content=str(turn.get("content", ""))[:2000]))
    messages.append(HumanMessage(content=user))

    try:
        if json_mode:
            result = model.bind(response_format={"type": "json_object"}).invoke(messages)
        else:
            result = model.invoke(messages)
        return str(result.content).strip(), None
    except Exception as exc:  # network, auth, rate limit, etc.
        name = type(exc).__name__
        if "Authentication" in name:
            return None, "OpenAI rejected the API key"
        if "RateLimit" in name:
            return None, "OpenAI rate limit reached"
        if "Timeout" in name:
            return None, "OpenAI timed out"
        return None, f"OpenAI call failed ({name})"
