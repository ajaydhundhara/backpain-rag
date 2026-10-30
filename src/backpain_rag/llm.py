"""The one place that talks to OpenAI.

Failure behaviour is decided here once: every call raises `LLMError` rather than
returning something partial. Almost every caller is a safety gate, and a gate
that cannot run must refuse -- so the decision to fail closed (or, for HyDE
alone, open) belongs at the call site, not hidden inside a retry loop.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TypeVar

from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


# --- degradation tracking --------------------------------------------------
# A refusal caused by an outage looks identical to a considered one. That is
# fine for the user, but not for the response cache, which would pin a
# transient outage for a whole TTL. Every fail-closed path records itself here.
# A ContextVar, not a global: one request's outage must not affect another's.
_degraded: ContextVar[tuple[str, ...]] = ContextVar("llm_degraded", default=())


def note_degraded(stage: str) -> None:
    _degraded.set(_degraded.get() + (stage,))


def degradations() -> tuple[str, ...]:
    return _degraded.get()


def clear_degradations() -> None:
    _degraded.set(())


_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        s = get_settings()
        if not s.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set (see .env.example)")
        _client = OpenAI(api_key=s.openai_api_key, max_retries=3, timeout=60)
    return _client


def parse(
    schema: type[T],
    system: str,
    user: str,
    model: str | None = None,
    temperature: float | None = 0.0,
) -> T:
    """Structured-output call. Raises LLMError rather than returning partial output."""
    from openai import OpenAIError

    s = get_settings()
    kwargs: dict = {
        "model": model or s.openai_guard_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": schema,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        resp = client().chat.completions.parse(**kwargs)
    except OpenAIError as e:
        # Reasoning models reject `temperature`; retry once without it.
        if "temperature" in str(e).lower() and "temperature" in kwargs:
            kwargs.pop("temperature")
            try:
                resp = client().chat.completions.parse(**kwargs)
            except OpenAIError as e2:
                raise LLMError(str(e2)) from e2
        else:
            raise LLMError(str(e)) from e

    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise LLMError("model returned no parseable structured output")
    return parsed


def embed(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Embed a batch of texts. Raises LLMError."""
    from openai import OpenAIError

    s = get_settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client().embeddings.create(
                model=s.openai_embed_model, input=batch, dimensions=s.embed_dim
            )
        except OpenAIError as e:
            raise LLMError(str(e)) from e
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
