"""Grounded generation: structured JSON with an answer and per-claim citations."""

from __future__ import annotations

from .config import get_settings
from .llm import LLMError, note_degraded, parse
from .models import GroundedAnswer, Retrieved
from .policy import SCOPE_STATEMENT

RULES = """You answer ONLY from the numbered passages provided. You have no other \
knowledge of medicine for the purposes of this task.

Rules:
1. Every factual statement must come from a passage and carry the marker(s) of \
the passage(s) it came from, e.g. [C2].
2. If the passages do not answer the question, set insufficient_context to true \
and leave claims empty. Do not fill gaps from memory. Partial answers are fine: \
say what the passages support and no more.
3. Write in the third person, attributed to the source and naming what kind \
of source it is, as the passage label shows: "the book describes...", \
"McKenzie recommends...", "the NICE guideline advises against...". Never call a \
book a guideline. Never address the reader as a patient and never use the \
imperative ("take", "you should", "try").
4. Do not introduce any number -- a dose, duration, age or threshold -- that \
does not appear in a passage you cite.
5. `answer` is the prose shown to the user. `claims` decomposes that same \
answer into individual factual statements with the markers they rely on (use \
the marker strings, e.g. "C2", as chunk_ids). Every claim must be traceable to \
a sentence in `answer`."""

SYSTEM = f"{SCOPE_STATEMENT}\n\n{RULES}"


def build_user_prompt(query: str, results: list[Retrieved]) -> str:
    context = "\n\n".join(f"{r.chunk.cite_label(r.marker)}\n{r.chunk.text}" for r in results)
    return f"Passages:\n\n{context}\n\nQuestion: {query}"


def generate(query: str, results: list[Retrieved]) -> GroundedAnswer:
    """On model failure returns insufficient_context, which gate 3 refuses."""
    try:
        return parse(GroundedAnswer, SYSTEM, build_user_prompt(query, results),
                     model=get_settings().openai_model)
    except LLMError:
        note_degraded("generator")
        return GroundedAnswer(answer="", claims=[], insufficient_context=True)
