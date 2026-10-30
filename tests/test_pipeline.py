"""End to end, with a scripted fake model standing in for every LLM call."""

import numpy as np

from backpain_rag import assistant as assistant_mod
from backpain_rag import generate as generate_mod
from backpain_rag import guardrails, retrieval
from backpain_rag.assistant import Assistant
from backpain_rag.index import LocalVectorStore, unit
from backpain_rag.models import Claim, GroundedAnswer, Intent, RefusalReason
from backpain_rag.retrieval import HybridRetriever, LLMReranker

from .conftest import fake_vector


def _build(chunks):
    vecs = np.stack([unit(fake_vector(c.embed_text())) for c in chunks]).astype(np.float16)
    r = HybridRetriever(chunks=chunks, store=LocalVectorStore([c.chunk_id for c in chunks], vecs))
    return Assistant(retriever=r, reranker=LLMReranker())


def _script(monkeypatch, *, intent=Intent.IN_SCOPE, top=9, answer=None, supported=True, framing=False):
    def gparse(schema, system, user, **kw):
        name = schema.__name__
        if name == "_IntentOut":
            return schema(intent=intent, reason="scripted")
        if name == "_Supports":
            n = user.count("Claim:")
            return schema(results=[{"index": i, "supported": supported} for i in range(n)])
        if name == "_Safety":
            return schema(personalized_advice=framing, diagnostic=False,
                          imperative_treatment=False, emergency_instruction=False)
        raise AssertionError(name)

    def rparse(schema, system, user, **kw):
        if schema.__name__ == "_Hypothetical":
            return schema(passages=["Imaging is not routinely offered for low back pain."])
        n = sum(1 for line in user.splitlines() if line.startswith("[") and "]" in line[:5])
        scores = [{"index": i, "score": top if i == 0 else 2} for i in range(n)]
        return schema(scores=scores)

    def genparse(schema, system, user, **kw):
        return answer or GroundedAnswer(
            answer="NICE advises against routinely offering imaging in non-specialist settings [C1].",
            claims=[Claim(text="advises against routine imaging", chunk_ids=["C1"])])

    monkeypatch.setattr(guardrails, "parse", gparse)
    monkeypatch.setattr(retrieval, "parse", rparse)
    monkeypatch.setattr(generate_mod, "parse", genparse)


def test_answers_with_citations(chunks, fake_embed, monkeypatch):
    _script(monkeypatch)
    r = _build(chunks).ask("Should imaging be used for low back pain?", use_cache=False)
    assert r.answered, r.trace
    assert r.citations and r.citations[0].marker == "C1"
    assert r.trace["stages"] == ["intent", "retrieve", "rerank", "confidence", "generate", "output_gate"]


def test_refused_query_never_retrieves(chunks, fake_embed, monkeypatch):
    _script(monkeypatch)
    a = _build(chunks)
    a.retriever.retrieve = lambda *x, **k: (_ for _ in ()).throw(AssertionError("retrieved"))
    r = a.ask("should i get an x-ray of my back", use_cache=False)
    assert r.refusal_reason is RefusalReason.PERSONALIZED_ADVICE
    assert r.trace["short_circuited"]


def test_low_confidence_refuses(chunks, fake_embed, monkeypatch):
    _script(monkeypatch, top=3)
    r = _build(chunks).ask("which mattress brand is best", use_cache=False)
    assert r.refusal_reason is RefusalReason.LOW_CONFIDENCE


def test_framing_violation_refuses(chunks, fake_embed, monkeypatch):
    _script(monkeypatch, framing=True)
    r = _build(chunks).ask("imaging for low back pain", use_cache=False)
    assert r.refusal_reason is RefusalReason.UNSAFE_OUTPUT


def test_degraded_responses_are_not_cached(chunks, fake_embed, monkeypatch):
    _script(monkeypatch)

    def boom(*a, **k):
        raise retrieval.LLMError("down")

    monkeypatch.setattr(retrieval, "parse", boom)  # reranker + hyde down
    a = _build(chunks)
    r1 = a.ask("imaging for low back pain")
    assert not r1.answered and "reranker" in r1.trace["degraded"]
    _script(monkeypatch)
    r2 = a.ask("imaging for low back pain")
    assert r2.answered and r2.trace.get("cache") != "hit"
    r3 = a.ask("imaging for low back pain")
    assert r3.trace.get("cache") == "hit"


def test_assistant_module_exports_refuse():
    assert callable(assistant_mod._refuse)
