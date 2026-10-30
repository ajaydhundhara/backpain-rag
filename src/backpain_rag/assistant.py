"""End-to-end query pipeline.

    query
     ├─ GATE 1  intent         personalized / emergency / off-domain -> refuse
     ├─ rewrite                 additive abbreviation expansion
     ├─ HyDE                    hypothetical passage blended into the dense vector
     ├─ retrieve                dense + BM25, fused with RRF
     ├─ rerank                  absolute 0-10 relevance per passage
     ├─ GATE 2  confidence     proceed / corrective retry once / refuse
     ├─ generate                JSON: answer + claims[] with markers
     ├─ GATE 3  output         citations -> numbers -> support -> framing
     └─ answer + citations + disclaimer

Every exit produces a `Response`. No path returns raw model text, and no path
skips a gate: a refusal anywhere returns the standard text for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import llm
from .cache import get_cache, key_for
from .config import get_settings
from .generate import generate
from .guardrails import ConfidenceVerdict, classify, evaluate_confidence, validate_output
from .models import Citation, ConfidenceAction, Intent, RefusalReason, Response, Retrieved
from .policy import DISCLAIMER, refusal_text
from .retrieval import HybridRetriever, LLMReranker, Reranker, correct, normalise

_INTENT_TO_REFUSAL = {
    Intent.EMERGENCY: RefusalReason.EMERGENCY,
    Intent.PERSONALIZED_ADVICE: RefusalReason.PERSONALIZED_ADVICE,
    Intent.OUT_OF_DOMAIN: RefusalReason.OUT_OF_DOMAIN,
}


def _refuse(query: str, reason: RefusalReason, trace: dict, top: float | None = None) -> Response:
    return Response(query=query, answered=False, answer=refusal_text(reason),
                    refusal_reason=reason, top_score=top, disclaimer=DISCLAIMER, trace=trace)


def _conf_trace(c: ConfidenceVerdict) -> dict:
    return {"passed": c.passed, "action": c.action.value, "top_score": c.top_score,
            "threshold": c.threshold, "reason": c.reason, "kept": len(c.kept)}


def _citations(kept: list[Retrieved]) -> list[Citation]:
    return [
        Citation(marker=r.marker, chunk_id=r.chunk.chunk_id, doc_id=r.chunk.doc_id,
                 title=r.chunk.title, section=r.chunk.section_path, pages=r.chunk.pages,
                 source_url=r.chunk.source_url, rerank_score=r.rerank_score)
        for r in kept
    ]


@dataclass
class Assistant:
    retriever: HybridRetriever
    reranker: Reranker

    @classmethod
    def build(cls) -> Assistant:
        return cls(retriever=HybridRetriever(), reranker=LLMReranker())

    def ask(self, query: str, screen: bool = True, use_cache: bool = True) -> Response:
        s = get_settings()
        cache = get_cache()
        ckey = f"resp:{s.pipeline_fingerprint}:{key_for(normalise(query).casefold(), screen)}"
        llm.clear_degradations()

        if use_cache:
            hit = cache.get_json(ckey)
            if hit is not None:
                cached = Response.model_validate(hit)
                cached.query = query
                cached.trace = dict(cached.trace) | {"cache": "hit"}
                return cached

        response = self._run(query, screen)

        degraded = list(llm.degradations())
        if degraded:
            response.trace["degraded"] = degraded
        # Never cache a refusal produced by an outage: that would turn a minute
        # of API trouble into a day of a real question being refused.
        if use_cache and not degraded:
            cache.set_json(ckey, response.model_dump(mode="json"), s.cache_response_ttl_s)
        return response

    def _run(self, query: str, screen: bool = True) -> Response:
        trace: dict = {"stages": []}

        # --- Gate 1 -------------------------------------------------------
        verdict = classify(query, use_model=screen)
        trace["intent"] = {"label": verdict.intent.value, "reason": verdict.reason,
                           "rule": verdict.matched_rule}
        trace["stages"].append("intent")
        if verdict.intent is not Intent.IN_SCOPE:
            trace["short_circuited"] = "no retrieval performed"
            return _refuse(query, _INTENT_TO_REFUSAL[verdict.intent], trace)

        # --- Retrieve + rerank --------------------------------------------
        results, rtrace = self.retriever.retrieve(query)
        trace["retrieval"] = rtrace
        trace["stages"].append("retrieve")

        results = self.reranker.rerank(query, results)
        trace["rerank"] = {"scored": sum(r.rerank_score is not None for r in results),
                           "top_scores": [r.rerank_score for r in results[:5]]}
        trace["stages"].append("rerank")

        # --- Gate 2 -------------------------------------------------------
        conf = evaluate_confidence(results)
        trace["confidence"] = _conf_trace(conf)
        trace["stages"].append("confidence")
        if conf.action is ConfidenceAction.CORRECT:
            results, ctrace = correct(self.retriever, self.reranker, query, results)
            trace["corrective"] = ctrace
            trace["stages"].append("corrective")
            conf = evaluate_confidence(results, allow_correction=False)
            trace["confidence_after_correction"] = _conf_trace(conf)
        if not conf.passed:
            return _refuse(query, RefusalReason.LOW_CONFIDENCE, trace, conf.top_score)

        # --- Generate -----------------------------------------------------
        answer = generate(query, conf.kept)
        trace["generation"] = {"claims": len(answer.claims),
                               "insufficient_context": answer.insufficient_context}
        trace["stages"].append("generate")

        # --- Gate 3 -------------------------------------------------------
        checked = validate_output(answer, conf.kept, query)
        trace["output_gate"] = {
            "passed": checked.passed, "detail": checked.detail,
            "invalid_citations": checked.invalid_citations,
            "unsupported_numbers": checked.unsupported_numbers,
            "unsupported_claims": checked.unsupported_claims,
            "safety_flags": checked.safety_flags,
        }
        trace["stages"].append("output_gate")
        if not checked.passed:
            assert checked.reason is not None
            return _refuse(query, checked.reason, trace, conf.top_score)

        answer = checked.repaired or answer
        return Response(query=query, answered=True, answer=answer.answer, claims=answer.claims,
                        citations=_citations(conf.kept), top_score=conf.top_score,
                        disclaimer=DISCLAIMER, trace=trace)
