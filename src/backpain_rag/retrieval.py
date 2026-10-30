"""Retrieval: rewrite -> HyDE -> hybrid search (dense + BM25, RRF) -> rerank,
plus the one-shot corrective retry used by gate 2's middle band.

The dense and lexical halves deliberately do *not* search with the same query.
The dense half searches with a HyDE-blended vector; the lexical half with the
literal rewritten string, because exact terms (drug names, "L5", "TENS") are
precisely what eighty words of generated prose would bury. Same corpus,
different query representations -- which is the whole reason to fuse them.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from .cache import get_cache, key_for
from .config import get_settings
from .index import BM25Index, Embedder, LocalVectorStore, StoreError, VectorStore, \
    reciprocal_rank_fusion, unit
from .ingest import load_chunks
from .llm import LLMError, note_degraded, parse
from .models import Chunk, Retrieved
from .policy import AMBIGUOUS, GLOSSARY


class RetrievalError(RuntimeError):
    pass


# --- 1. Conservative rewrite ----------------------------------------------------
# Additive, hand-curated, deliberately small. A model-generated expansion is how
# a question about one condition quietly starts retrieving another.

_WORD = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def normalise(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def rewrite(query: str) -> tuple[str, list[str]]:
    """Append expansions for known abbreviations. Returns (query, applied) --
    the log goes into the trace, because a silent rewrite is untraceable when
    an answer later looks wrong."""
    q = normalise(query)
    applied: list[str] = []
    extra: list[str] = []
    seen: set[str] = set()
    for token in _WORD.findall(q):
        low = token.lower()
        if low in AMBIGUOUS or low in seen:
            continue
        expansion = GLOSSARY.get(low)
        if expansion and expansion.lower() not in q.lower():
            applied.append(f"{token} -> {expansion}")
            extra.append(expansion)
            seen.add(low)
    return (f"{q} {' '.join(extra)}" if extra else q), applied


# --- 2. HyDE ----------------------------------------------------------------------
# Three rules bound it:
#  * The hypothetical passage is never shown, cited, reranked or generated from.
#    It is embedded and discarded. It can change *which* real passages are found,
#    never what is asserted.
#  * It is blended with the query vector, not substituted, so a hypothetical
#    that drifts to the wrong condition pulls retrieval only partway.
#  * Dense half only. BM25 keeps searching the literal query.
# This is the one stage that fails OPEN: it is a retrieval improvement, not a
# gate, so an unavailable model falls back to the plain query vector.

_HYDE_WORDS = 90

_HYDE_SYSTEM = """You write short passages that imitate back pain reference \
material: clinical guidelines, self-help and rehabilitation books (for example \
McKenzie-style exercise books), and patient-information leaflets.

Given a question, write the passage such a source would contain to answer it: \
declarative, with the vocabulary these sources use (named exercises and \
positions, posture, repetitions and frequency, red flags, interventions, \
populations).

Rules:
- About {words} words per passage. One paragraph, prose only, no markdown.
- Do not address a reader, do not hedge, no preamble.
- Write about the specific topic asked about and nothing else.
- If unsure of exact contents, still write the passage you would expect.

Return exactly {n} passage(s). This text is retrieval scaffolding: it is \
embedded and discarded, never shown to anyone and never used as an answer."""


class _Hypothetical(BaseModel):
    passages: list[str] = Field(description="one hypothetical passage per entry")


def hyde_passages(query: str, n: int) -> list[str]:
    """Generate (and cache) `n` hypothetical passages. Raises LLMError."""
    s = get_settings()
    model = s.hyde_model or s.openai_guard_model
    cache = get_cache()
    ckey = f"hyde:{key_for(model, s.hyde_version, n, query)}"
    hit = cache.get_json(ckey)
    if isinstance(hit, list) and hit and all(isinstance(p, str) for p in hit):
        return hit
    out = parse(
        _Hypothetical,
        _HYDE_SYSTEM.format(words=_HYDE_WORDS, n=n),
        f"Question: {query}",
        model=model,
        # Non-zero only when several are wanted: identical samples average to one.
        temperature=0.0 if n == 1 else 0.7,
    )
    passages = [p.strip() for p in out.passages if p and p.strip()][:n]
    if passages:
        cache.set_json(ckey, passages, s.cache_hyde_ttl_s)
    return passages


def query_vector(
    rewritten: str, embedder: Embedder, query_weight: float | None = None
) -> tuple[list[float], dict]:
    """The vector the dense half searches with, plus a trace of how it was made.

    `query_weight` overrides the configured blend for one call (the corrective
    pass leans harder on the hypothetical). Passed as an argument, never set on
    the shared settings object, which concurrent requests also read.
    """
    s = get_settings()
    base = embedder.embed_one(rewritten)
    if not s.hyde_enabled or s.hyde_n < 1:
        return base, {"used": False, "reason": "disabled"}
    try:
        passages = hyde_passages(rewritten, s.hyde_n)
    except LLMError as e:
        note_degraded("hyde")
        return base, {"used": False, "reason": f"model unavailable: {e}"[:200]}
    if not passages:
        return base, {"used": False, "reason": "model returned no passage"}

    w = min(1.0, max(0.0, s.hyde_query_weight if query_weight is None else query_weight))
    hypo = np.mean([unit(embedder.embed_one(p)) for p in passages], axis=0)
    blended = unit(w * unit(base) + (1.0 - w) * unit(hypo))
    return blended.tolist(), {"used": True, "n": len(passages), "query_weight": w,
                              "passages": passages}


# --- 3. Hybrid retrieval --------------------------------------------------------


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise RetrievalError("no chunks found in data/chunks/ -- run `rag ingest` first")
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Index(self.chunks)
        self.store = store if store is not None else LocalVectorStore.load()
        self.embedder = embedder or Embedder()

    def retrieve(
        self, query: str, k: int | None = None, hyde_query_weight: float | None = None
    ) -> tuple[list[Retrieved], dict]:
        k = k or get_settings().retrieve_k
        rewritten, applied = rewrite(query)

        qv, hyde_trace = query_vector(rewritten, self.embedder, hyde_query_weight)
        dense: list[tuple[Chunk, float]] = []
        orphans: list[str] = []
        for cid, score in self.store.search(qv, limit=k):
            chunk = self.by_id.get(cid)
            if chunk is None:
                # In the vector index, not in the local chunks: the two halves
                # have drifted. Report it rather than half-retrieve.
                orphans.append(cid)
                continue
            dense.append((chunk, score))

        lexical = self.bm25.search(rewritten, limit=k)

        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in dense], [c.chunk_id for c, _ in lexical]]
        )
        dense_s = {c.chunk_id: s for c, s in dense}
        lex_s = {c.chunk_id: s for c, s in lexical}
        results = [
            Retrieved(chunk=self.by_id[cid], vector_score=dense_s.get(cid),
                      lexical_score=lex_s.get(cid), fused_score=score)
            for cid, score in sorted(fused.items(), key=lambda x: -x[1])
        ][:k]

        return results, {
            "rewritten": rewritten,
            "expansions": applied,
            "hyde": hyde_trace,
            "dense_hits": len(dense),
            "lexical_hits": len(lexical),
            "fused_pool": len(fused),
            "orphan_chunks": orphans,
        }


# --- 4. Rerank ----------------------------------------------------------------------
# The score must be ABSOLUTE. Gate 2 refuses on the top-1 value, and the best of
# five irrelevant passages still ranks first -- so the prompt asks how well each
# passage answers the question on a fixed scale, never which one is best.


class Reranker(Protocol):
    def rerank(self, query: str, results: list[Retrieved]) -> list[Retrieved]: ...


class _Score(BaseModel):
    index: int
    score: int = Field(ge=0, le=10)


class _Scores(BaseModel):
    scores: list[_Score]


_RERANK_SYSTEM = """You score how well each passage answers a question, for a \
retrieval system over back pain reference material.

Score each passage independently on this fixed scale. Do NOT rank them against \
each other and do not spread scores out -- if every passage is irrelevant, \
every score should be low.

10 - directly and completely answers the question
7-9 - contains most of the answer
4-6 - related topic, contains part of the answer
1-3 - same general subject, does not answer the question
0 - unrelated

Return one score for every passage, using the passage numbers given."""


def _sort_scored(results: list[Retrieved]) -> list[Retrieved]:
    return sorted(results, key=lambda r: (r.rerank_score is not None, r.rerank_score or 0.0),
                  reverse=True)


class LLMReranker:
    def __init__(self, batch_size: int = 20) -> None:
        self.batch_size = batch_size

    def rerank(self, query: str, results: list[Retrieved]) -> list[Retrieved]:
        for start in range(0, len(results), self.batch_size):
            batch = results[start : start + self.batch_size]
            passages = "\n\n".join(
                f"[{i}] {r.chunk.title} — {r.chunk.section_path}\n{r.chunk.text}"
                for i, r in enumerate(batch)
            )
            try:
                out = parse(_Scores, _RERANK_SYSTEM, f"Question: {query}\n\nPassages:\n\n{passages}")
            except LLMError:
                # Fail closed: no scores -> gate 2 sees no confidence -> refusal.
                note_degraded("reranker")
                for r in batch:
                    r.rerank_score = None
                continue
            by_index = {s.index: s.score for s in out.scores}
            for i, r in enumerate(batch):
                raw = by_index.get(i)
                r.rerank_score = float(raw) if raw is not None else None
        return _sort_scored(results)


# --- 5. Corrective retry (CRAG without the web) -----------------------------------
# CRAG falls back to web search on poor retrieval. That cannot exist here: every
# claim must trace to a sha256-pinned source. So the retry targets the SAME
# corpus -- deeper (ranks the reranker never saw) and leaning on HyDE, since the
# literal query is what already failed. Safety properties:
#   * the bar does not move (caller re-judges at the same threshold),
#   * exactly one retry (caller passes allow_correction=False the second time),
#   * nothing is discarded (the pool is the union of both passes).


def correct(
    retriever: HybridRetriever, reranker: Reranker, query: str, previous: list[Retrieved]
) -> tuple[list[Retrieved], dict]:
    s = get_settings()
    scores = {r.chunk.chunk_id: r.rerank_score for r in previous if r.rerank_score is not None}
    before = max(scores.values(), default=0.0)
    try:
        retried, rtrace = retriever.retrieve(
            query, k=s.corrective_k, hyde_query_weight=s.corrective_hyde_query_weight
        )
    except (RetrievalError, StoreError, LLMError) as e:
        return previous, {"ran": False, "reason": f"retrieval failed: {e}"[:200]}

    merged: dict[str, Retrieved] = {r.chunk.chunk_id: r for r in previous}
    for r in retried:
        if r.chunk.chunk_id not in merged:
            merged[r.chunk.chunk_id] = r
    pool = list(merged.values())
    fresh = [r for r in pool if r.rerank_score is None and r.chunk.chunk_id not in scores]
    if fresh:
        # Only unseen passages -- sound because the scale is absolute.
        reranker.rerank(query, fresh)
    pool = _sort_scored(pool)
    after = max((r.rerank_score or 0.0 for r in pool), default=0.0)
    return pool, {
        "ran": True, "k": s.corrective_k,
        "hyde_query_weight": s.corrective_hyde_query_weight,
        "new_candidates": len(fresh), "pool": len(pool),
        "top_before": before, "top_after": after, "improved": after > before,
        "hyde": rtrace.get("hyde"),
    }
