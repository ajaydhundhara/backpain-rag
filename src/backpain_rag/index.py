"""Embeddings, the dense vector store, the lexical BM25 index, and RRF.

The vector store is a pair of local files (a float16 matrix and the chunk ids
in the same order) searched by brute-force cosine. For a guideline corpus of a
few thousand chunks that takes about a millisecond, needs no service to run,
and ships inside the deployment bundle. `VectorStore` is a Protocol, so a real
database (LanceDB, pgvector, Qdrant) can replace it without touching retrieval.

BM25 is rebuilt from data/chunks/*.jsonl at start-up rather than persisted.
That costs milliseconds and removes a whole class of bug where the lexical and
dense halves silently describe different corpora.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from . import llm
from .cache import get_cache, key_for
from .config import get_settings
from .models import Chunk, SourceDoc


class StoreError(RuntimeError):
    pass


# --- embeddings --------------------------------------------------------------


class Embedder:
    """Query embeddings are cached: a query vector is an exact mapping, safe to
    keep for a month, and it is paid on every request including refused ones."""

    def embed_one(self, text: str) -> list[float]:
        s = get_settings()
        cache = get_cache()
        ckey = f"emb:{key_for(s.openai_embed_model, s.embed_dim, text)}"
        hit = cache.get_json(ckey)
        if isinstance(hit, list):
            return hit
        vec = llm.embed([text])[0]
        cache.set_json(ckey, vec, s.cache_embed_ttl_s)
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return llm.embed(texts)


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a if n == 0.0 else a / n


# --- vector store ---------------------------------------------------------------


class VectorStore(Protocol):
    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]: ...


class LocalVectorStore:
    """Brute-force cosine over a normalised float16 matrix on disk."""

    def __init__(self, ids: list[str], matrix: np.ndarray) -> None:
        self.ids = ids
        self.matrix = matrix

    @classmethod
    def load(cls) -> LocalVectorStore:
        d = get_settings().index_dir
        ids_path, mat_path = d / "ids.json", d / "vectors.npy"
        if not ids_path.exists() or not mat_path.exists():
            raise StoreError("no vector index found -- run `rag index` first")
        ids = json.loads(ids_path.read_text())
        matrix = np.load(mat_path)
        if len(ids) != matrix.shape[0]:
            raise StoreError("vector index is corrupt: ids and vectors differ in length")
        return cls(ids, matrix)

    def save(self) -> None:
        d = get_settings().index_dir
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.matrix)
        (d / "ids.json").write_text(json.dumps(self.ids))

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = unit(vector).astype(np.float32)
        sims = self.matrix.astype(np.float32) @ q
        top = np.argsort(-sims)[:limit]
        return [(self.ids[i], float(sims[i])) for i in top]


# --- lexical --------------------------------------------------------------------

# Keep intra-word hyphens and digits: "l4-l5", "straight-leg", "b12".
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # Same contextual text the dense half embedded, so RRF fuses rankings
        # over the same documents.
        corpus = [tokenize(c.embed_text()) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = np.argsort(-scores)[:limit]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0.0]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """Fuse ranked id lists by rank, never by score.

    BM25 scores are unbounded and corpus-relative; cosine similarity is bounded.
    Normalising them onto one scale invents a comparison that does not exist.
    RRF only compares a document's position within its own ranking.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank + 1)
    return fused


# --- build ----------------------------------------------------------------------


def build(chunks: list[Chunk], docs: list[SourceDoc], on_progress=None) -> dict:
    """Embed every chunk, write the vector store and the index manifest."""
    if not chunks:
        raise StoreError("no chunks to index -- run `rag ingest` first")
    s = get_settings()
    vectors: list[list[float]] = []
    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vectors.extend(llm.embed([c.embed_text() for c in part]))
        if on_progress:
            on_progress(min(i + batch, len(chunks)), len(chunks))
    matrix = np.stack([unit(v) for v in vectors]).astype(np.float16)
    LocalVectorStore([c.chunk_id for c in chunks], matrix).save()

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "index_version": s.index_version,
        "embed_model": s.openai_embed_model,
        "embed_dim": s.embed_dim,
        "chunk_target_tokens": s.chunk_target_tokens,
        "chunk_overlap_tokens": s.chunk_overlap_tokens,
        "n_chunks": len(chunks),
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "sha256": d.sha256,
             "version": d.version, "n_chunks": sum(c.doc_id == d.doc_id for c in chunks)}
            for d in docs
        ],
    }
    s.index_manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_index_manifest() -> dict | None:
    p = get_settings().index_manifest_path
    return json.loads(p.read_text()) if p.exists() else None
