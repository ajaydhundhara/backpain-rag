import numpy as np

from backpain_rag import guardrails, retrieval
from backpain_rag.index import BM25Index, LocalVectorStore, reciprocal_rank_fusion, unit
from backpain_rag.ingest import pack, parse_text
from backpain_rag.models import ConfidenceAction, Retrieved
from backpain_rag.retrieval import HybridRetriever, correct, rewrite

from .conftest import fake_vector


def test_rewrite_is_additive_and_skips_ambiguous():
    q, applied = rewrite("Is TENS useful for LBP?")
    assert q.startswith("Is TENS useful for LBP?")
    assert "transcutaneous electrical nerve stimulation" in q and "low back pain" in q
    assert len(applied) == 2
    q2, applied2 = rewrite("Is PT helpful?")
    assert applied2 == [] and q2 == "Is PT helpful?"


def test_rrf_fuses_ranks_not_scores():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert max(fused, key=fused.get) == "b"


def test_bm25_prefers_exact_terms(chunks):
    hits = BM25Index(chunks).search("TENS low back pain", limit=3)
    assert hits[0][0].chunk_id == "nice::0004::00"


def test_vector_store_roundtrip(tmp_path):
    ids = ["a", "b"]
    m = np.stack([unit([1, 0]), unit([0, 1])]).astype(np.float16)
    LocalVectorStore(ids, m).save()
    s = LocalVectorStore.load()
    assert s.search([0.9, 0.1], 1)[0][0] == "a"


def _retriever(chunks):
    vecs = np.stack([unit(fake_vector(c.embed_text())) for c in chunks]).astype(np.float16)
    return HybridRetriever(chunks=chunks, store=LocalVectorStore([c.chunk_id for c in chunks], vecs))


def test_hybrid_retrieval_and_hyde_fail_open(chunks, fake_embed, monkeypatch):
    def boom(*a, **k):
        raise retrieval.LLMError("down")

    monkeypatch.setattr(retrieval, "parse", boom)
    results, trace = _retriever(chunks).retrieve("imaging for low back pain", k=3)
    assert results[0].chunk.chunk_id == "nice::0001::00"
    assert trace["hyde"]["used"] is False and "unavailable" in trace["hyde"]["reason"]


def test_hyde_blends_with_the_query(chunks, fake_embed, monkeypatch):
    monkeypatch.setattr(retrieval, "parse",
                        lambda schema, *a, **k: schema(passages=["Imaging is not routinely offered."]))
    _, trace = _retriever(chunks).retrieve("scans for back pain", k=3)
    assert trace["hyde"]["used"] and trace["hyde"]["query_weight"] == 0.5


class _FixedReranker:
    """Scores by chunk id; records what it was asked to score."""

    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def rerank(self, query, results):
        for r in results:
            self.seen.append(r.chunk.chunk_id)
            r.rerank_score = self.scores.get(r.chunk.chunk_id, 0.0)
        return sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)


def test_corrective_pass_unions_and_only_scores_new(chunks, fake_embed, monkeypatch):
    monkeypatch.setenv("HYDE_ENABLED", "false")
    from backpain_rag import config

    config.get_settings.cache_clear()
    r = _retriever(chunks)
    first = [Retrieved(chunk=chunks[0], rerank_score=6.5)]
    rr = _FixedReranker({"nice::0005::00": 9.0})
    pool, trace = correct(r, rr, "spinal decompression sciatica", first)
    assert trace["ran"] and trace["improved"]
    assert "nice::0001::00" not in rr.seen  # already scored: not re-sent
    assert {p.chunk.chunk_id for p in pool} >= {"nice::0001::00", "nice::0005::00"}
    assert guardrails.evaluate_confidence(pool, allow_correction=False).action is ConfidenceAction.PROCEED


def test_pack_respects_target_and_overlap():
    sents = [f"Sentence number {i} has a few words in it." for i in range(40)]
    pieces = pack(sents, target=60, overlap=15)
    assert len(pieces) > 3
    assert pieces[1].split(".")[0] in pieces[0]  # overlap carried


def test_markdown_sections_and_pages(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Back pain\n\n## Imaging\nDo not image routinely.\f\n## Exercise\nExercise helps.\n")
    secs = parse_text(p)
    assert [s.path for s in secs] == ["Back pain > Imaging", "Back pain > Exercise"]
    assert secs[1].page_start == 2
