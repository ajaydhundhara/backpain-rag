"""Test fixtures. No test touches the network: the LLM and embeddings are faked."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from backpain_rag import cache, config, db, llm, security
from backpain_rag.models import Chunk


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Fresh settings per test, with every data path redirected to tmp_path."""
    for var in ("DATABASE_URL", "REDIS_URL", "AUTH_ENABLED", "SES_SMTP_HOST", "MAIL_FROM",
                "OPENAI_API_KEY", "SESSION_SECRET", "HYDE_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config.Settings, "data_dir", property(lambda self: tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("SCRYPT_N", "1024")  # fast hashing in tests
    monkeypatch.setenv("OPENAI_API_KEY", "")
    (tmp_path / "data").mkdir()
    config.get_settings.cache_clear()
    cache.get_cache.cache_clear()
    db.reset_engine()
    security.reset_secret()
    llm.clear_degradations()
    yield
    config.get_settings.cache_clear()
    cache.get_cache.cache_clear()
    db.reset_engine()


def fake_vector(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words embedding: texts sharing words are close."""
    v = np.zeros(dim, dtype=np.float32)
    for w in text.lower().split():
        h = int(hashlib.md5(w.strip(".,?!:;()").encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


@pytest.fixture
def fake_embed(monkeypatch):
    monkeypatch.setattr(llm, "embed", lambda texts, batch_size=128: [fake_vector(t) for t in texts])


def make_chunk(cid: str, text: str, section: str = "Management", doc: str = "nice-ng59") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, title="Low back pain and sciatica", source_url="",
                 publisher="NICE", section_path=section, page_start=3, page_end=3, text=text,
                 n_tokens=len(text.split()), index_version="v1", ingested_at="2026-01-01")


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("nice::0001::00", "Do not routinely offer imaging in a non-specialist setting "
                   "for people with low back pain with or without sciatica.", "Imaging"),
        make_chunk("nice::0002::00", "Consider a group exercise programme within the NHS for people "
                   "with a specific episode or flare-up of low back pain.", "Exercise"),
        make_chunk("nice::0003::00", "Do not offer paracetamol alone for managing low back pain. "
                   "Consider oral NSAIDs at the lowest effective dose for the shortest possible "
                   "period of time.", "Pharmacological"),
        make_chunk("nice::0004::00", "Do not offer opioids for managing chronic sciatica. Do not offer "
                   "transcutaneous electrical nerve stimulation (TENS) for low back pain.",
                   "Pharmacological"),
        make_chunk("nice::0005::00", "Consider spinal decompression for people with sciatica when "
                   "non-surgical treatment has not improved pain or function and their "
                   "radiological findings are consistent with sciatic symptoms.", "Surgery"),
    ]
