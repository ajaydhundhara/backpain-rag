"""Shared data contracts. Every stage of the pipeline agrees on these shapes."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    """One manifest entry. A file not described here is never ingested."""

    doc_id: str
    title: str
    filename: str
    url: str = ""
    publisher: str = ""
    version: str | None = None
    published: str | None = None
    sha256: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    publisher: str
    section_path: str
    page_start: int
    page_end: int
    text: str
    n_tokens: int
    doc_version: str | None = None
    index_version: str
    ingested_at: str

    @property
    def pages(self) -> str:
        # 0 means the source carried no page evidence. "p.1" for a whole
        # document would be confidently wrong, which is worse than admitting it.
        if not self.page_start:
            return "n/a"
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    def cite_label(self, marker: str) -> str:
        return f"[{marker} | {self.doc_id} | {self.title} | §{self.section_path} | p.{self.pages}]"

    def embed_text(self) -> str:
        """What gets embedded and BM25-indexed: title and section path prefixed,
        so a body that never names its topic still matches a query that does.
        `text` itself stays clean for display and citation checking."""
        return f"{self.title} — {self.section_path}\n\n{self.text}"


class Retrieved(BaseModel):
    chunk: Chunk
    marker: str = ""
    vector_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class Intent(StrEnum):
    """Gate 1 outcomes. Only IN_SCOPE proceeds to retrieval."""

    IN_SCOPE = "in_scope"
    PERSONALIZED_ADVICE = "personalized_advice"
    EMERGENCY = "emergency"
    OUT_OF_DOMAIN = "out_of_domain"


class IntentVerdict(BaseModel):
    intent: Intent
    reason: str
    matched_rule: str | None = None


class ConfidenceAction(StrEnum):
    """Gate 2 outcomes. CORRECT is "retry retrieval once before deciding"."""

    PROCEED = "proceed"
    CORRECT = "correct"
    REFUSE = "refuse"


class Claim(BaseModel):
    """One factual statement. chunk_ids may not be empty: enforced by schema."""

    text: str
    chunk_ids: list[str] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    insufficient_context: bool = False


class RefusalReason(StrEnum):
    PERSONALIZED_ADVICE = "personalized_advice"
    EMERGENCY = "emergency"
    OUT_OF_DOMAIN = "out_of_domain"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_DECLINED = "model_declined"
    UNGROUNDED_OUTPUT = "ungrounded_output"
    UNSAFE_OUTPUT = "unsafe_output"


class Citation(BaseModel):
    marker: str
    chunk_id: str
    doc_id: str
    title: str
    section: str
    pages: str
    source_url: str = ""
    rerank_score: float | None = None


class Response(BaseModel):
    """The one object every exit of the pipeline returns: CLI, API and eval alike."""

    query: str
    answered: bool
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    top_score: float | None = None
    disclaimer: str = ""
    trace: dict = Field(default_factory=dict)
