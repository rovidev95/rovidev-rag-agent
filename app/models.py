"""Pydantic schemas shared across the API and the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document to ingest into the knowledge base."""

    id: str = Field(..., description="Stable unique id for the document.")
    text: str = Field(..., min_length=1)
    source: str = Field(..., description="Human-readable origin, e.g. a title or URL.")
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document]


class IngestResponse(BaseModel):
    ingested_documents: int
    total_chunks: int


class Citation(BaseModel):
    """A single grounded reference returned alongside an answer."""

    marker: int = Field(..., description="Citation number used inline as [n].")
    document_id: str
    source: str
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class TokenUsage(BaseModel):
    prompt_tokens: int
    context_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.context_tokens + self.completion_tokens


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    token_usage: TokenUsage
    grounded: bool = Field(
        ...,
        description="False when no relevant context was found (answer is a safe fallback).",
    )
