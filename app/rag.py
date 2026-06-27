"""The RAG pipeline: ingest -> retrieve -> ground -> answer with citations."""

from __future__ import annotations

import re

import numpy as np

from app.chunking import chunk_text
from app.embeddings.base import Embedder
from app.llm.base import LLM, LLMContext
from app.models import (
    AskResponse,
    Citation,
    Document,
    TokenUsage,
)
from app.tokens import estimate_tokens
from app.vectorstore import SearchResult, StoredChunk, VectorStore

_CITATION_RE = re.compile(r"\[(\d+)\]")


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        llm: LLM,
        store: VectorStore | None = None,
        *,
        max_context_tokens: int = 1500,
        min_score: float = 0.1,
        max_chars: int = 800,
        overlap_chars: int = 120,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.store = store or VectorStore()
        self.max_context_tokens = max_context_tokens
        self.min_score = min_score
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def ingest(self, documents: list[Document]) -> tuple[int, int]:
        all_chunks: list[StoredChunk] = []
        for doc in documents:
            # Re-ingesting a document replaces its previous chunks.
            self.store.delete_document(doc.id)
            pieces = chunk_text(
                doc.text, max_chars=self.max_chars, overlap_chars=self.overlap_chars
            )
            if not pieces:
                continue
            vectors = self.embedder.embed(pieces)
            for idx, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
                all_chunks.append(
                    StoredChunk(
                        id=f"{doc.id}::{idx}",
                        document_id=doc.id,
                        source=doc.source,
                        text=piece,
                        vector=np.asarray(vector, dtype=np.float64),
                        metadata=doc.metadata,
                    )
                )
        self.store.add_many(all_chunks)
        return len(documents), len(all_chunks)

    def ask(self, question: str, top_k: int = 4) -> AskResponse:
        query_vec = self.embedder.embed([question])[0]
        results = self.store.search(query_vec, k=top_k)
        relevant = [r for r in results if r.score >= self.min_score]

        if not relevant:
            answer = (
                "I could not find anything relevant to that question in the "
                "knowledge base."
            )
            return AskResponse(
                answer=answer,
                citations=[],
                token_usage=TokenUsage(
                    prompt_tokens=estimate_tokens(question),
                    context_tokens=0,
                    completion_tokens=estimate_tokens(answer),
                ),
                grounded=False,
            )

        contexts, citations = self._build_contexts(relevant)
        generation = self.llm.generate(question, contexts)

        used_markers = {int(m) for m in _CITATION_RE.findall(generation.answer)}
        final_citations = (
            [c for c in citations if c.marker in used_markers] or citations
        )

        context_tokens = sum(estimate_tokens(c.text) for c in contexts)
        return AskResponse(
            answer=generation.answer,
            citations=final_citations,
            token_usage=TokenUsage(
                prompt_tokens=estimate_tokens(question),
                context_tokens=context_tokens,
                completion_tokens=generation.completion_tokens,
            ),
            grounded=True,
        )

    def _build_contexts(
        self, results: list[SearchResult]
    ) -> tuple[list[LLMContext], list[Citation]]:
        contexts: list[LLMContext] = []
        citations: list[Citation] = []
        used = 0
        marker = 1

        for result in results:
            cost = estimate_tokens(result.chunk.text)
            if used + cost > self.max_context_tokens and contexts:
                break
            used += cost
            contexts.append(
                LLMContext(
                    marker=marker,
                    text=result.chunk.text,
                    source=result.chunk.source,
                )
            )
            citations.append(
                Citation(
                    marker=marker,
                    document_id=result.chunk.document_id,
                    source=result.chunk.source,
                    snippet=_snippet(result.chunk.text),
                    score=round(result.score, 4),
                )
            )
            marker += 1

        return contexts, citations


def _snippet(text: str, length: int = 160) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= length else text[:length].rstrip() + "…"
