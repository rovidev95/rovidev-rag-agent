"""In-memory vector store with cosine similarity search (numpy only).

Replace with pgvector/Qdrant behind the same interface for larger datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StoredChunk:
    id: str
    document_id: str
    source: str
    text: str
    vector: np.ndarray
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: StoredChunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()

    def add(self, chunk: StoredChunk) -> None:
        self._chunks.append(chunk)

    def add_many(self, chunks: list[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    def delete_document(self, document_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.document_id != document_id]
        return before - len(self._chunks)

    def search(self, query_vector: list[float], k: int = 4) -> list[SearchResult]:
        if not self._chunks or k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(query)
        if q_norm == 0.0:
            return []
        query = query / q_norm

        matrix = np.vstack([c.vector for c in self._chunks])
        # Stored vectors are pre-normalized, so dot == cosine.
        scores = matrix @ query

        top_idx = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
        ]
