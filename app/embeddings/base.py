from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors. Implementations must return L2-normalized
    vectors of a fixed ``dim`` so cosine similarity reduces to a dot product."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
