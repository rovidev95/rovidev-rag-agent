"""Embeddings via the hashing trick. No API key or model download required."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dropping stopwords reduces spurious similarity from hash collisions on short
# queries.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
        "and", "or", "in", "on", "for", "with", "at", "by", "from", "as", "it",
        "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
        "what", "how", "why", "when", "where", "who", "which", "do", "does",
        "did", "can", "could", "should", "would", "will", "my", "your", "our",
    }
)


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
        for token in tokens:
            vec[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]
