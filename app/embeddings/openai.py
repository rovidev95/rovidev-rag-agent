"""OpenAI embeddings adapter. Requires the ``openai`` extra."""

from __future__ import annotations

import math


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dim: int = 1536,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' extra: pip install '.[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = dim

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        return vec if norm == 0.0 else [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - needs API
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [self._normalize(item.embedding) for item in response.data]
