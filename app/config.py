"""Runtime configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # "offline" (default, no keys) or "openai".
    backend: str = "offline"
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    embedding_dim: int = 2048
    top_k: int = 4
    max_context_tokens: int = 1500
    min_score: float = 0.1


def build_pipeline(settings: Settings):
    """Wire the pipeline based on the configured backend."""
    from app.embeddings.hashing import HashingEmbedder
    from app.llm.extractive import ExtractiveLLM
    from app.rag import RAGPipeline

    if settings.backend == "openai":
        from app.embeddings.openai import OpenAIEmbedder
        from app.llm.openai import OpenAILLM

        embedder = OpenAIEmbedder(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
        llm = OpenAILLM(model=settings.chat_model, api_key=settings.openai_api_key)
    else:
        embedder = HashingEmbedder(dim=settings.embedding_dim)
        llm = ExtractiveLLM()

    return RAGPipeline(
        embedder=embedder,
        llm=llm,
        max_context_tokens=settings.max_context_tokens,
        min_score=settings.min_score,
    )
