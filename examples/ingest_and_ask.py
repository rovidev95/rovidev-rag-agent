"""Run a full RAG flow locally without any API keys.

    python -m examples.ingest_and_ask
"""

from __future__ import annotations

from app.embeddings.hashing import HashingEmbedder
from app.llm.extractive import ExtractiveLLM
from app.models import Document
from app.rag import RAGPipeline

KNOWLEDGE_BASE = [
    Document(
        id="onboarding",
        source="Onboarding handbook",
        text=(
            "New employees get a laptop on day one. Access to internal tools is "
            "granted within 24 hours. The first week focuses on shadowing a "
            "mentor and reading the architecture docs."
        ),
    ),
    Document(
        id="security",
        source="Security policy",
        text=(
            "All secrets must be stored in the vault, never in the repository. "
            "Two-factor authentication is mandatory. Production access requires "
            "a reviewed pull request and an approval from a team lead."
        ),
    ),
]


def main() -> None:
    pipeline = RAGPipeline(embedder=HashingEmbedder(dim=512), llm=ExtractiveLLM())
    docs, chunks = pipeline.ingest(KNOWLEDGE_BASE)
    print(f"Ingested {docs} documents -> {chunks} chunks\n")

    for question in [
        "How do I get production access?",
        "When does a new employee receive a laptop?",
        "What is the weather like today?",
    ]:
        result = pipeline.ask(question)
        print(f"Q: {question}")
        print(f"A: {result.answer}")
        if result.citations:
            for c in result.citations:
                print(f"   [{c.marker}] {c.source} (score={c.score}) — {c.snippet}")
        print(f"   tokens: {result.token_usage.total_tokens}, grounded={result.grounded}\n")


if __name__ == "__main__":
    main()
