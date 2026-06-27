from app.embeddings.hashing import HashingEmbedder
from app.llm.extractive import ExtractiveLLM
from app.models import Document
from app.rag import RAGPipeline

KB = [
    Document(
        id="webhooks",
        source="Webhooks guide",
        text=(
            "Stripe delivers webhooks at least once, so handlers must be "
            "idempotent. Record each event id before processing to avoid double "
            "fulfillment. Use exponential backoff with jitter when retrying "
            "failed deliveries."
        ),
    ),
    Document(
        id="pets",
        source="Pets blog",
        text=(
            "Cats are independent animals. Dogs are loyal companions. Many "
            "families enjoy keeping pets at home."
        ),
    ),
]


def _pipeline() -> RAGPipeline:
    pipe = RAGPipeline(embedder=HashingEmbedder(dim=4096), llm=ExtractiveLLM())
    pipe.ingest(KB)
    return pipe


def test_grounded_answer_has_citations():
    pipe = _pipeline()
    res = pipe.ask("Why must webhook handlers be idempotent?", top_k=4)
    assert res.grounded is True
    assert res.citations
    # The cited document should be the webhooks guide, not the pets blog.
    assert any(c.document_id == "webhooks" for c in res.citations)
    # Inline markers must reference real citations.
    assert "[" in res.answer


def test_token_usage_is_populated():
    pipe = _pipeline()
    res = pipe.ask("How should failed deliveries be retried?", top_k=4)
    assert res.token_usage.context_tokens > 0
    assert res.token_usage.total_tokens > 0


def test_unrelated_question_is_not_grounded():
    pipe = _pipeline()
    res = pipe.ask("What is the capital of France?", top_k=4)
    assert res.grounded is False
    assert res.citations == []


def test_reingest_replaces_document():
    pipe = _pipeline()
    before = len(pipe.store)
    pipe.ingest([KB[0]])  # re-ingest same doc id
    after = len(pipe.store)
    assert before == after  # no duplicate chunks
