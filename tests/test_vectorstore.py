import numpy as np

from app.embeddings.hashing import HashingEmbedder
from app.vectorstore import StoredChunk, VectorStore


def _chunk(emb, cid, doc, text):
    [vec] = emb.embed([text])
    return StoredChunk(
        id=cid,
        document_id=doc,
        source=doc,
        text=text,
        vector=np.asarray(vec, dtype=np.float64),
    )


def test_search_orders_by_similarity():
    emb = HashingEmbedder(dim=512)
    store = VectorStore()
    store.add(_chunk(emb, "1", "d1", "billing and stripe subscriptions"))
    store.add(_chunk(emb, "2", "d2", "cats and dogs are pets"))

    [qvec] = emb.embed(["stripe subscriptions billing"])
    results = store.search(qvec, k=2)
    assert results[0].chunk.id == "1"
    assert results[0].score >= results[1].score


def test_delete_document():
    emb = HashingEmbedder(dim=64)
    store = VectorStore()
    store.add(_chunk(emb, "1", "d1", "one"))
    store.add(_chunk(emb, "2", "d1", "two"))
    store.add(_chunk(emb, "3", "d2", "three"))
    removed = store.delete_document("d1")
    assert removed == 2
    assert len(store) == 1


def test_empty_store_returns_nothing():
    store = VectorStore()
    assert store.search([0.1, 0.2], k=3) == []
