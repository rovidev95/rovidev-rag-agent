import math

from app.embeddings.hashing import HashingEmbedder


def test_dimensions_and_normalization():
    emb = HashingEmbedder(dim=64)
    [vec] = emb.embed(["hello world hello"])
    assert len(vec) == 64
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-9


def test_similar_texts_score_higher():
    emb = HashingEmbedder(dim=512)
    a, b, c = emb.embed(
        [
            "stripe webhook idempotency retries",
            "stripe webhook idempotency and retries explained",
            "the weather in spain is sunny today",
        ]
    )
    sim_ab = sum(x * y for x, y in zip(a, b, strict=True))
    sim_ac = sum(x * y for x, y in zip(a, c, strict=True))
    assert sim_ab > sim_ac


def test_empty_text_is_zero_vector():
    emb = HashingEmbedder(dim=32)
    [vec] = emb.embed([""])
    assert all(v == 0.0 for v in vec)
