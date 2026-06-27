from app.chunking import chunk_text, split_sentences


def test_split_sentences():
    text = "Hello world. How are you? I am fine!"
    assert split_sentences(text) == ["Hello world.", "How are you?", "I am fine!"]


def test_chunk_respects_max_chars():
    text = " ".join(f"Sentence number {i} here." for i in range(50))
    chunks = chunk_text(text, max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_overlap_carries_context():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    chunks = chunk_text(text, max_chars=40, overlap_chars=15)
    assert len(chunks) >= 2


def test_hard_split_long_sentence():
    text = "x" * 250
    chunks = chunk_text(text, max_chars=100, overlap_chars=0)
    assert len(chunks) == 3


def test_empty_text():
    assert chunk_text("") == []


def test_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hi", max_chars=10, overlap_chars=10)
