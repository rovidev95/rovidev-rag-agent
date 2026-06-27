"""Sentence-aware text chunking with overlap."""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_text(
    text: str,
    *,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[str]:
    """Group sentences into chunks of at most ``max_chars`` with character overlap.

    Overlap carries trailing context from the previous chunk so a fact split
    across a boundary can still be retrieved.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail} {sentence}".strip()
        else:
            # A single sentence longer than max_chars: hard-split it.
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            current = ""

    if current:
        chunks.append(current)

    return chunks
