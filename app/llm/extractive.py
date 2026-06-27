"""Offline answerer: picks the most relevant sentences from the retrieved
context and tags them with inline ``[marker]`` citations."""

from __future__ import annotations

import re

from app.llm.base import Generation, LLMContext
from app.tokens import estimate_tokens

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "with", "what", "how", "why", "when", "where", "who",
    "does", "do", "did", "can", "could", "should", "would", "this", "that",
}


def _keywords(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


class ExtractiveLLM:
    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[LLMContext]) -> Generation:
        q_terms = _keywords(question)
        scored: list[tuple[float, int, str]] = []

        for ctx in contexts:
            for sentence in _SENTENCE_RE.split(ctx.text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(q_terms & _keywords(sentence))
                if overlap > 0:
                    scored.append((overlap, ctx.marker, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[: self.max_sentences]

        if not picked:
            answer = (
                "I could not find an answer to that question in the provided "
                "sources."
            )
            return Generation(answer=answer, completion_tokens=estimate_tokens(answer))

        parts = [f"{sentence} [{marker}]" for _, marker, sentence in picked]
        answer = " ".join(parts)
        return Generation(answer=answer, completion_tokens=estimate_tokens(answer))
