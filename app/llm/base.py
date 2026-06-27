from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMContext:
    """A retrieved passage made available to the model, tagged with a citation
    marker the model is expected to reference inline as ``[marker]``."""

    marker: int
    text: str
    source: str


@dataclass
class Generation:
    answer: str
    completion_tokens: int


@runtime_checkable
class LLM(Protocol):
    def generate(self, question: str, contexts: list[LLMContext]) -> Generation: ...
