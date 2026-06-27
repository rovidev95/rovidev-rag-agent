"""Token estimation and context budgeting.

Uses a ~4 chars/token heuristic to avoid a hard tiktoken dependency. Good enough
for budgeting; use tiktoken for exact counts when calling OpenAI.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Approximate token count. Stable and dependency-free."""
    if not text:
        return 0
    words = _WORD_RE.findall(text)
    # Blend word count with char/4 to better track sub-word tokenization.
    return max(len(words), (len(text) + 3) // 4)


def fit_to_budget(texts: list[str], max_tokens: int) -> list[str]:
    """Return the longest prefix of ``texts`` whose total tokens fit the budget."""
    selected: list[str] = []
    used = 0
    for text in texts:
        cost = estimate_tokens(text)
        if used + cost > max_tokens:
            break
        selected.append(text)
        used += cost
    return selected
