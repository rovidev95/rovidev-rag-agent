"""OpenAI chat adapter. Requires the ``openai`` extra.

Prompts the model to answer only from the given context and cite with ``[n]``.
"""

from __future__ import annotations

from app.llm.base import Generation, LLMContext

_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    "provided context passages. Cite every claim inline with the passage marker "
    "in square brackets, e.g. [1]. If the context does not contain the answer, "
    "say you could not find it. Never invent facts or sources."
)


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAILLM requires the 'openai' extra: pip install '.[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, contexts: list[LLMContext]) -> Generation:  # pragma: no cover
        context_block = "\n\n".join(
            f"[{c.marker}] (source: {c.source})\n{c.text}" for c in contexts
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        usage = response.usage
        completion_tokens = usage.completion_tokens if usage else 0
        return Generation(answer=answer, completion_tokens=completion_tokens)
