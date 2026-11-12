"""Cross-encoder rerank.

This stays off the default search path. Scoring even 20 pairs with a
transformer is often the step that breaks a 100ms budget. `/search` reranks
only when asked. `/answer` reranks when a model is configured, because a
wrong passage in the prompt is worse than a slower answer.
"""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        raw = self.model.predict(pairs)
        return [float(value) for value in raw]
