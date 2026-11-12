"""Dense encoders.

HashingEmbedder is a real vector function with no learned weights. It is what
the latency run and the tests use, because it does not download a model.
TransformerEmbedder is the semantic encoder: mean-pooled PubMedBERT, L2
normalized, written as a torch module so the Qdrant collection can be rebuilt
without changing the retriever.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from biomed_rag.tokenize import tokenize


def _fnv(token: str) -> int:
    value = 2166136261
    for byte in token.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vector = out[row]
            for token in tokenize(text):
                hashed = _fnv(token)
                vector[hashed % self.dim] += 1.0 if (hashed >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.maximum(norms, 1e-9, out=norms)
        out /= norms
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def mean_pool(last_hidden_state, attention_mask):
    """Mean pool a transformer sequence, ignoring pad tokens, then L2-normalize."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return torch.nn.functional.normalize(summed / counts, p=2, dim=1)


class TransformerEmbedder:
    """Sentence embedding via a Hugging Face encoder and the torch mean pool above."""

    name = "transformer"

    def __init__(self, model_name: str = "NeuML/pubmedbert-base-embeddings") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self.torch
        batched: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                hidden = self.model(**encoded).last_hidden_state
                pooled = mean_pool(hidden, encoded["attention_mask"])
                batched.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(batched, axis=0)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]


def build_embedder(kind: str, dim: int = 256, model_name: str = "NeuML/pubmedbert-base-embeddings") -> Embedder:
    if kind == "hashing":
        return HashingEmbedder(dim=dim)
    if kind == "transformer":
        return TransformerEmbedder(model_name=model_name)
    raise ValueError(f"unknown embedder {kind!r}")
