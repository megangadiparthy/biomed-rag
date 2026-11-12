"""Okapi BM25 over an inverted index.

Scoring touches the postings of the query terms only. A full scan of the
collection would not stay inside a 100ms budget once the corpus is large.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biomed_rag.tokenize import tokenize


@dataclass
class _Posting:
    docs: np.ndarray
    tfs: np.ndarray


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, _Posting] = {}

    @classmethod
    def build(cls, texts: list[str], doc_ids: list[str], k1: float = 1.2, b: float = 0.75) -> BM25Index:
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must be the same length")
        index = cls(k1=k1, b=b)
        index.doc_ids = list(doc_ids)
        bags: list[dict[str, int]] = []
        lengths = np.zeros(len(texts), dtype=np.float32)
        df: dict[str, int] = {}
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            bags.append(counts)
            lengths[i] = float(sum(counts.values()))
            for token in counts:
                df[token] = df.get(token, 0) + 1

        postings: dict[str, list[tuple[int, int]]] = {token: [] for token in df}
        for doc_index, counts in enumerate(bags):
            for token, tf in counts.items():
                postings[token].append((doc_index, tf))

        index.doc_len = lengths
        index.avgdl = float(lengths.mean()) if len(lengths) else 0.0
        n_docs = len(texts)
        index.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }
        index.postings = {
            token: _Posting(
                docs=np.asarray([doc for doc, _ in pairs], dtype=np.int32),
                tfs=np.asarray([tf for _, tf in pairs], dtype=np.float32),
            )
            for token, pairs in postings.items()
        }
        return index

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self.doc_ids:
            return []
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        k1 = self.k1
        norm = k1 * (1.0 - self.b)
        length_scale = (k1 * self.b / self.avgdl) if self.avgdl else 0.0
        for term in terms:
            posting = self.postings.get(term)
            if posting is None:
                continue
            tf = posting.tfs
            denom = tf + norm + length_scale * self.doc_len[posting.docs]
            scores[posting.docs] += self.idf[term] * (tf * (k1 + 1.0)) / denom

        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        k = min(top_k, int(positive.size))
        if positive.size > k:
            pick = np.argpartition(scores[positive], -k)[-k:]
            chosen = positive[pick]
        else:
            chosen = positive
        order = chosen[np.argsort(scores[chosen])[::-1]]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as handle:
            index = pickle.load(handle)
        if not isinstance(index, cls):
            raise TypeError(f"{path} did not contain a BM25Index")
        return index
