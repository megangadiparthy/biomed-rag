"""Turn a question into ordered passages, with a timing breakdown."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from biomed_rag.domain import Paper, Passage, chunk_paper
from biomed_rag.index.bm25 import BM25Index
from biomed_rag.index.embeddings import Embedder
from biomed_rag.index.qdrant_store import QdrantStore
from biomed_rag.retrieve.fusion import reciprocal_rank_fusion
from biomed_rag.retrieve.rerank import Reranker


@dataclass
class Hit:
    passage: Passage
    score: float
    dense_rank: int | None
    sparse_rank: int | None


@dataclass
class Timings:
    embed_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class SearchResult:
    query: str
    hits: list[Hit] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    reranker: str = "none"


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


class RetrievalEngine:
    def __init__(
        self,
        passages: list[Passage],
        bm25: BM25Index,
        store: QdrantStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.passages = passages
        self.by_id = {passage.chunk_id: passage for passage in passages}
        self.bm25 = bm25
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    @classmethod
    def from_papers(
        cls,
        papers: list[Paper],
        embedder: Embedder,
        store: QdrantStore,
        reranker: Reranker | None = None,
    ) -> RetrievalEngine:
        passages: list[Passage] = []
        for paper in papers:
            passages.extend(chunk_paper(paper))
        if len({passage.chunk_id for passage in passages}) != len(passages):
            raise ValueError("chunk ids must be unique")
        texts = [passage.indexed_text() for passage in passages]
        bm25 = BM25Index.build(texts, [passage.chunk_id for passage in passages])
        vectors = embedder.embed_documents(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
        store.reset(embedder.dim)
        store.upsert(passages, vectors)
        return cls(passages, bm25, store, embedder, reranker)

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 40,
        rerank: bool = False,
    ) -> SearchResult:
        started = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query is empty")
        top_k = max(1, top_k)
        candidate_k = max(top_k, candidate_k)

        embed_started = time.perf_counter()
        vector = self.embedder.embed_query(query)
        embed_done = time.perf_counter()

        dense = self.store.search(vector, candidate_k)
        dense_done = time.perf_counter()

        sparse = self.bm25.search(query, candidate_k)
        sparse_done = time.perf_counter()

        fused = reciprocal_rank_fusion(
            [[chunk_id for chunk_id, _ in dense], [chunk_id for chunk_id, _ in sparse]]
        )
        fusion_done = time.perf_counter()

        dense_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, start=1)}
        sparse_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(sparse, start=1)}
        chosen = fused[:candidate_k]
        reranker_name = "none"
        rerank_done = fusion_done
        if rerank and self.reranker is not None and chosen:
            texts = [self.by_id[chunk_id].text for chunk_id, _ in chosen if chunk_id in self.by_id]
            ids = [chunk_id for chunk_id, _ in chosen if chunk_id in self.by_id]
            scores = self.reranker.score(query, texts)
            order = sorted(range(len(ids)), key=lambda index: scores[index], reverse=True)
            chosen = [(ids[index], float(scores[index])) for index in order]
            reranker_name = self.reranker.name
            rerank_done = time.perf_counter()

        hits: list[Hit] = []
        for chunk_id, score in chosen[:top_k]:
            passage = self.by_id.get(chunk_id)
            if passage is None:
                continue
            hits.append(
                Hit(
                    passage=passage,
                    score=score,
                    dense_rank=dense_rank.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                )
            )
        finished = time.perf_counter()
        return SearchResult(
            query=query,
            hits=hits,
            timings=Timings(
                embed_ms=_ms(embed_started, embed_done),
                dense_ms=_ms(embed_done, dense_done),
                sparse_ms=_ms(dense_done, sparse_done),
                fusion_ms=_ms(sparse_done, fusion_done),
                rerank_ms=_ms(fusion_done, rerank_done),
                total_ms=_ms(started, finished),
            ),
            reranker=reranker_name,
        )
