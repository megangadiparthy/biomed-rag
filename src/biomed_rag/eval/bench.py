"""Latency at a chosen corpus size, and recall on the labeled demo set.

The large run uses synthetic abstracts. It measures the index, not medical
accuracy. Relevance is measured separately on data/demo.jsonl, which is also
synthetic and labeled by hand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from biomed_rag.domain import Paper, load_papers
from biomed_rag.index.embeddings import HashingEmbedder
from biomed_rag.index.qdrant_store import QdrantStore
from biomed_rag.retrieve.engine import RetrievalEngine


def synthesize(n_docs: int, seed: int = 7) -> list[Paper]:
    rng = np.random.default_rng(seed)
    vocab = [f"t{i}" for i in range(8000)]
    papers: list[Paper] = []
    for index in range(n_docs):
        picks = rng.integers(0, len(vocab), size=36)
        words = " ".join(vocab[int(pick)] for pick in picks)
        papers.append(
            Paper(
                pmid=f"syn-{index}",
                title=f"note {index}",
                abstract=words,
                source="synthetic",
            )
        )
    return papers


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[position])


def bench_latency(n_docs: int = 100_000, n_queries: int = 200, dim: int = 128, seed: int = 7) -> dict[str, object]:
    papers = synthesize(n_docs, seed=seed)
    embedder = HashingEmbedder(dim=dim)
    engine = RetrievalEngine.from_papers(papers, embedder, QdrantStore(":memory:"))
    rng = np.random.default_rng(seed + 1)
    vocab = [f"t{i}" for i in range(8000)]
    totals: list[float] = []
    dense: list[float] = []
    sparse: list[float] = []
    # One untimed query so the first Qdrant call is not the whole P95.
    engine.search("t1 t2 t3", top_k=8, candidate_k=40, rerank=False)
    for _ in range(n_queries):
        picks = rng.integers(0, len(vocab), size=4)
        query = " ".join(vocab[int(pick)] for pick in picks)
        started = time.perf_counter()
        result = engine.search(query, top_k=8, candidate_k=40, rerank=False)
        totals.append((time.perf_counter() - started) * 1000.0)
        dense.append(result.timings.dense_ms)
        sparse.append(result.timings.sparse_ms)
    return {
        "docs": n_docs,
        "passages": len(engine.passages),
        "queries": n_queries,
        "embedder": embedder.name,
        "dim": dim,
        "candidate_k": 40,
        "top_k": 8,
        "rerank": False,
        "qdrant": "embedded exact scan. A server collection uses HNSW m=16, ef=64.",
        "p50_ms": percentile(totals, 0.50),
        "p95_ms": percentile(totals, 0.95),
        "p99_ms": percentile(totals, 0.99),
        "dense_p95_ms": percentile(dense, 0.95),
        "sparse_p95_ms": percentile(sparse, 0.95),
        "note": "Synthetic abstracts. This is index latency, not medical relevance.",
    }


GOLD: list[tuple[str, set[str]]] = [
    ("short sleep psoriasis flare", {"demo-psor-1", "demo-psor-2"}),
    ("metformin insulin sensitivity", {"demo-met-1", "demo-met-2"}),
    ("brca1 contralateral breast", {"demo-brca-1", "demo-brca-2"}),
    ("systolic blood pressure stroke", {"demo-bp-1", "demo-bp-2"}),
    ("ssri major depression", {"demo-ssri-1", "demo-ssri-2"}),
]


def recall_at_k(engine: RetrievalEngine, k: int = 5) -> dict[str, object]:
    rows = []
    total = 0.0
    for query, expected in GOLD:
        result = engine.search(query, top_k=k, candidate_k=20, rerank=False)
        found = {hit.passage.pmid for hit in result.hits}
        overlap = len(expected & found)
        score = overlap / len(expected) if expected else 0.0
        total += score
        rows.append({"query": query, "expected": sorted(expected), "found": sorted(found), "recall": score})
    return {
        "k": k,
        "queries": len(GOLD),
        "recall": total / len(GOLD) if GOLD else 0.0,
        "rows": rows,
        "note": "Labeled synthetic abstracts in data/demo.jsonl. Not a PubMed benchmark.",
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def eval_demo(corpus: Path) -> dict[str, object]:
    papers = load_papers(corpus)
    engine = RetrievalEngine.from_papers(papers, HashingEmbedder(dim=256), QdrantStore(":memory:"))
    return recall_at_k(engine)
