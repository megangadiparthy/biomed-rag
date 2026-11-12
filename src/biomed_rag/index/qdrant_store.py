"""Qdrant collection of passage vectors.

Local mode uses the embedded Qdrant client (`:memory:` or a path on disk).
A URL uses a Qdrant server. Payload carries the citation fields so a hit can
be shown without a second lookup, and the engine still keeps the passages in
memory for the BM25 side.
"""

from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from biomed_rag.domain import Passage


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "biomed-rag:" + chunk_id))


class QdrantStore:
    def __init__(self, location: str, collection: str = "abstracts") -> None:
        self.location = location
        self.collection = collection
        if location == ":memory:":
            self.client = QdrantClient(":memory:")
        elif location.startswith("http://") or location.startswith("https://"):
            self.client = QdrantClient(url=location)
        else:
            self.client = QdrantClient(path=location)
        self.dim: int | None = None

    def reset(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
        )
        self.dim = dim

    def upsert(self, passages: list[Passage], vectors: np.ndarray, batch_size: int = 256) -> None:
        if len(passages) != len(vectors):
            raise ValueError("passages and vectors must be the same length")
        if not passages:
            return
        for start in range(0, len(passages), batch_size):
            stop = start + batch_size
            points = [
                qmodels.PointStruct(
                    id=_point_id(passage.chunk_id),
                    vector=vectors[index].tolist(),
                    payload={
                        "chunk_id": passage.chunk_id,
                        "pmid": passage.pmid,
                        "title": passage.title,
                        "text": passage.text,
                        "authors": list(passage.authors),
                        "journal": passage.journal,
                        "year": passage.year,
                        "mesh": list(passage.mesh),
                    },
                )
                for index, passage in enumerate(passages[start:stop], start=start)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        kwargs: dict[str, object] = {
            "collection_name": self.collection,
            "query": vector.tolist(),
            "limit": top_k,
            "with_payload": True,
        }
        if self.location.startswith("http://") or self.location.startswith("https://"):
            kwargs["search_params"] = qmodels.SearchParams(hnsw_ef=64)
        response = self.client.query_points(**kwargs)
        hits: list[tuple[str, float]] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                hits.append((str(chunk_id), float(point.score)))
        return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)
