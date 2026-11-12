from pathlib import Path

from fastapi.testclient import TestClient

from biomed_rag.api.app import create_app
from biomed_rag.citations import CitationError, compose_answer
from biomed_rag.domain import Paper, load_papers
from biomed_rag.eval.bench import eval_demo
from biomed_rag.index.embeddings import HashingEmbedder
from biomed_rag.index.qdrant_store import QdrantStore
from biomed_rag.retrieve.engine import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo.jsonl"


class _RejectingLLM:
    def complete(self, system: str, user: str) -> str:
        return "This claim has no source."


class _InventingLLM:
    def complete(self, system: str, user: str) -> str:
        return "A secret trial found the opposite. [9]"


def _engine() -> RetrievalEngine:
    papers = load_papers(DEMO)
    return RetrievalEngine.from_papers(papers, HashingEmbedder(dim=128), QdrantStore(":memory:"))


def test_rare_token_is_rank_one_even_when_dense_search_is_a_hash() -> None:
    papers = [
        Paper(pmid="plain", title="Ordinary note", abstract="Sleep and blood pressure were recorded.", source="synthetic"),
        Paper(pmid="rare", title="Marker note", abstract="The xqzv marker was the only endpoint in this note.", source="synthetic"),
    ]
    engine = RetrievalEngine.from_papers(papers, HashingEmbedder(dim=64), QdrantStore(":memory:"))
    result = engine.search("xqzv marker", top_k=2, candidate_k=5)
    assert result.hits[0].passage.pmid == "rare"
    assert result.hits[0].sparse_rank == 1


def test_labeled_demo_recall() -> None:
    report = eval_demo(DEMO)
    assert report["recall"] == 1.0


def test_search_and_answer_endpoints() -> None:
    client = TestClient(create_app(_engine()))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["passages"] == 10

    search = client.post("/search", json={"query": "short sleep psoriasis flare", "top_k": 5})
    assert search.status_code == 200
    body = search.json()
    found = {hit["pmid"] for hit in body["hits"]}
    assert found & {"demo-psor-1", "demo-psor-2"}
    assert body["timings_ms"]["total_ms"] >= 0

    answer = client.post("/answer", json={"query": "short sleep psoriasis flare"})
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["mode"] == "extractive"
    assert payload["citations"]
    assert "[" in payload["answer"]
    cited = {row["pmid"] for row in payload["citations"]}
    assert "demo-psor-1" in cited or "demo-psor-2" in cited


def test_answer_rejects_an_uncited_model_draft() -> None:
    engine = _engine()
    result = engine.search("short sleep psoriasis", top_k=3)
    try:
        compose_answer(result, llm=_RejectingLLM())
    except CitationError:
        return
    raise AssertionError("uncited draft should fail the guard")


def test_answer_endpoint_hides_an_invented_citation() -> None:
    client = TestClient(create_app(_engine(), llm=_InventingLLM()))
    response = client.post("/answer", json={"query": "short sleep psoriasis flare"})
    assert response.status_code == 422
    assert "citation" in response.json()["detail"]
