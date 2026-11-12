"""FastAPI surface: search, and an answer that cannot cite outside the hit list."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from biomed_rag.citations import CitationError, LLMClient, citation_view, compose_answer
from biomed_rag.retrieve.engine import RetrievalEngine, SearchResult


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    candidate_k: int = Field(default=40, ge=1, le=100)
    rerank: bool = False


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class TimingsModel(BaseModel):
    embed_ms: float
    dense_ms: float
    sparse_ms: float
    fusion_ms: float
    rerank_ms: float
    total_ms: float


class HitModel(BaseModel):
    rank: int
    chunk_id: str
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: int | None
    text: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class SearchResponse(BaseModel):
    query: str
    hits: list[HitModel]
    timings_ms: TimingsModel
    reranker: str
    passages: int


class CitationModel(BaseModel):
    n: int
    chunk_id: str
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: int | None
    text: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    mode: str
    citations: list[CitationModel]
    timings_ms: TimingsModel
    reranker: str


def _hits(result: SearchResult) -> list[HitModel]:
    rows: list[HitModel] = []
    for rank, hit in enumerate(result.hits, start=1):
        passage = hit.passage
        rows.append(
            HitModel(
                rank=rank,
                chunk_id=passage.chunk_id,
                pmid=passage.pmid,
                title=passage.title,
                authors=list(passage.authors),
                journal=passage.journal,
                year=passage.year,
                text=passage.text,
                score=hit.score,
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
            )
        )
    return rows


def _timings(result: SearchResult) -> TimingsModel:
    timing = result.timings
    return TimingsModel(
        embed_ms=timing.embed_ms,
        dense_ms=timing.dense_ms,
        sparse_ms=timing.sparse_ms,
        fusion_ms=timing.fusion_ms,
        rerank_ms=timing.rerank_ms,
        total_ms=timing.total_ms,
    )


def create_app(engine: RetrievalEngine, llm: LLMClient | None = None) -> FastAPI:
    app = FastAPI(title="Biomed RAG", version="0.1.0")
    app.state.engine = engine
    app.state.llm = llm

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "passages": len(engine.passages),
            "embedder": engine.embedder.name,
            "reranker": engine.reranker.name if engine.reranker else "none",
        }

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest) -> SearchResponse:
        try:
            result = engine.search(
                body.query,
                top_k=body.top_k,
                candidate_k=body.candidate_k,
                rerank=body.rerank,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SearchResponse(
            query=result.query,
            hits=_hits(result),
            timings_ms=_timings(result),
            reranker=result.reranker,
            passages=len(engine.passages),
        )

    @app.post("/answer", response_model=AnswerResponse)
    def answer(body: AnswerRequest) -> AnswerResponse:
        rerank = engine.reranker is not None
        try:
            result = engine.search(body.query, top_k=body.top_k, candidate_k=max(20, body.top_k), rerank=rerank)
            composed = compose_answer(result, llm=app.state.llm)
        except CitationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        citations = [CitationModel.model_validate(row) for row in citation_view(composed.search.hits)]
        return AnswerResponse(
            query=composed.search.query,
            answer=composed.text,
            mode=composed.mode,
            citations=citations,
            timings_ms=_timings(composed.search),
            reranker=composed.search.reranker,
        )

    return app
