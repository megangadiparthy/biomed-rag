"""Command line: serve, fetch, bench, eval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from biomed_rag.api.app import create_app
from biomed_rag.api.llm import OpenAICompatibleClient
from biomed_rag.domain import load_papers
from biomed_rag.eval.bench import bench_latency, eval_demo, write_json
from biomed_rag.index.embeddings import build_embedder
from biomed_rag.index.qdrant_store import QdrantStore
from biomed_rag.ingest.pipeline import fetch_to_jsonl
from biomed_rag.retrieve.engine import RetrievalEngine
from biomed_rag.retrieve.rerank import CrossEncoderReranker


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_engine(corpus: Path, qdrant: str, embedder_name: str, dim: int, rerank_model: str | None) -> RetrievalEngine:
    papers = load_papers(corpus)
    model_name = os.environ.get("BIOMED_ENCODER", "NeuML/pubmedbert-base-embeddings")
    embedder = build_embedder(embedder_name, dim=dim, model_name=model_name)
    reranker = CrossEncoderReranker(rerank_model) if rerank_model else None
    return RetrievalEngine.from_papers(papers, embedder, QdrantStore(qdrant), reranker)


def main(argv: list[str] | None = None) -> None:
    root = _root()
    parser = argparse.ArgumentParser(prog="biomed-rag", description="Hybrid PubMed retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the API")
    serve.add_argument("--corpus", type=Path, default=root / "data" / "demo.jsonl")
    serve.add_argument("--qdrant", default=":memory:")
    serve.add_argument("--embedder", choices=("hashing", "transformer"), default="hashing")
    serve.add_argument("--dim", type=int, default=256)
    serve.add_argument("--rerank-model", default="")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    fetch = sub.add_parser("fetch", help="download PubMed abstracts to jsonl")
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--limit", type=int, default=200)
    fetch.add_argument("--out", type=Path, default=root / "data" / "pubmed.jsonl")

    bench = sub.add_parser("bench", help="measure hybrid search latency")
    bench.add_argument("--docs", type=int, default=100_000)
    bench.add_argument("--queries", type=int, default=200)
    bench.add_argument("--dim", type=int, default=128)
    bench.add_argument("--out", type=Path, default=root / "results" / "latency.json")

    evaluate = sub.add_parser("eval", help="recall on the labeled demo corpus")
    evaluate.add_argument("--corpus", type=Path, default=root / "data" / "demo.jsonl")
    evaluate.add_argument("--out", type=Path, default=root / "results" / "quality.json")

    args = parser.parse_args(argv)
    if args.command == "serve":
        rerank_model = args.rerank_model or None
        engine = _build_engine(args.corpus, args.qdrant, args.embedder, args.dim, rerank_model)
        llm = None
        base = os.environ.get("BIOMED_LLM_BASE_URL", "")
        if base:
            llm = OpenAICompatibleClient(
                base_url=base,
                api_key=os.environ.get("BIOMED_LLM_API_KEY", ""),
                model=os.environ.get("BIOMED_LLM_MODEL", "gpt-4o-mini"),
            )
        uvicorn.run(create_app(engine, llm), host=args.host, port=args.port)
        return
    if args.command == "fetch":
        papers = fetch_to_jsonl(args.query, args.limit, args.out)
        print(f"wrote {len(papers)} abstracts to {args.out}")
        return
    if args.command == "bench":
        payload = bench_latency(n_docs=args.docs, n_queries=args.queries, dim=args.dim)
        write_json(args.out, payload)
        print(
            f"passages {payload['passages']}  "
            f"p50 {payload['p50_ms']:.2f} ms  "
            f"p95 {payload['p95_ms']:.2f} ms  "
            f"p99 {payload['p99_ms']:.2f} ms"
        )
        print(f"wrote {args.out}")
        return
    if args.command == "eval":
        payload = eval_demo(args.corpus)
        write_json(args.out, payload)
        print(f"recall@{payload['k']} {payload['recall']:.2f} over {payload['queries']} queries")
        print(f"wrote {args.out}")
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
