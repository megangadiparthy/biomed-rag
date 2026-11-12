# Biomed RAG

Search over biomedical abstracts. One query hits two indexes: a Qdrant vector index and a BM25 inverted index. The two lists are merged by rank, because a cosine and a BM25 score are not on the same scale. A cross-encoder can rerank the short list. That step is off unless you ask for it. Scoring passages with a transformer is the part that usually breaks a 100 ms budget.

An answer has to cite a passage that was just retrieved. A sentence with no passage number is rejected. So is a PMID that was not in the hit list. With no model key, the API writes a short extractive answer and holds it to the same rule.

This does not diagnose anyone.

## What is in the repo

`data/demo.jsonl` is ten synthetic abstracts, labeled by hand. It is there so the search and the citation check run with no download.

Real abstracts come from PubMed. Set `NCBI_EMAIL` first. NCBI asks for it.

```bash
python -m biomed_rag fetch --query "psoriasis sleep" --limit 200 --out data/pubmed.jsonl
```

The fetch pages in batches of 200 and can be run again. A sidecar file remembers where that query stopped.

## Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m biomed_rag serve
```

`POST /search` with `{"query": "short sleep psoriasis"}`. `POST /answer` returns the extractive text plus the passages it was allowed to cite. `GET /health` reports how many passages are loaded.

To point at a Qdrant server instead of the embedded client:

```bash
docker compose up -d
python -m biomed_rag serve --qdrant http://localhost:6333 --corpus data/pubmed.jsonl
```

The embedded client scans every vector. Qdrant warns once a collection passes about 20,000 points. The server path is the HNSW index (`m=16`, search `ef=64`).

## Encoders

The default embedder hashes tokens into a vector. It is deterministic and it needs no weights. It is what the tests and the timing run use. It does not know that two different words can mean the same thing.

The semantic encoder is mean-pooled PubMedBERT, L2 normalized, in PyTorch:

```bash
pip install -e ".[ml]"
python -m biomed_rag serve --embedder transformer --corpus data/pubmed.jsonl
```

The default model is `NeuML/pubmedbert-base-embeddings`. Override it with `BIOMED_ENCODER`.

Reranking, when you want it:

```bash
python -m biomed_rag serve --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

`/search` reranks only if the request sets `"rerank": true`. `/answer` reranks when that model is loaded.

An OpenAI-compatible model can write the answer. The citation check still runs, and a bad citation returns 422 instead of the draft.

```bash
BIOMED_LLM_BASE_URL=http://localhost:11434/v1 BIOMED_LLM_MODEL=llama3.1 python -m biomed_rag serve
```

## What the run found

From this machine, Apple silicon, Python 3.12. Numbers are in `results/latency.json` and `results/quality.json`.

Hybrid search, no rerank, hashing vectors of dimension 128, top 40 from each index, top 8 returned. 100,000 synthetic passages, 200 queries, after one warmup query:

| | Time |
| --- | --- |
| P50 | 26.7 ms |
| P95 | 27.7 ms |
| P99 | 29.0 ms |
| Dense P95 | 27.4 ms |
| BM25 P95 | 0.33 ms |

The dense leg is almost the whole budget, and on this run it was an exact scan inside embedded Qdrant, not HNSW. BM25 only walks the postings for the query terms, so it stays well under a millisecond here.

On the ten labeled abstracts, recall at 5 is 1.00 across the five questions in `eval/bench.py`. That checks the wiring. It is not a PubMed relevance benchmark. Relevance on real abstracts depends on which encoder you load.

A separate fetch of five live PubMed records for "psoriasis sleep" parsed, including PMID 37956475. Those records are not committed.
