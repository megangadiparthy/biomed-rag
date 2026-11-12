"""Page PubMed into a jsonl file, and resume a query that was already started."""

from __future__ import annotations

import json
from pathlib import Path

from biomed_rag.domain import Paper, dump_papers, load_papers
from biomed_rag.ingest.pubmed import PubMedClient


def fetch_to_jsonl(query: str, limit: int, path: Path, client: PubMedClient | None = None) -> list[Paper]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    client = client or PubMedClient()
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    have: list[Paper] = []
    retstart = 0
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("query") == query:
            have = load_papers(path)
            retstart = int(meta.get("retstart", len(have)))

    seen = {paper.pmid for paper in have}
    while len(have) < limit:
        page = min(200, limit - len(have))
        ids, total = client.search_ids(query, retmax=page, retstart=retstart)
        if not ids:
            break
        for paper in client.fetch(ids):
            if paper.pmid in seen:
                continue
            seen.add(paper.pmid)
            have.append(paper)
            if len(have) >= limit:
                break
        retstart += len(ids)
        dump_papers(path, have)
        meta_path.write_text(
            json.dumps({"query": query, "retstart": retstart, "count": total, "kept": len(have)}),
            encoding="utf-8",
        )
        if retstart >= total:
            break
    return have
