"""Papers, the passages we index, and the jsonl they live in."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    mesh: list[str] = Field(default_factory=list)
    source: str = "pubmed"


class Passage(BaseModel):
    chunk_id: str
    pmid: str
    title: str
    text: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    mesh: list[str] = Field(default_factory=list)

    def indexed_text(self) -> str:
        mesh = " ".join(self.mesh)
        return f"{self.title}. {mesh}. {self.text}"

    def citation_line(self) -> str:
        year = str(self.year) if self.year else "n.d."
        journal = self.journal or "unknown journal"
        return f"PMID {self.pmid}. {self.title} ({journal}, {year}). {self.text}"


def chunk_paper(paper: Paper, size: int = 700) -> list[Passage]:
    """Split an abstract on sentence boundaries. One paper can yield several passages."""
    raw = paper.abstract.strip() or paper.title.strip()
    sentences = [part.strip() for part in _SENTENCE.split(raw) if part.strip()]
    if not sentences:
        sentences = [raw]

    groups: list[str] = []
    buffer: list[str] = []
    length = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if buffer else 0)
        if buffer and length + extra > size:
            groups.append(" ".join(buffer))
            buffer = [sentence]
            length = len(sentence)
        else:
            buffer.append(sentence)
            length += extra
    if buffer:
        groups.append(" ".join(buffer))

    return [
        Passage(
            chunk_id=f"{paper.pmid}#{index}",
            pmid=paper.pmid,
            title=paper.title,
            text=text,
            authors=list(paper.authors),
            journal=paper.journal,
            year=paper.year,
            mesh=list(paper.mesh),
        )
        for index, text in enumerate(groups)
    ]


def load_papers(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(Paper.model_validate_json(line))
    return papers


def dump_papers(path: Path, papers: list[Paper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(paper.model_dump_json() + "\n")
