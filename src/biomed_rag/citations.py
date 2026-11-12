"""Citation tracking for whatever generates the answer.

Every sentence that states a finding has to point at a passage number from
this retrieval. A PMID in the answer has to be one of those passages. The
extractive writer exists so the API is usable with no model key, and it is
held to the same rule as an external LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from biomed_rag.domain import Passage
from biomed_rag.retrieve.engine import Hit, SearchResult
from biomed_rag.tokenize import tokenize

_CITE = re.compile(r"\[(\d+)\]")
_PMID = re.compile(r"\bPMID[:\s]+([A-Za-z0-9_-]+)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

REFUSAL = "I cannot tell from the retrieved abstracts."


class CitationError(ValueError):
    """The draft cited something that was not retrieved."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass
class Answer:
    text: str
    search: SearchResult
    mode: str


def build_prompt(query: str, passages: list[Passage]) -> tuple[str, str]:
    lines = [f"[{index}] {passage.citation_line()}" for index, passage in enumerate(passages, start=1)]
    system = (
        "You answer from the numbered passages only. "
        "End every sentence that states a finding with the passage number in brackets, such as [1]. "
        "If the passages do not support an answer, reply exactly: "
        f"{REFUSAL} "
        "Do not cite a passage number that is not in the list. Do not invent a PMID."
    )
    user = f"Question: {query}\n\nPassages:\n" + "\n".join(lines)
    return system, user


def guard(answer: str, passages: list[Passage]) -> None:
    allowed_numbers = set(range(1, len(passages) + 1))
    cited = {int(value) for value in _CITE.findall(answer)}
    unknown = cited - allowed_numbers
    if unknown:
        raise CitationError(f"citation not in the retrieved set: {sorted(unknown)}")

    allowed_pmids = {passage.pmid for passage in passages}
    mentioned = set(_PMID.findall(answer))
    invented = sorted(pmid for pmid in mentioned if pmid not in allowed_pmids)
    if invented:
        raise CitationError(f"PMID not in the retrieved set: {invented}")

    if answer.strip() == REFUSAL:
        return
    if not cited:
        raise CitationError("a finding has no passage number")


def extractive_answer(query: str, passages: list[Passage]) -> str:
    if not passages:
        return REFUSAL
    query_terms = set(tokenize(query))
    scored: list[tuple[int, int, str]] = []
    for number, passage in enumerate(passages, start=1):
        for sentence in _SENTENCE.split(passage.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap, number, sentence))
    if not scored:
        return REFUSAL
    scored.sort(key=lambda item: item[0], reverse=True)
    picked: list[str] = []
    used: set[int] = set()
    for _overlap, number, sentence in scored:
        if number in used:
            continue
        picked.append(f"{sentence} [{number}]")
        used.add(number)
        if len(picked) == 3:
            break
    if not picked:
        return REFUSAL
    return " ".join(picked)


def compose_answer(
    result: SearchResult,
    llm: LLMClient | None = None,
) -> Answer:
    passages = [hit.passage for hit in result.hits]
    if llm is None:
        text = extractive_answer(result.query, passages)
        mode = "extractive"
    else:
        system, user = build_prompt(result.query, passages)
        text = llm.complete(system, user).strip()
        mode = "llm"
    guard(text, passages)
    return Answer(text=text, search=result, mode=mode)


def citation_view(hits: list[Hit]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, hit in enumerate(hits, start=1):
        passage = hit.passage
        rows.append(
            {
                "n": number,
                "chunk_id": passage.chunk_id,
                "pmid": passage.pmid,
                "title": passage.title,
                "authors": list(passage.authors),
                "journal": passage.journal,
                "year": passage.year,
                "text": passage.text,
            }
        )
    return rows
