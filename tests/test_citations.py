import pytest

from biomed_rag.citations import REFUSAL, CitationError, build_prompt, extractive_answer, guard
from biomed_rag.domain import Passage


def _passage(pmid: str, text: str) -> Passage:
    return Passage(chunk_id=f"{pmid}#0", pmid=pmid, title=f"Title {pmid}", text=text)


def test_extractive_answer_only_cites_retrieved_passages() -> None:
    passages = [
        _passage("demo-psor-1", "Short sleep lined up with a psoriasis flare the next day."),
        _passage("demo-met-1", "Metformin improved insulin sensitivity."),
    ]
    text = extractive_answer("short sleep psoriasis", passages)
    guard(text, passages)
    assert "[1]" in text
    assert "[2]" not in text


def test_guard_rejects_an_unknown_passage_number() -> None:
    passages = [_passage("demo-psor-1", "Short sleep lined up with a psoriasis flare.")]
    with pytest.raises(CitationError):
        guard("Short sleep mattered. [2]", passages)


def test_guard_rejects_an_invented_pmid() -> None:
    passages = [_passage("demo-psor-1", "Short sleep lined up with a psoriasis flare.")]
    with pytest.raises(CitationError):
        guard("See PMID 999. [1]", passages)


def test_guard_allows_the_refusal_without_a_citation() -> None:
    guard(REFUSAL, [_passage("demo-psor-1", "Short sleep lined up with a psoriasis flare.")])


def test_prompt_lists_only_the_passages_it_was_given() -> None:
    passages = [_passage("demo-psor-1", "Short sleep lined up with a psoriasis flare.")]
    _system, user = build_prompt("sleep", passages)
    assert "PMID demo-psor-1" in user
    assert "[1]" in user
    assert "[2]" not in user
