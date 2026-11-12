from pathlib import Path

from biomed_rag.ingest.pubmed import parse_pubmed_xml

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_sample.xml"


def test_parse_pubmed_xml_keeps_citation_fields() -> None:
    papers = parse_pubmed_xml(FIXTURE.read_text(encoding="utf-8"))
    assert len(papers) == 1
    paper = papers[0]
    assert paper.pmid == "12345"
    assert paper.title == "Short sleep and next-day flares."
    assert paper.journal == "Sleep Medicine"
    assert paper.year == 2021
    assert paper.authors == ["A Ng", "Example Study Group"]
    assert "BACKGROUND: Sleep loss is common." in paper.abstract
    assert "RESULTS: Short sleep preceded flares." in paper.abstract
    assert paper.mesh == ["Psoriasis"]
    assert paper.source == "pubmed"
