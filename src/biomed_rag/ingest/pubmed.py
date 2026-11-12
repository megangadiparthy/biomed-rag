"""PubMed E-utilities client and MEDLINE XML parser.

NCBI asks for a tool name and an email. Set NCBI_TOOL and NCBI_EMAIL.
Without an API key the client stays under three requests a second.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable

import httpx

from biomed_rag.domain import Paper

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def _find(parent: ET.Element, path: str) -> ET.Element | None:
    current: ET.Element | None = parent
    for part in path.split("/"):
        if current is None:
            return None
        current = next((child for child in current if _local(child.tag) == part), None)
    return current


def _findall(parent: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in parent.iter() if _local(node.tag) == name]


def _year(article: ET.Element) -> int | None:
    year_node = _find(article, "Journal/JournalIssue/PubDate/Year")
    raw = _text(year_node)
    if not raw:
        medline = _find(article, "Journal/JournalIssue/PubDate/MedlineDate")
        raw = _text(medline)
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return None


def _authors(article: ET.Element) -> list[str]:
    names: list[str] = []
    for author in _findall(article, "Author"):
        collective = _text(next((child for child in author if _local(child.tag) == "CollectiveName"), None))
        if collective:
            names.append(collective)
            continue
        last = _text(next((child for child in author if _local(child.tag) == "LastName"), None))
        first = _text(next((child for child in author if _local(child.tag) == "ForeName"), None))
        joined = " ".join(part for part in (first, last) if part)
        if joined:
            names.append(joined)
    return names


def _abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for node in _findall(article, "AbstractText"):
        body = " ".join("".join(node.itertext()).split())
        if not body:
            continue
        label = node.attrib.get("Label")
        parts.append(f"{label}: {body}" if label else body)
    return " ".join(parts)


def parse_pubmed_xml(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for article_set in root.iter():
        if _local(article_set.tag) != "PubmedArticle":
            continue
        citation = next((child for child in article_set if _local(child.tag) == "MedlineCitation"), None)
        if citation is None:
            continue
        pmid = _text(next((child for child in citation if _local(child.tag) == "PMID"), None))
        article = next((child for child in citation if _local(child.tag) == "Article"), None)
        if not pmid or article is None:
            continue
        title_node = next((child for child in article if _local(child.tag) == "ArticleTitle"), None)
        title = " ".join("".join(title_node.itertext()).split()) if title_node is not None else ""
        abstract = _abstract(article)
        if not title and not abstract:
            continue
        mesh = []
        for descriptor in _findall(citation, "DescriptorName"):
            name = " ".join("".join(descriptor.itertext()).split())
            if name:
                mesh.append(name)
        journal = _text(_find(article, "Journal/Title"))
        papers.append(
            Paper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=_authors(article),
                journal=journal,
                year=_year(article),
                mesh=mesh,
                source="pubmed",
            )
        )
    return papers


class PubMedClient:
    def __init__(self, http: httpx.Client | None = None, min_interval: float = 0.34) -> None:
        self.http = http or httpx.Client(timeout=60.0)
        self.min_interval = min_interval
        self._last_call = 0.0
        self.tool = os.environ.get("NCBI_TOOL", "biomed_rag")
        self.email = os.environ.get("NCBI_EMAIL", "")
        self.api_key = os.environ.get("NCBI_API_KEY", "")

    def _params(self, extra: dict[str, str]) -> dict[str, str]:
        params = {"tool": self.tool, **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get(self, url: str, params: dict[str, str]) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.http.get(url, params=self._params(params))
                self._last_call = time.monotonic()
                if response.status_code in {429, 500, 502, 503}:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"PubMed request failed: {url}") from last_error

    def search_ids(self, query: str, retmax: int, retstart: int = 0) -> tuple[list[str], int]:
        response = self._get(
            ESEARCH,
            {
                "db": "pubmed",
                "retmode": "json",
                "term": query,
                "retmax": str(retmax),
                "retstart": str(retstart),
                "sort": "relevance",
            },
        )
        result = response.json()["esearchresult"]
        return list(result.get("idlist", [])), int(result.get("count", 0))

    def fetch(self, pmids: Iterable[str]) -> list[Paper]:
        ids = [pmid for pmid in pmids if pmid]
        if not ids:
            return []
        papers: list[Paper] = []
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            response = self._get(
                EFETCH,
                {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            )
            papers.extend(parse_pubmed_xml(response.text))
        return papers
