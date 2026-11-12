"""Tokenizer shared by BM25 and the hashing embedder.

Hyphenated tokens stay intact so a term like il-6 is one word, not two.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
        "from", "at", "as", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "we", "our", "you", "your",
        "they", "their", "not", "but", "than", "then", "into", "over", "after",
        "before", "between", "about", "using", "used", "study", "studies", "results",
        "method", "methods", "background", "conclusion", "objective", "patients",
        "patient", "who", "which", "what", "when", "where", "how", "can", "may",
        "also", "more", "most", "such", "both", "have", "has", "had", "were",
        "did", "does", "do", "if", "no", "yes", "via", "per", "vs", "versus",
    }
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN.findall(text.lower()) if len(tok) > 1 and tok not in STOPWORDS]
