from biomed_rag.index.bm25 import BM25Index


def test_rare_term_ranks_first() -> None:
    texts = [
        "sleep duration and plaque itch",
        "xqzv marker appears in this note only",
        "blood pressure and later stroke",
    ]
    index = BM25Index.build(texts, ["a", "b", "c"])
    hits = index.search("xqzv marker", top_k=3)
    assert [doc_id for doc_id, _score in hits] == ["b"]


def test_empty_query_returns_nothing() -> None:
    index = BM25Index.build(["alpha beta"], ["a"])
    assert index.search("the and of", top_k=5) == []
