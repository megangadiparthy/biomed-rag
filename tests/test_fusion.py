from biomed_rag.retrieve.fusion import reciprocal_rank_fusion


def test_document_in_both_lists_outranks_a_single_list_leader() -> None:
    # "both" is second in each list. "only-dense" is first in one list and missing in the other.
    fused = reciprocal_rank_fusion(
        [
            ["only-dense", "both", "other"],
            ["only-sparse", "both"],
        ]
    )
    scores = dict(fused)
    assert scores["both"] > scores["only-dense"]
    assert scores["both"] > scores["only-sparse"]
    assert fused[0][0] == "both"
