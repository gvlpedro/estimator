"""Unit tests for reciprocal_rank_fusion: pure rank-position fusion logic.

No I/O, no SQL, no mocks needed — every case below is a plain list of ids in,
a scored/sorted list of (id, score) pairs out.
"""

from app.generation.rag.retrieval.rrf import reciprocal_rank_fusion


def test_fuses_two_rankings_with_no_overlap_by_summing_independent_scores():
    vector_ranking = [1, 2]
    lexical_ranking = [3, 4]

    fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking], k=60)

    # id 1: rank 1 in vector only -> 1/(60+1); id 3: rank 1 in lexical only -> 1/(60+1)
    # Equal scores, stable sort keeps the first ranking's order for the tie.
    assert fused == [
        (1, 1 / 61),
        (3, 1 / 61),
        (2, 1 / 62),
        (4, 1 / 62),
    ]


def test_empty_rankings_list_returns_empty_result():
    fused = reciprocal_rank_fusion([], k=60)

    assert fused == []


def test_partial_overlap_produces_fused_order_that_differs_from_either_branch_alone():
    # id 5 is mid-pack in both branches; id 1 is top of vector but absent from
    # lexical; id 9 is top of lexical but absent from vector. Neither branch
    # alone ranks id 5 first, but fusion (summing both branches' contributions)
    # should surface it above the single-branch-only ids.
    vector_ranking = [1, 5, 2]
    lexical_ranking = [9, 5, 3]

    fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking], k=60)

    fused_ids = [chunk_id for chunk_id, _ in fused]
    assert fused_ids[0] == 5
    assert set(fused_ids) == {1, 2, 3, 5, 9}
    id_5_score = 1 / (60 + 2) + 1 / (60 + 2)
    assert fused[0] == (5, id_5_score)


def test_id_present_in_only_one_ranking_scores_only_that_rankings_contribution():
    vector_ranking = [1, 2, 3]
    lexical_ranking = [4, 5]

    fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking], k=60)

    fused_by_id = dict(fused)
    # id 3 only appears in the vector ranking, at rank 3 -> no lexical contribution added.
    assert fused_by_id[3] == 1 / (60 + 3)
