"""Mechanism tests only; synthetic signals do not establish ranking quality."""

from datetime import date

from paper_agent.models import Paper
from paper_agent.services.value_reranker import value_rerank

DAY = date(2026, 11, 6)


def candidates(count=45):
    return [Paper(paper_id=f"p{i}", title=f"Paper {i}", rank=i + 1, year=2020,
                  ranking_details={"rrf_rank": i + 1, "cosine_similarity": .9 - i / 100}) for i in range(count)]


def test_irrelevant_high_citation_tail_cannot_enter_and_originals_stay_intact():
    papers = candidates()
    papers[44].metadata["citation_normalized_percentile"] = 1.
    papers[44].citation_count = 10000000
    vectors = {p.paper_id: [1., 0.] for p in papers}
    ranked, audit = value_rerank(papers, vectors, as_of=DAY)
    assert audit["applied"] and ranked[44].paper_id == "p44"
    assert ranked[0] is not papers[0]
    assert "value_score" not in papers[0].ranking_details
    assert ranked[0].ranking_details["rrf_rank"] == 1
    assert len({p.paper_id for p in ranked}) == len(papers)


def test_missing_citations_are_unknown_not_zero_and_raw_totals_not_used():
    papers = candidates(4)
    papers[0].citation_count = 1000000
    vectors = {p.paper_id: [1., 0.] for p in papers}
    ranked, audit = value_rerank(papers, vectors, as_of=DAY)
    assert audit["impact_known_count"] == 0
    assert all(p.ranking_details["value_features"]["impact_missing"] for p in ranked)
    assert all(p.ranking_details["value_features"]["impact_percentile"] is None for p in ranked)


def test_near_ties_prefer_complementary_content_and_normalized_impact():
    papers = candidates()
    vectors = {p.paper_id: [1., 0.] for p in papers}
    vectors["p2"] = [0., 1.]
    papers[2].metadata["citation_normalized_percentile"] = .8
    ranked, _ = value_rerank(papers, vectors, as_of=DAY, anchor_count=1)
    assert [p.paper_id for p in ranked[:2]] == ["p0", "p2"]


def test_invalid_or_missing_vectors_preserve_baseline():
    papers = candidates(2)
    for vectors in ({}, {"p0": [0.], "p1": [1.]}, {"p0": [1.], "p1": [float("nan")]}):
        ranked, audit = value_rerank(papers, vectors, as_of=DAY)
        assert ranked is papers and audit["applied"] is False


def test_baseline_top_k_is_protected_from_citation_and_diversity_adjustments():
    papers = candidates()
    for p in papers[5:]:
        p.metadata["citation_normalized_percentile"] = 1.
    vectors = {p.paper_id: [1., i / 100] for i, p in enumerate(papers)}
    ranked, audit = value_rerank(papers, vectors, as_of=DAY, anchor_count=5)
    assert [p.paper_id for p in ranked[:5]] == [p.paper_id for p in papers[:5]]
    assert audit["protected_count"] == 5


def test_explicit_recency_preference_and_unknown_future_dates_are_audited():
    papers = candidates(3)
    papers[0].year = 2030
    papers[1].year = None
    vectors = {p.paper_id: [1.] for p in papers}
    ranked, _ = value_rerank(papers, vectors, as_of=DAY, prefer_recent=True)
    by_id = {p.paper_id: p for p in ranked}
    assert by_id["p0"].ranking_details["value_features"]["age_years"] is None
    assert by_id["p1"].ranking_details["value_features"]["age_years"] is None
    assert by_id["p0"].ranking_details["value_features"]["weights"] == [.75, .1, .15]
