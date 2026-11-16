"""Ordering and honest fallback fixtures; actual quality is evaluated separately."""

import asyncio

from paper_agent.models import Paper
from paper_agent.services import cross_encoder_reranker as module


def test_joint_scores_rerank_only_bounded_pool_and_preserve_baseline_fields(monkeypatch):
    papers = [Paper(paper_id=str(i), title=f"Paper {i}", rank=i + 1, initial_score=.8,
                    ranking_details={"cosine_similarity": .9}) for i in range(45)]
    monkeypatch.setattr(module, "_predict", lambda q, pool: (list(range(len(pool))), ["2"]))
    ranked, audit = asyncio.run(module.cross_encoder_rerank("query", papers))
    assert [p.paper_id for p in ranked[:2]] == ["19", "18"]
    assert ranked[20:] == papers[20:]
    assert {p.paper_id for p in ranked[:20]} == {p.paper_id for p in papers[:20]}
    assert audit["applied"] and audit["truncated_pair_ids"] == ["2"]
    assert ranked[0].initial_score == .8 and ranked[0].ranking_details["cosine_similarity"] == .9
    assert "cross_encoder_score" not in papers[0].ranking_details


def test_missing_model_and_invalid_scores_preserve_baseline(monkeypatch):
    papers = [Paper(paper_id="p", title="Paper")]
    def unavailable(*args):
        raise OSError("Offline cache incomplete")
    monkeypatch.setattr(module, "_predict", unavailable)
    ranked, audit = asyncio.run(module.cross_encoder_rerank("query", papers))
    assert ranked is papers and not audit["applied"]
    monkeypatch.setattr(module, "_predict", lambda *args: ([float("nan")], []))
    ranked, audit = asyncio.run(module.cross_encoder_rerank("query", papers))
    assert ranked is papers and not audit["applied"]
