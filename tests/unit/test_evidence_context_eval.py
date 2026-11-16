"""Scoring and frozen sampling checks, not evidence of real retrieval quality."""

import runpy
from pathlib import Path

import pytest

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[2]
                           / "scripts/evaluate_evidence_context.py"))


def test_complete_evidence_requires_whole_paragraph_and_allows_alternative_annotation():
    references = [["The complete evidence paragraph.", "Second observation."], ["Alternate support."]]
    score = SCRIPT["score_context"]("The complete evidence paragraph. Extra words", references)
    assert score["evidence_recall"] == 0.5
    assert not score["complete_evidence"]
    assert SCRIPT["score_context"]("Alternate\n support.", references)["complete_evidence"]
    assert SCRIPT["score_context"]("The complete evidence", references)["evidence_recall"] == 0
    with pytest.raises(ValueError):
        SCRIPT["score_context"]("Anything", [[]])


def test_sampling_is_deterministic_paper_disjoint_and_excludes_unavailable_evidence():
    answer = {"unanswerable": False, "evidence": ["Real paragraph."]}
    paper = {"title": "Paper", "abstract": "Abstract", "figures_and_tables": [],
             "full_text": [{"section_name": None, "paragraphs": ["Real paragraph."]}],
             "qas": [{"question_id": "q1", "question": "Question one?", "answers": [{"answer": answer}]},
                     {"question_id": "q2", "question": "Question two?", "answers": [{"answer": answer}]}]}
    first, counts = SCRIPT["select_examples"]({"p1": paper, "p2": paper}, 2, 1)
    second, _ = SCRIPT["select_examples"]({"p2": paper, "p1": paper}, 2, 1)
    assert first == second
    assert len({row["paper_id"] for row in first}) == 2
    assert counts["eligible_papers"] == 2
    prefix, _ = SCRIPT["select_examples"]({"p1": paper, "p2": paper}, 1, 1)
    later, _ = SCRIPT["select_examples"]({"p1": paper, "p2": paper}, 1, 1, offset=1)
    assert first == prefix + later
    assert {row["paper_id"] for row in prefix}.isdisjoint(row["paper_id"] for row in later)
    assert SCRIPT["gold_references"](paper["qas"][0], "Unrelated text") == []
    answer["unanswerable"] = True
    assert SCRIPT["gold_references"](paper["qas"][0], "Real paragraph.") == []
