"""Evidence reports use exact original quotes and honest partial scope."""
import asyncio
import json

import pytest

from paper_agent.agents.research_agent import summarize_paper
from paper_agent.config import Settings
from paper_agent.models import Paper


def test_evidence_report_preserves_offsets_and_rejects_fabrication(tmp_path, monkeypatch):
    text = "# Results\n\nThe treatment improved accuracy by 12 percent.\n"
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    start = text.index("The treatment")
    span = {"id": "P1", "start": start, "end": len(text) - 1,
            "text": text[start:-1], "section": "Results"}
    monkeypatch.setattr("paper_agent.agents.evidence_context.build_evidence_context",
                        lambda *args, **kwargs: ("[P1] Results\n" + span["text"],
                                                {"selected_spans": [span], "context_bytes": 60}))
    class Model:
        def __init__(self, quote, marker="P1"):
            self.quote = quote
            self.marker = marker
            self.calls = 0

        async def json(self, system, user):
            self.calls += 1
            request = json.loads(user)
            assert request["source"]["scope"] == "partial_fulltext"
            return {"overview": "提高了准确率", "deep_analysis": "准确率提高12%。[" + self.marker + "]",
                    "findings": ["准确率提高12%"],
                    "citations": [{"id": "P1", "quote": self.quote, "claim": "准确率提高12%"}]}

    settings = Settings(research_profile="evidence", data_dir=tmp_path)
    paper = Paper(paper_id="p", title="Study", markdown_path=str(path),
                  fulltext_quality={"complete": True})
    model = Model("improved accuracy by 12 percent.")
    summary = asyncio.run(summarize_paper("What changed?", paper, settings=settings, llm=model))
    assert not summary.errors and model.calls == 1
    assert summary.summary_scope == "partial_fulltext"
    ref = summary.evidence[0]
    assert text[ref.start_char:ref.end_char] == ref.quote
    assert ref.quote in summary.markdown
    assert paper.metadata["research_audit"]["validated_citations"] == 1
    bad = asyncio.run(summarize_paper("What changed?", paper, settings=settings, llm=Model("99 percent")))
    assert bad.summary_scope == "failed" and bad.errors[0].code == "invalid_model_output"
    invalid_marker = asyncio.run(summarize_paper(
        "What changed?", paper, settings=settings,
        llm=Model("improved accuracy by 12 percent.", marker="P999")))
    assert invalid_marker.errors[0].code == "invalid_model_output"
    # A selected paragraph may be cited in the body without a separate model quote.
    # The system supplies its exact original text and keeps the locator usable.
    second = {**span, "id": "P2"}
    monkeypatch.setattr("paper_agent.agents.evidence_context.build_evidence_context",
                        lambda *args, **kwargs: ("[P1] " + span["text"] + "\n[P2] " + second["text"],
                                                {"selected_spans": [span, second], "context_bytes": 120}))
    restored = asyncio.run(summarize_paper(
        "What changed?", paper, settings=settings,
        llm=Model("improved accuracy by 12 percent.", marker="P2")))
    assert not restored.errors and len(restored.evidence) == 2
    assert restored.evidence[1].quote == second["text"]
    assert "[P2] 正文引用 [P2]" in restored.markdown


@pytest.mark.parametrize("budget", [10, 500])
def test_budget_does_not_call_model_when_prompt_cannot_fit(tmp_path, monkeypatch, budget):
    path = tmp_path / "paper.md"
    path.write_text("A " * 2000, encoding="utf-8")
    monkeypatch.setattr("paper_agent.agents.evidence_context.build_evidence_context",
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("budget")))
    class Never:
        async def json(self, *args):
            raise AssertionError("must not call model")
    paper = Paper(paper_id="p", title="Study", markdown_path=str(path),
                  fulltext_quality={"complete": True})
    summary = asyncio.run(summarize_paper("What changed?", paper, settings=Settings(
        research_profile="evidence", data_dir=tmp_path, analysis_input_token_budget=budget), llm=Never()))
    assert summary.errors[0].code == "llm_budget"


@pytest.mark.parametrize("complete,expected_scope", [(True, "fulltext"), (False, "partial_fulltext")])
def test_grounded_report_keeps_complete_source_and_scope(tmp_path, complete, expected_scope):
    text = "# 实验\r\n\r\n  Cases include case, number, and gender.\n\n## Limits\nOnly evaluated on English.\n"
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8", newline="")

    class Model:
        calls = 0

        async def json(self, system, user):
            self.calls += 1
            request = json.loads(user)
            assert list(request)[-1] == "idea"
            assert request["source"]["scope"] == expected_scope
            assert "# 实验\n" in request["source"]["text"]
            assert "## Limits\n" in request["source"]["text"]
            assert "Only evaluated on English." in request["source"]["text"]
            assert system.index("citations：") < system.index("overview：")
            return {"citations": [{"id": "P1", "claim": "具体特征为格、数和性",
                                   "quote": "case, number, and gender"}],
                    "overview": "特征为格、数和性。", "deep_analysis": "特征包括格、数和性。[P1]",
                    "findings": ["特征为格、数和性"]}

    paper = Paper(paper_id="grounded", title="Features", markdown_path=str(path),
                  fulltext_quality={"complete": complete})
    model = Model()
    summary = asyncio.run(summarize_paper("哪些形态特征？", paper, settings=Settings(
        paper_analysis_strategy="grounded", data_dir=tmp_path), llm=model))
    assert model.calls == 1 and not summary.errors
    assert summary.summary_scope == expected_scope
    loaded_source = path.read_text(encoding="utf-8")
    ref = summary.evidence[0]
    assert loaded_source[ref.start_char:ref.end_char] == ref.quote
    audit = paper.metadata["research_audit"]
    assert audit["strategy"] == "grounded_fulltext"
    assert audit["context"]["selection_is_fulltext"] is True
    assert audit["source_scope"] == expected_scope
    assert audit["validated_citations"] == 1


def test_grounded_rejects_fabricated_quote_without_an_extra_call(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("# Results\n\nThe score was 12.\n", encoding="utf-8")

    class Model:
        calls = 0

        async def json(self, *args):
            self.calls += 1
            return {"citations": [{"id": "P1", "claim": "score was 99", "quote": "score was 99"}],
                    "overview": "99", "deep_analysis": "99 [P1]", "findings": ["99"]}

    model = Model()
    paper = Paper(paper_id="grounded", title="Results", markdown_path=str(path),
                  fulltext_quality={"complete": True})
    summary = asyncio.run(summarize_paper("score?", paper, settings=Settings(
        paper_analysis_strategy="grounded", data_dir=tmp_path), llm=model))
    assert model.calls == 1
    assert summary.summary_scope == "failed" and summary.errors[0].code == "invalid_model_output"


def test_grounded_budget_rejects_without_compression_or_model_call(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("A complete source. " * 2000, encoding="utf-8")

    class Never:
        async def json(self, *args):
            raise AssertionError("must not call model")

    paper = Paper(paper_id="grounded", title="Long source", markdown_path=str(path),
                  fulltext_quality={"complete": True})
    summary = asyncio.run(summarize_paper("What changed?", paper, settings=Settings(
        paper_analysis_strategy="grounded", data_dir=tmp_path, analysis_input_token_budget=2000), llm=Never()))
    assert summary.errors[0].code == "llm_budget"
    audit = paper.metadata["research_audit"]
    assert audit["llm_calls"] == 0 and audit["input_units"] > 2000
    assert audit["context"]["input_bytes"] == len(path.read_bytes())
