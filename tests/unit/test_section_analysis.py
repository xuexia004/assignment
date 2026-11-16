"""Coverage and scheduling fixtures, not actual paper quality or speed results."""

import asyncio
import json
from pathlib import Path

import pytest

from paper_agent.agents.research_agent import summarize_paper
from paper_agent.agents.section_analysis import split_document
from paper_agent.config import Settings
from paper_agent.demo_fixtures import DemoLLM
from paper_agent.models import Paper
from paper_agent.services.pipeline import ResearchPipeline


def document():
    return "# Introduction\n" + "Premise and conditions.\n\n" * 800 + "# Method\n" + "Method under assumption A.\n\n" * 800 + "# Conclusion\nResult only under condition A.\n"


def test_sections_cover_all_source_characters_in_order_without_splitting_headings():
    text = document()
    parts = split_document(text)
    assert len(parts) == 3
    assert "".join(p["text"] for p in parts) == text
    assert parts[0]["start"] == 0 and parts[-1]["end"] == len(text)
    assert all(a["end"] == b["start"] for a, b in zip(parts, parts[1:]))


def test_plain_text_without_markdown_headings_uses_paragraph_boundaries():
    text = "\n\n".join("A complete sentence with conditions and numeric results. " * 5 for _ in range(11))
    parts = split_document(text)
    assert len(parts) == 3 and "".join(p["text"] for p in parts) == text
    assert all(text[p["start"] - 2:p["start"]] == "\n\n" for p in parts[1:])


class SectionModel(DemoLLM):
    def __init__(self, *, invalid=False):
        super().__init__()
        self.active = self.peak = 0
        self.invalid = invalid
        self.sections = []
        self.reducers = []

    async def json(self, system, user):
        payload = json.loads(user)
        if payload.get("task") not in {"analyze_paper_section", "reduce_paper_sections"}:
            return await super().json(system, user)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(.005)  # Scheduling fixture only, not a speed benchmark.
            if payload["task"] == "analyze_paper_section":
                self.sections.append(payload)
                return {"notes": "结论仅在条件 A 下成立。", "key_facts": ["条件 A"],
                        "context_ids": ["invented" if self.invalid else "part-2"]}
            self.reducers.append(payload)
            return {"overview": "论文研究了一个有前提的方法。", "deep_analysis": "## 方法与条件\n\n结论仅在条件 A 下成立。",
                    "findings": ["方法有明确适用条件。"]}
        finally:
            self.active -= 1


@pytest.mark.parametrize("invalid", [False, True])
def test_full_coverage_context_backread_and_invalid_reference_fail_closed(tmp_path, invalid):
    path = tmp_path / "source.md"
    text = document()
    path.write_text(text, encoding="utf-8")
    paper = Paper(paper_id="real-source-fixture", title="Fixture paper", markdown_path=str(path),
                  fulltext_quality={"complete": True})
    model = SectionModel(invalid=invalid)
    result = asyncio.run(summarize_paper("What conditions apply?", paper,
                         settings=Settings(data_dir=tmp_path, paper_analysis_strategy="sections"), llm=model))
    audit = paper.metadata["research_audit"]
    assert audit["coverage_complete"] is True
    assert "".join(p["source_text"] for p in sorted(model.sections, key=lambda p: p["section_id"])) == text
    assert len(model.sections) == 3
    if invalid:
        assert result.summary_scope == "failed" and not model.reducers
        assert result.errors
    else:
        assert result.summary_scope == "fulltext"
        assert audit["llm_calls"] == 4
        assert model.reducers[0]["requested_original_context"][0]["text"] == split_document(text)[1]["text"]
        assert Path(result.summary_path).is_file()


def test_combined_paper_and_section_concurrency_uses_one_request_limit(tmp_path, monkeypatch):
    from paper_agent.demo_fixtures import prepare_demo_paper
    async def prepare(paper, *, settings):
        result = await prepare_demo_paper(paper, settings=settings)
        Path(result.markdown_path).write_text(document(), encoding="utf-8")
        return result
    model = SectionModel()
    settings = Settings(data_dir=tmp_path, output_dir=tmp_path / "runs", processing="concurrent",
                        llm_concurrency=2, paper_analysis_strategy="sections")
    result = asyncio.run(ResearchPipeline(settings, llm=model, download_fn=prepare)
                         .run_research_pipeline("Check conditions", 3))
    assert model.peak == 2
    assert len(model.sections) == 9 and len(model.reducers) == 3
    assert result.metrics["analysis_strategy"] == "section_map_reduce"
    assert all(a["coverage_complete"] for a in result.metrics["paper_analysis"].values())
    assert result.report and result.report.markdown


def test_budget_rejection_makes_no_partial_source_model_calls(tmp_path):
    text = document()
    path = tmp_path / "source.md"
    path.write_text(text, encoding="utf-8")
    paper = Paper(paper_id="fixture", title="Fixture", markdown_path=str(path), fulltext_quality={"complete": True})
    model = SectionModel()
    result = asyncio.run(summarize_paper("Check conditions", paper,
                         settings=Settings(data_dir=tmp_path, paper_analysis_strategy="sections",
                                           analysis_input_token_budget=1000), llm=model))
    assert result.summary_scope == "failed" and result.errors[0].code == "llm_budget"
    assert not model.sections and not model.reducers
