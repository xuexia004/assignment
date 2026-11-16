"""Real SDK transport with a public-text excerpt and mocked model/selection.

The excerpt below is from the QASPER abstract (https://arxiv.org/abs/2105.03011).
``complete=True`` describes this tiny test input only. The injected text, selected
span and answer exercise live-mode routing; they are not paper-quality evidence.
"""

import asyncio
import json
from pathlib import Path

import pytest
from mcp import Client

from paper_agent.config import Settings
from paper_agent.main_agent import server_environment, unpack_result
from paper_agent.models import Paper, PipelineResult
from paper_agent.server import create_server
from paper_agent.services.pipeline import ResearchPipeline


@pytest.mark.parametrize("idea", ["How many questions are in QASPER?", "QASPER 有多少个问题？"])
@pytest.mark.parametrize("profile", ["evidence", "evidence_fast", "grounded"])
def test_evidence_summary_via_real_mcp_preserves_scope_and_quote(tmp_path, monkeypatch, idea, profile):
    quote = ("We therefore present QASPER, a dataset of 5,049 questions over "
             "1,585 Natural Language Processing papers.")
    text = "# QASPER\n\n" + quote + "\n"
    source_path = tmp_path / "public-excerpt.md"
    source_path.write_text(text, encoding="utf-8")
    start = text.index(quote)
    span = {"id": "P1", "start": start, "end": start + len(quote),
            "section": "QASPER", "text": quote, "partial_parent": False}
    retrieval_queries = []

    def selected_context(question, source, max_bytes=12000, method="hybrid"):
        assert source == text
        assert method == ("cascade" if profile == "evidence_fast" else "hybrid")
        retrieval_queries.append(question)
        context = "[P1] QASPER\n" + quote
        assert len(context.encode("utf-8")) <= max_bytes
        return context, {"actual_method": "mocked_test_selector", "selected_spans": [span],
                         "context_bytes": len(context.encode("utf-8"))}

    monkeypatch.setattr("paper_agent.agents.evidence_context.build_evidence_context", selected_context)

    class Model:
        calls = 0

        async def json(self, system, user):
            self.calls += 1
            payload = json.loads(user)
            assert payload["task"] == "summarize_paper"
            assert payload["idea"] == idea
            assert payload["source"]["scope"] == ("fulltext" if profile == "grounded" else "partial_fulltext")
            assert quote in payload["source"]["text"]
            if profile == "grounded":
                assert list(payload)[-1] == "idea"
                assert payload["source"]["text"].replace("[P1]\n", "") == text
            return {"overview": "QASPER 包含 5,049 个问题。",
                    "deep_analysis": "QASPER 包含 5,049 个问题。[P1]",
                    "findings": ["QASPER 包含 5,049 个问题。"],
                    "citations": [{"id": "P1", "claim": "问题数量为 5,049。", "quote": quote}]}

    settings = Settings(mode="live", research_profile=profile, top_k=1,
                        output_dir=tmp_path / "runs", data_dir=tmp_path / "files")
    env = server_environment(settings)
    assert env["PAPER_AGENT_RESEARCH_PROFILE"] == profile
    assert env["PAPER_AGENT_PAPER_ANALYSIS_STRATEGY"] == profile
    model = Model()
    pipeline = ResearchPipeline(settings, llm=model)
    paper = Paper(paper_id="qasper-excerpt-fixture", title="QASPER abstract excerpt",
                  markdown_path=str(source_path), fulltext_quality={"complete": True},
                  metadata={"test_fixture": "public_abstract_excerpt_not_full_paper"})
    run = PipelineResult(idea=idea, query="QASPER question count", top_k=1, mode="live",
                         execution_mode=settings.execution_mode, status="fulltexts_ready",
                         papers=[paper], selected_papers=[paper], metrics={
                             "experiment": settings.experiment_snapshot(top_k=1),
                             "completed_stages": ["search", "selection", "fulltexts"],
                             "analysis_paper_ids": [paper.paper_id], "llm_usage": {},
                             "model_provenance": "injected", "test_fixture": True,
                         })
    run.output_dir = str(pipeline.store.directory(run.run_id))
    pipeline.store.save(run)

    async def scenario():
        async with Client(create_server(pipeline)) as client:
            response = await client.call_tool("summarize_selected_papers", {"run_id": run.run_id})
            assert not response.is_error
            result = unpack_result(response)
            repeated = unpack_result(await client.call_tool(
                "summarize_selected_papers", {"run_id": run.run_id}))
            assert repeated["summaries"] == result["summaries"]
            return result

    result = asyncio.run(scenario())
    assert model.calls == 1
    expected_query = "QASPER question count" if "QASPER 有" in idea else idea
    assert retrieval_queries == ([] if profile == "grounded" else [expected_query])
    assert result["status"] == "summarized"
    strategy = "grounded_fulltext" if profile == "grounded" else "evidence_context"
    assert result["metrics"]["analysis_strategy"] == strategy
    assert result["metrics"]["usable_partial_summary_count"] == (0 if profile == "grounded" else 1)
    assert result["metrics"]["fulltext_summary_count"] == (1 if profile == "grounded" else 0)
    assert result["metrics"]["physical_llm_requests"] == 0
    audit = result["metrics"]["paper_analysis"][paper.paper_id]
    assert audit["validated_citations"] == 1 and audit["llm_calls"] == 1
    summary = result["summaries"][0]
    assert not summary["errors"]
    assert summary["summary_scope"] == ("fulltext" if profile == "grounded" else "partial_fulltext")
    reference = summary["evidence"][0]
    assert text[reference["start_char"]:reference["end_char"]] == reference["quote"] == quote
    assert "[P1]" in summary["markdown"]
    assert Path(summary["summary_path"]).is_file()
    saved = pipeline.store.load(run.run_id)
    assert saved.summaries[0].evidence[0].quote == quote
    assert saved.metrics["analysis_strategy"] == strategy
