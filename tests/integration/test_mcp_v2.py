"""Real SDK v2 transports with offline business dependencies and model mocks."""

import asyncio
import json
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from paper_agent.config import Settings
from paper_agent.demo_fixtures import DemoLLM, demo_papers, prepare_demo_paper
from paper_agent.main_agent import (
    ALL_RESEARCH_TOOLS,
    STAGE_TOOLS,
    run_connected_research,
    run_research,
    server_environment,
    unpack_result,
)
from paper_agent.models import ErrorInfo, PipelineResult
from paper_agent.server import create_server
from paper_agent.services.pipeline import ResearchPipeline


@pytest.mark.parametrize("crossref_enabled", [False, True])
def test_six_tools_via_real_in_memory_mcp(tmp_path, crossref_enabled):
    async def scenario():
        settings = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "files")
        async def crossref(query, limit, *, settings=None, client=None):
            papers = demo_papers("arxiv", 2)
            for paper in papers:
                paper.sources = ["crossref"]
            return papers
        pipeline = ResearchPipeline(settings, search_crossref_fn=crossref if crossref_enabled else None)
        async with Client(create_server(pipeline)) as client:
            tools = (await client.list_tools()).tools
            assert {tool.name for tool in tools} == ALL_RESEARCH_TOOLS
            assert all(isinstance(tool.input_schema, dict) for tool in tools)
            result = unpack_result(await client.call_tool("search_papers", {"idea": "agent memory token costs", "k": 3}))
            early = await client.call_tool("generate_research_report", {"run_id": result["run_id"]})
            assert early.is_error
            invalid = await client.call_tool("prepare_fulltexts", {"run_id": "../outside"})
            assert invalid.is_error
            for name in STAGE_TOOLS[1:]:
                result = unpack_result(await client.call_tool(name, {"run_id": result["run_id"]}))
            assert result["status"] == "demo_completed"
            assert len(result["summaries"]) == 3
            assert ("crossref" in result["source_counts"]) is crossref_enabled
            direct = unpack_result(await client.call_tool("run_research_pipeline", {"idea": "agent memory", "k": 3}))
            assert direct["status"] == "demo_completed"
    asyncio.run(scenario())


def test_stdio_client_runs_scripted_demo_and_records_calls(tmp_path):
    settings = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "files", research_profile="baseline")
    result = asyncio.run(run_research("如何利用长期记忆减少 Agent 的 Token 消耗", settings, 3))
    assert result["status"] == "demo_completed"
    events = [json.loads(line) for line in (Path(result["output_dir"]) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [event for event in events if event["stage"] == "main_agent_tool_call"]
    assert [call["tool"] for call in calls] == list(STAGE_TOOLS)
    assert all(call["controller"] == "scripted_demo" for call in calls)
    assert [call["tool"] for call in result["tool_calls"]] == list(STAGE_TOOLS)
    assert all(call["duration_seconds"] >= 0 for call in result["tool_calls"])
    assert {"selector", "research_summarize", "research_report"} <= {event["stage"] for event in events}
    assert "total" in {timing["stage"] for timing in result["timings"]}
    assert result["metrics"]["physical_llm_requests"] == 0
    assert result["metrics"]["experiment"]["profile"] == "baseline"
    assert result["metrics"]["experiment"]["config"]["top_k"] == 3
    assert result["metrics"]["experiment"]["config"]["pdf_parser"] == "legacy"
    persisted = json.loads((Path(result["output_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert persisted["metrics"]["main_agent_wall_seconds"] > 0
    assert persisted["metrics"] == result["metrics"]


def test_section_analysis_completes_via_main_agent_and_real_mcp(tmp_path, monkeypatch):
    # Force the small labeled fixture through the long-paper route; not a timing result.
    monkeypatch.setattr("paper_agent.agents.section_analysis.MIN_SECTION_CHARS", 1)
    class SectionFixture(DemoLLM):
        async def json(self, system, user):
            payload = json.loads(user)
            if payload.get("task") == "analyze_paper_section":
                return {"notes": "Synthetic fixture: conditional method evidence.", "key_facts": [], "context_ids": []}
            if payload.get("task") == "reduce_paper_sections":
                return {"overview": "合成夹具的论文概览。", "deep_analysis": "合成夹具的分析正文，不代表真实论文结论。",
                        "findings": ["合成夹具研究了记忆复用。"]}
            return await super().json(system, user)
    async def scenario():
        settings = Settings(data_dir=tmp_path / "files", output_dir=tmp_path / "runs",
                            processing="concurrent", paper_analysis_strategy="sections")
        async with Client(create_server(ResearchPipeline(settings, llm=SectionFixture()))) as client:
            return await run_connected_research(client, "agent memory", settings, 3)
    result = asyncio.run(scenario())
    assert result["status"] == "demo_completed"
    assert result["metrics"]["analysis_strategy"] == "section_map_reduce"
    assert all(a["coverage_complete"] for a in result["metrics"]["paper_analysis"].values())
    assert Path(result["report"]["report_path"]).is_file()
    assert result["metrics"]["physical_llm_requests"] == 0


@pytest.mark.parametrize("report_strategy", ["legacy", "structured"])
def test_layout_pdf_to_whole_paper_notes_and_report_via_main_agent_mcp(tmp_path, report_strategy):
    from paper_agent.services.pdf_converter import convert_pdf

    async def prepare(paper, *, settings):
        result = await prepare_demo_paper(paper, settings=settings)
        result.markdown_path, result.page_map, result.fulltext_quality = None, [], {"complete": False}
        return result

    async def scenario():
        settings = Settings(data_dir=tmp_path / "files", output_dir=tmp_path / "runs", processing="concurrent",
                            report_strategy=report_strategy)
        pipeline = ResearchPipeline(settings, download_fn=prepare, convert_fn=convert_pdf)
        async with Client(create_server(pipeline)) as client:
            return await run_connected_research(client, "agent memory", settings, 3)

    result = asyncio.run(scenario())
    assert result["status"] == "demo_completed"
    analyzed = set(result["metrics"]["analysis_paper_ids"])
    assert len(analyzed) == 3
    assert all(p["metadata"]["conversion"]["parser"] == "pymupdf4llm.layout"
               for p in result["papers"] if p["paper_id"] in analyzed)
    assert result["metrics"]["analysis_strategy"] == "direct_paper_reports"
    assert result["metrics"]["report_generation"]["strategy"] == report_strategy
    assert result["metrics"]["experiment"]["config"]["report_strategy"] == report_strategy
    assert all(a["llm_calls"] == 1 for a in result["metrics"]["paper_analysis"].values())
    assert all(a["source_parser"] == "pymupdf4llm.layout" for a in result["metrics"]["paper_analysis"].values())
    assert Path(result["report"]["report_path"]).is_file()
    assert result["metrics"]["physical_llm_requests"] == 0


def test_comparison_profiles_keep_markdown_and_reports_independent(tmp_path):
    from paper_agent.services.pdf_converter import convert_pdf

    async def scenario():
        saved, paths = [], []
        for profile in ("baseline", "layout"):
            settings = Settings(data_dir=tmp_path / "files", output_dir=tmp_path / "runs", research_profile=profile)
            pipeline = ResearchPipeline(settings, download_fn=prepare_demo_paper, convert_fn=convert_pdf)
            async with Client(create_server(pipeline)) as client:
                result = await run_connected_research(client, "agent memory", settings, 3)
            ids = set(result["metrics"]["analysis_paper_ids"])
            current = [Path(p["markdown_path"]) for p in result["papers"] if p["paper_id"] in ids]
            assert len(current) == 3 and result["metrics"]["experiment"]["profile"] == profile
            for path, original in saved:
                assert path.read_bytes() == original
            saved.extend((path, path.read_bytes()) for path in current)
            paths.extend(current)
            assert Path(result["report"]["report_path"]).is_file()
            if profile == "baseline":
                baseline_id = result["run_id"]
            else:
                with pytest.raises(ValueError, match="对照配置"):
                    pipeline._load(baseline_id)
        assert len(set(paths)) == 6

    asyncio.run(scenario())


def test_main_agent_mcp_recovers_acquisition_and_conversion_failures_to_report(tmp_path):
    async def download(paper, *, settings):
        if paper.paper_id == "demo:001":
            result = paper.model_copy(deep=True)
            result.errors.append(ErrorInfo(code="fixture_403", message="Fixture access denied", stage="download"))
            result.status = "download_failed"
            return result
        return await prepare_demo_paper(paper, settings=settings)

    async def convert(paper, *, settings):
        if paper.paper_id != "demo:004":
            return paper
        result = paper.model_copy(deep=True)
        result.markdown_path = None
        result.page_map = []
        result.fulltext_quality = {"complete": False}
        result.status = "conversion_failed"
        return result

    async def scenario():
        config = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "files", processing="concurrent")
        pipeline = ResearchPipeline(config, download_fn=download, convert_fn=convert)
        async with Client(create_server(pipeline)) as client:
            return await run_connected_research(client, "agent memory", config, 3)

    result = asyncio.run(scenario())
    assert [c["tool"] for c in result["tool_calls"]] == list(STAGE_TOOLS)
    assert result["metrics"]["analysis_paper_ids"] == ["demo:002", "demo:003", "demo:005"]
    assert result["metrics"]["readable_count"] == 3
    assert result["metrics"]["physical_llm_requests"] == 0
    assert result["report"]["markdown"] and Path(result["report"]["report_path"]).is_file()
    assert any(e["code"] == "fixture_403" for e in result["errors"])


def test_stdio_v2_discovery_is_real(tmp_path):
    import sys
    async def scenario():
        config = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "files")
        params = StdioServerParameters(command=sys.executable, args=["-m", "paper_agent.server"],
                                       env=server_environment(config))
        async with Client(params, read_timeout_seconds=60) as client:
            assert {tool.name for tool in (await client.list_tools()).tools} == ALL_RESEARCH_TOOLS
    asyncio.run(scenario())


class TerminalMockPipeline:
    """Protocol-only fake: it deliberately reports failed rather than live success."""
    async def run_research_pipeline(self, idea, k=5, query=""):
        return PipelineResult(idea=idea, top_k=k, query=query, mode="live", status="failed")


class ToolChoiceMock:
    async def chat(self, messages, tools=None):
        request = json.loads(messages[1]["content"])
        return {"role": "assistant", "tool_calls": [{"id": "mock-call", "type": "function",
                "function": {"name": "run_research_pipeline", "arguments": json.dumps(request)}}]}


def test_main_agent_accepts_model_selected_full_pipeline_tool(tmp_path):
    async def scenario():
        config = Settings(mode="live", output_dir=tmp_path / "runs", data_dir=tmp_path / "files")
        async with Client(create_server(TerminalMockPipeline())) as client:
            result = await run_connected_research(client, "agent memory", config, 3, llm=ToolChoiceMock())
            assert result["status"] == "failed"
            events = (config.output_dir / result["run_id"] / "trace.jsonl").read_text(encoding="utf-8")
            assert '"controller": "model_tool_call"' in events
    asyncio.run(scenario())


def test_main_agent_recovers_from_tool_error_feedback(tmp_path):
    class RecoveringMock:
        def __init__(self):
            self.calls = 0
            self.tool_feedback = []

        async def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return {"role": "assistant", "tool_calls": [{
                    "id": "bad-call", "type": "function",
                    "function": {"name": "generate_research_report",
                                 "arguments": json.dumps({"run_id": "missing"})},
                }]}
            if self.calls == 2:
                for message in messages:
                    if message.get("role") == "tool":
                        self.tool_feedback.append(message.get("content", ""))
                request = json.loads(messages[1]["content"])
                return {"role": "assistant", "tool_calls": [{
                    "id": "full-call", "type": "function",
                    "function": {"name": "run_research_pipeline", "arguments": json.dumps(request)},
                }]}
            return {"role": "assistant", "content": "terminal"}

    async def scenario():
        config = Settings(mode="live", output_dir=tmp_path / "runs", data_dir=tmp_path / "files")
        model = RecoveringMock()
        async with Client(create_server(TerminalMockPipeline())) as client:
            result = await run_connected_research(client, "agent memory", config, 3, llm=model)
        return result, model

    result, model = asyncio.run(scenario())
    assert result["status"] == "failed"
    assert any('"code": "tool_call_rejected"' in item or '"code":"tool_call_rejected"' in item
               for item in model.tool_feedback)
    assert model.calls == 2


def test_main_agent_rejects_premature_text_completion(tmp_path):
    class NoTools:
        async def chat(self, messages, tools=None):
            return {"role": "assistant", "content": "Finished."}
    async def scenario():
        config = Settings(mode="live", output_dir=tmp_path / "runs", data_dir=tmp_path / "files")
        async with Client(create_server(TerminalMockPipeline())) as client:
            with pytest.raises(RuntimeError, match="终态"):
                await run_connected_research(client, "agent memory", config, 3, llm=NoTools())
            with pytest.raises(ValueError, match="合成"):
                await run_connected_research(client, "agent memory", config, 3, llm=DemoLLM())
    asyncio.run(scenario())


def test_environment_forwards_budgets_and_cache_policy(tmp_path):
    config = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "files", cache_mode="off",
                      llm_max_calls=7, processing="concurrent", download_concurrency=2,
                      merged_candidate_limit=30, selector_input_token_budget=48000,
                      report_input_token_budget=64000, max_fulltext_attempts=30, fulltext_timeout=45,
                      report_strategy="structured")
    env = server_environment(config)
    assert env["LLM_MAX_CALLS"] == "7"
    assert env["PAPER_AGENT_CACHE_MODE"] == "off"
    assert env["PAPER_AGENT_PROCESSING"] == "concurrent"
    assert env["PAPER_AGENT_DOWNLOAD_CONCURRENCY"] == "2"
    assert env["PAPER_AGENT_MERGED_CANDIDATE_LIMIT"] == "30"
    assert env["PAPER_AGENT_SELECTOR_INPUT_TOKEN_BUDGET"] == "48000"
    assert env["PAPER_AGENT_ANALYSIS_INPUT_TOKEN_BUDGET"] == "256000"
    assert env["PAPER_AGENT_PAPER_ANALYSIS_STRATEGY"] == "direct"
    assert env["PAPER_AGENT_REPORT_STRATEGY"] == "structured"
    assert env["PAPER_AGENT_VALUE_RANKING_ENABLED"] == "False"
    assert env["PAPER_AGENT_CROSS_ENCODER_ENABLED"] == "False"
    assert env["PAPER_AGENT_REPORT_INPUT_TOKEN_BUDGET"] == "64000"
    assert env["PAPER_AGENT_MAX_FULLTEXT_ATTEMPTS"] == "30"
    assert env["PAPER_AGENT_CROSSREF_ENABLED"] == "True"
    assert env["PAPER_AGENT_FULLTEXT_TIMEOUT"] == "45"
    assert env["PAPER_AGENT_PDF_PARSER"] == "layout"
    assert env["PAPER_AGENT_DOWNLOAD_CANDIDATE_TIMEOUT"] == "45.0"
