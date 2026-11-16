"""Deterministic orchestration checks; fixtures are not live model validation."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from paper_agent.config import Settings
from paper_agent.demo_fixtures import DemoLLM, demo_papers, prepare_demo_paper
from paper_agent.models import ErrorInfo
from paper_agent.services.pipeline import ResearchPipeline


def settings_for(tmp_path, **kwargs):
    return Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "files", **kwargs)


def test_demo_runs_both_task_agents_and_persists_scopes(tmp_path):
    settings = settings_for(tmp_path)
    result = asyncio.run(ResearchPipeline(settings).run_research_pipeline("agent memory token cost", 3))
    assert result.status == "demo_completed"
    assert result.source_counts == {"arxiv": 12, "openalex": 12}
    assert len(result.papers) == 18
    assert all(summary.summary_scope == "demo_fixture" for summary in result.summaries)
    assert result.metrics["physical_llm_requests"] == 0
    assert result.metrics["online_complete"] is False
    assert result.metrics["processing_complete"] is True
    assert result.delivery_status == "preliminary_notes"
    assert result.brief is not None
    assert not result.brief.findings
    assert result.brief.report_artifact_id == result.report.report_artifact_id
    trace = [json.loads(line) for line in (Path(result.output_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"selector", "research_summarize", "research_report"} <= {event["stage"] for event in trace}
    assert {timing.stage for timing in result.timings} == {"search", "selection", "fulltexts", "summarize", "report", "total"}


def test_optional_joint_reranking_keeps_selector_set_and_reserve_tail(tmp_path, monkeypatch):
    from paper_agent.agents.selector_agent import build_selection_payload
    from paper_agent.models import Paper
    from paper_agent.services import cross_encoder_reranker

    # Explicitly synthetic retrieval/ranking signals; no network, model or quality claim.
    async def source(query, limit, *, settings, client=None):
        return [Paper(paper_id=f"fixture:{i}", title=f"Agent memory fixture {i}",
                      abstract=f"Complete synthetic abstract {i} about agent memory.", sources=["arxiv"])
                for i in range(30)]

    async def empty(query, limit, *, settings, client=None):
        return []

    monkeypatch.setattr(cross_encoder_reranker, "_predict", lambda q, pool: (list(range(len(pool))), []))
    async def scenario():
        settings = settings_for(tmp_path, mode="live", ranker="keyword", crossref_enabled=False, source_limit=30)
        baseline = await ResearchPipeline(settings, search_arxiv_fn=source, search_openalex_fn=empty).search_papers("agent memory", 3, "agent memory")
        enabled = replace(settings, cross_encoder_enabled=True)
        ranked = await ResearchPipeline(enabled, search_arxiv_fn=source, search_openalex_fn=empty).search_papers("agent memory", 3, "agent memory")
        return baseline, ranked, enabled

    baseline, ranked, settings = asyncio.run(scenario())
    assert len(baseline.papers) == len(ranked.papers) == 30
    assert [p.paper_id for p in ranked.papers[:20]] == [p.paper_id for p in reversed(baseline.papers[:20])]
    assert [p.paper_id for p in ranked.papers[20:]] == [p.paper_id for p in baseline.papers[20:]]
    assert ranked.metrics["ranking"]["cross_encoder"]["applied"]
    assert ranked.metrics["comparison_inputs"] == baseline.metrics["comparison_inputs"]
    payload, _ = build_selection_payload("agent memory", ranked.papers[:20], 3, settings)
    assert payload["candidates"][0]["abstract"] == ranked.papers[0].abstract
    assert "cross_encoder_score" in payload["candidates"][0]["ranking"]


@pytest.mark.parametrize(("processing", "expected_peak"), [("serial", 1), ("concurrent", 2)])
def test_paper_reports_use_selected_execution_mode_and_all_finish_before_synthesis(
    tmp_path, processing, expected_peak,
):
    class ReportModel(DemoLLM):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.peak = 0
            self.completed = set()
            self.report_inputs = None

        async def json(self, system, user):
            payload = json.loads(user)
            if payload.get("task") == "summarize_paper":
                self.active += 1
                self.peak = max(self.peak, self.active)
                # Scheduling fixture only, not an actual performance measurement.
                await asyncio.sleep(.01)
                response = await super().json(system, user)
                self.completed.add(payload["paper"]["paper_id"])
                self.active -= 1
                return {key: response[key] for key in ("overview", "deep_analysis", "findings")}
            if payload.get("task") == "generate_research_report":
                assert self.active == 0
                assert self.completed == {item["paper_id"] for item in payload["summaries"]}
                self.report_inputs = payload["summaries"]
            return await super().json(system, user)

    model = ReportModel()
    result = asyncio.run(ResearchPipeline(
        settings_for(tmp_path, processing=processing, llm_concurrency=2), llm=model,
    ).run_research_pipeline("agent memory", 3, "agent memory"))
    assert model.peak == expected_peak
    assert len(model.completed) == 3
    assert [item["markdown"] for item in model.report_inputs] == [summary.markdown for summary in result.summaries]
    assert [call["task"] for call in model.call_log].count("summarize_paper") == 3
    assert [call["task"] for call in model.call_log].count("generate_research_report") == 1
    assert "research_chunk_audit" not in result.metrics
    assert "merge_input_audit" not in result.metrics
    assert result.metrics["analysis_strategy"] == "direct_paper_reports"
    assert result.metrics["summary_status_counts"] == {"complete": 3}
    assert result.metrics["report_input_audit"]["compacted"] is False


def test_original_selection_survives_ordered_replacement(tmp_path):
    async def fail_first(paper, *, settings):
        if paper.paper_id == "demo:001":
            failed = paper.model_copy(deep=True)
            failed.status = "download_failed"
            failed.errors.append(ErrorInfo(code="fixture_download_failure", message="Explicit test failure", stage="download", paper_id=paper.paper_id))
            return failed
        return await prepare_demo_paper(paper, settings=settings)

    result = asyncio.run(ResearchPipeline(settings_for(tmp_path), download_fn=fail_first).run_research_pipeline("agent memory token cost", 3))
    assert [paper.paper_id for paper in result.selected_papers] == ["demo:001", "demo:002", "demo:003"]
    assert result.metrics["analysis_paper_ids"] == ["demo:002", "demo:003", "demo:004"]
    assert result.metrics["replacements"] == [{"original_paper_id": "demo:001", "replacement_paper_id": "demo:004"}]
    fulltext_timing = next(timing for timing in result.timings if timing.stage == "fulltexts")
    assert fulltext_timing.error_count == 1
    assert result.status == "demo_completed"


def test_one_source_failure_cannot_complete(tmp_path):
    async def unavailable(*args, **kwargs):
        raise RuntimeError("explicit source fixture failure")

    result = asyncio.run(ResearchPipeline(settings_for(tmp_path), search_arxiv_fn=unavailable).run_research_pipeline("agent memory token cost", 3))
    assert result.status == "partial"
    assert result.metrics["source_success"] == {"arxiv": False, "openalex": True}
    assert result.metrics["processing_complete"] is True
    assert next(t for t in result.timings if t.stage == "search").error_count == 1


@pytest.mark.parametrize("fails", [False, True])
def test_optional_crossref_preserves_core_results_and_audits_coverage(tmp_path, fails):
    async def crossref(query, limit, *, settings=None, client=None):
        if fails:
            from paper_agent.sources.crossref_source import CrossrefSourceError
            raise CrossrefSourceError("rate_limit", "Fixture Crossref rate limit", retryable=True)
        papers = demo_papers("arxiv", 1)
        papers[0].sources = ["crossref"]
        papers[0].source_records = [{"source": "crossref", "id": "fixture"}]
        return papers

    result = asyncio.run(ResearchPipeline(settings_for(tmp_path), search_crossref_fn=crossref)
                         .run_research_pipeline("agent memory", 3))
    assert result.status == "demo_completed"
    assert result.metrics["required_sources"] == ["arxiv", "openalex"]
    assert result.metrics["optional_sources"] == ["crossref"]
    assert result.metrics["source_success"]["crossref"] is not fails
    assert result.source_counts["crossref"] == int(not fails)
    assert result.metrics["source_coverage"]["crossref"]["merged_count"] == int(not fails)
    if fails:
        assert any(e.stage == "crossref_search" and e.code == "rate_limit" for e in result.errors)
    assert result.metrics["processing_complete"] is True
    assert next(t for t in result.timings if t.stage == "search").error_count == int(fails)


def test_explicit_query_is_used_verbatim_without_planner(tmp_path):
    async def forbidden(*args, **kwargs):
        pytest.fail("An explicit English query must not invoke the planner")

    async def source(query, limit, *, settings=None, client=None):
        return demo_papers("arxiv", 4)

    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path), query_planner_fn=forbidden,
                                    search_arxiv_fn=source, search_openalex_fn=source)
        return await pipeline.search_papers("agent memory token cost", 3, "exact memory query")

    result = asyncio.run(scenario())
    assert result.query == "exact memory query"
    assert result.queries == ["exact memory query"]
    assert result.metrics["query_plan"]["mode"] == "explicit"
    assert result.metrics["source_requests"] == {"arxiv": 0, "openalex": 0}
    assert result.source_counts == {"arxiv": 4, "openalex": 4}


def test_untranslated_idea_never_hits_sources(tmp_path):
    async def untranslated(*args, **kwargs):
        return {"queries": [], "ranking_query": "", "mode": "needs_translation", "warnings": [], "errors": []}

    async def forbidden(*args, **kwargs):
        pytest.fail("A source request must not occur without an English query")

    result = asyncio.run(ResearchPipeline(settings_for(tmp_path), query_planner_fn=untranslated,
                                         search_arxiv_fn=forbidden, search_openalex_fn=forbidden).run_research_pipeline("长期记忆与经验复用", 3))
    assert result.status == "failed"
    assert result.errors[0].code == "translation_required"
    assert not result.papers


def test_stage_dependencies_run_id_and_resume_configuration(tmp_path):
    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path))
        run = await pipeline.search_papers("agent memory", 3)
        with pytest.raises(ValueError, match="selection"):
            await pipeline.prepare_fulltexts(run.run_id)
        with pytest.raises(ValueError, match="run_id"):
            await pipeline.select_top_k_papers("../secret")
        concurrent = ResearchPipeline(replace(pipeline.settings, processing="concurrent"))
        with pytest.raises(ValueError, match="原始模式"):
            await concurrent.select_top_k_papers(run.run_id)
        live = ResearchPipeline(replace(pipeline.settings, mode="live"))
        with pytest.raises(ValueError, match="原始模式"):
            await live.select_top_k_papers(run.run_id)
    asyncio.run(scenario())


def test_frozen_input_does_not_invent_online_source_success(tmp_path):
    result = asyncio.run(ResearchPipeline(settings_for(tmp_path)).run_research_pipeline(
        "agent memory", 3, papers=demo_papers("arxiv", 6)))
    assert result.metrics["frozen_input"] is True
    assert result.metrics["source_success"] == {"arxiv": False, "openalex": False}
    assert result.metrics["processing_complete"] is True
    assert result.metrics["online_complete"] is False
    assert result.status == "partial"


def test_partial_documents_do_not_complete(tmp_path):
    async def partial(paper, *, settings):
        value = paper.model_copy(deep=True)
        value.status = "partial_fulltext"
        value.fulltext_quality["complete"] = False
        return value

    result = asyncio.run(ResearchPipeline(settings_for(tmp_path), convert_fn=partial).run_research_pipeline("agent memory", 3))
    assert result.status == "partial"
    assert result.metrics["fulltext_count"] == 0
    assert result.metrics["processing_complete"] is False
    assert len(result.metrics["attempted_paper_ids"]) == 3
    assert result.metrics["pdf_acquired_count"] == 3
    assert result.metrics["replacement_attempts"] == []


def test_pdf_replacement_follows_rank_without_preferring_url_or_target(tmp_path):
    attempted = []

    async def download(paper, *, settings):
        attempted.append(paper.paper_id)
        if paper.paper_id in {"demo:001", "demo:004"}:
            result = paper.model_copy(deep=True)
            result.status = "download_failed"
            return result
        return await prepare_demo_paper(paper, settings=settings)

    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path), download_fn=download)
        run = await pipeline.search_papers("agent memory", 3)
        run = await pipeline.select_top_k_papers(run.run_id)
        # A later candidate has both a PDF URL and a target match. Neither
        # should move it ahead of the next ranked candidate.
        for paper in run.papers:
            if paper.paper_id == "demo:006":
                paper.pdf_url = "https://example.invalid/later.pdf"
        run.metrics["target_coverage"] = {
            "targets": [{"target": "fixture", "direct_title_paper_ids": ["demo:006"]}],
        }
        pipeline.store.save(run)
        return await pipeline.prepare_fulltexts(run.run_id)

    result = asyncio.run(scenario())
    assert attempted == ["demo:001", "demo:002", "demo:003", "demo:004", "demo:005"]
    assert result.metrics["analysis_paper_ids"] == ["demo:002", "demo:003", "demo:005"]
    assert result.metrics["replacements"] == [
        {"original_paper_id": "demo:001", "replacement_paper_id": "demo:005"},
    ]
    assert result.selected_papers[0].paper_id == "demo:001"


def test_failed_conversions_try_bounded_reserves_then_preserve_original_pdfs(tmp_path):
    async def failed_conversion(paper, *, settings):
        result = paper.model_copy(deep=True)
        result.status = "conversion_failed"
        result.markdown_path = None
        result.page_map = []
        result.fulltext_quality = {"complete": False}
        result.errors.append(ErrorInfo(code="conversion_failed", message="Fixture conversion failure", stage="conversion"))
        return result

    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path), convert_fn=failed_conversion)
        run = await pipeline.search_papers("agent memory", 3)
        run = await pipeline.select_top_k_papers(run.run_id)
        return await pipeline.prepare_fulltexts(run.run_id)

    result = asyncio.run(scenario())
    assert result.metrics["pdf_acquired_count"] == 3
    assert result.metrics["fulltext_count"] == 0
    assert result.metrics["partial_fulltext_count"] == 0
    assert len(result.metrics["replacement_attempts"]) == 6
    assert result.metrics["readable_count"] == 0
    assert result.metrics["downloaded_pdf_count"] == 9
    assert result.metrics["fulltext_attempt_limit_reached"] is True
    assert result.metrics["replacements"] == []
    assert result.metrics["analysis_paper_ids"] == ["demo:001", "demo:002", "demo:003"]
    assert all(paper.errors for paper in result.papers[:3])


@pytest.mark.parametrize("processing", ["serial", "concurrent"])
def test_unreadable_pdfs_are_replaced_in_rank_order_and_partial_bodies_are_kept(tmp_path, processing):
    async def convert(paper, *, settings):
        result = paper.model_copy(deep=True)
        if paper.paper_id in {"demo:001", "demo:004"}:
            result.status = "conversion_failed"
            result.markdown_path = None
            result.page_map = []
            result.fulltext_quality = {"complete": False}
            result.errors.append(ErrorInfo(code="conversion_failed", message="Fixture no body", stage="conversion"))
        elif paper.paper_id == "demo:002":
            result.status = "partial_fulltext"
            result.fulltext_quality["complete"] = False
        return result

    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path, processing=processing), convert_fn=convert)
        result = await pipeline.run_research_pipeline("agent memory", 3)
        persisted = pipeline.store.load(result.run_id)
        assert persisted.metrics == result.metrics
        # Completed stage remains idempotent across a fresh pipeline instance.
        resumed = ResearchPipeline(pipeline.settings, convert_fn=convert)
        assert (await resumed.prepare_fulltexts(result.run_id)).metrics == result.metrics
        return result

    result = asyncio.run(scenario())
    assert [p.paper_id for p in result.selected_papers] == ["demo:001", "demo:002", "demo:003"]
    assert result.metrics["attempted_paper_ids"] == [f"demo:{i:03d}" for i in range(1, 6)]
    assert result.metrics["analysis_paper_ids"] == ["demo:002", "demo:003", "demo:005"]
    assert result.metrics["replacements"] == [{"original_paper_id": "demo:001", "replacement_paper_id": "demo:005"}]
    assert result.metrics["readable_count"] == 3
    assert result.metrics["fulltext_count"] == 2
    assert result.metrics["partial_fulltext_count"] == 1
    assert result.metrics["downloaded_pdf_count"] == 5
    assert result.metrics["pdf_acquired_count"] == 3
    assert result.report and result.report.markdown
    assert {r.paper_id for r in result.report.references} == set(result.metrics["analysis_paper_ids"])
    assert any(p.paper_id == "demo:001" and Path(p.pdf_path).is_file() and p.errors for p in result.papers)


@pytest.mark.parametrize("body", [b" \n\t", b"valid start" * 7000 + b"\xff"],
                         ids=["whitespace", "invalid-utf8-tail"])
def test_limited_recovery_does_not_reinsert_original_of_an_already_filled_slot(tmp_path, body):
    async def convert(paper, *, settings):
        if paper.paper_id not in {"demo:001", "demo:002"}:
            return paper
        result = paper.model_copy(deep=True)
        # Empty text and invalid UTF-8 (even after a valid prefix) are unreadable.
        Path(result.markdown_path).write_bytes(body)
        result.status = "conversion_failed"
        result.fulltext_quality["complete"] = False
        return result

    result = asyncio.run(ResearchPipeline(
        settings_for(tmp_path, max_fulltext_attempts=1), convert_fn=convert,
    ).run_research_pipeline("agent memory", 3))
    assert result.metrics["readable_count"] == 2
    assert result.metrics["analysis_paper_ids"] == ["demo:003", "demo:004", "demo:002"]
    assert result.metrics["replacements"] == [{"original_paper_id": "demo:001", "replacement_paper_id": "demo:004"}]
    assert result.metrics["fulltext_attempt_limit_reached"] is True
    assert next(s for s in result.summaries if s.paper_id == "demo:002").summary_scope == "demo_fixture"
    assert result.metrics["fulltext_count"] == 2


@pytest.mark.parametrize("processing", ["serial", "concurrent"])
def test_fulltext_stage_deadline_preserves_completed_work_and_stops_reserves(tmp_path, processing):
    async def download(paper, *, settings):
        if paper.paper_id == "demo:001":
            return await prepare_demo_paper(paper, settings=settings)
        await asyncio.Event().wait()

    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path, processing=processing, fulltext_timeout=.1),
                                    download_fn=download)
        result = await asyncio.wait_for(pipeline.run_research_pipeline("agent memory", 3), timeout=5)
        assert pipeline.store.load(result.run_id).metrics == result.metrics
        return result

    result = asyncio.run(scenario())
    assert result.metrics["fulltext_time_limit_reached"] is True
    assert result.metrics["replacement_attempts"] == []
    assert result.metrics["analysis_paper_ids"] == ["demo:001"]
    assert result.metrics["readable_count"] == 1
    assert result.metrics["fulltext_count"] == 1
    assert {e.paper_id for e in result.errors if e.code == "fulltext_timeout"} == {"demo:002", "demo:003"}
    assert result.status == "partial"


def test_cancelling_fulltexts_keeps_completed_paper_checkpoint(tmp_path):
    async def scenario():
        entered = asyncio.Event()

        async def download(paper, *, settings):
            if paper.paper_id == "demo:001":
                return await prepare_demo_paper(paper, settings=settings)
            entered.set()
            await asyncio.Event().wait()

        pipeline = ResearchPipeline(settings_for(tmp_path), download_fn=download)
        run = await pipeline.search_papers("agent memory", 3)
        run = await pipeline.select_top_k_papers(run.run_id)
        task = asyncio.create_task(pipeline.prepare_fulltexts(run.run_id))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        restored = pipeline.store.load(run.run_id)
        assert "fulltexts" not in restored.metrics["completed_stages"]
        assert Path(restored.papers[0].markdown_path).is_file()
        assert restored.papers[0].metadata["fulltext_provenance"]["role"] == "original"
        assert len(restored.selected_papers) == 3
    asyncio.run(scenario())


def test_concurrency_is_bounded_and_candidate_order_stable(tmp_path):
    async def scenario():
        active, peak = 0, 0
        async def measured_download(paper, *, settings):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            # Controlled scheduling fixture; not a performance measurement.
            await asyncio.sleep(.002)
            result = await prepare_demo_paper(paper, settings=settings)
            active -= 1
            return result
        config = settings_for(tmp_path, processing="concurrent", download_concurrency=2)
        result = await ResearchPipeline(config, download_fn=measured_download).run_research_pipeline("agent memory", 5)
        assert peak == 2
        assert result.metrics["analysis_paper_ids"] == [f"demo:{i:03d}" for i in range(1, 6)]
        assert result.metrics["attempted_paper_ids"] == result.metrics["analysis_paper_ids"]
        assert result.status == "demo_completed"
    asyncio.run(scenario())


def test_live_rejects_synthetic_model(tmp_path):
    with pytest.raises(ValueError, match="合成"):
        ResearchPipeline(settings_for(tmp_path, mode="live"), llm=DemoLLM())


@pytest.mark.parametrize("processing", ["serial", "concurrent"])
def test_readable_progress_tracks_each_paper_and_preserves_failure(tmp_path, processing):
    class OneFailure(DemoLLM):
        async def json(self, system, user):
            payload = json.loads(user)
            if payload.get("task") == "summarize_paper" and payload["paper"]["paper_id"] == "demo:002":
                return {"overview": "invalid fixture without analysis"}
            await asyncio.sleep(0)
            return await super().json(system, user)

    result = asyncio.run(ResearchPipeline(
        settings_for(tmp_path, processing=processing, llm_concurrency=1), llm=OneFailure(),
    ).run_research_pipeline("agent memory", 3))
    events = [json.loads(line) for line in (Path(result.output_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    for paper_id in result.metrics["analysis_paper_ids"]:
        analysis = [e for e in events if e["stage"] == "research_summarize" and e.get("paper_id") == paper_id]
        assert [e["event"] for e in analysis] == ["queued", "start", "finish"]
        assert all(e["title"] in e["message"] for e in analysis)
        for step in ("download", "conversion"):
            assert any(e.get("paper_id") == paper_id and e.get("step") == step
                       and e.get("event") == "start" and e.get("message") for e in events)
    finished = [e for e in events if e["stage"] == "research_summarize" and e.get("event") == "finish"]
    assert [e["completed"] for e in finished] == [1, 2, 3]
    assert finished[-1]["total"] == 3 and finished[-1]["usable"] == 2
    failed = next(e for e in finished if e["paper_id"] == "demo:002")
    assert failed["scope"] == "failed" and "失败" in failed["message"]
    assert "invalid_model_output" in failed["error_codes"]
    report_start = next(e for e in events if e["stage"] == "research_report" and e.get("event") == "start")
    assert "2/3" in report_start["message"]
    assert events.index(report_start) > events.index(finished[-1])
    assert result.status == "partial"
    assert result.metrics["summary_status_counts"] == {"complete": 2, "failed": 1}


def test_waiting_progress_stops_with_cancelled_activity(tmp_path, monkeypatch):
    monkeypatch.setattr("paper_agent.services.pipeline.PROGRESS_INTERVAL_SECONDS", .01)

    async def scenario():
        pipeline = ResearchPipeline(settings_for(tmp_path))
        run = pipeline._new("agent memory", 3, "")
        path = Path(run.output_dir) / "trace.jsonl"

        async def work():
            async with pipeline._activity(run.run_id, "research_summarize", "正在等待模型响应",
                                          paper_id="fixture", title="Fixture paper"):
                await asyncio.Event().wait()

        task = asyncio.create_task(work())
        for _ in range(100):
            await asyncio.sleep(.005)
            if path.is_file() and '"waiting"' in path.read_text(encoding="utf-8"):
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        recorded = path.read_text(encoding="utf-8")
        await asyncio.sleep(.03)
        assert path.read_text(encoding="utf-8") == recorded
        return [json.loads(line) for line in recorded.splitlines()]

    events = asyncio.run(scenario())
    assert events[0]["event"] == "start" and events[-1]["event"] == "cancelled"
    waiting = [event for event in events if event["event"] == "waiting"]
    assert waiting and all("已等待" in event["message"] for event in waiting)
    assert all(event["elapsed_seconds"] >= 0 for event in waiting)
