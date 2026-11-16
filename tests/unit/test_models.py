from dataclasses import replace

import pytest
from pydantic import ValidationError

from paper_agent.config import Settings
from paper_agent.models import (
    ErrorInfo,
    PageMapping,
    Paper,
    PaperReference,
    PaperSummary,
    PipelineResult,
    ResearchReport,
    StageTiming,
    ToolCallRecord,
)


def test_pipeline_contract_roundtrip_and_independent_defaults():
    first = Paper(paper_id="arxiv:1234.56789", title="Agent memory", arxiv_version="v2")
    second = Paper(paper_id="oa:W2", title="Another paper")
    first.sources.append("arxiv")
    assert second.sources == []
    summary = PaperSummary(
        paper_id=first.paper_id, summary_scope="partial_fulltext", warnings=["One unread page"]
    )
    report = ResearchReport(
        idea="Memory", references=[PaperReference(paper_id=first.paper_id, title=first.title)]
    )
    result = PipelineResult(
        idea="Memory", papers=[first, second], selected_papers=[first], summaries=[summary],
        report=report, timings=[StageTiming(stage="search", duration_seconds=1.2)],
        errors=[ErrorInfo(code="PARTIAL", message="One unread page", stage="conversion")],
    )
    restored = PipelineResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.top_k == 5
    assert restored.summaries[0].summary_scope == "partial_fulltext"


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
def test_scores_are_normalized(score):
    with pytest.raises(ValidationError):
        Paper(paper_id="p", title="P", initial_score=score)


def test_invalid_scope_offsets_and_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        PaperSummary(paper_id="p", summary_scope="pretend_fulltext")
    with pytest.raises(ValidationError):
        PageMapping(page_number=1, start_char=10, end_char=5)
    with pytest.raises(ValidationError):
        Paper(paper_id="p", title="P", silently_ignored=True)


def test_tool_call_records_accumulate_in_pipeline_result():
    result = PipelineResult(idea="Memory")
    assert result.tool_calls == []
    result.tool_calls.append(
        ToolCallRecord(tool="search_papers", arguments={"idea": "Memory", "k": 3},
                       duration_seconds=0.5, status="demo_completed")
    )
    result.tool_calls.append(
        ToolCallRecord(tool="select_top_k_papers", arguments={"run_id": result.run_id},
                       duration_seconds=0.2, status="demo_completed", controller="model_tool_call")
    )
    restored = PipelineResult.model_validate_json(result.model_dump_json())
    assert [call.tool for call in restored.tool_calls] == ["search_papers", "select_top_k_papers"]
    assert restored.tool_calls[0].controller == "model_tool_call"
    with pytest.raises(ValidationError):
        ToolCallRecord(tool="search_papers", duration_seconds=-1)
    with pytest.raises(ValidationError):
        ToolCallRecord(tool="search_papers", controller="hardcoded_chain")


def test_settings_compatible_and_secret_repr_safe(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_api_key="secret-value", openalex_api_key="key")
    settings.validate()
    assert settings.pdf_dir == tmp_path / "pdfs"
    assert settings.markdown_dir == tmp_path / "markdown"
    assert settings.execution_mode == "sequential"
    assert replace(settings, processing="concurrent").execution_mode == "concurrent"
    assert "secret-value" not in repr(settings)
    assert "openalex_api_key" not in repr(settings)
    assert not (tmp_path / "pdfs").exists()


def test_report_strategy_is_independent_and_survives_environment_and_snapshot(monkeypatch):
    assert Settings().report_strategy == "structured"
    monkeypatch.setenv("PAPER_AGENT_REPORT_STRATEGY", "structured")
    monkeypatch.setenv("PAPER_AGENT_RESEARCH_PROFILE", "baseline")
    current = Settings.from_env()
    assert current.report_strategy == "structured"
    assert current.paper_analysis_strategy == "direct"
    baseline = replace(current, report_strategy="legacy")
    assert current.experiment_snapshot()["config"]["report_strategy"] == "structured"
    assert current.experiment_snapshot()["config_sha256"] != baseline.experiment_snapshot()["config_sha256"]
    for invalid in ("unknown", True, None, []):
        with pytest.raises(ValueError, match="Report strategy"):
            replace(current, report_strategy=invalid).validate()


def test_comparison_profiles_change_only_declared_methods_and_snapshots_exclude_secrets():
    baseline = Settings(research_profile="baseline", llm_api_key="test-private-value", openalex_api_key="test-openalex-value")
    base = baseline.experiment_snapshot(top_k=3)
    for profile, expected in (("sources", {"crossref_enabled"}), ("ranking", {"cross_encoder_enabled"}), ("layout", {"pdf_parser"}), ("evidence", {"paper_analysis_strategy"}), ("evidence_fast", {"paper_analysis_strategy"}), ("grounded", {"paper_analysis_strategy"})):
        settings = replace(baseline, research_profile=profile)
        new = settings.experiment_snapshot(top_k=3)
        assert {key for key in base["config"] if base["config"][key] != new["config"][key]} == expected
        assert new["profile"] == profile
    assert base["config"]["top_k"] == 3 and base["config"]["pdf_parser"] == "legacy"
    assert base["config_sha256"] == baseline.experiment_snapshot(top_k=3)["config_sha256"]
    assert "private" not in str(base) and "openalex-value" not in str(base)
    assert "llm_api_key" not in base["config"]
    with pytest.raises(ValueError):
        Settings(research_profile="invalid")


def test_environment_configuration_and_invalid_boolean(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PAPER_AGENT_TOP_K", "7")
    monkeypatch.setenv("PAPER_AGENT_CACHE_ENABLED", "false")
    monkeypatch.setenv("PAPER_AGENT_FULLTEXT_TIMEOUT", "42")
    settings = Settings.from_env()
    assert settings.top_k == 7
    assert settings.cache_enabled is False
    assert settings.fulltext_timeout == 42
    assert settings.crossref_enabled is True
    assert settings.pdf_parser == "layout" and settings.cross_encoder_enabled is False
    assert settings.pdf_dir == tmp_path / "pdfs"
    monkeypatch.setenv("PAPER_AGENT_CACHE_ENABLED", "maybe")
    with pytest.raises(ValueError, match="boolean"):
        Settings.from_env()


def test_reranking_strategies_are_explicit_and_invalid_parser_is_rejected(monkeypatch):
    monkeypatch.setenv("PAPER_AGENT_CROSS_ENCODER_ENABLED", "true")
    monkeypatch.setenv("PAPER_AGENT_PDF_PARSER", "legacy")
    settings = Settings.from_env()
    assert settings.cross_encoder_enabled and settings.pdf_parser == "legacy"
    with pytest.raises(ValueError, match="either"):
        replace(settings, value_ranking_enabled=True).validate()
    with pytest.raises(ValueError, match="PDF_PARSER"):
        Settings(pdf_parser="unknown").validate()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_fulltext_stage_timeout_is_bounded(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        Settings(fulltext_timeout=timeout).validate()
