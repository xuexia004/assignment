"""Synthetic regression only; live report quality is measured separately."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from paper_agent.agents.research_agent import REPORT_SYSTEM, _size, generate_report
from paper_agent.config import Settings
from paper_agent.llm_client import LLMBudgetError, LLMError, LLMOutputError
from paper_agent.models import PaperSummary


def notes():
    return [PaperSummary(
        paper_id=f"arxiv:2601.0000{i}", title=f"Fixture Paper {i}",
        summary_scope="fulltext", overview="笔记概述", deep_analysis="有条件的分析",
        markdown=f"# 完整笔记{i}\n方法、证据、条件和局限。END_OF_NOTE_{i}",
    ) for i in range(5)]


def valid_output():
    return {
        "overview": "在相同任务设置下比较记忆复用成本；收益取决于检索和维护开销。",
        "related_work": ["这些工作共同研究复用，但证据的任务范围不同。"],
        "comparison": [{"paper_ids": [n.paper_id], "method": "复用已存储经验。",
                        "evidence": "给定笔记讨论同任务的重复推理。",
                        "limitations": "跨任务收益未证实。"} for n in notes()],
        "researched_questions": ["何时复用经验能减少重复推理？"],
        "limitations": ["不同任务不能直接作数值排名。"],
        "research_directions": ["在固定任务上测量包括检索维护在内的总成本。"],
        "references": [{"paper_id": n.paper_id, "title": n.title} for n in notes()],
    }


class RecordedModel:
    def __init__(self, *responses):
        self.responses = list(responses) or [valid_output()]
        self.requests = []

    async def json(self, system, user):
        self.requests.append((system, json.loads(user)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


def run(tmp_path, model, *, strategy="structured", summaries=None, audit=None, **settings):
    return asyncio.run(generate_report(
        "怎样降低推理成本？", notes() if summaries is None else summaries,
        settings=Settings(data_dir=tmp_path, report_strategy=strategy, **settings),
        llm=model, audit=audit,
    ))


def test_structured_report_renders_all_fields_without_dropping_input(tmp_path):
    model, audit, summaries = RecordedModel(), {}, notes()
    original = copy.deepcopy([n.model_dump() for n in summaries])
    report = run(tmp_path, model, audit=audit, summaries=summaries)
    assert not report.errors and report.overview
    assert [n.model_dump() for n in summaries] == original
    assert [n["markdown"] for n in model.requests[0][1]["summaries"]] == [n.markdown for n in summaries]
    assert len(model.requests) == audit["llm_calls"] == 1
    assert audit["repair_count"] == 0 and audit["attempts"][0]["status"] == "valid"
    assert audit["input_compacted"] is False
    assert "END_OF_NOTE" not in json.dumps(audit)
    assert Path(report.report_path).read_text(encoding="utf-8") == report.markdown
    for text in ("研究主题概述", "相关工作分类", "论文比较", "已有证据与方法的局限", "待验证的研究方向"):
        assert text in report.markdown
    assert all(f"《{n.title}》" in report.markdown for n in summaries)
    assert "arxiv:" not in report.markdown


def test_legacy_retains_one_original_prompt_and_no_repairs(tmp_path):
    bad = valid_output()
    bad["related_work"] = "bad string instead of array"
    model, audit = RecordedModel(bad, valid_output()), {}
    report = run(tmp_path, model, strategy="legacy", audit=audit)
    assert len(model.requests) == 1 and model.requests[0][0] == REPORT_SYSTEM
    assert "validation_feedback" not in model.requests[0][1]
    assert report.errors[0].details["reason"] == "schema_validation_failed"
    assert audit["repair_count"] == 0


@pytest.mark.parametrize("kind", ["schema", "json", "length"])
def test_one_repair_uses_complete_original_notes_and_validates_again(tmp_path, kind):
    bad = valid_output()
    bad["related_work"] = "not an array"
    first = {"schema": bad, "json": LLMOutputError("invalid", reason="json_parse_failed"),
             "length": LLMOutputError("truncated", reason="output_truncated", finish_reason="length")}[kind]
    model, audit = RecordedModel(first, valid_output()), {}
    report = run(tmp_path, model, audit=audit)
    assert not report.errors
    assert audit["llm_calls"] == 2 and audit["repair_count"] == 1
    assert [a["status"] for a in audit["attempts"]] == ["invalid_output", "valid"]
    first_payload, repaired_payload = [p for _, p in model.requests]
    feedback = repaired_payload.pop("validation_feedback")
    assert first_payload == repaired_payload
    assert feedback["reason"] in {"schema_validation_failed", "json_parse_failed", "output_truncated"}
    assert all(n["markdown"].endswith(f"END_OF_NOTE_{i}") for i, n in enumerate(repaired_payload["summaries"]))


@pytest.mark.parametrize("error,code", [
    (LLMOutputError("filtered", reason="output_truncated", finish_reason="content_filter"), "invalid_model_output"),
    (LLMOutputError("unspecified", reason="output_truncated"), "invalid_model_output"),
    (LLMError("private transport diagnostic"), "llm_failed"),
    (LLMBudgetError("budget"), "llm_budget"),
])
def test_no_extra_calls_for_filter_transport_or_budget(tmp_path, error, code):
    model = RecordedModel(error, valid_output())
    report = run(tmp_path, model)
    assert len(model.requests) == 1
    assert report.errors[0].code == code
    assert "private transport diagnostic" not in report.model_dump_json()


def test_repair_failure_keeps_failure_and_references_without_third_call(tmp_path):
    bad = valid_output()
    bad["references"][0]["paper_id"] = "arxiv:invented"
    model, audit = RecordedModel(bad, bad, valid_output()), {}
    report = run(tmp_path, model, audit=audit)
    assert len(model.requests) == 2 and audit["repair_count"] == 1
    assert not report.overview
    assert report.errors[0].details["reason"] == "report_validation_failed"
    assert [r.paper_id for r in report.references] == [n.paper_id for n in notes()]


@pytest.mark.parametrize("kind", ["empty_overview", "no_evidence", "uncovered_paper", "duplicate_reference"])
def test_structured_empty_shell_or_bad_references_cannot_pass(tmp_path, kind):
    bad = valid_output()
    if kind == "empty_overview":
        bad["overview"] = " "
    elif kind == "no_evidence":
        bad["comparison"][0]["evidence"] = " "
    elif kind == "uncovered_paper":
        bad["comparison"].pop()
    else:
        bad["references"].append(bad["references"][0])
    report = run(tmp_path, RecordedModel(bad, bad))
    assert not report.overview
    assert report.errors[0].code == "invalid_model_output"


def test_input_and_repair_budgets_are_checked_before_request(tmp_path):
    model = RecordedModel()
    report = run(tmp_path, model, report_input_token_budget=100)
    assert not model.requests and report.errors[0].code == "llm_budget"
    first = RecordedModel()
    run(tmp_path, first)
    system, payload = first.requests[0]
    bad = valid_output()
    bad["related_work"] = "wrong"
    model, audit = RecordedModel(bad, valid_output()), {}
    report = run(tmp_path, model, audit=audit, report_input_token_budget=_size(system, payload))
    assert len(model.requests) == 1 and report.errors[0].code == "llm_budget"
    assert audit["repair_count"] == 0


def test_repaired_report_preserves_partial_inputs(tmp_path):
    summaries = notes()
    summaries[0].summary_scope = "partial_fulltext"
    bad = valid_output()
    bad["limitations"] = "wrong"
    report = run(tmp_path, RecordedModel(bad, valid_output()), summaries=summaries)
    assert report.overview
    assert [e.code for e in report.errors] == ["partial_report"]
    assert report.errors[0].details["incomplete_summary_ids"] == [summaries[0].paper_id]


def test_cancellation_propagates_without_repair(tmp_path):
    model, audit = RecordedModel(asyncio.CancelledError(), valid_output()), {}
    with pytest.raises(asyncio.CancelledError):
        run(tmp_path, model, audit=audit)
    assert len(model.requests) == 1 and audit["attempts"][0]["status"] == "cancelled"
