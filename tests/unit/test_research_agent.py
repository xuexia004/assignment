"""Direct N-paper reports plus one synthesis; fixtures do not prove live quality."""
import asyncio
import copy
import hashlib
import json
from functools import wraps
from pathlib import Path

import pytest

from paper_agent.agents.research_agent import (
    build_research_brief,
    generate_report,
    summarize_paper,
)
from paper_agent.config import Settings
from paper_agent.demo_fixtures import DemoLLM
from paper_agent.models import (
    ErrorInfo,
    EvidenceReference,
    PageMapping,
    Paper,
    PaperReference,
    PaperSummary,
    ResearchReport,
)


def run_async(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return wrapped


def settings_at(tmp_path, **kwargs):
    return Settings(data_dir=tmp_path / "data", **kwargs)


def content_for(request):
    return {
        "overview": "本文报告一种记忆方法及其评测结果。",
        "deep_analysis": "## 研究问题\n\n如何复用记忆？\n\n## 实验与结果\n\n完整阅读所提供的论文后生成单篇报告。",
        "research_question": "如何复用记忆？", "core_idea": "复用已有材料。",
        "methods": ["论文描述的记忆方法。"], "experiments": ["论文提供的实验。"],
        "findings": ["该论文描述记忆复用方法。"], "limitations": ["需要更多场景验证。"],
        "relevance": "与记忆研究有关。", "author_claims": ["作者报告了实验。"],
        "system_inferences": [], "speculative_ideas": [],
    }


class RecordingLLM:
    def __init__(self, mutate=None):
        self.requests = []
        self.mutate = mutate

    async def json(self, system, user):
        request = json.loads(user)
        self.requests.append((system, request))
        if request["task"] == "generate_research_report":
            ids = [summary["paper_id"] for summary in request["summaries"]]
            response = {"overview": "根据 [" + ids[0] + "] 等单篇报告进行综合。",
                        "related_work": ["记忆方法。"],
                        "comparison": [{"paper_ids": ids, "method": "比较输入中的单篇报告。"}],
                        "researched_questions": ["已有工作研究了哪些问题？"],
                        "limitations": ["尚未独立复现。"], "research_directions": ["验证新的研究方向。"],
                        "references": [{"paper_id": item["paper_id"], "title": item["title"]}
                                       for item in request["summaries"]]}
        else:
            response = content_for(request)
        return self.mutate(request, response) if self.mutate else response


def full_paper(tmp_path, *, paper_id="arxiv:1234.5678", texts=None, quality=True):
    texts = texts or ["# Method\nA grounded method reduces repeated work.\n", "# Experiments\nThe authors report a small experiment and discuss limitations.\n"]
    text = "".join(texts)
    path = tmp_path / ("source-" + str(abs(hash(paper_id))) + ".md")
    path.write_text(text, encoding="utf-8")
    page_map = []
    offset = 0
    for number, page_text in enumerate(texts, 1):
        page_map.append(PageMapping(page_number=number, text=page_text, start_char=offset, end_char=offset + len(page_text)))
        offset += len(page_text)
    return Paper(paper_id=paper_id, title="A traceable paper", markdown_path=str(path),
                 page_map=page_map, status="converted", fulltext_quality={"complete": quality})

def test_atomic_artifact_write_with_long_windows_filename(tmp_path):
    from paper_agent.agents.research_agent import _write_pair

    # The final name fits MAX_PATH; appending the name again for a temporary
    # file used to exceed it inside a nested desktop job directory.
    parent = tmp_path / ("nested-" + "a" * 55)
    name_length = max(20, 238 - len(str(parent.resolve())) - 1)
    target = parent / ("r" * (name_length - 3) + ".md")
    report = ResearchReport(idea="fixture", markdown="fixture report")
    _write_pair(target, report)
    assert target.read_text(encoding="utf-8") == "fixture report"
    assert json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))["idea"] == "fixture"
    assert not list(parent.glob("*.tmp"))


@run_async
async def test_full_markdown_enters_one_model_call_and_report_is_persisted(tmp_path):
    text = "# 正文\n" + "Whole document. " * 9000 + "END_OF_PAPER_EXPERIMENT"
    paper = full_paper(tmp_path, texts=[text])
    original = Path(paper.markdown_path).read_text(encoding="utf-8")
    model = RecordingLLM()
    summary = await summarize_paper("memory", paper, settings=settings_at(
        tmp_path, chunk_chars=1000, max_chunks=1, llm_input_token_budget=3000), llm=model)
    assert len(model.requests) == 1
    assert model.requests[0][1]["task"] == "summarize_paper"
    assert model.requests[0][1]["source"]["text"] == original
    assert summary.summary_scope == "fulltext" and not summary.errors
    assert summary.overview and summary.deep_analysis and summary.findings
    assert Path(summary.summary_path).read_text(encoding="utf-8") == summary.markdown
    assert json.loads(Path(summary.summary_path).with_suffix(".json").read_text(encoding="utf-8"))["deep_analysis"] == summary.deep_analysis
    assert not list(Path(summary.summary_path).parent.glob("*.chunks.json"))
    assert paper.metadata["research_audit"]["llm_calls"] == 1
    assert paper.metadata["research_audit"]["quote_validation"] is False
    assert all(evidence.quote is None and evidence.page_number is None for evidence in summary.evidence)
    assert all(evidence.evidence_type == "system_inference" for evidence in summary.evidence)


@run_async
async def test_source_scope_comes_from_converter_without_page_quote_gate(tmp_path):
    paper = full_paper(tmp_path)
    paper.page_map = []
    summary = await summarize_paper("memory", paper, settings=settings_at(tmp_path), llm=RecordingLLM())
    assert summary.summary_scope == "fulltext" and not summary.errors
    paper.fulltext_quality["complete"] = False
    summary = await summarize_paper("memory", paper, settings=settings_at(tmp_path), llm=RecordingLLM())
    assert summary.summary_scope == "partial_fulltext"
    assert summary.overview and summary.deep_analysis


@run_async
async def test_missing_markdown_uses_only_available_abstract(tmp_path):
    model = RecordingLLM()
    paper = Paper(paper_id="p1", title="Unavailable PDF", abstract="Actual abstract.")
    summary = await summarize_paper("memory", paper, settings=settings_at(tmp_path), llm=model)
    assert summary.summary_scope == "abstract_only"
    assert model.requests[0][1]["source"]["text"] == paper.abstract


@run_async
async def test_no_source_or_idea_has_no_model_calls(tmp_path):
    model = RecordingLLM()
    assert (await summarize_paper("memory", Paper(paper_id="p1", title="Missing"), settings=settings_at(tmp_path), llm=model)).summary_scope == "failed"
    assert (await summarize_paper(" ", full_paper(tmp_path), settings=settings_at(tmp_path), llm=model)).summary_scope == "failed"
    assert model.requests == []


@run_async
async def test_full_document_over_budget_fails_without_chunking_or_truncation(tmp_path):
    model = RecordingLLM()
    paper = full_paper(tmp_path, texts=["Long source. " * 5000])
    summary = await summarize_paper("memory", paper, settings=settings_at(tmp_path, analysis_input_token_budget=4000), llm=model)
    assert summary.summary_scope == "failed"
    assert [error.code for error in summary.errors] == ["llm_budget"]
    assert model.requests == []
    assert paper.metadata["research_audit"]["source_chars"] == len(Path(paper.markdown_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("bad", ["extra", "empty", "wrong_type", "claimed_scope"])
@run_async
async def test_only_basic_report_shape_is_checked_without_extra_model_rounds(tmp_path, bad):
    def mutate(request, response):
        if bad == "extra":
            response["summary_path"] = "untrusted-path"
        elif bad == "empty":
            response["deep_analysis"] = " "
        elif bad == "wrong_type":
            response["findings"] = "not a list"
        else:
            response["summary_scope"] = "fulltext"
        return response
    model = RecordingLLM(mutate)
    summary = await summarize_paper("memory", full_paper(tmp_path), settings=settings_at(tmp_path), llm=model)
    assert summary.summary_scope == "failed"
    assert len(model.requests) == 1
    assert summary.errors[0].code == "invalid_model_output"


@run_async
async def test_model_errors_are_not_written_into_artifacts(tmp_path):
    def mutate(request, response):
        raise RuntimeError("a-private-provider-credential")
    summary = await summarize_paper("memory", full_paper(tmp_path), settings=settings_at(tmp_path), llm=RecordingLLM(mutate))
    assert "a-private-provider-credential" not in summary.model_dump_json()
    assert "a-private-provider-credential" not in Path(summary.summary_path).read_text(encoding="utf-8")
    missing = await summarize_paper("memory", full_paper(tmp_path), settings=settings_at(tmp_path))
    assert missing.errors[0].code == "llm_configuration"


@run_async
async def test_three_paper_reports_then_one_synthesis_with_every_full_report(tmp_path):
    model = RecordingLLM()
    papers = [full_paper(tmp_path, paper_id=f"p{i}") for i in range(3)]
    summaries = [await summarize_paper("memory", paper, settings=settings_at(tmp_path), llm=model) for paper in papers]
    for i, summary in enumerate(summaries):
        summary.markdown += "\n" + (f"Full paper report {i}. " * 500) + f"LAST_REPORT_SECTION_{i}"
    before = copy.deepcopy([summary.model_dump() for summary in summaries])
    report = await generate_report("memory", summaries, settings=settings_at(tmp_path, report_input_token_budget=100000), llm=model)
    assert [request["task"] for _, request in model.requests] == ["summarize_paper"] * 3 + ["generate_research_report"]
    submitted = model.requests[-1][1]["summaries"]
    assert [item["markdown"] for item in submitted] == [summary.markdown for summary in summaries]
    assert [item["paper_id"] for item in submitted] == [paper.paper_id for paper in papers]
    assert [summary.model_dump() for summary in summaries] == before
    assert not report.errors
    assert Path(report.report_path).read_text(encoding="utf-8") == report.markdown
    brief, status = build_research_brief(papers=papers, summaries=summaries, report=report)
    assert status == "result" and brief.findings


@run_async
async def test_report_input_over_budget_is_not_silently_compacted(tmp_path):
    model = RecordingLLM()
    summaries = [PaperSummary(paper_id=f"p{i}", title=f"Paper {i}", summary_scope="fulltext", overview="Overview", deep_analysis="Report", markdown="ALL_REPORT_SECTIONS " * 2000) for i in range(3)]
    report = await generate_report("memory", summaries, settings=settings_at(tmp_path, report_input_token_budget=4000), llm=model)
    assert model.requests == []
    assert any(error.code == "llm_budget" for error in report.errors)
    assert all(summary.markdown.endswith("ALL_REPORT_SECTIONS ") for summary in summaries)


@run_async
async def test_failed_paper_report_is_not_invented_and_remaining_reports_still_synthesize(tmp_path):
    model = RecordingLLM()
    summaries = [PaperSummary(paper_id=f"p{i}", title=f"Paper {i}", summary_scope="fulltext", overview="Overview", deep_analysis="Whole report") for i in range(2)]
    summaries.append(PaperSummary(paper_id="missing", title="Failed report", summary_scope="failed"))
    report = await generate_report("memory", summaries, settings=settings_at(tmp_path), llm=model)
    assert len(model.requests) == 1
    assert [item["paper_id"] for item in model.requests[0][1]["summaries"]] == ["p0", "p1"]
    assert report.overview and report.errors[0].code == "partial_report"
    assert report.errors[0].details["failed_summary_ids"] == ["missing"]


@run_async
async def test_synthesis_keeps_actual_paper_ids_without_quote_validation(tmp_path):
    summaries = [PaperSummary(paper_id=f"p{i}", title=f"Paper {i}", summary_scope="fulltext", overview="Overview", deep_analysis="Report") for i in range(3)]
    def mutate(request, response):
        response["references"][0]["paper_id"] = "invented"
        return response
    model = RecordingLLM(mutate)
    report = await generate_report("memory", summaries, settings=settings_at(tmp_path), llm=model)
    assert len(model.requests) == 1
    assert any(error.code == "invalid_model_output" for error in report.errors)
    assert [reference.paper_id for reference in report.references] == ["p0", "p1", "p2"]


@run_async
async def test_dynamic_report_body_preserves_structure_and_renders_named_citations(tmp_path):
    summaries = [PaperSummary(paper_id=f"arxiv:2604.1729{i}", title=f"Research Paper {i}",
                              summary_scope="fulltext", overview="概览", deep_analysis="单篇完整报告")
                 for i in range(3)]

    def mutate(request, response):
        response["markdown"] = ("# 如何选择研究路线\n\n## 先看适用条件\n\n"
                                "[arxiv:2604.17290] 给出第一条路线。\n\n"
                                "## 一个值得检验的差异\n\n风险范围 [0,1]；研究建议需要实证验证。")
        return response

    model = RecordingLLM(mutate)
    report = await generate_report("比较研究路线", summaries, settings=settings_at(tmp_path), llm=model)
    assert not report.errors
    assert len(model.requests) == 1
    assert report.markdown.startswith("# 如何选择研究路线\n\n## 先看适用条件")
    assert "《Research Paper 0》 给出第一条路线。" in report.markdown
    assert "## 一个值得检验的差异" in report.markdown
    assert "[0,1]" in report.markdown
    assert "## 研究主题概述" not in report.markdown
    assert "## 参考文献" in report.markdown
    assert "arxiv:" not in report.markdown
    assert report.references[0].paper_id == "arxiv:2604.17290"
    assert Path(report.report_path).read_text(encoding="utf-8") == report.markdown


@run_async
async def test_demo_is_explicit_and_still_uses_one_call_per_report(tmp_path):
    model = DemoLLM()
    summaries = []
    for i in range(3):
        paper = full_paper(tmp_path, paper_id=f"demo:{i}")
        paper.metadata["synthetic"] = True
        summaries.append(await summarize_paper("memory", paper, settings=settings_at(tmp_path), llm=model))
    assert all(summary.summary_scope == "demo_fixture" for summary in summaries)
    report = await generate_report("memory", summaries, settings=settings_at(tmp_path), llm=model)
    assert report.overview.startswith("DEMO ONLY") and not report.errors
    assert [item["task"] for item in model.call_log] == ["summarize_paper"] * 3 + ["generate_research_report"]
    assert model.usage.requests == 0


def brief_inputs(count=3, scope="fulltext"):
    papers = [
        Paper(
            paper_id=f"p{index}", title=f"Paper {index}",
            selection_reason=f"Selected for question {index}",
            pdf_artifact_id=f"pdf-{index}", markdown_artifact_id=f"markdown-{index}",
        )
        for index in range(1, count + 1)
    ]
    summaries = [
        PaperSummary(
            paper_id=f"p{index}", title=f"Paper {index}", summary_scope=scope,
            overview="A complete paper-level overview.",
            deep_analysis="A complete paper-level analysis.",
            research_question="A question", core_idea="An idea",
            findings=[f"Finding {index}-1", f"Finding {index}-2"],
            evidence=[
                EvidenceReference(
                    paper_id=f"p{index}", quote=f"Source quote {index}-1",
                    claim=f"Finding {index}-1", page_number=index,
                    pdf_artifact_id=f"pdf-{index}",
                    markdown_artifact_id=f"markdown-{index}",
                ),
                EvidenceReference(
                    paper_id=f"p{index}", quote=f"Source quote {index}-2",
                    claim=f"Finding {index}-2", page_number=index + 1,
                ),
            ],
        )
        for index in range(1, count + 1)
    ]
    report = ResearchReport(
        idea="memory", overview="A source-bounded synthesis.",
        references=[PaperReference(paper_id=paper.paper_id, title=paper.title) for paper in papers],
        report_artifact_id=hashlib.sha256(b"report:report").hexdigest()[:20],
    )
    return papers, summaries, report


def test_brief_builds_traceable_result_without_mutating_inputs():
    papers, summaries, report = brief_inputs()
    original = (
        [paper.model_dump() for paper in papers],
        [summary.model_dump() for summary in summaries],
        report.model_dump(),
    )
    brief, status = build_research_brief(
        papers=papers, summaries=summaries, report=report,
    )
    assert status == "result"
    assert len(brief.findings) == 3
    assert [finding.statement for finding in brief.findings] == [
        "Finding 1-1", "Finding 2-1", "Finding 3-1",
    ]
    for finding, summary in zip(brief.findings, summaries):
        assert finding.evidence[0].paper_id == summary.paper_id
        assert finding.evidence[0].quote == summary.evidence[0].quote
        assert finding.evidence[0] is not summary.evidence[0]
    assert [paper.evidence_level for paper in brief.key_papers] == ["fulltext"] * 3
    assert brief.key_papers[0].selection_reason == "Selected for question 1"
    assert brief.next_step.action
    assert brief.report_artifact_id == report.report_artifact_id
    assert original == (
        [paper.model_dump() for paper in papers],
        [summary.model_dump() for summary in summaries],
        report.model_dump(),
    )


def test_brief_deduplicates_citation_and_attribution_variants_without_losing_contrasts():
    papers, summaries, report = brief_inputs(count=1)
    positive = "作者报告所提框架相比基线 LLM 降低幻觉并改善校准（f12）"
    variant = "所提框架相比基线LLM降低幻觉并改善校准"
    negative = "所提框架相比基线LLM未降低幻觉并改善校准"
    summaries[0].findings = [positive, variant, negative]
    summaries[0].evidence = [
        EvidenceReference(paper_id="p1", claim=positive, quote="A fixture positive result."),
        EvidenceReference(paper_id="p1", claim=negative, quote="A fixture contrasting result."),
    ]
    brief, _ = build_research_brief(papers=papers, summaries=summaries, report=report)
    assert [finding.statement for finding in brief.findings] == [
        "作者报告所提框架相比基线 LLM 降低幻觉并改善校准", negative,
    ]
    assert len(summaries[0].findings) == 3


@pytest.mark.parametrize("scope", ["abstract_only", "partial_fulltext"])
def test_brief_preserves_reading_scope_as_preliminary(scope):
    papers, summaries, report = brief_inputs(scope=scope)
    brief, status = build_research_brief(
        papers=papers, summaries=summaries, report=report,
    )
    assert status == "preliminary_notes"
    assert len(brief.findings) == 3
    assert {paper.evidence_level for paper in brief.key_papers} == {scope}
    assert any(scope == "abstract_only" and "摘要级" in item.description or
               scope == "partial_fulltext" and "部分全文" in item.description
               for item in brief.limitations)
    assert "完整阅读" in brief.next_step.action


def test_brief_retains_reliable_partial_material_and_failed_paper_gap():
    papers, summaries, _ = brief_inputs()
    summaries[2] = PaperSummary(
        paper_id="p3", title="Paper 3", summary_scope="failed",
        errors=[ErrorInfo(code="llm_failed", stage="summarize", message="模型分析失败。")],
    )
    brief, status = build_research_brief(
        papers=papers, summaries=summaries, report=None,
    )
    assert status == "preliminary_notes"
    assert [paper.paper_id for paper in brief.key_papers] == ["p1", "p2"]
    assert len(brief.findings) == 3
    assert {finding.evidence[0].paper_id for finding in brief.findings} == {"p1", "p2"}
    assert any(item.paper_ids == ["p3"] for item in brief.limitations)
    assert any("完整综合报告" in item.description for item in brief.limitations)


def test_brief_full_notes_without_report_are_not_promoted():
    papers, summaries, _ = brief_inputs()
    brief, status = build_research_brief(
        papers=papers, summaries=summaries, report=None,
    )
    assert status == "preliminary_notes"
    assert brief.report_artifact_id is None
    assert len(brief.findings) == 3


def test_brief_omits_unsupported_findings_and_never_pads_to_three():
    papers, summaries, report = brief_inputs(count=4)
    summaries[0].findings = ["Unsupported statement"]
    summaries[0].evidence = []
    brief, status = build_research_brief(
        papers=papers, summaries=summaries, report=report,
    )
    assert status == "preliminary_notes"
    assert [finding.statement for finding in brief.findings] == [
        "Finding 2-1", "Finding 3-1", "Finding 4-1",
    ]
    assert all(finding.statement != "Unsupported statement" for finding in brief.findings)
    assert all(finding.evidence for finding in brief.findings)
    assert any("缺少可关联" in item.description for item in brief.limitations)


def test_brief_matches_numbers_to_the_supporting_quote_not_first_evidence():
    papers, summaries, report = brief_inputs(count=3)
    summaries[0].findings = [
        "扰动导致 8.8% 退化：ε=0 时 96.88%，ε=0.2 时 88.12%（f35、f63、e6）",
        "没有对应引文的额外断言",
    ]
    summaries[0].evidence = [
        EvidenceReference(
            paper_id="p1", claim="Background framing only.",
            quote="A general statement without the reported measurements.", page_number=1,
        ),
        EvidenceReference(
            paper_id="p1", claim="扰动使通过率从 96.88% 降至 88.12%。",
            quote="Perturbations cause 8.8% degradation: from 96.88% to 88.12%.",
            page_number=9, pdf_artifact_id="pdf-1",
        ),
    ]
    brief, _ = build_research_brief(
        papers=papers, summaries=summaries, report=report,
    )
    finding = next(item for item in brief.findings if "8.8%" in item.statement)
    assert "96.88%" in finding.statement and "88.12%" in finding.statement
    assert "f35" not in finding.statement and "e6" not in finding.statement
    assert finding.evidence[0].page_number == 9
    assert "96.88%" in finding.evidence[0].quote
    assert all("没有对应引文" not in item.statement for item in brief.findings)
    assert any("未找到可核验的引文支持" in item.description for item in brief.limitations)


def test_brief_no_result_has_reason_and_actionable_next_step():
    paper = Paper(paper_id="p1", title="Unavailable")
    summary = PaperSummary(paper_id="p1", title="Unavailable", summary_scope="failed")
    brief, status = build_research_brief(
        papers=[paper], summaries=[summary], report=None,
    )
    assert status == "no_result"
    assert not brief.findings and not brief.key_papers
    assert brief.limitations
    assert brief.next_step.action and brief.next_step.rationale
    assert brief.report_artifact_id is None
