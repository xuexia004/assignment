"""Paired answer scoring, independent selection, and spending boundaries."""

import asyncio
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.config import Settings
from paper_agent.llm_client import LLMUsage
from paper_agent.models import ErrorInfo, EvidenceReference, PaperSummary

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/benchmark_evidence_answers.py"))


def make_entry(tmp_path, qid="q1"):
    text = "The result was 42 points. " * 10
    path = tmp_path / f"{qid}.md"
    path.write_text(text, encoding="utf-8")
    return {"question_id": qid, "paper_id": "paper-" + qid, "question": "What was the result?",
            "title": "Test", "text_path": str(path), "input_sha256": SCRIPT["digest"](text),
            "question_sha256": SCRIPT["digest"]("What was the result?"), "text": text,
            "gold_answers": [{"unanswerable": False, "extractive_spans": ["42 points"],
                              "yes_no": None, "free_form_answer": ""}],
            "references": [{"answer": "42 points", "type": "extractive"}]}


def test_f1_and_reference_answers_match_qasper_normalization():
    assert SCRIPT["token_f1"]("The 42 points!", "42 points") == 1
    assert SCRIPT["token_f1"]("42", "42 points") == pytest.approx(2 / 3)
    assert SCRIPT["token_f1"]("none", "42 points") == 0
    annotations = [{"unanswerable": False, "yes_no": False},
                   {"unanswerable": True, "yes_no": True},
                   {"unanswerable": False, "extractive_spans": ["X", "Y"], "free_form_answer": "Z"},
                   {"unanswerable": False, "free_form_answer": "paraphrase"}]
    assert SCRIPT["reference_answers"](annotations) == [
        {"answer": "No", "type": "boolean"},
        {"answer": "Unanswerable", "type": "none"},
        {"answer": "X, Y", "type": "extractive"},
        {"answer": "paraphrase", "type": "abstractive"},
    ]


def test_selection_is_sorted_and_rejects_changed_frozen_input(tmp_path):
    entries = [make_entry(tmp_path, qid) for qid in ["z", "a"]]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "test", "examples": entries}), encoding="utf-8")
    selected, _ = SCRIPT["selected_inputs"](manifest, 1)
    assert selected[0]["question_id"] == "a"
    Path(entries[1]["text_path"]).write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        SCRIPT["selected_inputs"](manifest, 1)


class Model:
    def __init__(self, fail=False):
        self.settings = SimpleNamespace(llm_max_output_tokens=512, llm_model="fixture")
        self.usage = LLMUsage()
        self.call_log = []
        self.fail = fail
        self.prompts = []

    async def json(self, system, user):
        self.prompts.append((system, user))
        self.usage.requests += 1
        self.call_log.append({"http_status": 200})
        if self.fail:
            self.usage.failed_requests += 1
            raise RuntimeError("sensitive transport diagnostic must not be persisted")
        self.usage.prompt_tokens += 100
        self.usage.completion_tokens += 10
        self.usage.total_tokens += 110
        self.usage.successful_requests += 1
        return {"answer": "42 points", "unanswerable": False}


def context_builder(question, text, *, max_bytes, method):
    assert max_bytes == 12000 and method == "hybrid"
    return "The result was 42 points.", {"actual_method": "hybrid"}


def test_pair_order_usage_and_all_failed_slots_remain_in_denominator(tmp_path):
    entry = make_entry(tmp_path)
    args = SimpleNamespace(output=tmp_path / "run", repeats=2, max_calls=4, max_units=20000)
    model = Model(fail=True)
    result = asyncio.run(SCRIPT["run_comparison"](args, [entry], {}, model, context_builder))
    assert [row["strategy"] for row in result["rows"]] == [
        "wholetext", "evidence_context", "evidence_context", "wholetext"]
    assert all(row["status"] == "error" for row in result["rows"])
    assert result["summary"]["wholetext"]["planned"] == 2
    assert result["summary"]["wholetext"]["answer_f1_all_planned"] == 0
    assert result["reserved_calls"] == 4
    saved = (args.output / "result.json").read_text(encoding="utf-8")
    assert "sensitive transport" not in saved
    assert all("gold_answers" not in prompt and "references" not in prompt for _, prompt in model.prompts)
    with pytest.raises(FileExistsError):
        asyncio.run(SCRIPT["run_comparison"](args, [entry], {}, model, context_builder))


def test_budget_stop_preserves_all_questions_and_skipped_pairs(tmp_path):
    entries = [make_entry(tmp_path, qid) for qid in ["a", "b"]]
    args = SimpleNamespace(output=tmp_path / "run", repeats=1, max_calls=1, max_units=20000)
    result = asyncio.run(SCRIPT["run_comparison"](args, entries, {}, Model(), context_builder))
    assert len(result["rows"]) == 4
    assert [row["status"] for row in result["rows"]] == [
        "success", "budget_exhausted", "not_attempted", "not_attempted"]
    assert result["summary"]["wholetext"]["answer_f1_all_planned"] == 0.5
    assert result["reserved_calls"] == 1
    assert not result["completed"]


def test_dry_run_never_reads_config_or_creates_output(tmp_path, capsys, monkeypatch):
    entry = make_entry(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "test", "examples": [entry]}), encoding="utf-8")
    secret_config = tmp_path / "secret-config.json"
    secret_config.write_text("sensitive", encoding="utf-8")
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == secret_config:
            raise AssertionError("Credentials were read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    output = tmp_path / "output"
    SCRIPT["main"](["--inputs", str(manifest), "--output", str(output), "--limit", "1",
                    "--config", str(secret_config), "--prompt-key"])
    assert json.loads(capsys.readouterr().out)["planned_json_calls"] == 4
    assert not output.exists()
    with pytest.raises(SystemExit):
        SCRIPT["main"](["--inputs", str(manifest), "--output", str(output), "--live"])


@pytest.mark.parametrize("evidence_strategy", ["evidence", "evidence_fast"])
def test_production_smoke_shares_ledger_and_verifies_real_offsets(tmp_path, evidence_strategy):
    entry = make_entry(tmp_path)
    output = tmp_path / "output"
    (output / "inputs").mkdir(parents=True)
    (output / "inputs" / "01.md").write_text(entry["text"], encoding="utf-8")
    args = SimpleNamespace(output=output, production_evidence_strategy=evidence_strategy)
    raw = Model()
    raw.settings = Settings(llm_max_output_tokens=512)
    budgeted = SCRIPT["BudgetedModel"](raw, 4, 20000, output / "budget.json")
    observed_caps = []

    async def summarize(idea, paper, *, settings, llm):
        observed_caps.append(raw.settings.llm_max_output_tokens)
        assert settings.research_profile == "current"
        assert paper.metadata["evidence_query"] == entry["question"]
        await llm.json("production", "source")
        settings.summaries_dir.mkdir(parents=True)
        report = settings.summaries_dir / "note.md"
        report.write_text("结论为42。", encoding="utf-8")
        evidence = []
        if settings.paper_analysis_strategy == evidence_strategy:
            evidence = [EvidenceReference(paper_id=paper.paper_id, quote="42 points",
                                          start_char=15, end_char=24)]
        return PaperSummary(paper_id=paper.paper_id, title=paper.title,
                            overview="概览", deep_analysis="结论为42。", evidence=evidence,
                            summary_path=str(report))

    async def exercise():
        await budgeted.json("QA", "source")
        await budgeted.json("QA", "source")
        rows = [{"question_id": entry["question_id"], "repeat": 0, "strategy": strategy,
                 "status": "success", "answer_f1": 1, "total_seconds": 1} for strategy in SCRIPT["STRATEGIES"]]
        result = {"rows": rows}
        await SCRIPT["run_production_smoke"](args, [entry], raw, budgeted, result,
                                              output / "result.json", summarize)
        return result

    result = asyncio.run(exercise())
    assert observed_caps == [2048, 2048]
    assert raw.settings.llm_max_output_tokens == 512
    assert budgeted.calls == result["reserved_calls"] == 4
    smoke = result["production_smoke"]
    assert smoke["completed"]
    assert smoke["evidence_strategy"] == evidence_strategy
    assert smoke["rows"][1]["strategy"] == evidence_strategy
    assert smoke["rows"][1]["verified_quote_count"] == 1
    assert smoke["rows"][1]["all_quote_offsets_exact"]
    assert smoke["rows"][1]["raw_output"] == {"answer": "42 points", "unanswerable": False}


def test_production_only_skips_qa_and_keeps_rejected_model_json(tmp_path, monkeypatch, capsys):
    from paper_agent.agents import research_agent

    entry = make_entry(tmp_path)
    args = SimpleNamespace(output=tmp_path / "output", repeats=2, max_calls=8, max_units=700000,
                           production_only=True, production_smoke=True)
    raw = Model()
    raw.settings = Settings(llm_max_output_tokens=512)

    async def invalid_summary(idea, paper, *, settings, llm):
        await llm.json("production", "source")
        return PaperSummary(paper_id=paper.paper_id, errors=[
            ErrorInfo(code="invalid_model_output", message="invalid", stage="summarize")])

    def unexpected_context(*args, **kwargs):
        raise AssertionError("QA context builder was invoked")

    monkeypatch.setattr(research_agent, "summarize_paper", invalid_summary)
    result = asyncio.run(SCRIPT["run_comparison"](args, [entry], {}, raw, unexpected_context))
    assert not result["qa_executed"]
    assert result["rows"] == [] and result["completed"] is None
    assert result["reserved_calls"] == 2
    assert all(row["status"] == "error" and row["raw_output"] ==
               {"answer": "42 points", "unanswerable": False}
               for row in result["production_smoke"]["rows"])
    capsys.readouterr()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "test", "examples": [entry]}), encoding="utf-8")
    SCRIPT["main"](["--inputs", str(manifest), "--output", str(tmp_path / "dry"), "--limit", "1",
                    "--production-only", "--max-calls", "8", "--max-units", "700000"])
    assert json.loads(capsys.readouterr().out)["planned_json_calls"] == 2


def test_three_routes_balance_order_dispatch_methods_and_keep_pairwise_results(tmp_path):
    entries = [make_entry(tmp_path, qid) for qid in ["a", "b", "c"]]
    strategies = ["wholetext", "evidence_context", "cascade"]
    args = SimpleNamespace(output=tmp_path / "three", repeats=2, max_calls=18,
                           max_units=40000, strategies=strategies)
    methods = []

    def build(question, text, *, max_bytes, method):
        assert max_bytes == 12000
        methods.append(method)
        return "42 points", {"actual_method": method}

    result = asyncio.run(SCRIPT["run_comparison"](args, entries, {}, Model(), build))
    assert result["completed"] and result["reserved_calls"] == 18
    assert methods.count("hybrid") == methods.count("cascade") == 6
    schedule = result["order_schedule"]
    for index in range(3):
        assert schedule[index + 3]["strategies"] == list(reversed(schedule[index]["strategies"]))
        assert {order["strategies"][index] for order in schedule[:3]} == set(strategies)
    assert set(result["summary"]["pairwise"]) == {
        "evidence_context_vs_wholetext", "cascade_vs_wholetext", "cascade_vs_evidence_context"}
    assert all(pair["planned_pairs"] == pair["successful_pairs"] == 6
               for pair in result["summary"]["pairwise"].values())
    assert all(pair["question_level"]["planned_questions"] == 3
               for pair in result["summary"]["pairwise"].values())
    assert result["summary"]["paired"] == result["summary"]["pairwise"]["evidence_context_vs_wholetext"]
    assert result["summary"]["cascade"]["answer_f1_all_planned"] == 1


def test_paired_statistics_aggregate_repeats_before_comparing_questions():
    rows = []
    for question_id, repeat, baseline, candidate in [("a", 0, 0, 1), ("a", 1, 0, 1),
                                                    ("b", 0, 1, 0)]:
        for strategy, score in [("wholetext", baseline), ("cascade", candidate)]:
            rows.append({"question_id": question_id, "repeat": repeat, "strategy": strategy,
                         "status": "success", "answer_f1": score, "total_seconds": 1})
    summary = SCRIPT["paired_summary"](rows, "wholetext", "cascade")
    assert summary["planned_pairs"] == 3
    assert summary["mean_answer_f1_difference_all_planned"] == pytest.approx(1 / 3)
    questions = summary["question_level"]
    assert questions["planned_questions"] == 2
    assert questions["mean_answer_f1_difference_all_questions"] == 0
    assert questions["answer_f1_wins_ties_losses"] == {"wins": 1, "ties": 0, "losses": 1}
    assert [row["planned_repeats"] for row in questions["rows"]] == [2, 1]


def test_extended_cli_budget_and_three_route_dry_run(tmp_path, capsys):
    entry = make_entry(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "test", "examples": [entry]}), encoding="utf-8")
    base = ["--inputs", str(manifest), "--output", str(tmp_path / "unused"), "--limit", "1",
            "--max-calls", "80", "--max-units", "3000000"]
    SCRIPT["main"]([*base, "--strategies", "wholetext", "evidence_context", "cascade",
                    "--production-smoke", "--production-evidence-strategy", "evidence_fast"])
    output = json.loads(capsys.readouterr().out)
    assert output["planned_json_calls"] == 8
    assert output["strategies"] == ["wholetext", "evidence_context", "cascade"]
    assert output["production_evidence_strategy"] == "evidence_fast"
    assert not (tmp_path / "unused").exists()
    with pytest.raises(SystemExit):
        SCRIPT["main"]([*base, "--strategies", "wholetext", "wholetext"])
    with pytest.raises(SystemExit):
        SCRIPT["main"]([*base, "--max-calls", "81"])
