"""Scoring parity, source-bound citations and complete budget accounting."""

import asyncio
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent.llm_client import LLMOutputError, LLMUsage

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/benchmark_quality_answers.py"))


def annotation(**values):
    return {"unanswerable": False, "extractive_spans": [], "free_form_answer": "", "yes_no": None, **values}


def entry(tmp_path, qid="q1"):
    text = "# Results\n\nThe result was 42 points."
    question = "What was the result?"
    path = tmp_path / f"{qid}.md"
    path.write_text(text, encoding="utf-8")
    answers = [annotation(extractive_spans=["42 points"])]
    return {"paper_id": "paper-" + qid, "question_id": qid, "title": "Example", "text": text,
            "question": question, "text_path": str(path), "gold_answers": answers,
            "references": SCRIPT["reference_answers"](answers),
            "input_sha256": SCRIPT["digest"](text), "question_sha256": SCRIPT["digest"](question)}


def builder(text):
    return "[P1]\n" + text, {"actual_method": "fixture_fulltext", "selected_spans": [
        {"id": "P1", "text": text, "start": 0, "end": len(text)}]}


def cascade(question, text, *, max_bytes, method):
    assert max_bytes == 12000 and method == "cascade"
    context, audit = builder(text)
    return context, {**audit, "actual_method": "cascade"}


class Model:
    def __init__(self, *, invalid_quote=False):
        self.settings = SimpleNamespace(llm_model="fixture", llm_max_output_tokens=512)
        self.usage = LLMUsage()
        self.call_log, self.prompts = [], []
        self.invalid_quote = invalid_quote

    async def json(self, system, user):
        self.prompts.append((system, user))
        self.usage.requests += 1
        self.usage.successful_requests += 1
        self.usage.prompt_tokens += 100
        self.usage.completion_tokens += 20
        self.usage.total_tokens += 120
        self.call_log.append({"http_status": 200})
        output = {"answer": "42 points", "unanswerable": False}
        if "citations (a list" in system:
            output["citations"] = [{"id": "P1", "quote": "invented" if self.invalid_quote else "42 points"}]
        return output


def test_official_all_annotation_precedence_and_normalization():
    refs = SCRIPT["reference_answers"]([
        annotation(unanswerable=True, extractive_spans=["must not take priority"]),
        annotation(extractive_spans=["X", "Y"], free_form_answer="Z", yes_no=True),
        annotation(free_form_answer="A paraphrase", yes_no=True),
        annotation(yes_no=True), annotation(yes_no=False),
    ])
    assert refs == [{"answer": "Unanswerable", "type": "none"},
                    {"answer": "X, Y", "type": "extractive"},
                    {"answer": "A paraphrase", "type": "abstractive"},
                    {"answer": "Yes", "type": "boolean"}, {"answer": "No", "type": "boolean"}]
    assert SCRIPT["score_answer"]("Unanswerable", refs) == 1
    assert SCRIPT["token_f1"]("The X, Y!", "X Y") == 1
    assert SCRIPT["token_f1"]("", "") == 0
    assert SCRIPT["token_f1"]("42", "42 BIBREF16") == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        SCRIPT["reference_answers"]([annotation()])


def test_selection_preserves_manifest_order_and_rejects_duplicate_papers(tmp_path):
    entries = [entry(tmp_path, qid) for qid in ("z", "a")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "dev", "evaluation_role": "holdout", "examples": entries}), encoding="utf-8")
    selected, _ = SCRIPT["selected_inputs"](manifest)
    assert [item["question_id"] for item in selected] == ["z", "a"]
    entries[1]["paper_id"] = entries[0]["paper_id"]
    manifest.write_text(json.dumps({"split": "dev", "evaluation_role": "holdout", "examples": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        SCRIPT["selected_inputs"](manifest)


def test_quotes_need_exact_original_span_and_answerable_needs_quote():
    source = "The result was 42 points."
    _, audit = builder(source)
    raw = {"citations": [{"id": "P1", "quote": "42 points"}], "answer": "42 points", "unanswerable": False}
    checks = SCRIPT["check_citations"](raw, audit, source)
    assert checks == [{"valid": True, "id": "P1", "quote": "42 points", "start_char": 15, "end_char": 24}]
    assert SCRIPT["grounded_answer"](raw, checks)["answer"] == "42 points"
    for citation in ({"id": "P2", "quote": "42 points"}, {"id": "P1", "quote": "42.0 points"}):
        raw["citations"] = [citation]
        checks = SCRIPT["check_citations"](raw, audit, source)
        with pytest.raises(LLMOutputError):
            SCRIPT["grounded_answer"](raw, checks)
    raw["citations"] = []
    with pytest.raises(LLMOutputError):
        SCRIPT["grounded_answer"](raw, [])
    raw.update(unanswerable=True, answer="unsupported")
    assert SCRIPT["grounded_answer"](raw, [])["answer"] == "Unanswerable"


def test_runner_retains_invalid_raw_output_no_repair_and_never_sends_gold(tmp_path):
    args = SimpleNamespace(output=tmp_path / "run", strategies=["wholetext", "grounded"],
                           max_calls=2, max_units=20000, implementation_sha256="fixture")
    model = Model(invalid_quote=True)
    result = asyncio.run(SCRIPT["run_comparison"](args, [entry(tmp_path)], {}, model, cascade, builder))
    assert [row["status"] for row in result["rows"]] == ["success", "error"]
    assert result["rows"][1]["raw_output"]["citations"][0]["quote"] == "invented"
    assert result["rows"][1]["answer_f1"] == 0
    assert result["reserved_calls"] == 2 and len(model.prompts) == 2
    assert all(set(json.loads(prompt)) == {"question", "source"} for _, prompt in model.prompts)
    assert list(json.loads(model.prompts[1][1])) == ["source", "question"]
    with pytest.raises(FileExistsError):
        asyncio.run(SCRIPT["run_comparison"](args, [entry(tmp_path)], {}, model, cascade, builder))


def test_budget_stop_keeps_all_planned_slots_and_confidence_rejects_incomplete(tmp_path):
    args = SimpleNamespace(output=tmp_path / "run", strategies=["wholetext", "grounded"],
                           max_calls=1, max_units=20000, implementation_sha256="fixture")
    result = asyncio.run(SCRIPT["run_comparison"](args, [entry(tmp_path, "a"), entry(tmp_path, "b")],
                                                 {}, Model(), cascade, builder))
    assert [row["status"] for row in result["rows"]] == ["success", "budget_exhausted", "not_attempted", "not_attempted"]
    assert result["reserved_calls"] == 1
    assert result["summary"]["wholetext"]["answer_f1_all_planned"] == 0.5
    assert not result["confidence"]["grounded_vs_wholetext"]["quality_numeric_gate"]


def test_dry_run_does_not_read_credentials_or_create_output(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "train", "evaluation_role": "development",
                                    "examples": [entry(tmp_path)]}), encoding="utf-8")
    config = tmp_path / "private-config.json"
    config.write_text("not JSON: must not be read", encoding="utf-8")
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path == config:
            raise AssertionError("read credential file during dry run")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    output = tmp_path / "new-output"
    SCRIPT["main"](["--inputs", str(manifest), "--output", str(output), "--config", str(config), "--prompt-key"])
    assert json.loads(capsys.readouterr().out)["planned_json_calls"] == 4
    assert not output.exists()


def test_three_consecutive_provider_failures_stop_without_retries(tmp_path):
    class UnavailableModel(Model):
        async def json(self, system, user):
            self.prompts.append((system, user))
            self.usage.requests += 1
            self.usage.failed_requests += 1
            event = ({"status": "http_error", "http_status": 500},
                     {"status": "http_error", "http_status": 503},
                     {"status": "transport_error", "error_type": "ConnectError"})[len(self.call_log)]
            self.call_log.append(event)
            raise RuntimeError("provider failure")

    args = SimpleNamespace(output=tmp_path / "run", strategies=["wholetext", "grounded"],
                           max_calls=6, max_units=20000, implementation_sha256="fixture")
    model = UnavailableModel()
    result = asyncio.run(SCRIPT["run_comparison"](
        args, [entry(tmp_path, qid) for qid in ("a", "b", "c")], {}, model, cascade, builder))
    assert len(model.prompts) == result["reserved_calls"] == 3
    assert result["stop_reason"] == "provider_failure_circuit_open"
    assert [row["status"] for row in result["rows"]] == ["error"] * 3 + ["not_attempted"] * 3
    assert not result["completed"]
