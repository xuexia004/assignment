"""Independently check official annotation precedence and immutable rescoring."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/rescore_evidence_answers.py"))


def annotation(**values):
    return {"unanswerable": False, "extractive_spans": [], "free_form_answer": "",
            "yes_no": None, **values}


def test_official_reference_precedence_and_token_normalization():
    refs = SCRIPT["reference_answers"]([
        annotation(unanswerable=True, extractive_spans=["Ignored"], yes_no=True),
        annotation(extractive_spans=["X", "Y"], free_form_answer="Ignored", yes_no=False),
        annotation(free_form_answer="Text", yes_no=True),
        annotation(yes_no=False), annotation(yes_no=True),
    ])
    assert refs == [{"answer": "Unanswerable", "type": "none"},
                    {"answer": "X, Y", "type": "extractive"},
                    {"answer": "Text", "type": "abstractive"},
                    {"answer": "No", "type": "boolean"},
                    {"answer": "Yes", "type": "boolean"}]
    score = SCRIPT["score_answer"]
    assert score("the x y!", refs) == 1
    assert score("Unanswerable", refs) == 1
    assert score("x", refs) == pytest.approx(2 / 3)
    assert score("", [{"answer": "", "type": "extractive"}]) == 0
    assert score("Newsroom", [{"answer": "dataset Newsroom BIBREF16", "type": "extractive"}]) == .5
    with pytest.raises(ValueError):
        SCRIPT["reference_answers"]([annotation()])


def fixture(tmp_path):
    text = "The exact dimensions are INLINEFORM10."
    source = tmp_path / "source.md"
    source.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"inputs": [{"question_id": "q", "question": "What size?", "text_path": str(source),
                        "input_sha256": digest,
                        "gold_answers": [annotation(free_form_answer="1x3", evidence=[text]),
                                         annotation(unanswerable=True)]}],
            "rows": [{"question_id": "q", "input_index": 0, "repeat": 0, "strategy": "candidate",
                      "status": "success", "answer_f1": 0.0, "output": {"answer": "Unanswerable"},
                      "context_path": str(source), "context_sha256": digest,
                      "audit": {"selected_spans": [{"id": "P1", "start": 0, "end": len(text),
                                                     "text": text}]}}]}


def test_corrects_abstention_without_changing_original_or_failed_denominator(tmp_path):
    result = fixture(tmp_path)
    result["rows"].append({**result["rows"][0], "repeat": 1, "status": "error"})
    before = copy.deepcopy(result)
    corrected = SCRIPT["rescore"](result)
    assert result == before
    assert corrected["summary"]["candidate"] == {
        "planned": 2, "successful": 1, "legacy_f1": 0.0, "corrected_f1": .5, "changed_rows": 1}
    assert SCRIPT["evidence_audit"](result)[0]["contexts"][0]["evidence_matches"][0]["span_ids"] == ["P1"]
    result["rows"][0]["audit"]["selected_spans"][0]["start"] = 1
    with pytest.raises(ValueError, match="span"):
        SCRIPT["evidence_audit"](result)


def test_cli_preserves_input_and_rejects_overwrite_or_modified_source(tmp_path):
    result = fixture(tmp_path)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    original_bytes = path.read_bytes()
    destination = tmp_path / "rescored"
    args = ["--input", str(path), "--output-dir", str(destination)]
    SCRIPT["main"](args)
    assert path.read_bytes() == original_bytes
    saved = json.loads((destination / "corrected-scores.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["legacy_f1"] == 0 and saved["rows"][0]["corrected_f1"] == 1
    with pytest.raises(SystemExit):
        SCRIPT["main"](args)
    Path(result["inputs"][0]["text_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        SCRIPT["main"](["--input", str(path), "--output-dir", str(tmp_path / "another")])
