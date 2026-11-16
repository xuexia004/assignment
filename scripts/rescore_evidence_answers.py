"""Rescore saved QA answers against every QASPER annotation, with no model calls.

Original measurements remain immutable. A new directory receives corrected and
legacy scores together, plus source/context integrity and evidence-presence audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_quality_answers import reference_answers, score_answer  # noqa: E402

OFFICIAL_SCORER = "https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rescore(result: dict) -> dict:
    """Retain every planned QA slot; failed calls still receive zero credit."""
    inputs = result["inputs"]
    references = [reference_answers(entry["gold_answers"]) for entry in inputs]
    rows, seen = [], set()
    for row in result["rows"]:
        entry = inputs[row["input_index"]]
        if row["question_id"] != entry["question_id"]:
            raise ValueError("QA row and input question disagree")
        key = (row["question_id"], row["repeat"], row["strategy"])
        if key in seen:
            raise ValueError("Duplicate QA slot")
        seen.add(key)
        answer = row.get("output", {}).get("answer", "")
        if row["status"] == "success" and not isinstance(answer, str):
            raise ValueError("Successful QA answer must be text")
        corrected = (score_answer(answer, references[row["input_index"]])
                     if row["status"] == "success" else 0.0)
        rows.append({key: row[key] for key in
                     ("question_id", "input_index", "repeat", "strategy", "status")}
                    | {"answer": answer, "legacy_f1": row["answer_f1"],
                       "corrected_f1": corrected})
    summary = {}
    for strategy in dict.fromkeys(row["strategy"] for row in rows):
        group = [row for row in rows if row["strategy"] == strategy]
        summary[strategy] = {
            "planned": len(group), "successful": sum(row["status"] == "success" for row in group),
            "legacy_f1": statistics.mean(row["legacy_f1"] for row in group),
            "corrected_f1": statistics.mean(row["corrected_f1"] for row in group),
            "changed_rows": sum(row["legacy_f1"] != row["corrected_f1"] for row in group),
        }
    return {"scope": "Saved QA rows only; no model calls or new method results; production smoke excluded",
            "protocol": "Official max token F1 over all annotations, including Unanswerable",
            "official_scorer": OFFICIAL_SCORER,
            "independent_questions": len({row["question_id"] for row in rows}),
            "references": [{"question_id": entry["question_id"], "answers": refs}
                           for entry, refs in zip(inputs, references, strict=True)],
            "summary": summary, "rows": rows}


def evidence_audit(result: dict) -> list[dict]:
    """Check saved text/offsets; exact evidence presence is not answer sufficiency."""
    audited = []
    for index, entry in enumerate(result["inputs"]):
        text = Path(entry["text_path"]).read_text(encoding="utf-8")
        if digest(text) != entry["input_sha256"]:
            raise ValueError("Original text hash differs")
        group = [row for row in result["rows"] if row["input_index"] == index]
        evidence = list(dict.fromkeys(fragment for annotation in entry["gold_answers"]
                                     for fragment in annotation.get("evidence", [])))
        contexts = []
        for row in group:
            if row["status"] != "success":
                continue
            context = Path(row["context_path"]).read_text(encoding="utf-8")
            if digest(context) != row["context_sha256"]:
                raise ValueError("Saved context hash differs")
            spans = row.get("audit", {}).get("selected_spans", [])
            if any(text[span["start"]:span["end"]] != span["text"] for span in spans):
                raise ValueError("Selected span differs from original text")
            contexts.append({
                "strategy": row["strategy"], "repeat": row["repeat"],
                "context_sha256": row["context_sha256"],
                "selected_span_ids": [span["id"] for span in spans],
                "evidence_matches": [{"evidence_index": i, "present": fragment in context,
                                      "span_ids": [span["id"] for span in spans
                                                   if fragment in span["text"]]}
                                     for i, fragment in enumerate(evidence)],
            })
        audited.append({
            "question_id": entry["question_id"], "question": entry["question"],
            "input_sha256": entry["input_sha256"],
            "unanswerable_annotations": sum(bool(a["unanswerable"]) for a in entry["gold_answers"]),
            "annotation_count": len(entry["gold_answers"]),
            "evidence": [{"index": i, "sha256": digest(fragment),
                          "present_in_fulltext": fragment in text}
                         for i, fragment in enumerate(evidence)], "contexts": contexts,
        })
    return audited


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a new directory")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output exists; preserve all prior artifacts by using a new directory")
    source = args.input.read_bytes()
    original = json.loads(source)
    corrected = rescore(original)
    corrected["source_path"] = str(args.input.resolve())
    corrected["source_sha256"] = hashlib.sha256(source).hexdigest()
    audit = {"source_sha256": corrected["source_sha256"], "official_scorer": OFFICIAL_SCORER,
             "interpretation": "Evidence presence alone does not establish that answer details survived extraction",
             "questions": evidence_audit(original)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("corrected-scores.json", corrected), ("audit.json", audit)):
        with (args.output_dir / name).open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
    print(json.dumps(corrected["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
