"""QASPER paragraph-evidence evaluation with deterministic paper-disjoint samples.

No model credentials or paid generation are used. Gold evidence is only read by
the scorer, never passed to the retrieval function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA_ROOT = ROOT / "data/validation/evidence-context-20261107"
SOURCE = "https://huggingface.co/datasets/allenai/qasper"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.split())


def document_text(paper: dict) -> str:
    """Keep original paragraphs and available figure/table captions intact."""
    parts = [f"# {paper['title']}", "## Abstract", paper["abstract"]]
    for section in paper["full_text"]:
        if section["section_name"]:
            parts.append("## " + section["section_name"])
        parts.extend(section["paragraphs"])
    captions = [item["caption"] for item in paper.get("figures_and_tables", [])]
    if captions:
        parts.extend(["## Figure and table captions", *captions])
    return "\n\n".join(part for part in parts if part)


def gold_references(question: dict, text: str) -> list[list[str]]:
    normalized = normalize(text)
    references = []
    for annotation in question["answers"]:
        answer = annotation["answer"]
        evidence = list(dict.fromkeys(normalize(item) for item in answer["evidence"]
                                      if normalize(item)))
        if (not answer["unanswerable"] and evidence
                and all(item in normalized for item in evidence)):
            references.append(evidence)
    return references


def select_examples(data: dict, limit: int, minimum_bytes: int,
                    offset: int = 0) -> tuple[list[dict], dict]:
    """One question per paper, hash ordering fixed before observing retrieval."""
    selected = []
    counts = Counter()
    for paper_id in sorted(data, key=lambda value: digest("paper-context-v1:" + value)):
        paper = data[paper_id]
        text = document_text(paper)
        counts["papers_total"] += 1
        if len(text.encode("utf-8")) < minimum_bytes:
            counts["papers_below_minimum_bytes"] += 1
            continue
        eligible = []
        for question in paper["qas"]:
            references = gold_references(question, text)
            if references:
                eligible.append((question, references))
        if not eligible:
            counts["papers_without_available_answerable_evidence"] += 1
            continue
        counts["eligible_papers"] += 1
        if counts["eligible_papers"] <= offset or len(selected) >= limit:
            continue
        question, references = min(eligible, key=lambda item: digest(item[0]["question_id"]))
        selected.append({"paper_id": paper_id, "question_id": question["question_id"],
                         "title": paper["title"], "question": question["question"],
                         "text": text, "gold_references": references,
                         "gold_answers": [annotation["answer"] for annotation in question["answers"]],
                         "input_sha256": digest(text),
                         "question_sha256": digest(question["question"])})
    return selected, dict(counts)


def score_context(context: str, references: list[list[str]]) -> dict:
    """Match complete annotated paragraphs, allowing whitespace changes only."""
    if not references or any(not reference for reference in references):
        raise ValueError("Every example requires a nonempty evidence reference")
    normalized = normalize(context)
    hits = [sum(item in normalized for item in reference) for reference in references]
    recalls = [hit / len(reference) for hit, reference in zip(hits, references, strict=True)]
    best = max(range(len(recalls)), key=recalls.__getitem__)
    return {"evidence_recall": recalls[best], "complete_evidence": recalls[best] == 1.0,
            "evidence_hits": hits[best], "evidence_count": len(references[best]),
            "reference_count": len(references), "reference_recalls": recalls}


def summarize(rows: list[dict], methods: list[str]) -> dict:
    summary = {}
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summary[method] = {
            "questions": len(group),
            "evidence_recall": statistics.mean(row["evidence_recall"] for row in group),
            "complete_evidence_rate": statistics.mean(row["complete_evidence"] for row in group),
            "median_context_bytes": statistics.median(row["context_bytes"] for row in group),
            "mean_retained_bytes_ratio": statistics.mean(row["context_bytes"] / row["input_bytes"]
                                                          for row in group),
            "median_elapsed_seconds": statistics.median(row["elapsed_seconds"] for row in group),
            "budget_fit_rate": statistics.mean(row["within_budget"] for row in group),
            "actual_methods": dict(Counter(row["actual_method"] for row in group)),
        }
    paired = {(row["question_id"], row["method"]): row for row in rows}
    for new, baseline in (("hybrid", "bm25"), ("cascade", "hybrid"), ("cascade", "bm25")):
        if new not in methods or baseline not in methods:
            continue
        ids = [row["question_id"] for row in rows if row["method"] == new]
        differences = [paired[(qid, new)]["evidence_recall"]
                       - paired[(qid, baseline)]["evidence_recall"] for qid in ids]
        rng = random.Random(20261107)
        estimates = sorted(statistics.mean(rng.choices(differences, k=len(differences)))
                           for _ in range(2000))
        summary[f"{new}_vs_{baseline}"] = {
            "paired_mean_recall_difference": statistics.mean(differences),
            "paired_bootstrap_95_ci": [estimates[49], estimates[1949]],
            "improved": sum(value > 0 for value in differences),
            "regressed": sum(value < 0 for value in differences),
            "unchanged": sum(value == 0 for value in differences),
        }
    return summary


def run(args: argparse.Namespace) -> None:
    data_path = args.data_root / f"qasper-{args.split}-v0.3.json"
    raw = data_path.read_text(encoding="utf-8")
    examples, counts = select_examples(json.loads(raw), args.limit, args.minimum_bytes, args.offset)
    if len(examples) < args.limit:
        raise ValueError(f"Only {len(examples)} eligible papers for requested {args.limit}")
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve earlier observations")
    args.output.mkdir(parents=True)
    input_dir = args.output / "inputs"
    input_dir.mkdir()
    records = []
    for example in examples:
        path = input_dir / f"{example['paper_id']}.md"
        path.write_text(example["text"], encoding="utf-8")
        records.append({key: value for key, value in example.items() if key != "text"}
                       | {"text_path": str(path.resolve())})
    manifest = {"source": SOURCE, "dataset_version": "0.3", "license": "CC BY 4.0",
                "source_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "split": args.split, "byte_budget": args.max_bytes,
                "minimum_document_bytes": args.minimum_bytes,
                "eligible_offset": args.offset,
                "selection": "paper-context-v1 SHA256 order; one eligible question per paper in question-id SHA256 order",
                "eligibility": "answerable; at least one nonempty evidence reference fully present in supplied text",
                "selection_counts": counts, "examples": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"selected": len(examples), "counts": counts}))
        return

    from paper_agent.agents.evidence_context import build_evidence_context

    method_path = ROOT / "src/paper_agent/agents/evidence_context.py"
    method_sha = hashlib.sha256(method_path.read_bytes()).hexdigest()
    if args.split == "test" and args.frozen_method_sha != method_sha:
        raise ValueError("Test evaluation requires the reviewed method SHA256")
    rows = []
    result = {"manifest_sha256": digest(json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
              "method_sha256": method_sha, "method_names": args.methods,
              "scope": "supplied-text paragraph retrieval, no LLM answer or PDF quality claims",
              "timing_scope": "sequential local wall-clock; shared model/cache; first call includes initialization",
              "byte_budget": args.max_bytes, "rows": rows}
    for example in examples:
        for method in args.methods:
            start = time.perf_counter()
            if method == "full":
                context, audit = example["text"], {"actual_method": "full"}
            elif method == "head":
                context = example["text"].encode("utf-8")[:args.max_bytes].decode("utf-8", "ignore")
                audit = {"actual_method": "head"}
            else:
                context, audit = build_evidence_context(example["question"], example["text"],
                                                        max_bytes=args.max_bytes, method=method)
            elapsed = time.perf_counter() - start
            context_path = args.output / f"{example['question_id']}-{method}.md"
            context_path.write_text(context, encoding="utf-8")
            row = {"paper_id": example["paper_id"], "question_id": example["question_id"],
                   "method": method, "actual_method": audit.get("actual_method", method),
                   "input_sha256": example["input_sha256"],
                   "input_bytes": len(example["text"].encode("utf-8")),
                   "context_sha256": digest(context), "context_bytes": len(context.encode("utf-8")),
                   "within_budget": len(context.encode("utf-8")) <= args.max_bytes,
                   "elapsed_seconds": elapsed, "context_path": str(context_path.resolve()),
                   "audit": audit, **score_context(context, example["gold_references"])}
            rows.append(row)
        result["summary"] = summarize(rows, args.methods)
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        print(json.dumps({"completed": len(rows) // len(args.methods), "paper_id": example["paper_id"],
                          "recall": {name: values.get("evidence_recall")
                                     for name, values in result["summary"].items()
                                     if name in args.methods}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many eligible papers in fixed hash order")
    parser.add_argument("--minimum-bytes", type=int, default=24000)
    parser.add_argument("--max-bytes", type=int, default=12000)
    parser.add_argument("--methods", nargs="+", choices=["full", "head", "bm25", "dense", "hybrid", "cascade"],
                        default=["full", "head", "bm25", "dense", "hybrid"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--frozen-method-sha")
    arguments = parser.parse_args()
    if arguments.limit < 1 or arguments.max_bytes < 1 or arguments.minimum_bytes < 1 or arguments.offset < 0:
        parser.error("Limits and budgets must be positive")
    if len(arguments.methods) != len(set(arguments.methods)):
        parser.error("Methods must be unique")
    run(arguments)
