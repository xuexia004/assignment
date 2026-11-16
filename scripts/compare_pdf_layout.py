"""Compare whole-paper layout extraction on fixed local public PDFs; no keys/LLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf  # noqa: E402
import pymupdf4llm  # noqa: E402

from paper_agent.services.pdf_converter import (  # noqa: E402
    _default_parser,
    _layout_parser,
    _page_chunks,
)


def numbers(text):
    return Counter(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)", text))


def compare(paths, output, repeats):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ValueError("Choose a fresh output directory to preserve prior measurements")
    rows = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pymupdf.open(path) as document:
            source = "\n".join(page.get_text() for page in document)
            title, count = document.metadata.get("title", ""), len(document)
        source_numbers = numbers(source)
        for repeat in range(repeats):
            order = ["legacy", "layout"] if repeat % 2 == 0 else ["layout", "legacy"]
            for parser in order:
                started = time.perf_counter()
                with pymupdf.open(path) as document:
                    chunks = (_default_parser if parser == "legacy" else _layout_parser)(document)
                pages = _page_chunks(chunks, count)
                text = "\n\n".join(pages.get(i, "") for i in range(1, count + 1))
                elapsed = time.perf_counter() - started
                target = output / f"{digest[:12]}-{parser}-{repeat}.md"
                target.write_text(text, encoding="utf-8")
                headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", text)
                row = {"path": str(path.resolve()), "title": title, "pdf_sha256": digest,
                       "parser": parser, "repeat": repeat, "elapsed_seconds": elapsed,
                       "page_count": count, "extracted_pages": len(pages), "chars": len(text),
                       "headings": headings, "markdown_table_rows": len(re.findall(r"(?m)^\|.*\|\s*$", text)),
                       "source_numeric_occurrences": sum(source_numbers.values()),
                       "numeric_occurrence_recall_heuristic": sum((source_numbers & numbers(text)).values()) / max(1, sum(source_numbers.values())),
                       "artifact": str(target.resolve())}
                rows.append(row)
                (output / "result.json").write_text(json.dumps({"paid_calls": 0,
                    "scope": "local_pdf_extraction_structure_not_llm_comprehension_or_semantic_accuracy",
                    "version": pymupdf4llm.__version__, "ocr": False, "rows": rows},
                    ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({k: row[k] for k in ["title", "parser", "repeat", "elapsed_seconds", "chars", "numeric_occurrence_recall_heuristic"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/opensource-followup-20261106/pdf-layout")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    compare(args.paths, args.output, args.repeats)
