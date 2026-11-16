"""Independent-process cold and warm-model latency for fixed public paper contexts.

Every worker uses one method and a new process. A separate cold-document probe
loads its required models. Vector memoization is cleared before every warm case.
No credentials, network calls or answer generation are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
METHOD_PATH = ROOT / "src/paper_agent/agents/evidence_context.py"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_examples(path: Path, limit: int, reverse: bool = False) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    examples = sorted(manifest["examples"], key=lambda entry: entry["question_id"])[:limit]
    if len(examples) < limit:
        raise ValueError("Not enough examples in frozen input")
    for entry in examples:
        text = Path(entry["text_path"]).read_text(encoding="utf-8")
        if digest(text.encode("utf-8")) != entry["input_sha256"]:
            raise ValueError("Input hash mismatch")
        entry["text"] = text
    return list(reversed(examples)) if reverse else examples


def worker(args) -> None:
    from paper_agent.agents.evidence_context import build_evidence_context
    from paper_agent.services import semantic_reranker

    examples = load_examples(args.inputs, args.limit, args.reverse)
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    cold_started = time.perf_counter()
    _, cold_audit = build_evidence_context(examples[0]["question"], examples[0]["text"],
                                           max_bytes=args.max_bytes, method=args.worker)
    cold_probe = {"question_id": examples[0]["question_id"],
                  "cold_document_seconds": time.perf_counter() - cold_started,
                  "actual_method": cold_audit["actual_method"]}
    for entry in examples:
        with semantic_reranker._CACHE_LOCK:
            semantic_reranker._VECTOR_CACHE.clear()
        started = time.perf_counter()
        context, audit = build_evidence_context(entry["question"], entry["text"],
                                                max_bytes=args.max_bytes, method=args.worker)
        rows.append({"question_id": entry["question_id"], "paper_id": entry["paper_id"],
                     "input_sha256": entry["input_sha256"], "method": args.worker,
                     "model_cold": False, "elapsed_seconds": time.perf_counter() - started,
                     "context_bytes": len(context.encode("utf-8")),
                     "context_sha256": digest(context.encode("utf-8")),
                     "actual_method": audit["actual_method"], "fallback_reason": audit["fallback_reason"]})
        result = {"method_sha256": digest(METHOD_PATH.read_bytes()), "cold_probe": cold_probe, "rows": rows}
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def run(args) -> None:
    method_sha = digest(METHOD_PATH.read_bytes())
    if args.frozen_method_sha != method_sha:
        raise ValueError("Timing requires the frozen method SHA")
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    rows, processes = [], []
    for repeat in range(args.repeats):
        methods = ("hybrid", "cascade") if repeat % 2 == 0 else ("cascade", "hybrid")
        for method in methods:
            destination = args.output / f"repeat-{repeat}-{method}"
            command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                       "--worker", method, "--inputs", str(args.inputs.resolve()),
                       "--limit", str(args.limit), "--max-bytes", str(args.max_bytes),
                       "--output", str(destination.resolve())]
            if repeat % 2:
                command.append("--reverse")
            started = time.perf_counter()
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                     encoding="utf-8", timeout=600)
            processes.append({"repeat": repeat, "method": method,
                              "process_wall_seconds": time.perf_counter() - started,
                              "exit_code": process.returncode})
            (args.output / f"repeat-{repeat}-{method}.log").write_text(
                process.stdout + process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"Worker failed; see {method} log")
            result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
            if result["method_sha256"] != method_sha:
                raise ValueError("Method changed during timing")
            rows.extend({**row, "repeat": repeat} for row in result["rows"])
            processes[-1]["cold_probe"] = result["cold_probe"]
            print(json.dumps({"repeat": repeat, "method": method, "measured": len(result["rows"])}),
                  flush=True)
    summary = {}
    for method in ("hybrid", "cascade"):
        group = [row for row in rows if row["method"] == method]
        warm = [row["elapsed_seconds"] for row in group if not row["model_cold"]]
        summary[method] = {"cold_document_seconds": [process["cold_probe"]["cold_document_seconds"]
                                                   for process in processes if process["method"] == method],
                           "warm_model_new_document_median_seconds": statistics.median(warm),
                           "successful_actual_method_count": sum(row["actual_method"] == method for row in group),
                           "measured": len(group)}
    pairs = {}
    for row in rows:
        pairs.setdefault((row["question_id"], row["repeat"]), {})[row["method"]] = row
    ratios = [pair["cascade"]["elapsed_seconds"] / pair["hybrid"]["elapsed_seconds"]
              for pair in pairs.values()
              if all(not row["model_cold"] for row in pair.values())]
    summary["paired_warm"] = {"count": len(ratios), "median_cascade_over_hybrid": statistics.median(ratios)}
    result = {"method_sha256": method_sha, "manifest_sha256": digest(args.inputs.read_bytes()),
              "benchmark_sha256": digest(Path(__file__).read_bytes()),
              "timing_scope": "one method per fresh process; separate cold-document probe including model import/load; all measured questions warm models with vector cache cleared before every question",
              "process_scope": "worker process wall-clock includes Python start and imports",
              "order": "method order and paper order alternate by repeat",
              "byte_budget": args.max_bytes, "summary": summary, "rows": rows, "processes": processes}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-bytes", type=int, default=12000)
    parser.add_argument("--frozen-method-sha")
    parser.add_argument("--worker", choices=["hybrid", "cascade"])
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.limit < 2 or args.repeats < 1 or args.max_bytes < 1:
        parser.error("Need at least 2 papers, 1 repeat and a positive byte budget")
    if args.worker:
        worker(args)
    else:
        run(args)
