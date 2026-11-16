"""Reviewable paper-internal comparison; live runs require --live --yes."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_agent.agents.research_agent import summarize_paper  # noqa: E402
from paper_agent.agents.section_analysis import MIN_SECTION_CHARS, split_document  # noqa: E402
from paper_agent.config import Settings  # noqa: E402
from paper_agent.llm_client import ChatClient, LLMBudgetError  # noqa: E402
from paper_agent.models import Paper  # noqa: E402
from paper_agent.service_control import default_data_root  # noqa: E402


class BudgetedModel:
    """Reserve UTF-8 input units plus output capacity before any paid request."""
    def __init__(self, model, max_calls, max_units, ledger_path=None):
        self.model, self.max_calls, self.max_units = model, max_calls, max_units
        self.calls = self.reserved_units = 0
        self.ledger_path = ledger_path
        self.lock = asyncio.Lock()

    async def json(self, system, user):
        units = len((system + user).encode("utf-8")) + 64 + self.model.settings.llm_max_output_tokens
        async with self.lock:
            if self.calls >= self.max_calls or self.reserved_units + units > self.max_units:
                raise LLMBudgetError("Reviewed comparison budget exhausted")
            self.calls += 1
            self.reserved_units += units
            if self.ledger_path:
                self.ledger_path.write_text(json.dumps({"reserved_calls": self.calls,
                    "reserved_units": self.reserved_units, "max_calls": self.max_calls,
                    "max_units": self.max_units}), encoding="utf-8")
        return await self.model.json(system, user)


def selected_inputs(path):
    entries = json.loads(path.read_text(encoding="utf-8"))
    entries = sorted((e for e in entries if e["chars"] >= MIN_SECTION_CHARS), key=lambda e: e["chars"])
    if len(entries) < 3:
        raise ValueError("Need three complete, untruncated long-paper snapshots")
    return [entries[0], entries[len(entries) // 2], entries[-1]]


async def run(args, inputs):
    # Read credentials only after both explicit live switches; no Workbench/history side effects.
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    if args.prompt_key:
        values["api_key"] = getpass.getpass("Model API key (hidden): ")
    base = Settings(mode="live", llm_base_url=values.get("base_url", "https://api.deepseek.com"),
                    llm_model=values.get("model", "deepseek-flash"), llm_api_key=values.get("api_key", ""),
                    http_proxy=values.get("http_proxy", ""), llm_thinking_enabled=False,
                    llm_timeout=180, llm_max_retries=0, llm_concurrency=2,
                    llm_max_calls=args.max_calls, llm_max_total_tokens=args.max_units,
                    data_dir=args.output / "artifacts", output_dir=args.output / "runs")
    raw_model = ChatClient(base)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    model = BudgetedModel(raw_model, args.max_calls, args.max_units, args.output / "budget.json")
    for repeat in range(args.repeats):
        for entry in inputs:
            path = Path(entry["path"])
            text = path.read_text(encoding="utf-8")
            order = ["direct", "sections"] if repeat % 2 == 0 else ["sections", "direct"]
            for strategy in order:
                paper = Paper(paper_id=entry["paper_id"], title=entry["title"], abstract=entry["abstract"],
                              markdown_path=str(path), fulltext_quality={"complete": True},
                              metadata={"source": entry["source"], "dataset_text_not_independently_checked_pdf": True})
                before = raw_model.usage.to_dict()
                started = time.perf_counter()
                trial = replace(base, paper_analysis_strategy=strategy,
                                summaries_dir=args.output / "artifacts" / f"repeat-{repeat}" / strategy)
                summary = await summarize_paper("说明论文主要方法、适用假设、关键对照与实验数字，并指出局限。", paper,
                                                settings=trial, llm=model)
                after = raw_model.usage.to_dict()
                row = {"paper_id": paper.paper_id, "title": paper.title, "repeat": repeat,
                       "strategy": strategy, "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                       "elapsed_seconds": time.perf_counter() - started,
                       "scope": summary.summary_scope, "summary_path": summary.summary_path,
                       "errors": [e.model_dump(mode="json") for e in summary.errors],
                       "usage": {k: after[k] - before[k] for k in after},
                       "audit": paper.metadata["research_audit"]}
                rows.append(row)
                result = {"live": True, "model": base.llm_model, "thinking": False,
                          "scope": "fixed_public_dataset_text_paper_analysis_not_end_to_end_or_quality_proof",
                          "max_calls": args.max_calls, "max_conservative_units": args.max_units,
                          "reserved_units": model.reserved_units, "usage": after, "rows": rows}
                (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({k: row[k] for k in ("paper_id", "repeat", "strategy", "elapsed_seconds", "scope")}), flush=True)
                if (any(e.code == "llm_budget" for e in summary.errors)
                        or any(c.get("http_status") in {401, 403, 404} for c in raw_model.call_log)):
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/validation/analysis-inputs-20261106/manifest.json")
    parser.add_argument("--config", type=Path, default=default_data_root() / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/sections-live-20261106")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument("--max-units", type=int, default=1000000)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--prompt-key", action="store_true", help="Read a transient key with terminal echo disabled")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 1 <= args.max_calls <= 30 or not 0 < args.max_units <= 1000000:
        parser.error("repeats=1..5, max-calls=1..30, max-units=1..1000000")
    inputs = selected_inputs(args.inputs)
    if args.live and not args.yes:
        parser.error("Live comparison requires --live --yes")
    if args.live and not args.prompt_key and not args.config.is_file():
        parser.error("Model configuration not found; use --config with the actual desktop config.json path")
    if args.live and any((args.output / name).exists() for name in ("budget.json", "result.json")):
        parser.error("Output already contains a live ledger/result; do not overwrite or reset the reviewed budget")
    if not args.live:
        print(json.dumps({"dry_run": True, "reads_credentials": False, "planned_json_calls":
                          sum(1 + len(split_document(Path(e['path']).read_text(encoding='utf-8'))) + 1 for e in inputs) * args.repeats,
                          "max_calls": args.max_calls, "max_conservative_units": args.max_units,
                          "papers": [{k: e[k] for k in ("paper_id", "title", "chars", "bytes")} for e in inputs]}, ensure_ascii=False, indent=2))
    else:
        asyncio.run(run(args, inputs))
