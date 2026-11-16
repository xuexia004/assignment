"""Public metadata-only validation. No model, saved key or .env is read."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_agent.config import Settings  # noqa: E402
from paper_agent.models import Paper  # noqa: E402
from paper_agent.services.pipeline import ResearchPipeline  # noqa: E402
from paper_agent.services.retrieval import merge_dedupe_rank  # noqa: E402


async def validate(args):
    output = ROOT / "data" / "validation" / "sources" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = Settings(mode="live", ranker="keyword", processing="concurrent", source_limit=args.limit,
                        output_dir=output / "runs", data_dir=output / "files",
                        http_timeout=args.timeout, http_max_retries=0, cache_enabled=False)
    rows = []
    for query in args.query:
        # An explicit query bypasses the planner; only the source-search stage runs.
        run = await ResearchPipeline(settings, llm=object()).search_papers(query, 3, query)
        records = [Paper.model_validate(record) for paper in run.papers
                   for record in paper.metadata.get("retrieval_records", [])]
        baseline = merge_dedupe_rank([p for p in records if "crossref" not in p.sources], query)
        core_ok = all(run.metrics["source_success"].get(s) for s in ("arxiv", "openalex"))
        row = {"query": query, "run_id": run.run_id, "source_counts": run.source_counts,
               "source_success": run.metrics["source_success"], "source_requests": run.metrics["source_requests"],
               "source_coverage": run.metrics["source_coverage"], "baseline_unique_count": len(baseline),
               "combined_unique_count": len(run.papers), "incremental_count": len(run.papers) - len(baseline),
               "core_sources_complete": core_ok, "errors": [e.model_dump(mode="json") for e in run.errors],
               "comparison_scope": "same_observed_records_with_or_without_crossref_not_relevance_quality"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = {"scope": "real_public_metadata_only", "llm_calls": 0, "queries": rows,
              "limit_per_source": args.limit, "timeout_seconds": args.timeout,
              "notes": "Counts are coverage, not quality; failed core sources limit the comparison."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=15)
    options = parser.parse_args()
    if not options.query:
        options.query = ["retrieval augmented generation", "large language model uncertainty calibration",
                         "robot manipulation task planning"]
    if not 1 <= options.limit <= 100 or not 0 < options.timeout <= 60:
        parser.error("limit must be 1..100 and timeout 0..60 seconds")
    asyncio.run(validate(options))
