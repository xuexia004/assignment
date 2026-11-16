"""Fixed public LitSearch candidate regression; no new metadata or paid calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_value_ranking import ndcg, recall  # noqa: E402

from paper_agent.models import Paper  # noqa: E402
from paper_agent.services.cross_encoder_reranker import cross_encoder_rerank  # noqa: E402
from paper_agent.services.retrieval import merge_dedupe_rank  # noqa: E402
from paper_agent.services.semantic_reranker import semantic_rerank  # noqa: E402


async def evaluate(dataset, output):
    if output.exists():
        raise ValueError("Preserve prior measurements; choose a fresh output file")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    papers = [Paper.model_validate(r) for r in data["records"]]
    rows = []
    for query in data["queries"]:
        baseline, baseline_audit = await semantic_rerank(query["query"], merge_dedupe_rank(papers, query["query"]))
        if baseline_audit["actual_ranker"] != "semantic":
            raise RuntimeError("Actual semantic baseline is required")
        ranked, audit = await cross_encoder_rerank(query["query"], baseline)
        if not audit["applied"]:
            raise RuntimeError("Actual cross-encoder is required: " + str(audit))
        gold = {f"litsearch:{i}" for i in query["corpusids"]}
        row = {"query": query["query"], "gold_ids": sorted(gold), "audit": audit,
               "baseline_top5": [p.title for p in baseline[:5]], "reranked_top5": [p.title for p in ranked[:5]]}
        for k in (5, 20):
            row[f"baseline_ndcg{k}"] = ndcg(baseline, gold, k)
            row[f"reranked_ndcg{k}"] = ndcg(ranked, gold, k)
        row["baseline_recall20"] = recall(baseline, gold, 20)
        row["reranked_recall20"] = recall(ranked, gold, 20)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ["baseline_ndcg5", "reranked_ndcg5"]}), flush=True)
    result = {"scope": data["scope"], "dataset": str(dataset.resolve()), "paid_calls": 0,
              "queries": rows, "missing_gold": data["missing_gold"],
              "mean_metrics": {k: statistics.mean(r[k] for r in rows) for k in rows[0] if "ndcg" in k or "recall" in k},
              "median_warm_rerank_seconds": statistics.median(r["audit"]["duration_seconds"] for r in rows[1:]),
              "first_rerank_seconds": rows[0]["audit"]["duration_seconds"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "queries"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(evaluate(args.dataset, args.output))
