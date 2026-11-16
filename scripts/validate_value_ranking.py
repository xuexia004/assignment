"""Fixed, small LitSearch regression with real public metadata; no paid model."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_agent.models import Paper  # noqa: E402
from paper_agent.services.retrieval import merge_dedupe_rank  # noqa: E402
from paper_agent.services.semantic_reranker import cached_paper_vectors, semantic_rerank  # noqa: E402
from paper_agent.services.value_reranker import value_rerank  # noqa: E402


def prepare(snapshot: Path, output: Path, count: int, offset: int):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((snapshot / "runs").glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        records.extend(r for p in run["papers"] for r in p["metadata"].get("retrieval_records", []))
    with httpx.Client(timeout=25, trust_env=False, transport=httpx.HTTPTransport(retries=1)) as client:
        # urllib honors the Windows system proxy used for this public dataset host.
        params = {"dataset": "princeton-nlp/LitSearch", "config": "query", "split": "full", "offset": offset, "length": count}
        with urlopen("https://datasets-server.huggingface.co/rows?" + urlencode(params), timeout=25) as response:
            queries = [r["row"] for r in json.load(response)["rows"]]
        ids = sorted({i for q in queries for i in q["corpusids"]})
        response = client.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                               params={"fields": "title,abstract,year,externalIds"},
                               json={"ids": [f"CorpusId:{i}" for i in ids]})
        response.raise_for_status()
        resolved = response.json()
        if not isinstance(resolved, list) or len(resolved) != len(ids):
            raise ValueError("Public corpus-id resolution is incomplete")
        for corpusid, item in zip(ids, resolved, strict=True):
            if not item:
                continue  # Missing gold stays in the denominator; do not drop its query.
            external = item.get("externalIds") or {}
            records.append(Paper(paper_id=f"litsearch:{corpusid}", title=item["title"],
                                 abstract=item.get("abstract") or "", year=item.get("year"),
                                 doi=external.get("DOI"), arxiv_id=external.get("ArXiv"),
                                 sources=["semantic_scholar"], metadata={"litsearch_corpusid": corpusid}).model_dump())
        # Refresh only normalized impact of the fixed observed OpenAlex identities.
        work_ids = sorted({r.get("metadata", {}).get("openalex_id") for r in records
                           if r.get("metadata", {}).get("openalex_id")})
        response = client.get("https://api.openalex.org/works", params={
            "filter": "openalex_id:" + "|".join(i.rsplit("/", 1)[-1] for i in work_ids[:100]),
            "per_page": 100, "select": "id,citation_normalized_percentile,cited_by_count"})
        response.raise_for_status()
        impact = {p["id"]: p.get("citation_normalized_percentile") for p in response.json()["results"]}
        for record in records:
            value = impact.get(record.get("metadata", {}).get("openalex_id"))
            if isinstance(value, dict):
                record.setdefault("metadata", {})["citation_normalized_percentile"] = value.get("value")
    data = {"queries": queries, "records": records, "as_of": date.today().isoformat(),
            "query_offset": offset,
            "scope": "first_fixed_queries_with_gold_added_to_fixed_observed_background_not_full_litsearch",
            "query_source": "https://huggingface.co/datasets/princeton-nlp/LitSearch",
            "missing_gold": [i for i, item in zip(ids, resolved, strict=True) if not item],
            "paid_model_calls": 0}
    (output / "dataset.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def ndcg(ranked, gold, k):
    def relevant(paper):
        return bool(gold.intersection(paper.metadata.get("merged_paper_ids", [paper.paper_id])))
    dcg = sum(relevant(p) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.


def recall(ranked, gold, k):
    found = {i for p in ranked[:k] for i in p.metadata.get("merged_paper_ids", [p.paper_id])}
    return len(found & gold) / len(gold) if gold else 0.


async def evaluate(data, output):
    papers = [Paper.model_validate(r) for r in data["records"]]
    rows = []
    for query in data["queries"]:
        baseline = merge_dedupe_rank(papers, query["query"])
        baseline, audit = await semantic_rerank(query["query"], baseline)
        if audit["actual_ranker"] != "semantic":
            raise RuntimeError("Semantic baseline unavailable: " + str(audit.get("fallback_reason")))
        ranked, value_audit = value_rerank(baseline, cached_paper_vectors(baseline[:40]),
                                          as_of=date.fromisoformat(data["as_of"]))
        gold = {f"litsearch:{i}" for i in query["corpusids"]}
        row = {"query": query["query"], "gold_ids": sorted(gold),
               "baseline_ndcg5": ndcg(baseline, gold, 5), "value_ndcg5": ndcg(ranked, gold, 5),
               "baseline_ndcg20": ndcg(baseline, gold, 20), "value_ndcg20": ndcg(ranked, gold, 20),
               "baseline_recall20": recall(baseline, gold, 20), "value_recall20": recall(ranked, gold, 20),
               "baseline_top5": [p.title for p in baseline[:5]], "value_top5": [p.title for p in ranked[:5]],
               "audit": value_audit}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("baseline_ndcg5", "value_ndcg5")}), flush=True)
    result = {"scope": data["scope"], "candidate_count": len(merge_dedupe_rank(papers, "background")),
              "queries": rows, "paid_model_calls": 0,
              "mean_baseline_ndcg5": sum(r["baseline_ndcg5"] for r in rows) / len(rows),
              "mean_value_ndcg5": sum(r["value_ndcg5"] for r in rows) / len(rows),
              "mean_baseline_recall20": sum(r["baseline_recall20"] for r in rows) / len(rows),
              "mean_value_recall20": sum(r["value_recall20"] for r in rows) / len(rows),
              "limits": "Candidate-stage regression only; not full-corpus performance, utility labels or final LLM Top-k."}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "queries"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/validation/sources/20261106T120919Z")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/value-ranking-20261106")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("count must be 1..100")
    data_path = args.output / "dataset.json"
    dataset = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else prepare(args.snapshot, args.output, args.count, args.offset)
    if not args.prepare_only:
        asyncio.run(evaluate(dataset, args.output))
