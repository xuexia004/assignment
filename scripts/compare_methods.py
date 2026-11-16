"""Freeze paired evidence and produce a classroom report; no keys or paid calls.

Default: replay existing observations. --run repeats only local ranking/PDF work.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_agent.config import RESEARCH_PROFILES  # noqa: E402

EVIDENCE = ROOT / "data/validation/opensource-followup-20261106"
DATASETS = [ROOT / f"data/validation/{name}/dataset.json" for name in
            ("value-ranking-20261106", "value-ranking-holdout-20261106")]
SOURCES = ROOT / "data/validation/sources/20261106T120919Z/summary.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(source, target):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError("Input changed while preserving evidence")
    return {"original": str(source.resolve()), "file": str(target.resolve()), "sha256": digest}


def ranking_summary(result, dataset):
    rows, queries = result["queries"], dataset["queries"]
    if len(rows) != len(queries) or not rows:
        raise ValueError("Ranking queries do not match the frozen dataset")
    for row, query in zip(rows, queries, strict=True):
        audit = row["audit"]
        if (row["query"] != query["query"] or set(row["gold_ids"]) != {f"litsearch:{i}" for i in query["corpusids"]}
                or not audit["applied"] or audit["pool_limit"] != 20
                or set(audit["baseline_ids"]) != set(audit["ordered_ids"])):
            raise ValueError("Ranking evidence is not the paired top-20 comparison")
    return {"queries": len(rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "baseline_ndcg5", "reranked_ndcg5", "baseline_recall20", "reranked_recall20")},
            "improved": sum(r["reranked_ndcg5"] > r["baseline_ndcg5"] for r in rows),
            "regressed": sum(r["reranked_ndcg5"] < r["baseline_ndcg5"] for r in rows),
            "median_rerank_seconds": result["median_warm_rerank_seconds"]}


def pdf_summary(result):
    summary = []
    for digest in dict.fromkeys(row["pdf_sha256"] for row in result["rows"]):
        rows = [r for r in result["rows"] if r["pdf_sha256"] == digest]
        rounds = {r["repeat"] for r in rows}
        if any(sorted(r["parser"] for r in rows if r["repeat"] == repeat) != ["layout", "legacy"] for repeat in rounds):
            raise ValueError("Every PDF repeat needs both parser observations")
        baseline = statistics.median(r["elapsed_seconds"] for r in rows if r["parser"] == "legacy")
        method = statistics.median(r["elapsed_seconds"] for r in rows if r["parser"] == "layout")
        summary.append({"title": rows[0]["title"], "sha256": digest, "repeats": len(rounds),
                        "baseline_seconds": baseline, "layout_seconds": method,
                        "reduction_percent": 100 * (1 - method / baseline)})
    return summary


def report(output, manifest, sources, rankings, pdf):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# 基线与新方法对照", "",
             f"证据方式：{'重新运行本地模型与解析器' if manifest['mode'] == 'local_run' else '导入已有真实测量，当前没有重新计时'}。本命令付费调用：0。", "",
             "下面是模块对照，不是综合增强的端到端实验。排序使用固定构造候选，含先前各来源记录；不能把这些候选指标当成最终 LLM 精选或报告质量。", "",
             "## 来源覆盖", "", "本表始终导入既有观察记录，不请求公开接口。同批记录移除/保留 Crossref；不是相关性或全文成功率。", "",
             "| 查询 | 双源基线 | 加 Crossref | 增量 |", "|---|---:|---:|---:|"]
    lines += [f"| {cell(r['query'])} | {r['baseline_unique_count']} | {r['combined_unique_count']} | {r['incremental_count']} |" for r in sources["queries"]]
    lines += ["", "## 候选排序", "", "| 查询组 | 题数 | 基线 nDCG@5 | 联合重排 nDCG@5 | 基线/新 Recall@20 | 改善/退化 | 重排中位秒 |", "|---|---:|---:|---:|---|---|---:|"]
    lines += [f"| {i + 1} | {r['queries']} | {r['baseline_ndcg5']:.6f} | {r['reranked_ndcg5']:.6f} | {r['baseline_recall20']:.6f} / {r['reranked_recall20']:.6f} | {r['improved']} / {r['regressed']} | {r['median_rerank_seconds']:.2f} |" for i, r in enumerate(rankings)]
    lines += ["", "Recall@20 相等由原候选集合不变保证，不是新召回收益。", "", "## 整篇 PDF 转换", "",
              "| 论文 | 配对轮数 | legacy 中位秒 | layout 中位秒 | 耗时降低 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {cell(r['title'])} | {r['repeats']} | {r['baseline_seconds']:.2f} | {r['layout_seconds']:.2f} | {r['reduction_percent']:.1f}% |" for r in pdf]
    lines += ["", "转换耗时不含下载和 LLM；复杂表头、公式与部分参考文献仍可能有提取错误。", "",
              "## 可追溯文件", "", "- [实验清单与输入哈希](manifest.json)", "- [机器可读对照](comparison.json)",
              "- raw/：原始测量；inputs/：固定输入；artifacts/：各解析器原文输出。", "",
              "旧版已知负结果继续保留在原实验目录，本报告没有用新结果覆盖它们。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"mode": "local_run" if args.run else "replay", "paid_calls_this_command": 0,
                "created_at": datetime.now(timezone.utc).isoformat(), "profiles": RESEARCH_PROFILES,
                "inputs": [], "raw_results": [], "tool_code_sha256": {},
                "code_hash_scope": "current_tool_at_invocation_not_original_code_of_imported_observations"}
    for name in ("src/paper_agent/config.py", "src/paper_agent/services/cross_encoder_reranker.py",
                 "src/paper_agent/services/semantic_reranker.py", "src/paper_agent/services/pdf_converter.py",
                 "src/paper_agent/services/retrieval.py", "src/paper_agent/services/author_keywords.py",
                 "scripts/validate_value_ranking.py", "scripts/validate_cross_encoder.py",
                 "scripts/compare_pdf_layout.py", "scripts/compare_methods.py"):
        manifest["tool_code_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    manifest["report_tool_versions"] = {name: importlib.metadata.version(name) for name in ("pymupdf4llm", "sentence-transformers", "torch")}
    parser_source = args.evidence / "pdf-layout/result.json"
    old_pdf = read(parser_source) if parser_source.is_file() else {"rows": []}
    pdf_paths = args.pdfs or [Path(r["path"]) for r in old_pdf["rows"] if r["parser"] == "legacy" and r["repeat"] == 0]
    if not pdf_paths:
        raise ValueError("No fixed PDFs found; provide --pdfs with the preserved inputs")
    frozen_pdf = []
    for i, path in enumerate(pdf_paths):
        record = freeze(path, output / "inputs" / f"paper-{i + 1}.pdf")
        if not args.run and record["sha256"] not in {r["pdf_sha256"] for r in old_pdf["rows"]}:
            raise ValueError("PDF content differs from imported measurements")
        manifest["inputs"].append(record)
        frozen_pdf.append(Path(record["file"]))
    rankings = []
    for i, dataset in enumerate(args.datasets):
        record = freeze(dataset, output / "inputs" / f"dataset-{i + 1}.json")
        manifest["inputs"].append(record)
        target = output / "raw" / f"ranking-{i + 1}.json"
        if args.run:
            from validate_cross_encoder import evaluate
            await evaluate(Path(record["file"]), target)
            manifest["raw_results"].append({"file": str(target.resolve()),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "generated_locally": True})
        else:
            name = ["crossencoder-top20-first20.json", "crossencoder-top20-second20.json"][i]
            result = read(args.evidence / name)
            if hashlib.sha256(Path(result["dataset"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Ranking input differs from imported measurements")
            manifest["raw_results"].append(freeze(args.evidence / name, target))
        rankings.append(ranking_summary(read(target), read(Path(record["file"]))))
    if args.run:
        from compare_pdf_layout import compare
        compare(frozen_pdf, output / "pdf-layout", args.repeats)
        parser_source = output / "pdf-layout/result.json"
    record = freeze(parser_source, output / "raw/pdf-layout.json")
    manifest["raw_results"].append(record)
    for row in read(parser_source)["rows"]:
        freeze(Path(row["artifact"]), output / "artifacts" / Path(row["artifact"]).name)
    source_record = freeze(args.sources, output / "raw/sources.json")
    manifest["raw_results"].append(source_record)
    manifest["source_coverage_is_imported"] = True
    sources = read(Path(source_record["file"]))
    summary = {"mode": manifest["mode"], "scope": "component_comparisons_not_end_to_end_or_llm_quality",
               "source_coverage": sources["queries"], "ranking": rankings, "pdf": pdf_summary(read(output / "raw/pdf-layout.json"))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report(output, manifest, sources, rankings, summary["pdf"])
    print(json.dumps({"output": str(output), "mode": manifest["mode"], "paid_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Repeat local ranking/PDF comparison, never a live research job")
    parser.add_argument("--plan", action="store_true", help="Print frozen method definitions without reading inputs or models")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--datasets", type=Path, nargs=2, default=DATASETS)
    parser.add_argument("--pdfs", type=Path, nargs="+")
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/comparisons" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    if args.plan:
        print(json.dumps({"profiles": RESEARCH_PROFILES, "paid_calls": 0, "mode": "local_run" if args.run else "replay"}, ensure_ascii=False, indent=2))
    else:
        asyncio.run(run(args))
