"""Reject unpaired measurements instead of producing misleading comparisons."""

import runpy
from pathlib import Path

import pytest

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/compare_methods.py"))


def test_different_queries_or_candidates_cannot_be_merged_into_a_ranking_comparison():
    dataset = {"queries": [{"query": "original", "corpusids": [1]}]}
    row = {"query": "different", "gold_ids": ["litsearch:1"],
           "audit": {"applied": True, "pool_limit": 20, "baseline_ids": ["p"], "ordered_ids": ["p"]}}
    with pytest.raises(ValueError, match="paired"):
        SCRIPT["ranking_summary"]({"queries": [row]}, dataset)
    row["query"] = "original"
    row["audit"]["ordered_ids"] = ["another"]
    with pytest.raises(ValueError, match="paired"):
        SCRIPT["ranking_summary"]({"queries": [row]}, dataset)


def test_different_pdf_hashes_cannot_be_reported_as_a_pair():
    result = {"rows": [{"pdf_sha256": "one", "repeat": 0, "parser": "legacy"},
                       {"pdf_sha256": "two", "repeat": 0, "parser": "layout"}]}
    with pytest.raises(ValueError, match="both"):
        SCRIPT["pdf_summary"](result)
