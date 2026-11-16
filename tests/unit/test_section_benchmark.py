"""The live comparison must reserve concurrent requests before spending."""

import asyncio
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from paper_agent.llm_client import LLMBudgetError

BudgetedModel = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/benchmark_sections.py"))["BudgetedModel"]


def test_concurrent_reservations_cannot_exceed_calls_or_units_and_failures_stay_charged(tmp_path):
    class Model:
        settings = SimpleNamespace(llm_max_output_tokens=100)
        calls = 0

        async def json(self, system, user):
            self.calls += 1
            await asyncio.sleep(0)
            raise RuntimeError("Simulated transport failure")

    async def exercise(call_limit, unit_limit):
        raw = Model()
        ledger = tmp_path / f"{call_limit}-{unit_limit}.json"
        model = BudgetedModel(raw, call_limit, unit_limit, ledger)
        results = await asyncio.gather(*(model.json("sys", "input") for _ in range(4)), return_exceptions=True)
        assert raw.calls == model.calls == 2
        assert model.reserved_units == 344
        assert sum(isinstance(r, LLMBudgetError) for r in results) == 2
        saved = json.loads(ledger.read_text(encoding="utf-8"))
        assert saved["reserved_calls"] == 2 and saved["reserved_units"] == 344

    asyncio.run(exercise(2, 10000))
    asyncio.run(exercise(10, 344))
