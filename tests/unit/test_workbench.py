"""Workbench job lifecycle: trace events must never be lost to the terminal-status race."""

import asyncio
import json
from pathlib import Path

import pytest

from paper_agent.config import Settings
from paper_agent.models import PipelineResult
from paper_agent.storage import PipelineStore
from paper_agent.web import Workbench, _trace_records


def _runner(job: dict, settings: Settings, progress):
    async def run() -> dict:
        run = PipelineResult(idea=job["idea"], mode="demo", status="demo_completed", top_k=job["top_k"])
        store = PipelineStore(settings.output_dir)
        directory = store.directory(run.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "trace.jsonl").write_text("\n".join(json.dumps(item) for item in (
            {"stage": "main_agent_tool_call", "tool": "search_papers"},
            {"stage": "main_agent_tool_call", "tool": "generate_research_report"},
        )) + "\n", encoding="utf-8")
        store.save(run)
        progress(f"  status=demo_completed, run_id={run.run_id}")
        return run.model_dump(mode="json")

    return run()


def test_terminal_status_is_visible_only_after_all_trace_events(tmp_path):
    async def scenario() -> dict:
        settings = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "data")
        bench = Workbench(settings, root=tmp_path / "web", config_path=tmp_path / "web" / "config.json",
                          research_runner=_runner)
        session = bench.create_session()
        job = bench.submit(session["session_id"], {"idea": "agent memory", "query": "", "top_k": 3,
                                                   "mode": "demo", "execution_mode": "sequential"})
        await bench.tasks[job["job_id"]]
        return bench.jobs[job["job_id"]]

    job = asyncio.run(scenario())
    assert job["status"] == "completed"
    events = job["events"]
    last_trace = max(index for index, event in enumerate(events) if event["kind"] == "trace")
    final_status = max(index for index, event in enumerate(events)
                       if event["kind"] == "status" and event["data"].get("status") == "completed")
    assert final_status > last_trace
    tools = [event["data"]["tool"] for event in events
             if event["kind"] == "trace" and event["data"].get("stage") == "main_agent_tool_call"]
    assert tools == ["search_papers", "generate_research_report"]


def test_artifact_names_are_safe_for_desktop_open_export():
    assert Workbench._artifact_name("A/B: C?", ".pdf") == "A_B_ C_.pdf"
    assert Workbench._artifact_name("  ..  ", ".md") == "artifact.md"
    assert len(Workbench._artifact_name("x" * 300, ".md")) <= 143


def test_sync_run_reads_trace_before_run_json_exists(tmp_path):
    async def scenario() -> dict:
        settings = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "data")
        bench = Workbench(settings, root=tmp_path / "web", config_path=tmp_path / "web" / "config.json")
        job = {"job_id": "j1", "session_id": "s1", "idea": "agent memory", "query": "", "top_k": 3,
               "mode": "demo", "execution_mode": "sequential", "status": "running", "created_at": "",
               "updated_at": "", "run_id": "run1", "result": None, "events": [], "error": None,
               "artifacts": []}
        bench.jobs[job["job_id"]] = job
        directory = settings.output_dir / "run1"
        directory.mkdir(parents=True)
        (directory / "trace.jsonl").write_text(
            json.dumps({"stage": "selector", "event": "start"}) + "\n", encoding="utf-8")
        await bench._sync_run(job, settings)
        return job

    job = asyncio.run(scenario())
    assert [event["kind"] for event in job["events"]] == ["trace"]
    assert job["events"][0]["data"]["stage"] == "selector"


def test_sync_run_failure_cannot_strand_terminal_status_or_task_slot(tmp_path, monkeypatch):
    async def scenario() -> tuple[dict, Workbench]:
        settings = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "data")
        bench = Workbench(settings, root=tmp_path / "web", config_path=tmp_path / "web" / "config.json",
                          research_runner=_runner)

        async def broken(*_arguments):
            raise RuntimeError("trace directory vanished")

        monkeypatch.setattr(bench, "_sync_run", broken)
        session = bench.create_session()
        job = bench.submit(session["session_id"], {"idea": "agent memory", "query": "", "top_k": 3,
                                                   "mode": "demo", "execution_mode": "sequential"})
        await bench.tasks[job["job_id"]]
        return bench.jobs[job["job_id"]], bench

    job, bench = asyncio.run(scenario())
    assert job["status"] == "completed"
    assert not bench.tasks


def test_incremental_trace_waits_for_utf8_tail_and_detects_replacement(tmp_path):
    path = tmp_path / "trace.jsonl"
    first = (json.dumps({"message": "第一篇分析完成"}, ensure_ascii=False) + "\n").encode()
    second = (json.dumps({"message": "第二篇正在分析"}, ensure_ascii=False) + "\n").encode()
    path.write_bytes(first + second[:-3])
    rows = _trace_records(path, None)
    assert len(rows) == 1 and rows[0][1]["offset"] == len(first)
    cursor = rows[-1][1]
    assert _trace_records(path, cursor) == []
    with path.open("ab") as stream:
        stream.write(second[-3:])
    rows = _trace_records(path, cursor)
    assert [item["message"] for item, _ in rows] == ["第二篇正在分析"]
    cursor = rows[-1][1]
    replacement = tmp_path / "replacement"
    replacement.write_bytes(first + second)
    replacement.replace(path)
    assert _trace_records(path, cursor) == []
    # A truncate-and-regrow on the same path can exceed the old offset.
    changed = (json.dumps({"message": "新日志" * 100}, ensure_ascii=False) + "\n").encode()
    path.write_bytes(changed)
    rows = _trace_records(path, cursor)
    assert len(rows) == 1 and rows[0][0]["message"] == "新日志" * 100
    path.write_bytes(first)
    rows = _trace_records(path, rows[-1][1])
    assert len(rows) == 1 and rows[0][1]["offset"] == len(first)


def test_trace_poll_reads_constant_bytes_when_nothing_is_appended(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b'{"message":"fixture"}\n' * 10000)
    cursor = _trace_records(path, None)[-1][1]
    original = Path.open
    measurements = {"bytes": 0}

    class Meter:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def __getattr__(self, key):
            return getattr(self.stream, key)
        def read(self, *args):
            value = self.stream.read(*args)
            measurements["bytes"] += len(value)
            return value
        def readline(self, *args):
            value = self.stream.readline(*args)
            measurements["bytes"] += len(value)
            return value

    def measured(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        return Meter(stream) if self == path else stream

    monkeypatch.setattr(Path, "open", measured)
    for _ in range(5):
        assert _trace_records(path, cursor) == []
    assert measurements["bytes"] <= 5 * 192
    assert measurements["bytes"] < path.stat().st_size // 100


def test_trace_batch_skips_corruption_and_recovers_journal_cursor(tmp_path, monkeypatch):
    async def scenario():
        settings = Settings(data_dir=tmp_path, output_dir=tmp_path / "runs")
        bench = Workbench(settings, root=tmp_path / "web")
        session = bench.create_session()
        job = {"job_id": "a" * 32, "session_id": session["session_id"], "idea": "agent memory",
               "mode": "demo", "execution_mode": "sequential", "status": "completed",
               "created_at": "", "updated_at": "", "run_id": "run1", "result": None,
               "events": [], "artifacts": [], "error": None}
        bench.jobs[job["job_id"]] = job
        bench.persist_job(job)
        directory = settings.output_dir / "run1"
        directory.mkdir(parents=True)
        trace = directory / "trace.jsonl"
        trace.write_text('{"stage":"search","message":"检索完成"}\nnot json\n[]\n'
                         '{"stage":"research_report","message":"综合报告完成"}\n', encoding="utf-8")
        writes = []
        original = bench.persist_job
        monkeypatch.setattr(bench, "persist_job", lambda value: (writes.append(1), original(value)))
        await bench._sync_run(job, settings)
        assert len(writes) == 1
        assert [event["message"] for event in job["events"]][::3] == ["检索完成", "综合报告完成"]
        assert sum(bool(event["data"].get("log_record_invalid")) for event in job["events"]) == 2
        # Simulate a crash after journal append but before the snapshot commits.
        with trace.open("a", encoding="utf-8") as stream:
            stream.write('{"stage":"research_report","message":"最终记录"}\n')
        monkeypatch.setattr(bench, "persist_job", lambda value: (_ for _ in ()).throw(PermissionError("fixture lock")))
        with pytest.raises(PermissionError):
            await bench._sync_run(job, settings)
        reopened = Workbench(settings, root=bench.root)
        restored = reopened.jobs[job["job_id"]]
        assert restored["_trace_lines"] == 5
        await reopened._sync_run(restored, settings)
        assert [event["seq"] for event in reopened.events_after(restored, 0)] == [1, 2, 3, 4, 5]
        assert restored["events"][-1]["message"] == "最终记录"

    asyncio.run(scenario())


def test_legacy_line_cursor_is_migrated_without_replaying_old_events(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"message":"old"}\n{"message":"new"}\n', encoding="utf-8")
    rows = _trace_records(path, None, legacy_lines=1)
    assert len(rows) == 1 and rows[0][0]["message"] == "new"
    assert _trace_records(path, rows[-1][1], legacy_lines=2) == []


def test_persistence_failure_still_releases_the_job_slot(tmp_path, monkeypatch):
    async def scenario() -> tuple[dict, Workbench]:
        settings = Settings(output_dir=tmp_path / "runs", data_dir=tmp_path / "data")
        bench = Workbench(settings, root=tmp_path / "web", config_path=tmp_path / "web" / "config.json",
                          research_runner=_runner)
        session = bench.create_session()
        job = bench.submit(session["session_id"], {"idea": "agent memory", "query": "", "top_k": 3,
                                                   "mode": "demo", "execution_mode": "sequential"})
        monkeypatch.setattr("paper_agent.web.atomic_json",
                            lambda *_arguments: (_ for _ in ()).throw(PermissionError(13, "locked")))
        await bench.tasks[job["job_id"]]
        return bench.jobs[job["job_id"]], bench

    job, bench = asyncio.run(scenario())
    assert job["status"] == "completed"
    assert not bench.tasks
