"""Resumable CLI runs: a run survives its bridge process and is adopted.

The case this exists for (2026-09-23): an agent building agntchat edits the
desktop app, `tauri dev` restarts it, the bridge dies mid-task. The CLI run
must keep going and the next bridge must deliver its result.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from agentchat.backends import cli_runs


@pytest.fixture(autouse=True)
def _runs_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNTCHAT_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_ID", "agent-1")


def _as_other_process(monkeypatch):
    monkeypatch.setattr(cli_runs, "PROCESS_TOKEN", "a-later-bridge")


SCRIPT = (
    'echo \'{"type":"system","subtype":"init"}\'; sleep 0.3; '
    'echo \'{"type":"result","subtype":"success","is_error":false,"result":"done"}\''
)


async def _spawn(key, script=SCRIPT, resumable=True):
    token = cli_runs.run_key_var.set(key)
    try:
        return await cli_runs.spawn(
            ["/bin/sh", "-c", script],
            prompt="", env=dict(os.environ), limit=2**20,
            resumable=resumable, subprocess_kwargs={},
        )
    finally:
        cli_runs.run_key_var.reset(token)


async def _read_all(run):
    out = b""
    reader = run.stdout()
    while True:
        chunk = await reader.read(4096)
        if not chunk:
            return out
        out += chunk


@pytest.mark.asyncio
async def test_a_later_process_adopts_a_running_run_and_reads_it_to_the_end(monkeypatch):
    run = await _spawn(("task", "t-1"))
    _as_other_process(monkeypatch)

    assert cli_runs.resumable_task_ids() == ["t-1"]
    adopted = cli_runs.find_adoptable(("task", "t-1"))
    assert adopted is not None and adopted.adopted and adopted.returncode is None

    lines = [json.loads(l) for l in (await _read_all(adopted)).splitlines()]
    assert [e["type"] for e in lines] == ["system", "result"]
    await run.proc.wait()
    adopted.discard()
    assert cli_runs.find_adoptable(("task", "t-1")) is None


@pytest.mark.asyncio
async def test_a_run_that_finished_while_no_bridge_was_up_is_still_adoptable(monkeypatch):
    run = await _spawn(("message", "m-1"))
    await run.proc.wait()
    _as_other_process(monkeypatch)

    assert cli_runs.find_adoptable(("message", "m-1")) is not None
    assert cli_runs.resumable_task_ids() == []  # messages are requeued server-side anyway


@pytest.mark.asyncio
async def test_own_runs_are_never_adopted():
    run = await _spawn(("task", "t-2"))
    assert cli_runs.find_adoptable(("task", "t-2")) is None
    assert cli_runs.resumable_task_ids() == []
    await run.proc.wait()


@pytest.mark.asyncio
async def test_dead_runs_without_a_result_and_non_resumable_runs_are_dropped(monkeypatch):
    dead = await _spawn(("task", "t-3"), script="echo '{\"type\":\"system\"}'")
    unsafe = await _spawn(("task", "t-4"), resumable=False)
    await dead.proc.wait()
    await unsafe.proc.wait()
    _as_other_process(monkeypatch)

    assert cli_runs.resumable_task_ids() == []
    assert not (cli_runs.runs_dir() / "task-t-3").exists()
    assert not (cli_runs.runs_dir() / "task-t-4").exists()


@pytest.mark.asyncio
async def test_without_a_run_key_the_run_is_piped_and_not_resumable():
    run = await cli_runs.spawn(
        ["/bin/sh", "-c", "echo hi"], prompt="", env=dict(os.environ),
        limit=2**20, resumable=True, subprocess_kwargs={},
    )
    assert run.record_dir is None and not run.resumable
    assert (await run.stdout().read(100)).strip() == b"hi"
    await run.proc.wait()
