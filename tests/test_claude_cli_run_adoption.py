"""A bridge restart mid-turn: the next bridge adopts the running CLI.

End to end through `ClaudeCliBackend._generate_streaming`: the first
"process" starts a run under a run key and is shut down mid-read; a later
process handling the same task gets the original run's result instead of
spawning a second CLI (2026-09-23, Codie building the desktop app while
`tauri dev` restarted it).
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from agentchat.backends import _cli_utils, cli_runs
from agentchat.backends.claude_cli import ClaudeCliBackend


def _backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(
        cli_path="/bin/sh", api_url="http://localhost",
        agent_id="agent-test", api_key="key-test",
    )


async def _noop(_event):
    return None


RESULT = {"type": "result", "subtype": "success", "is_error": False, "result": "built it", "num_turns": 3}
SLOW_CLI = ["/bin/sh", "-c", f"sleep 0.6; printf '%s\\n' '{json.dumps(RESULT)}'"]
MUST_NOT_RUN = ["/bin/sh", "-c", "exit 7"]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNTCHAT_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_ID", "agent-test")
    monkeypatch.setattr(_cli_utils, "_shutting_down", False)


@pytest.mark.asyncio
async def test_restarted_bridge_adopts_the_run_and_returns_its_result(monkeypatch):
    token = cli_runs.run_key_var.set(("task", "task-1"))
    try:
        first = asyncio.ensure_future(_backend()._generate_streaming(SLOW_CLI, _noop, prompt=""))
        await asyncio.sleep(0.2)
        # The bridge gets its shutdown signal mid-turn and goes away.
        _cli_utils.begin_shutdown()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        # A new bridge process, same task redelivered.
        monkeypatch.setattr(_cli_utils, "_shutting_down", False)
        monkeypatch.setattr(cli_runs, "PROCESS_TOKEN", "next-bridge")
        assert cli_runs.resumable_task_ids() == ["task-1"]

        result = await _backend()._generate_streaming(MUST_NOT_RUN, _noop, prompt="")
    finally:
        cli_runs.run_key_var.reset(token)

    assert result.text == "built it"
    assert result.metadata["cli_num_turns"] == 3
    assert not (cli_runs.runs_dir() / "task-task-1").exists()


@pytest.mark.asyncio
async def test_adopted_run_that_died_without_a_result_is_interrupted(monkeypatch):
    from agentchat.backends import BackendInterruptedError

    token = cli_runs.run_key_var.set(("task", "task-2"))
    try:
        first = asyncio.ensure_future(
            _backend()._generate_streaming(["/bin/sh", "-c", "sleep 0.4"], _noop, prompt="")
        )
        await asyncio.sleep(0.1)
        _cli_utils.begin_shutdown()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        monkeypatch.setattr(_cli_utils, "_shutting_down", False)
        monkeypatch.setattr(cli_runs, "PROCESS_TOKEN", "next-bridge")
        run = cli_runs.find_adoptable()
        assert run is not None  # still alive at this point

        with pytest.raises(BackendInterruptedError):
            await _backend()._generate_streaming(MUST_NOT_RUN, _noop, prompt="")
    finally:
        cli_runs.run_key_var.reset(token)


@pytest.mark.asyncio
async def test_cancel_outside_shutdown_still_kills_the_run():
    token = cli_runs.run_key_var.set(("task", "task-3"))
    try:
        first = asyncio.ensure_future(
            _backend()._generate_streaming(["/bin/sh", "-c", "sleep 30"], _noop, prompt="")
        )
        await asyncio.sleep(0.2)
        meta = json.loads((cli_runs.runs_dir() / "task-task-3" / "meta.json").read_text())
        first.cancel()  # e.g. the human pressed Stop
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        cli_runs.run_key_var.reset(token)

    for _ in range(50):
        if not cli_runs.pid_alive(meta["pid"]):
            break
        await asyncio.sleep(0.05)
    assert not cli_runs.pid_alive(meta["pid"])
    assert not (cli_runs.runs_dir() / "task-task-3").exists()
