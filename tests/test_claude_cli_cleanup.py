"""Tests for subprocess process-group cleanup in the Claude CLI backend.

Regression cover for the leak where a cancelled handler (executor outer
`wait_for` timeout, or bridge shutdown) left the Claude CLI subprocess —
and its computer-use MCP grandchild — running and still driving the desktop.
"""

import asyncio
import os
import time

import pytest

from agentchat.backends._cli_utils import (
    kill_all_cli_processes,
    kill_process_group,
    track_cli_process,
)


async def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    """Poll until `pid` no longer exists (tolerates zombie-reap lag)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_kill_process_group_reaps_child_and_grandchild():
    """killpg reaps the whole tree — not just the direct child.

    A plain `proc.kill()` would orphan the grandchild; for computer use
    that grandchild is the MCP server still controlling the desktop.
    """
    # Shell (group leader, via start_new_session) backgrounds a long-lived
    # grandchild, prints its PID, then waits.
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh", "-c", "sleep 300 & echo $!; wait",
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    assert proc.stdout is not None
    grandchild_pid = int((await proc.stdout.readline()).decode().strip())

    os.kill(grandchild_pid, 0)  # sanity: grandchild is alive

    kill_process_group(proc)

    assert await _wait_dead(proc.pid), "parent (CLI) still alive after kill"
    assert await _wait_dead(grandchild_pid), "grandchild (MCP server) leaked"
    await asyncio.wait_for(proc.wait(), timeout=5)


@pytest.mark.asyncio
async def test_kill_process_group_noop_on_exited_process():
    """Safe no-op once the process has already exited cleanly."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh", "-c", "true", start_new_session=True,
    )
    await proc.wait()
    assert proc.returncode is not None
    kill_process_group(proc)  # must not raise


@pytest.mark.asyncio
async def test_kill_all_cli_processes_reaps_tracked_runs_on_shutdown():
    """The shutdown sweep kills every tracked, still-running CLI tree.

    Regression (2026-09-23): the desktop app restarted mid-task, the bridge
    exited without cancelling its handlers, and the CLI — its own session
    leader — ran on as an orphan, committing work nobody heard about.
    """
    running = track_cli_process(await asyncio.create_subprocess_exec(
        "/bin/sh", "-c", "sleep 300 & echo $!; wait",
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    ))
    assert running.stdout is not None
    grandchild_pid = int((await running.stdout.readline()).decode().strip())

    finished = track_cli_process(await asyncio.create_subprocess_exec(
        "/bin/sh", "-c", "true", start_new_session=True,
    ))
    await finished.wait()

    assert kill_all_cli_processes() == 1

    assert await _wait_dead(running.pid), "tracked CLI survived shutdown"
    assert await _wait_dead(grandchild_pid), "tracked CLI's MCP child survived shutdown"
    await asyncio.wait_for(running.wait(), timeout=5)
    assert kill_all_cli_processes() == 0
