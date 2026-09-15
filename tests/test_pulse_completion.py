"""A pulse is completed by the server-side `pulse_report` tool, inside the
model run. The bridge must not complete it a second time, and must never
post pulse text.

Prod artifact (Botty, 2026-09-15, every 5 minutes): each pulse turn logged
`pulse_report {}` → "Pulse complete — nothing to report.", then two seconds
later a `complete_task` whose summary was "I wasn't able to complete this —
the task timed out … PULSE_OK <pulse_state>{"last_check":"2026-05-03…"}" —
refused by the backend as "ALREADY complete". The text was the bridge's
old pulse path joining the last 10 agent messages of the work conversation,
which TaskAssignmentWorker had seeded from the agent's pulse DM tail: a May
timeout bubble and a May `<pulse_state>` message, replayed on every run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent_bridge import _pulse_completion_fields
from agentchat.executor import ExecutorClient, GatewayTask


def _result(tool_names: list[str], *, errored: set[str] | None = None):
    errored = errored or set()
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(name=n, is_error=(n in errored)) for n in tool_names
        ],
        metadata={},
    )


def test_pulse_report_call_marks_task_completed_by_tool():
    fields = _pulse_completion_fields(_result(["get_memory", "pulse_report"]))
    assert fields == {"silent": True, "completed_via_tool": "pulse_report"}


def test_pulse_without_report_is_still_silent():
    """The model forgot the finisher: the bridge completes silently, never
    with a posted response, and the backend treats it as nothing to report."""
    assert _pulse_completion_fields(_result(["get_memory"])) == {"silent": True}


def test_refused_pulse_report_does_not_count():
    fields = _pulse_completion_fields(
        _result(["pulse_report"], errored={"pulse_report"})
    )
    assert fields == {"silent": True}


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


def _pulse_task() -> GatewayTask:
    return GatewayTask.from_dict(
        {
            "id": "queue-1",
            "taskId": "task-1",
            "task": {
                "title": "Pulse Check",
                "conversationId": "pulse-conv",
                "source": "pulse",
                "metadata": {"work_conversation_id": "work-conv"},
            },
        }
    )


def _fake_post():
    calls: list[tuple[str, dict]] = []

    async def post(path, json=None, **_kw):
        if path == "/api/mcp":
            calls.append((json["params"]["name"], json["params"]["arguments"]))
            return {"result": {"isError": False, "content": [{"type": "text", "text": "ok"}]}}
        return {}

    return post, calls


@pytest.mark.asyncio
async def test_executor_skips_completion_when_pulse_report_already_ran(executor):
    @executor.on_task
    async def handler(_task):
        return {
            "summary": "PULSE_OK <pulse_state>{\"last_check\":\"2026-05-03T19:39:00Z\"}</pulse_state>",
            "silent": True,
            "completed_via_tool": "pulse_report",
        }

    post, calls = _fake_post()
    with patch.object(executor, "_post", new=AsyncMock(side_effect=post)):
        await executor._handle_task(_pulse_task())

    assert calls == [], f"no MCP call expected after pulse_report, got {calls}"


@pytest.mark.asyncio
async def test_executor_never_posts_pulse_text(executor):
    """Handler text on a pulse task is bookkeeping: the completion is silent
    and carries no `response`, even though non-pulse summary-only results
    are promoted to the posted response."""

    @executor.on_task
    async def handler(_task):
        return {"summary": "some prose the model wrote instead of calling pulse_report"}

    post, calls = _fake_post()
    with patch.object(executor, "_post", new=AsyncMock(side_effect=post)):
        await executor._handle_task(_pulse_task())

    assert [name for name, _ in calls] == ["complete_task"]
    args = calls[0][1]
    assert args["silent"] is True
    assert "response" not in args
