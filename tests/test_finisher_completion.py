"""A task the model closes itself through `complete_task` / `fail_task`
inside the run is not completed a second time by the bridge.

Prod artifact (Botty, 2026-09-15): the speaker-list self-task logged the
model's `complete_task` at 10:30:47 ("Task … completed."), then the
executor's own `complete_task` at 10:30:48 with a different, non-silent
summary — refused by the backend as "ALREADY complete". The API backends
stop their tool loop on a terminal tool; the CLI backends run their own
loop, so the bridge only sees the finisher in the tally after the run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent_bridge import _finisher_completion_fields
from agentchat.backends import ModelResult, ToolCall
from agentchat.executor import ExecutorClient, GatewayTask


def _result(tool_names: list[str], *, errored: set[str] | None = None, cli: list[str] | None = None):
    errored = errored or set()
    return ModelResult(
        text="",
        model="test",
        elapsed_seconds=0.0,
        tool_calls=[
            ToolCall(id=f"tu_{i}", name=n, arguments={}, result="", is_error=(n in errored))
            for i, n in enumerate(tool_names)
        ],
        metadata={"cli_tool_uses": [{"name": n} for n in (cli or [])]},
    )


def test_complete_task_call_marks_task_completed_by_tool():
    fields = _finisher_completion_fields(_result(["get_memory", "complete_task"]))
    assert fields == {"completed_via_tool": "complete_task"}


def test_fail_task_counts_too():
    assert _finisher_completion_fields(_result(["fail_task"])) == {"completed_via_tool": "fail_task"}


def test_cli_tally_with_mcp_prefix_and_kebab_name_counts():
    """claude_cli reports `mcp__agentgram__complete-task` in the tally."""
    fields = _finisher_completion_fields(_result([], cli=["mcp__agentgram__complete-task"]))
    assert fields == {"completed_via_tool": "complete_task"}


def test_no_finisher_means_bridge_completes():
    assert _finisher_completion_fields(_result(["get_memory", "send_message"])) == {}


def test_refused_finisher_does_not_count():
    fields = _finisher_completion_fields(_result(["complete_task"], errored={"complete_task"}))
    assert fields == {}


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


def _self_task() -> GatewayTask:
    return GatewayTask.from_dict(
        {
            "id": "queue-1",
            "taskId": "task-1",
            "task": {
                "title": "Look up speaker list",
                "conversationId": "group-conv",
                "source": "message_triage",
                "metadata": {},
            },
        }
    )


@pytest.mark.asyncio
async def test_executor_skips_completion_when_model_completed_the_task(executor):
    @executor.on_task
    async def handler(_task):
        return {
            "summary": "Done — task closed out with the full speaker list.",
            "response": "Done — task closed out with the full speaker list.",
            "completed_via_tool": "complete_task",
        }

    calls: list[tuple[str, dict]] = []

    async def post(path, json=None, **_kw):
        if path == "/api/mcp":
            calls.append((json["params"]["name"], json["params"]["arguments"]))
            return {"result": {"isError": False, "content": [{"type": "text", "text": "ok"}]}}
        return {}

    with patch.object(executor, "_post", new=AsyncMock(side_effect=post)):
        await executor._handle_task(_self_task())

    assert calls == [], f"no MCP completion expected after complete_task, got {calls}"
