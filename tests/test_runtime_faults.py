"""Runtime facts reach the backend: failed turns and the MCP server's status."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import agent_bridge
from agentchat.backends import BackendToolsUnavailableError, ModelResult


class _RecordingExecutor:
    def __init__(self, fail: bool = False) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.fail = fail

    async def _post(self, path, json=None, extra_headers=None):  # noqa: A002
        if self.fail:
            raise RuntimeError("backend down")
        self.posts.append((path, json))
        return {}


def _stats(session: dict) -> dict:
    r = ModelResult(text="", model="m", elapsed_seconds=0.0, metadata={"cli_session_tools": session})
    return agent_bridge._turn_stats(r)


def test_turn_stats_carry_the_agentgram_status():
    s = _stats({"mcp": 0, "tool_search": True, "mcp_servers": [{"name": "agentgram", "status": "failed"}]})
    assert s["mcp_server"] == "failed"


def test_expected_but_unlisted_server_is_absent():
    assert _stats({"mcp": 0, "tool_search": True, "mcp_servers": [], "expect_mcp": True})["mcp_server"] == "absent"


def test_no_status_when_no_mcp_was_configured():
    assert "mcp_server" not in _stats({"mcp": 0, "tool_search": False, "mcp_servers": [], "expect_mcp": False})


def test_fault_kind_names_unreachable_tools():
    assert agent_bridge._model_fault_kind(BackendToolsUnavailableError("x")) == "tools_unreachable"
    assert agent_bridge._model_fault_kind(RuntimeError("x")) == "model_error"


async def _report_and_drain(ex, kind, msg, detail):
    await agent_bridge._report_runtime_fault(ex, "k", kind, msg, detail)
    # Fire-and-forget: the report runs as its own task.
    await asyncio.gather(*agent_bridge._fault_reports)


def test_fault_report_posts_the_turn_it_belongs_to():
    ex = _RecordingExecutor()
    msg = SimpleNamespace(conversation_id="c1", message_id="m1")
    asyncio.run(_report_and_drain(ex, "reply_post_failed", msg, "422"))
    assert ex.posts == [(
        "/api/agents/me/runtime-faults",
        {"kind": "reply_post_failed", "conversation_id": "c1", "message_id": "m1", "detail": "422"},
    )]


def test_fault_report_never_raises():
    msg = SimpleNamespace(conversation_id="c1", message_id="m1")
    asyncio.run(_report_and_drain(_RecordingExecutor(fail=True), "model_error", msg, "boom"))


def test_fault_report_does_not_wait_for_the_backend():
    class _Slow(_RecordingExecutor):
        async def _post(self, path, json=None, extra_headers=None):  # noqa: A002
            await asyncio.sleep(10)

    async def run():
        msg = SimpleNamespace(conversation_id="c1", message_id="m1")
        await asyncio.wait_for(agent_bridge._report_runtime_fault(_Slow(), "k", "model_error", msg, "x"), 0.5)
        for t in list(agent_bridge._fault_reports):
            t.cancel()

    asyncio.run(run())


def test_task_fault_report_names_the_task():
    # Only unreachable tools are reported from task turns; the executor
    # spawns this for BackendToolsUnavailableError and nothing else.
    from agentchat.executor import ExecutorClient, GatewayTask

    ex = _RecordingExecutor()
    task = GatewayTask(id="q1", task_id="t1", title="T", conversation_id="c1")
    asyncio.run(ExecutorClient._report_task_fault(ex, task, BackendToolsUnavailableError("agentgram failed")))
    [(path, body)] = ex.posts
    assert path == "/api/agents/me/runtime-faults"
    assert body["kind"] == "tools_unreachable"
    assert body["conversation_id"] == "c1"
    assert body["meta"] == {"task_id": "t1"}
