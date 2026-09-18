"""`tool_result_is_error` reads the executor's failure contract.

`ToolExecutor.execute` never raises: every failure — a raised SDK call, an
unknown tool, a rejected placeholder argument, the server's structured
refusal — comes back as a JSON object with a top-level `error` key. That is
the only signal the API-backend tool loops have for stamping
`ToolCall.is_error`, which the bridge reads after the run to tell a call
that happened from one that was refused.
"""

from __future__ import annotations

import json

from agentchat.backends import ToolCall, tool_result_is_error


def test_error_object_is_an_error():
    assert tool_result_is_error(json.dumps({"error": "Task task-1 not found."}))
    assert tool_result_is_error(json.dumps({"error": "boom", "code": "missing_conversation_id"}))


def test_success_shapes_are_not_errors():
    assert not tool_result_is_error("")
    assert not tool_result_is_error(json.dumps({"status": "sent", "message_id": "m1"}))
    assert not tool_result_is_error(json.dumps({"error": None, "ok": True}))
    assert not tool_result_is_error(json.dumps([{"error": "inside a list"}]))
    assert not tool_result_is_error("Turn ended.")
    assert not tool_result_is_error('{"error": "unterminated')


def test_tool_call_defaults_to_success():
    tc = ToolCall(id="x", name="send_message", arguments={}, result="")
    assert tc.is_error is False
    assert ToolCall(id="x", name="send_message", arguments={}, result="", is_error=True).is_error
