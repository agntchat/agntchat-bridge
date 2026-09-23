"""A task run that ends with end_turn(blocked | awaiting_input) is parked,
not completed (bridge 2.11.9)."""

from __future__ import annotations

from agentchat.backends import ToolCall

import agent_bridge


class _Result:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.metadata = {}


def _call(name, arguments, is_error=False):
    return ToolCall(id="t1", name=name, arguments=arguments, result="", is_error=is_error)


def test_blocked_and_awaiting_input_park_the_task():
    for reason in ("blocked", "awaiting_input"):
        r = _Result([_call("mcp__agentgram__end_turn", {"reason": reason})])
        assert agent_bridge._task_parked_reason(r) == reason


def test_other_reasons_and_refused_calls_do_not_park():
    assert agent_bridge._task_parked_reason(_Result([_call("end_turn", {"reason": "task_complete"})])) is None
    assert agent_bridge._task_parked_reason(_Result([_call("end_turn", {"reason": "no_action_needed"})])) is None
    assert agent_bridge._task_parked_reason(_Result([_call("end_turn", {"reason": "blocked"}, is_error=True)])) is None
    assert agent_bridge._task_parked_reason(_Result([])) is None
