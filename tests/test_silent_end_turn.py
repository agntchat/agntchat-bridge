"""Coverage for the silent-end_turn prose drop.

When the model calls `end_turn` to be silent this turn (reason
`no_action_needed` / `thread_redirect`) and ALSO emits prose, that prose is
the declined turn leaking out ("Trip King's got the hotel lookup — nothing
for me to add here", posted 2s after the tool call, conv 0b86e6ed). The
bridge must not post it. A TERMINATOR reason (`task_complete`,
`awaiting_input`, `blocked`) is the opposite case — the model delivered its
answer as final text and end_turn just closes the turn — so the text stays.

These pin `_tool_call_arguments` / `_silent_end_turn_called` across the ways
backends surface the call: `result.tool_calls[].arguments` (API backends)
and `result.metadata["cli_tool_uses"][].arguments` (CLI internal loops).
"""

from __future__ import annotations

from agent_bridge import _silent_end_turn_called, _tool_call_arguments, _tool_was_called
from agentchat.backends import ModelResult, ToolCall
from agentchat.backends.claude_cli import _tool_calls_from_cli_tally


def _call(name, arguments=None, *, is_error=False) -> ToolCall:
    return ToolCall(id=f"tu_{name}", name=name, arguments=dict(arguments or {}), result="", is_error=is_error)


def _result(tool_calls=None, cli_tool_uses=None) -> ModelResult:
    """A real `ModelResult`, the shape every backend hands the bridge —
    never a namespace fake, which would let a field the dataclass lacks
    (`is_error`, before 2.10.4) pass the tally filters in the test while
    the production object silently failed them."""
    return ModelResult(
        text="",
        model="test",
        elapsed_seconds=0.0,
        tool_calls=[_call(n, a) for n, a in (tool_calls or [])],
        metadata={"cli_tool_uses": [dict(tu) for tu in (cli_tool_uses or [])]},
    )


def _cli_result(cli_tool_uses, text="") -> ModelResult:
    """A `ModelResult` built the way `ClaudeCliBackend.generate` builds it
    from the CLI's tool-use tally: the same list lands BOTH in
    `metadata["cli_tool_uses"]` and, hoisted through
    `_tool_calls_from_cli_tally`, in `tool_calls`."""
    tally = [dict(tu) for tu in cli_tool_uses]
    return ModelResult(
        text=text,
        model="claude-cli",
        elapsed_seconds=0.0,
        tool_calls=_tool_calls_from_cli_tally(tally),
        metadata={"cli_tool_uses": tally, "cli_internal_loop": True},
    )


class TestToolCallArguments:
    def test_none_when_not_called(self):
        assert _tool_call_arguments(_result(tool_calls=[("send_message", {})]), "end_turn") is None

    def test_reads_tool_calls_arguments(self):
        r = _result(tool_calls=[("end_turn", {"reason": "task_complete"})])
        assert _tool_call_arguments(r, "end_turn") == {"reason": "task_complete"}

    def test_reads_namespaced_cli_tool_uses_arguments(self):
        r = _result(
            cli_tool_uses=[{"name": "mcp__agentgram__end_turn", "arguments": {"reason": "blocked"}}]
        )
        assert _tool_call_arguments(r, "end_turn") == {"reason": "blocked"}

    def test_uncaptured_arguments_yield_empty_dict(self):
        r = _result(cli_tool_uses=[{"name": "mcp__agentgram__end_turn"}])
        assert _tool_call_arguments(r, "end_turn") == {}

    def test_last_call_wins(self):
        r = _result(
            tool_calls=[
                ("end_turn", {"reason": "no_action_needed"}),
                ("end_turn", {"reason": "task_complete"}),
            ]
        )
        assert _tool_call_arguments(r, "end_turn") == {"reason": "task_complete"}


class TestSilentEndTurnCalled:
    def test_not_called(self):
        assert not _silent_end_turn_called(_result())

    def test_no_action_needed_is_silent(self):
        r = _result(cli_tool_uses=[{"name": "mcp__agentgram__end_turn", "arguments": {"reason": "no_action_needed"}}])
        assert _silent_end_turn_called(r)

    def test_thread_redirect_is_silent(self):
        assert _silent_end_turn_called(_result(tool_calls=[("end_turn", {"reason": "thread_redirect"})]))

    def test_missing_reason_defaults_to_silent(self):
        # Server coerces a missing/unknown reason to no_action_needed.
        assert _silent_end_turn_called(_result(tool_calls=[("end_turn", {})]))
        assert _silent_end_turn_called(_result(tool_calls=[("end_turn", {"reason": "   "})]))

    def test_terminator_reasons_keep_the_text(self):
        for reason in ("task_complete", "awaiting_input", "blocked"):
            assert not _silent_end_turn_called(
                _result(tool_calls=[("end_turn", {"reason": reason})])
            ), reason


class TestRejectedEndTurn:
    """A tool call the server REFUSED did not end anything.

    The backend rejects `end_turn(no_action_needed)` when a human addressed
    the agent directly; the CLI parser stamps that verdict as `is_error` on
    the recorded tool use. Reading the refused call as chosen silence dropped
    the 1373-char answer Botty wrote right after it (conv 425d14b0).

    The first fix (2.9.3) filtered `is_error` on both tallies but
    `ToolCall` had no such field, so the hoisted `tool_calls` copy of the
    same refused call still counted and the answer kept getting dropped.
    """

    def test_rejected_silent_end_turn_is_not_silence(self):
        r = _result(
            cli_tool_uses=[
                {
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "no_action_needed"},
                    "is_error": True,
                }
            ]
        )
        assert not _silent_end_turn_called(r)
        assert not _tool_was_called(r, "end_turn")
        assert _tool_call_arguments(r, "end_turn") is None

    def test_rejected_then_successful_call_resolves_to_the_successful_one(self):
        r = _result(
            cli_tool_uses=[
                {
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "no_action_needed"},
                    "is_error": True,
                },
                {
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "blocked"},
                    "is_error": False,
                },
            ]
        )
        assert _tool_was_called(r, "end_turn")
        assert _tool_call_arguments(r, "end_turn") == {"reason": "blocked"}
        assert not _silent_end_turn_called(r)

    def test_api_backend_tool_calls_honor_is_error(self):
        r = ModelResult(
            text="",
            model="test",
            elapsed_seconds=0.0,
            tool_calls=[_call("end_turn", {"reason": "no_action_needed"}, is_error=True)],
        )
        assert not _tool_was_called(r, "end_turn")
        assert not _silent_end_turn_called(r)

    def test_tool_call_carries_is_error(self):
        # The dataclass itself must own the flag; a `getattr(..., False)`
        # filter over an object without it is dead code.
        assert ToolCall(id="x", name="end_turn", arguments={}, result="").is_error is False
        assert ToolCall(id="x", name="end_turn", arguments={}, result="", is_error=True).is_error is True

    def test_hoisted_cli_tally_keeps_the_refusal(self):
        # claude_cli.generate rebuilds result.tool_calls from cli_tool_uses;
        # the verdict must survive that rebuild or the metadata filter is
        # undone by the tool_calls copy of the same call.
        r = _cli_result(
            [
                {
                    "id": "tu1",
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "no_action_needed"},
                    "is_error": True,
                }
            ],
            text="Here are the Amazon links you asked for.",
        )
        [tc] = r.tool_calls
        assert tc.is_error is True
        assert not _tool_was_called(r, "end_turn")
        assert not _silent_end_turn_called(r)
        assert _tool_call_arguments(r, "end_turn") is None

    def test_hoisted_cli_tally_counts_the_accepted_call(self):
        r = _cli_result(
            [
                {
                    "id": "tu1",
                    "name": "mcp__agentgram__end_turn",
                    "arguments": {"reason": "no_action_needed"},
                    "is_error": False,
                }
            ]
        )
        [tc] = r.tool_calls
        assert tc.is_error is False
        assert _tool_was_called(r, "end_turn")
        assert _silent_end_turn_called(r)

    def test_hoisted_cli_tally_without_verdict_counts_as_success(self):
        # A tool use the CLI never delivered a tool_result for (stream cut
        # short) has no verdict; it is not read as refused.
        r = _cli_result([{"id": "tu1", "name": "mcp__agentgram__end_turn", "arguments": {"reason": "blocked"}}])
        assert r.tool_calls[0].is_error is False
        assert _tool_was_called(r, "end_turn")


class TestRejectedSendMessage:
    """A refused / guardrail-blocked `send_message` delivered nothing.

    `_tu_sent_via_tool` gates whether the bridge posts the model's text
    itself; counting a failed call as delivered swallows the reply.
    """

    def test_failed_send_message_is_not_delivery(self):
        r = ModelResult(
            text="",
            model="test",
            elapsed_seconds=0.0,
            tool_calls=[_call("send_message", {"content": "hi"}, is_error=True)],
        )
        assert not _tool_was_called(r, "send_message")

    def test_successful_send_message_counts(self):
        r = ModelResult(
            text="",
            model="test",
            elapsed_seconds=0.0,
            tool_calls=[_call("send_message", {"content": "hi"})],
        )
        assert _tool_was_called(r, "send_message")

    def test_failed_then_successful_send_counts_once_delivered(self):
        r = ModelResult(
            text="",
            model="test",
            elapsed_seconds=0.0,
            tool_calls=[
                _call("send_message", {"content": "hi"}, is_error=True),
                _call("send_message", {"content": "hi"}),
            ],
        )
        assert _tool_was_called(r, "send_message")
