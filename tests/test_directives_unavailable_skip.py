"""The directives-unavailable skip in handle_message must clear the
"thinking" bubble, exactly like the other pre-model early returns.

When a human sends a message the backend paints an InstantAgentSignal
stream for the targeted agent within ~50ms. Every `return None` in
handle_message that happens before the model call has to cancel that
stream, or the bubble ghosts for ~60s until the TimeoutServer sweep and
the human sees an agent that "started working, then nothing". The
skipMessage, skipTrivialMessage and triage-TASK returns already did; the
no-promptDirectives return (server directive pipeline down, no cached
copy) did not. Pins 2.10.6.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.executor import ExecutorClient, GatewayMessage

import agent_bridge


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


def _msg(**overrides) -> GatewayMessage:
    fields = dict(
        id="queue-1",
        message_id="trigger-1",
        conversation_id="conv-1",
        content="please do the thing",
        directives=None,  # preloader timed out: no promptDirectives on the wire
    )
    fields.update(overrides)
    return GatewayMessage(**fields)


@pytest.mark.asyncio
async def test_skip_cancels_the_signal_bubble_and_returns_none(executor, caplog):
    """The bubble cancel is a status=cancelled stream update from this
    agent (the endpoint cancels by senderId, so the stream id only needs
    to be unique), and the helper's return value is what handle_message
    returns — None, so the executor acks without posting anything."""
    msg = _msg()

    with (
        patch.object(executor, "send_stream_update", new=AsyncMock()) as stream,
        caplog.at_level("WARNING"),
    ):
        result = await agent_bridge._skip_directives_unavailable(
            executor, msg, msg.conversation_id, executor_key="test",
        )

    assert result is None
    stream.assert_awaited_once_with(
        "conv-1", "signal-cancel:queue-1", status="cancelled",
    )


@pytest.mark.asyncio
async def test_skip_logs_the_conversation_not_the_content(executor, caplog):
    """Operators need to find the skipped turn (conversation + message id);
    the message body must not leak into the bridge journal."""
    msg = _msg(content="SECRET-BODY-DO-NOT-LOG")

    with (
        patch.object(executor, "send_stream_update", new=AsyncMock()),
        caplog.at_level("WARNING"),
    ):
        await agent_bridge._skip_directives_unavailable(
            executor, msg, msg.conversation_id, executor_key="test",
        )

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings, "expected a WARNING that the message was skipped"
    assert any("directives unavailable" in r.getMessage() for r in warnings)
    assert any("conv-1" in r.getMessage() for r in warnings)
    assert all("SECRET-BODY-DO-NOT-LOG" not in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_skip_is_best_effort_when_the_cancel_fails(executor):
    """A failed cancel must not turn a silent skip into a crashed handler
    (the bubble then expires on its own, as before)."""
    msg = _msg()

    with patch.object(
        executor, "send_stream_update", new=AsyncMock(side_effect=RuntimeError("down")),
    ):
        result = await agent_bridge._skip_directives_unavailable(
            executor, msg, msg.conversation_id, executor_key="test",
        )

    assert result is None


@pytest.mark.asyncio
async def test_cancel_is_a_noop_without_a_conversation(executor):
    msg = _msg(conversation_id="")

    with patch.object(executor, "send_stream_update", new=AsyncMock()) as stream:
        await agent_bridge._cancel_signal_bubble(executor, msg)

    stream.assert_not_awaited()


def test_handle_message_returns_the_skip_helper_before_any_model_call():
    """Guard the call site, not just the helper.

    handle_message is a closure inside main(), so the contract is pinned
    structurally: the no-promptDirectives branch must `return await` the
    helper (which cancels the bubble) as the FIRST thing after the check,
    and that check must sit before the model is ever reached. Re-inlining
    a bare `return None` there would reintroduce the ghost bubble while
    every test above still passed.
    """
    src = open(agent_bridge.__file__, encoding="utf-8").read()
    marker = 'if not directives.get("promptDirectives"):\n'
    # The task path has the same check (it fails the task instead);
    # the message path is the one that follows the `msg.directives` cache read.
    start = src.index("def handle_message(")
    branch = src.index(marker, start) + len(marker)
    next_line = src[branch:].lstrip().split("\n", 1)[0]
    assert next_line == (
        "return await _skip_directives_unavailable(executor, msg, conv_id, executor_key)"
    )
    # No model call between the start of the handler and the skip.
    assert "backend.chat(" not in src[start:branch]
