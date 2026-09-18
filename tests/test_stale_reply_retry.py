"""Coverage for the one-shot stale-reply retry and the shared post-parse
"nothing emitted" guard (bridge 2.10.5).

Stale retry: the server's 409 `stale_context` says "these messages arrived
while you drafted" and attaches them. When they are peer bubbles or side
notes the draft is still the answer to the human, so every anchored
reply/card send (`send_with_stale_retry`) re-posts the SAME content ONCE
with `last_seen_message_id` advanced past the newest of them. A second 409
propagates so the caller drops the draft exactly as before.

Post-parse guard: the tool_use branch already re-checked for an empty turn
after parsing; single_shot only had the raw-result guard, so a cards-only
or task_request-only reply (task creation disallowed) posted nothing with no
fallback. `_post_parse_fallback_needed` is the one guard both branches call,
and a successful `create_task` counts as "something emitted" — the task card
is the acknowledgement.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge import _post_parse_fallback_needed, send_parsed_presentations
from agentchat.backends import ModelResult, ToolCall
from agentchat.errors import StaleContextError
from agentchat.executor import newest_message_id, send_with_stale_retry


def _stale(new_messages) -> StaleContextError:
    return StaleContextError("API error 409: stale_context", new_messages=new_messages)


def _executor(side_effect) -> MagicMock:
    ex = MagicMock()
    ex.send_message = AsyncMock(side_effect=side_effect)
    return ex


# ---------------------------------------------------------------------------
# newest_message_id
# ---------------------------------------------------------------------------


class TestNewestMessageId:
    def test_max_by_inserted_at(self):
        msgs = [
            {"id": "m2", "insertedAt": "2026-09-18T10:00:02Z"},
            {"id": "m3", "insertedAt": "2026-09-18T10:00:03Z"},
            {"id": "m1", "insertedAt": "2026-09-18T10:00:01Z"},
        ]
        assert newest_message_id(msgs) == "m3"

    def test_falls_back_to_last_element_without_timestamps(self):
        assert newest_message_id([{"id": "a"}, {"id": "b"}]) == "b"
        # One missing insertedAt disables the timestamp ordering entirely.
        assert newest_message_id([{"id": "a", "insertedAt": "z"}, {"id": "b"}]) == "b"

    def test_none_when_nothing_usable(self):
        assert newest_message_id([]) is None
        assert newest_message_id(None) is None
        assert newest_message_id([{"insertedAt": "x"}, "junk"]) is None


# ---------------------------------------------------------------------------
# send_with_stale_retry
# ---------------------------------------------------------------------------


class TestSendWithStaleRetry:
    @pytest.mark.asyncio
    async def test_first_409_retries_once_with_advanced_anchor(self, caplog):
        """(a) first 409 → retry anchored past the new messages → success,
        posted exactly once more."""
        stale = _stale([
            {"id": "m2", "insertedAt": "2026-09-18T10:00:02Z"},
            {"id": "m3", "insertedAt": "2026-09-18T10:00:03Z"},
        ])
        ex = _executor([stale, {"id": "posted"}])

        with caplog.at_level(logging.INFO, logger="agentchat.executor"):
            out = await send_with_stale_retry(
                ex, "conv-1", "the answer",
                metadata={"model": "m"},
                last_seen_message_id="trigger-1",
                log_label="exec tool_use",
            )

        assert out == {"id": "posted"}
        assert ex.send_message.await_count == 2
        first, second = ex.send_message.await_args_list
        assert first.args == ("conv-1", "the answer")
        assert first.kwargs["last_seen_message_id"] == "trigger-1"
        assert first.kwargs["metadata"] == {"model": "m"}
        # Same content, same metadata, anchor moved to the newest new message.
        assert second.args == ("conv-1", "the answer")
        assert second.kwargs["last_seen_message_id"] == "m3"
        assert second.kwargs["metadata"] == {"model": "m"}
        assert any(
            "Stale reply: re-posting once with anchor advanced past 2 message(s)" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_second_409_propagates_after_exactly_two_attempts(self):
        """(b) the retry is stale too → the second StaleContextError reaches
        the caller (which drops the draft and cancels the stream); no third
        attempt."""
        first = _stale([{"id": "m2", "insertedAt": "2026-09-18T10:00:02Z"}])
        second = _stale([{"id": "m9", "insertedAt": "2026-09-18T10:00:09Z"}])
        ex = _executor([first, second])

        with pytest.raises(StaleContextError) as excinfo:
            await send_with_stale_retry(
                ex, "conv-1", "the answer",
                last_seen_message_id="trigger-1",
                log_label="exec single_shot",
            )

        assert excinfo.value is second
        assert ex.send_message.await_count == 2
        assert ex.send_message.await_args_list[1].kwargs["last_seen_message_id"] == "m2"

    @pytest.mark.asyncio
    async def test_409_without_usable_messages_is_not_retried(self):
        ex = _executor([_stale([])])
        with pytest.raises(StaleContextError):
            await send_with_stale_retry(
                ex, "conv-1", "x", last_seen_message_id="t", log_label="l",
            )
        assert ex.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_409_naming_the_current_anchor_is_not_retried(self):
        # Nothing to advance past — re-posting would just 409 again.
        ex = _executor([_stale([{"id": "t"}])])
        with pytest.raises(StaleContextError):
            await send_with_stale_retry(
                ex, "conv-1", "x", last_seen_message_id="t", log_label="l",
            )
        assert ex.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_success_is_a_single_send(self):
        ex = _executor([{"id": "posted"}])
        await send_with_stale_retry(
            ex, "conv-1", "x", last_seen_message_id="t", log_label="l",
        )
        ex.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_other_errors_pass_through_untouched(self):
        ex = _executor([RuntimeError("boom")])
        with pytest.raises(RuntimeError):
            await send_with_stale_retry(
                ex, "conv-1", "x", last_seen_message_id="t", log_label="l",
            )
        ex.send_message.assert_awaited_once()


class TestPresentationCardStaleRetry:
    @pytest.mark.asyncio
    async def test_card_retries_once_then_counts_as_sent(self):
        stale = _stale([{"id": "peer-1", "insertedAt": "2026-09-18T10:00:00Z"}])
        ex = _executor([stale, {"id": "card"}])

        sent = await send_parsed_presentations(
            ex, "conv-1", [{"result_type": "links", "title": "Links", "items": [1, 2]}],
            last_seen_message_id="trigger-1",
        )

        assert sent == 1
        assert ex.send_message.await_count == 2
        first, second = ex.send_message.await_args_list
        assert first.kwargs["message_type"] == "ResultPresentation"
        assert first.kwargs["last_seen_message_id"] == "trigger-1"
        assert second.kwargs["message_type"] == "ResultPresentation"
        assert second.kwargs["content_structured"] == first.kwargs["content_structured"]
        assert second.kwargs["last_seen_message_id"] == "peer-1"

    @pytest.mark.asyncio
    async def test_card_dropped_on_second_409(self):
        ex = _executor([_stale([{"id": "p1"}]), _stale([{"id": "p2"}])])
        sent = await send_parsed_presentations(
            ex, "conv-1", [{"result_type": "links", "items": []}],
            last_seen_message_id="trigger-1",
        )
        assert sent == 0
        assert ex.send_message.await_count == 2


# ---------------------------------------------------------------------------
# _post_parse_fallback_needed
# ---------------------------------------------------------------------------


def _call(name, arguments=None, *, is_error=False) -> ToolCall:
    return ToolCall(id=f"tu_{name}", name=name, arguments=dict(arguments or {}), result="", is_error=is_error)


def _result(text="", tool_calls=(), cli_tool_uses=()) -> ModelResult:
    return ModelResult(
        text=text,
        model="test",
        elapsed_seconds=0.0,
        tool_calls=list(tool_calls),
        metadata={"cli_tool_uses": [dict(tu) for tu in cli_tool_uses]},
    )


def _guard(result, **overrides) -> bool:
    kwargs = dict(
        reply="",
        presentations=[],
        task_requests=[],
        dm_blocks=[],
        human_expects_reply=True,
        failed=False,
    )
    kwargs.update(overrides)
    return _post_parse_fallback_needed(result, **kwargs)


class TestPostParseFallbackNeeded:
    def test_single_shot_reply_parsed_to_empty_gets_fallback(self):
        """(c) single_shot has no end_turn / send_message tools; a reply that
        parsed down to nothing with the human waiting posts the fallback."""
        assert _guard(_result()) is True
        # The single_shot branch passes its own flags: no tool, no failure.
        assert _guard(_result(), ended_turn=False, sent_via_tool=False) is True

    def test_task_request_only_reply_with_task_creation_disallowed(self):
        # parse_task_requests stripped the tag and, with task creation
        # disallowed, kept none of them: nothing left to post → fallback.
        assert _guard(_result(text="<task_request>...</task_request>"), task_requests=[]) is True
        # With task creation allowed the request itself is the emission.
        assert _guard(_result(), task_requests=[{"title": "do it"}]) is False

    def test_anything_emitted_is_not_a_fallback(self):
        assert _guard(_result(), reply="hi") is False
        assert _guard(_result(), presentations=[{"result_type": "x"}]) is False
        assert _guard(_result(), dm_blocks=[{"target": "A", "body": "b"}]) is False
        # Whitespace-only text is still nothing.
        assert _guard(_result(), reply="  \n ") is True

    def test_silence_is_allowed_when_nobody_is_waiting(self):
        assert _guard(_result(), human_expects_reply=False) is False

    def test_failed_turn_keeps_its_failure_copy(self):
        assert _guard(_result(), failed=True) is False

    def test_deliberate_end_turn_or_tool_delivery_is_not_broken(self):
        assert _guard(_result(), ended_turn=True) is False
        assert _guard(_result(), sent_via_tool=True) is False

    def test_successful_create_task_counts_as_emitted(self):
        """(d) tool_use turn that created a task and wrote no text: the task
        card is the acknowledgement — no apology on top of it."""
        r = _result(tool_calls=[_call("create_task", {"title": "Research links"})])
        assert _guard(r, ended_turn=False, sent_via_tool=False) is False

    def test_successful_create_task_via_cli_tally_counts_as_emitted(self):
        r = _result(cli_tool_uses=[{"name": "mcp__agentgram__create_task", "arguments": {"title": "t"}}])
        assert _guard(r) is False

    def test_refused_create_task_does_not_count(self):
        # A create_task the server rejected produced no card, so the turn
        # really did emit nothing.
        r = _result(tool_calls=[_call("create_task", {"title": "t"}, is_error=True)])
        assert _guard(r) is True

    def test_result_none_is_tolerated(self):
        # single_shot passes result=None on a failed run (failed=True), but
        # the tool tally lookup must not blow up on it either way.
        assert _guard(None) is True
        assert _guard(None, failed=True) is False
