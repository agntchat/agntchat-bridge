"""An "auto" agent's per-turn model and effort arrive on the message.

The backend (Agentchat.Agents.AutoModel) stamps `turnOverride` on a queued
message for an agent whose model_config.model is "auto". The bridge sets the
MODEL_OVERRIDE and EFFORT_OVERRIDE contextvars from it for that handler task
only — the same isolation the task path's model_override relies on.
"""

from __future__ import annotations

import asyncio
import contextvars

from agentchat.backends import EFFORT_OVERRIDE, MODEL_OVERRIDE
from agentchat.executor import GatewayMessage


def test_gateway_message_carries_turn_override():
    msg = GatewayMessage.from_dict(
        {
            "id": "qm1",
            "messageId": "m1",
            "conversationId": "c1",
            "turnOverride": {"model": "claude-haiku-4-5", "effort": "low", "tier": 1, "source": "turn_class"},
        }
    )
    assert msg.turn_override == {
        "model": "claude-haiku-4-5",
        "effort": "low",
        "tier": 1,
        "source": "turn_class",
    }


def test_gateway_message_without_turn_override_is_none():
    msg = GatewayMessage.from_dict({"id": "qm1", "messageId": "m1", "conversationId": "c1"})
    assert msg.turn_override is None


def test_effort_override_defaults_to_none_and_is_task_scoped():
    assert EFFORT_OVERRIDE.get() is None

    async def turn(effort: str) -> str | None:
        EFFORT_OVERRIDE.set(effort)
        await asyncio.sleep(0)
        return EFFORT_OVERRIDE.get()

    async def run():
        a, b = await asyncio.gather(
            asyncio.create_task(turn("low")), asyncio.create_task(turn("high"))
        )
        return a, b, EFFORT_OVERRIDE.get()

    a, b, parent = asyncio.run(run())
    assert (a, b) == ("low", "high")
    assert parent is None


def test_claude_cli_reads_effort_override_per_request():
    from agentchat.backends.claude_cli import ClaudeCliBackend

    backend = ClaudeCliBackend.__new__(ClaudeCliBackend)
    backend._effort = "medium"

    assert backend._request_effort() == "medium"

    ctx = contextvars.copy_context()

    def inside():
        EFFORT_OVERRIDE.set("high")
        return backend._request_effort()

    assert ctx.run(inside) == "high"
    # The override did not escape the context it was set in.
    assert backend._request_effort() == "medium"

    def invalid():
        EFFORT_OVERRIDE.set("turbo")
        return backend._request_effort()

    assert contextvars.copy_context().run(invalid) == "medium"
    assert MODEL_OVERRIDE.get() is None


def test_effective_model_name_reflects_the_override():
    from agentchat.backends.claude_cli import ClaudeCliBackend

    backend = ClaudeCliBackend.__new__(ClaudeCliBackend)
    backend._model = "claude-sonnet-4-6"
    assert backend.effective_model_name == "claude-cli (claude-sonnet-4-6)"

    def inside():
        MODEL_OVERRIDE.set("claude-haiku-4-5")
        return backend.effective_model_name

    assert contextvars.copy_context().run(inside) == "claude-cli (claude-haiku-4-5)"
    assert backend.effective_model_name == "claude-cli (claude-sonnet-4-6)"
