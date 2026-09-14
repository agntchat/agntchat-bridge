"""Agent queries: agent-owned LLM work run on the agent's own seat.

The property under test is provider-agnosticism. Nothing in this path may
assume a Claude seat — the handler must run whatever backend the bridge was
started with, and report back the model that actually answered. A regression
would not crash; it would silently re-lock non-Claude users to Claude.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agentchat.backends import MAX_TOKENS_OVERRIDE, ModelResult
from agentchat.executor import AgentQuery


def test_from_dict_reads_the_camelcase_wire_shape():
    query = AgentQuery.from_dict(
        {
            "id": "q1",
            "agentId": "a1",
            "label": "soul_revision",
            "systemPrompt": "You rewrite souls.",
            "userContent": "Make me terser.",
            "maxTokens": 8192,
            "tier": "deep",
            "expiresAt": "2026-09-14T12:00:00Z",
        }
    )

    assert query.id == "q1"
    assert query.label == "soul_revision"
    assert query.system_prompt == "You rewrite souls."
    assert query.user_content == "Make me terser."
    assert query.max_tokens == 8192
    assert query.tier == "deep"


def test_tier_defaults_to_fast_and_carries_no_model():
    query = AgentQuery.from_dict(
        {"id": "q1", "systemPrompt": "s", "userContent": "u"}
    )

    assert query.tier == "fast"
    # The server must never name a model: the seat decides.
    assert not hasattr(query, "model")


def test_null_tier_falls_back_to_fast():
    # The server omits `tier` via reject_nil when it is unset.
    query = AgentQuery.from_dict(
        {"id": "q1", "systemPrompt": "s", "userContent": "u", "tier": None}
    )

    assert query.tier == "fast"


class _FakeBackend:
    """Stands in for whatever seat the agent actually runs on."""

    def __init__(self, model: str):
        self._model = model
        self._max_tokens = 4096
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> ModelResult:
        self.calls.append((system_prompt, user_prompt))
        return ModelResult(
            text="answered",
            model=self._model,
            elapsed_seconds=0.2,
            usage={"input_tokens": 10, "output_tokens": 4},
        )


async def _handle(backend, query: AgentQuery) -> dict[str, Any]:
    """The handler contract agent_bridge.py registers, in miniature."""
    if query.max_tokens:
        MAX_TOKENS_OVERRIDE.set(query.max_tokens)
    result = await backend.generate(query.system_prompt, query.user_content)
    return {
        "text": result.text or "",
        "model": result.model,
        "usage": result.usage or {},
    }


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5", "gpt-5-codex", "some-local-llm-v3"],
)
def test_any_seat_can_answer_and_reports_its_own_model(model):
    backend = _FakeBackend(model)
    query = AgentQuery.from_dict(
        {"id": "q1", "systemPrompt": "sys", "userContent": "user"}
    )

    result = asyncio.run(_handle(backend, query))

    assert result["text"] == "answered"
    # Reported back verbatim — an OpenAI seat attributes as honestly as a
    # Claude one.
    assert result["model"] == model
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 4}


def test_query_carries_no_directives_or_history():
    backend = _FakeBackend("claude-sonnet-5")
    query = AgentQuery.from_dict(
        {"id": "q1", "systemPrompt": "sys", "userContent": "user"}
    )

    asyncio.run(_handle(backend, query))

    # Exactly what the server sent — no directive stack, no transcript.
    # That is what keeps this the cost of an API call, not of a full turn.
    assert backend.calls == [("sys", "user")]


def test_max_tokens_override_widens_a_narrow_backend_default():
    backend = _FakeBackend("gpt-5-codex")
    query = AgentQuery.from_dict(
        {
            "id": "q1",
            "systemPrompt": "sys",
            "userContent": "user",
            "maxTokens": 32768,
        }
    )

    async def run():
        await _handle(backend, query)
        # A 32K consolidation must not truncate against a 4096 default.
        return MAX_TOKENS_OVERRIDE.get()

    assert asyncio.run(run()) == 32768
