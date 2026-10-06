"""Server-owned prompt-cache horizon (bridge 2.11.24).

The backend derives one prefix-stability horizon and ships it on every turn
as ``behavioralConfig.promptCache``. The bridge applies two numbers from it
and decides nothing:

1. ``anthropicTtl`` ("5m" | "1h") goes on all four cache_control breakpoints
   the Anthropic API backend places — the 5m marker keeps the bare wire shape,
   the 1h marker is explicit.
2. ``historyWindowTtlSeconds`` becomes the TTL of the ``_cached_get_messages``
   window, so the rebase that changes the history prefix happens only once
   the Anthropic entry it asked for has expired anyway.

A missing or malformed block leaves the 5-minute defaults in place.
"""

import pytest

import agent_bridge
from agentchat.backends import ChatMessage, ModelBackend
from agentchat.backends.anthropic import AnthropicBackend, _cache_marker, _coalesce_messages


def _markers(api_messages):
    out = []
    for msg in api_messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    out.append(block["cache_control"])
    return out


class TestMarkerShape:
    def test_5m_is_the_bare_marker(self):
        assert _cache_marker("5m") == {"type": "ephemeral"}

    def test_1h_is_explicit(self):
        assert _cache_marker("1h") == {"type": "ephemeral", "ttl": "1h"}

    def test_every_breakpoint_helper_honours_the_ttl(self):
        system = AnthropicBackend._cached_system("You are Ada.", ttl="1h")
        assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

        tools = AnthropicBackend._cached_tools(
            [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}], ttl="1h"
        )
        assert tools[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in tools[0]

        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="plan?", cache_boundary=True),
            ChatMessage(role="user", content="tail"),
        ]
        api = AnthropicBackend._with_history_cache(
            AnthropicBackend._apply_cache_boundary(_coalesce_messages(messages), ttl="1h"),
            ttl="1h",
        )
        markers = _markers(api)
        assert len(markers) == 2
        assert all(m == {"type": "ephemeral", "ttl": "1h"} for m in markers)

    def test_default_ttl_keeps_the_old_wire_shape(self):
        system = AnthropicBackend._cached_system("You are Ada.")
        assert system[0]["cache_control"] == {"type": "ephemeral"}


class TestBackendSetter:
    def _backend(self):
        # Bypass __init__: it builds an SDK client. Only the TTL slot matters.
        b = AnthropicBackend.__new__(AnthropicBackend)
        b._cache_ttl = "5m"
        return b

    def test_accepts_known_ttls(self):
        b = self._backend()
        b.set_prompt_cache_ttl("1h")
        assert b._cache_ttl == "1h"
        b.set_prompt_cache_ttl("5m")
        assert b._cache_ttl == "5m"

    def test_ignores_unknown_values(self):
        b = self._backend()
        b.set_prompt_cache_ttl("1h")
        for bad in ("2h", "", None, 3600):
            b.set_prompt_cache_ttl(bad)  # type: ignore[arg-type]
        assert b._cache_ttl == "1h"

    def test_base_backend_is_a_no_op(self):
        class Dummy(ModelBackend):
            async def generate(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

            @property
            def model_name(self):  # pragma: no cover
                return "dummy"

        assert Dummy().set_prompt_cache_ttl("1h") is None


class TestHistoryWindowTtl:
    @pytest.fixture(autouse=True)
    def _restore(self):
        before = agent_bridge._conv_cache_ttl
        yield
        agent_bridge._conv_cache_ttl = before

    def test_default_is_five_minutes(self):
        assert agent_bridge._CONV_CACHE_TTL_DEFAULT == 300

    def test_server_value_is_applied(self):
        agent_bridge._set_conv_cache_ttl(3600)
        assert agent_bridge._conv_cache_ttl == 3600.0
        agent_bridge._set_conv_cache_ttl("1800")
        assert agent_bridge._conv_cache_ttl == 1800.0

    def test_garbage_is_ignored(self):
        agent_bridge._set_conv_cache_ttl(3600)
        for bad in (None, "soon", 0, -5):
            agent_bridge._set_conv_cache_ttl(bad)
        assert agent_bridge._conv_cache_ttl == 3600.0

    @pytest.mark.asyncio
    async def test_window_survives_past_five_minutes_when_horizon_is_an_hour(self, monkeypatch):
        """The whole point: at 20 idle minutes a 1h horizon merges instead of refetching + rebasing."""
        agent_bridge._conv_message_cache.clear()
        agent_bridge._set_conv_cache_ttl(3600)

        fetches = []

        class Executor:
            async def get_messages(self, conversation_id, limit=20):
                fetches.append(limit)
                return [{"id": f"m{i}", "insertedAt": f"t{i}"} for i in range(limit)]

        clock = {"now": 1000.0}
        import time as _real_time
        monkeypatch.setattr(_real_time, "monotonic", lambda: clock["now"])

        ex = Executor()
        first = await agent_bridge._cached_get_messages(ex, "conv", limit=4)
        assert [m["id"] for m in first] == ["m0", "m1", "m2", "m3"]

        clock["now"] += 20 * 60  # twenty idle minutes
        new = [{"id": "m4", "insertedAt": "t4"}]
        second = await agent_bridge._cached_get_messages(ex, "conv", limit=4, preloaded=new)
        # Warm path: the anchored window grew instead of rebasing to the newest 4.
        assert [m["id"] for m in second] == ["m0", "m1", "m2", "m3", "m4"]
        assert fetches == [4]

        agent_bridge._set_conv_cache_ttl(300)
        clock["now"] += 20 * 60
        third = await agent_bridge._cached_get_messages(ex, "conv", limit=4, preloaded=new)
        # Expired window: full "fetch" from the payload, which rebases.
        assert [m["id"] for m in third] == ["m4"]
