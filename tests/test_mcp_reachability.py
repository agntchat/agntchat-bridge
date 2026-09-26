"""A CLI session with an MCP server but no way to reach its tools is refused."""

from __future__ import annotations

from agentchat.backends.claude_cli import ClaudeCliBackend, _mcp_reachable, _session_tools_from_init


def test_tool_search_is_always_in_the_builtin_list():
    # Deferred MCP tools load through it; without it the platform tools are
    # configured but unreachable for models the CLI serves deferred tools to.
    assert "ToolSearch" in ClaudeCliBackend._CLI_TOOLS


def test_init_event_is_summarised():
    ev = {
        "type": "system", "subtype": "init", "model": "claude-opus-5",
        "tools": ["Bash", "Read", "ToolSearch", "mcp__agentgram__list_calendars"],
        "mcp_servers": [{"name": "agentgram", "status": "connected"}],
    }
    s = _session_tools_from_init(ev)
    assert s == {
        "total": 4, "mcp": 1, "tool_search": True,
        "mcp_servers": [{"name": "agentgram", "status": "connected"}], "model": "claude-opus-5",
    }


def test_reachable_when_attached_or_via_tool_search():
    assert _mcp_reachable({"mcp": 71, "tool_search": False}, True)[0]
    assert _mcp_reachable({"mcp": 0, "tool_search": True}, True)[0]


def test_unreachable_when_neither_and_mcp_expected():
    ok, why = _mcp_reachable({"mcp": 0, "tool_search": False, "total": 8, "mcp_servers": [{"name": "agentgram", "status": "pending"}]}, True)
    assert not ok
    assert "pending" in why


def test_failed_server_is_unreachable_even_with_tool_search():
    # 2026-09-22..26: the agentgram server failed to spawn on every turn and
    # ToolSearch alone passed this check, so the agent ran toolless for days.
    ok, why = _mcp_reachable({"mcp": 0, "tool_search": True, "total": 9, "mcp_servers": [{"name": "agentgram", "status": "failed"}]}, True)
    assert not ok
    assert "failed" in why


def test_pending_server_with_tool_search_is_reachable():
    # The deferred-startup race: ToolSearch waits for the server.
    assert _mcp_reachable({"mcp": 0, "tool_search": True, "mcp_servers": [{"name": "agentgram", "status": "pending"}]}, True)[0]


def test_nothing_to_reach_without_mcp():
    assert _mcp_reachable({"mcp": 0, "tool_search": False}, False) == (True, "no MCP expected")


def test_turn_stats_reads_tool_evidence():
    import agent_bridge
    from agentchat.backends import ModelResult, ToolCall

    r = ModelResult(text="", model="m", elapsed_seconds=0.0,
                    metadata={"cli_session_tools": {"mcp": 0, "tool_search": True}},
                    tool_calls=[ToolCall(id="1", name="x", arguments={}, result="")], iterations=3)
    assert agent_bridge._turn_stats(r) == {"tool_calls": 1, "iterations": 3, "mcp_tools_attached": 0, "tool_search": True}
