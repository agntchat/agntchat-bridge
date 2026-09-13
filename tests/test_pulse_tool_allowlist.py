"""Pulse turns carry a server-sent `toolAllowlist` of BARE tool names, and the
bridge filters its advertised tool defs down to that list.

Tool defs come in two wire shapes: Anthropic (`{"name": ...}`) and OpenAI
(`{"type": "function", "function": {"name": ...}}`). Only the `anthropic`
backend gets the first shape — `claude_cli`, `codex_cli` and the rest all get
the OpenAI one. The original filter read `d.get("name", d.get("type"))`, which
for an OpenAI-shaped def yields the literal string "function", matches nothing,
and silently strips every tool from the turn.

Prod damage (2026-09-13): the Gmail agent's pulse turns logged
"toolAllowlist active ... 0/80 tool defs advertised" and the agent, genuinely
toolless, told its owner "Gmail access is currently unavailable — reconnect
via Profile → Connected Accounts" on an OAuth token that had been refreshed
eight minutes earlier and was used successfully eight minutes later.
"""

import agent_bridge
from agent_bridge import (
    _resolved_tools_to_anthropic,
    _resolved_tools_to_openai,
    _tool_def_name,
)

RESOLVED = [
    {"name": "list_emails", "description": "d", "inputSchema": {"type": "object"}},
    {"name": "send_message", "description": "d", "inputSchema": {"type": "object"}},
    {"name": "loop_report", "description": "d", "inputSchema": {"type": "object"}},
]


def _apply_allowlist(tool_defs, allowlist):
    """The production filter expression, verbatim (agent_bridge.py ~4214)."""
    allowed = set(allowlist)
    return [d for d in (tool_defs or []) if _tool_def_name(d) in allowed]


def test_openai_shaped_defs_survive_the_allowlist():
    """The regression: claude_cli agents get OpenAI-shaped defs."""
    defs = _resolved_tools_to_openai(RESOLVED)
    kept = _apply_allowlist(defs, ["list_emails", "send_message"])

    assert [_tool_def_name(d) for d in kept] == ["list_emails", "send_message"]


def test_anthropic_shaped_defs_still_survive_the_allowlist():
    defs = _resolved_tools_to_anthropic(RESOLVED)
    kept = _apply_allowlist(defs, ["list_emails", "send_message"])

    assert [_tool_def_name(d) for d in kept] == ["list_emails", "send_message"]


def test_allowlist_still_excludes_what_the_server_left_out():
    """The filter must actually filter — not just pass everything through."""
    for defs in (_resolved_tools_to_openai(RESOLVED), _resolved_tools_to_anthropic(RESOLVED)):
        kept = _apply_allowlist(defs, ["list_emails", "send_message"])
        assert "loop_report" not in [_tool_def_name(d) for d in kept]


def test_tool_def_name_reads_both_shapes():
    assert _tool_def_name({"name": "list_emails"}) == "list_emails"
    assert _tool_def_name({"type": "function", "function": {"name": "list_emails"}}) == "list_emails"


def test_tool_def_name_never_returns_the_type_discriminator():
    """The exact bug: "function" must never be mistaken for a tool name.

    If this regresses, a pulse allowlist silently matches zero defs again.
    """
    assert _tool_def_name({"type": "function", "function": {"name": "x"}}) != "function"
    assert _tool_def_name({"type": "function"}) is None
    assert _tool_def_name({}) is None


def test_module_uses_the_helper_not_a_raw_name_lookup():
    """Guard the call site itself, not just the helper.

    The helper is only useful if the filter goes through it; re-inlining
    `d.get("name", d.get("type"))` anywhere would reintroduce the outage
    while every test above still passed.
    """
    src = open(agent_bridge.__file__).read()
    assert 'd.get("name", d.get("type"))' not in src
