"""Regression: the Claude CLI's "workflow"/"workflows" keyword trigger must be
off on every invocation.

The CLI scans the whole piped prompt for that word and, when it finds it,
injects a meta turn telling the model to use the Workflow tool. We pipe the
rendered transcript plus the server's volatileContext, so one stored memory
reading "trip planning workflow" fired it on every single turn — and the
Workflow tool isn't in --tools, so the agent received an unexecutable
instruction attributed to its owner. Agents read that as a prompt injection
and told the owner so mid-conversation (2026-09-08, 2026-09-12).
"""

from __future__ import annotations

import json

from agentchat.backends.claude_cli import ClaudeCliBackend


def _backend(max_tokens: int | None = None) -> ClaudeCliBackend:
    kwargs = {
        "cli_path": "/bin/true",
        "api_url": "http://localhost",
        "agent_id": "agent-test",
        "api_key": "ak_test",
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ClaudeCliBackend(**kwargs)


def _settings(cmd: list[str]) -> dict:
    assert "--settings" in cmd, "--settings must always be passed"
    return json.loads(cmd[cmd.index("--settings") + 1])


def test_keyword_trigger_disabled_without_max_tokens() -> None:
    cmd, _prompt, _cleanup = _backend()._base_cmd("a prompt about our workflow")
    assert _settings(cmd)["workflowKeywordTriggerEnabled"] is False


def test_keyword_trigger_disabled_alongside_max_tokens() -> None:
    cmd, _prompt, _cleanup = _backend(max_tokens=16000)._base_cmd("hi")
    settings = _settings(cmd)
    assert settings["workflowKeywordTriggerEnabled"] is False
    assert settings["maxOutputTokens"] == 16000
