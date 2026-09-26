"""The tool catalog reaches the MCP server as a file, never as an env/argv string.

Linux caps one env or argv string at 128 KiB (MAX_ARG_STRLEN). A 125-tool
GitHub-heavy catalog serializes to ~137 KB, and passing it inline made the
server's spawn fail on every turn (2026-09-22..26).
"""

from __future__ import annotations

import json
import os

from agentchat.backends.claude_cli import ClaudeCliBackend
from agentchat.backends.codex_cli import CodexCliBackend

_BIG = [{"name": f"tool_{i}", "description": "x" * 2000} for i in range(125)]
_LIMIT = 131072


def _assert_catalog_file(env: dict, cleanup: list[str]) -> None:
    assert "AGENTGRAM_TOOL_DEFS" not in env
    path = env["AGENTGRAM_TOOL_DEFS_FILE"]
    assert path in cleanup
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == _BIG
    assert all(len(str(v).encode()) < _LIMIT for v in env.values())


def test_claude_cli_writes_catalog_to_a_file():
    b = ClaudeCliBackend(cli_path="/bin/true", api_url="http://localhost", agent_id="agent-1", api_key="ak_x")
    b._mcp_server_script = "/x/agntchat_mcp_server.py"
    cleanup: list[str] = []
    try:
        cfg = json.loads(b._build_mcp_config(_BIG, "c", "t", "o", "s", "l", cleanup_paths=cleanup))
        _assert_catalog_file(cfg["mcpServers"]["agentgram"]["env"], cleanup)
    finally:
        for p in cleanup:
            os.unlink(p)


def test_codex_cli_writes_catalog_to_a_file():
    b = CodexCliBackend(cli_path="/bin/true", api_url="http://localhost", agent_id="agent-1", api_key="ak_x")
    b._mcp_server_script = "/x/agntchat_mcp_server.py"
    cleanup: list[str] = []
    try:
        ov = b._mcp_overrides("c", "t", "o", "s", "l", _BIG, cleanup)
        assert all(len(a.encode()) < _LIMIT for a in ov)
        env = {}
        for a in ov:
            if a.startswith("mcp_servers.agentgram.env."):
                k, v = a[len("mcp_servers.agentgram.env."):].split("=", 1)
                env[k] = json.loads(v)
        _assert_catalog_file(env, cleanup)
    finally:
        for p in cleanup:
            os.unlink(p)
