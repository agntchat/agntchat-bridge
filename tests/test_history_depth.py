"""The server's per-turn `directives.historyDepth` trims the rendered
history to the newest N messages (bridge 2.11.6).

The number comes from Agentchat.Agents.PromptGate.history_depth/2 — the
turn class's `history` level — so the bridge only applies it: anything
that is not a positive integer leaves the window alone.
"""

from __future__ import annotations

from agentchat.backends import ChatMessage

import agent_bridge


def _history(n: int) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=f"m{i}", source_id=f"id{i}") for i in range(n)]


def test_keeps_the_newest_depth_messages():
    out = agent_bridge._apply_history_depth(_history(12), 4)
    assert [m.content for m in out] == ["m8", "m9", "m10", "m11"]


def test_a_window_already_within_depth_is_untouched():
    msgs = _history(3)
    assert agent_bridge._apply_history_depth(msgs, 4) is msgs


def test_absent_or_invalid_depth_leaves_the_window_alone():
    msgs = _history(12)
    for depth in (None, 0, -1, "4", 4.0, True, {}):
        assert agent_bridge._apply_history_depth(msgs, depth) is msgs
