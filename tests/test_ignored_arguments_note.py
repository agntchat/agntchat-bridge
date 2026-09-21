"""A tool argument the SDK method cannot take is reported back to the model
(bridge 2.11.7), instead of being dropped with only a log line.

Gmail passed `message` to find_or_create_dm on 2026-09-21; the kwargs fit
dropped it, the thread opened empty, and the task closed as "pinged Kal, no
answer yet". The model can only act on what the result tells it.
"""

from __future__ import annotations

import json

from agentchat.tools.executor import _note_ignored_arguments


def test_dict_results_carry_the_ignored_list_and_a_note():
    out = json.loads(_note_ignored_arguments(json.dumps({"id": "c1"}), ["message"]))
    assert out["id"] == "c1"
    assert out["_ignored_arguments"] == ["message"]
    assert "message" in out["_note"]
    assert "ignored" in out["_note"]


def test_plain_text_results_get_a_trailing_note():
    out = _note_ignored_arguments("DM ready", ["message", "topic"])
    assert out.startswith("DM ready")
    assert "message, topic" in out
