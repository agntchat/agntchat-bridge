"""A tool call missing a schema-required argument is refused before dispatch
with the field named and described (bridge 2.11.10)."""

from __future__ import annotations

import asyncio
import json

from agentchat.tools.executor import ToolExecutor


class _Client:
    async def get_calendar_event(self, event_id: str, *, calendar_id: str | None = None):
        return {"id": event_id, "calendar_id": calendar_id}


def _executor():
    tools = [{
        "name": "get_calendar_event",
        "executorMethod": "get_calendar_event",
        "inputSchema": {
            "type": "object",
            "required": ["event_id"],
            "properties": {
                "event_id": {"type": "string", "description": "The event's id from list_calendar_events."},
                "calendar_id": {"type": "string"},
            },
        },
    }]
    return ToolExecutor(_Client(), context={}, resolved_tools=tools)


def test_missing_required_argument_is_refused_with_the_field_described():
    out = json.loads(asyncio.run(_executor().execute("get_calendar_event", {"calendar_id": "cal"})))
    assert out["missing"] == ["event_id"]
    assert "event_id" in out["error"]
    assert "list_calendar_events" in out["detail"]


def test_present_required_argument_dispatches():
    out = json.loads(asyncio.run(_executor().execute("get_calendar_event", {"event_id": "e1", "calendar_id": "cal"})))
    assert out["id"] == "e1"
