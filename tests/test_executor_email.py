"""Tests for send_email in ExecutorClient.

Body normalization (escape sequences, markdown backslash stripping) is handled
server-side by GoogleWorkspace.normalize_email_body/1.  The SDK passes the body
through as-is so the backend is the single source of truth.

These tests mock _post so no network calls are made.
"""

import pytest
from unittest.mock import AsyncMock, patch

from agentchat.executor import ExecutorClient


@pytest.fixture
def executor(base_url, agent_id, api_key):
    return ExecutorClient(base_url, agent_id, api_key, "test-executor")


def _captured_body(mock_post: AsyncMock) -> str:
    """Extract the body field from the last _post call's json payload."""
    return mock_post.call_args.kwargs["json"]["body"]


class TestSendEmailPassthrough:
    """SDK passes body through to the backend without modification."""

    @pytest.mark.asyncio
    async def test_body_passed_through_unchanged(self, executor):
        body = "Dear James\\,\\n\\nThank you\\!"
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email(body, to="to@example.com", subject="Subject")
            assert _captured_body(mock) == body

    @pytest.mark.asyncio
    async def test_real_newlines_passed_through(self, executor):
        body = "Line one\nLine two\n\nLine four"
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email(body, to="to@example.com", subject="Subject")
            assert _captured_body(mock) == body

    @pytest.mark.asyncio
    async def test_payload_routing(self, executor):
        """Verifies the body is sent to the correct endpoint."""
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email("Hello\\!", to="to@example.com", subject="Hi")
            path = mock.call_args.args[0]
            assert path == "/api/google/gmail/send"
            payload = mock.call_args.kwargs["json"]
            assert payload["to"] == "to@example.com"
            assert payload["subject"] == "Hi"
            assert payload["body"] == "Hello\\!"

    @pytest.mark.asyncio
    async def test_cc_bcc_forwarded(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email("Body", to="to@example.com", subject="Hi",
                cc=["cc@example.com"], bcc=["bcc@example.com"],
            )
            payload = mock.call_args.kwargs["json"]
            assert payload["cc"] == ["cc@example.com"]
            assert payload["bcc"] == ["bcc@example.com"]

    @pytest.mark.asyncio
    async def test_reply_forwards_thread_id_and_omits_subject(self, executor):
        """A reply carries reply_to_message_id and no subject — the backend
        derives the subject from the thread, and sending one that doesn't
        match would make Gmail reject the threadId."""
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email(
                "to@example.com", "Sounds good.",
                reply_to_message_id="18c2f4a",
            )
            payload = mock.call_args.kwargs["json"]
            assert payload["reply_to_message_id"] == "18c2f4a"
            assert "subject" not in payload

    @pytest.mark.asyncio
    async def test_draft_reply_forwards_thread_id(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.save_draft(
                "to@example.com", "Draft body",
                reply_to_message_id="18c2f4a",
            )
            assert mock.call_args.args[0] == "/api/google/gmail/drafts"
            assert mock.call_args.kwargs["json"]["reply_to_message_id"] == "18c2f4a"


class TestRunContextHeaders:
    """Gmail writes carry the turn's task/conversation so the backend can tell
    a pulse's send from one the owner asked for in chat (bridge 2.11.9)."""

    @pytest.mark.asyncio
    async def test_send_email_sends_run_context_headers(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email(
                "Body", to="to@example.com", subject="Hi",
                run_context={"task_id": "t-1", "conversation_id": "c-1"},
            )
            assert mock.call_args.kwargs["extra_headers"] == {
                "X-Task-Id": "t-1",
                "X-Active-Conversation": "c-1",
            }

    @pytest.mark.asyncio
    async def test_save_draft_omits_empty_context(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.save_draft(
                "Body", to="to@example.com", subject="Hi",
                run_context={"task_id": None, "conversation_id": "c-1"},
            )
            assert mock.call_args.kwargs["extra_headers"] == {"X-Active-Conversation": "c-1"}

    @pytest.mark.asyncio
    async def test_create_reminder_sends_run_context_headers(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.create_reminder(
                "10m", "Retry the deck", action_instruction="Retry.",
                run_context={"task_id": "t-1", "conversation_id": "c-1"},
            )
            assert mock.call_args.args[0] == "/api/agents/me/reminders"
            assert mock.call_args.kwargs["extra_headers"] == {
                "X-Task-Id": "t-1",
                "X-Active-Conversation": "c-1",
            }

    @pytest.mark.asyncio
    async def test_no_context_no_headers(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={})) as mock:
            await executor.send_email("Body", to="to@example.com", subject="Hi")
            assert mock.call_args.kwargs["extra_headers"] is None

    @pytest.mark.asyncio
    async def test_tool_executor_injects_turn_context(self, executor):
        from agentchat.tools.executor import ToolExecutor

        tools = [{
            "name": "send_email",
            "executorMethod": "send_email",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["body"],
            },
        }]
        te = ToolExecutor(
            executor,
            context={"task_id": "pulse-1", "conversation_id": "conv-1"},
            resolved_tools=tools,
        )
        with patch.object(executor, "_post", new=AsyncMock(return_value={"message_id": "m1"})) as mock, \
             patch("agentchat.tools.executor.verify_action", new=AsyncMock(return_value=None)):
            result = await te.execute("send_email", {"to": "a@b.c", "subject": "S", "body": "B"})
            assert mock.call_args.kwargs["extra_headers"] == {
                "X-Task-Id": "pulse-1",
                "X-Active-Conversation": "conv-1",
            }
            assert "_ignored_arguments" not in result


class TestSaveAgentMemoryRunContext:
    """A memory saved mid-turn says which room and task it came from — the
    backend records the source room and links the task, so recall stops
    serving a mid-task note as current work once the task closes."""

    @pytest.mark.asyncio
    async def test_tool_executor_sends_turn_context_on_memory_save(self, executor):
        from agentchat.tools.executor import ToolExecutor

        tools = [{
            "name": "save_agent_memory",
            "executorMethod": "save_agent_memory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "key": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["category", "key", "content"],
            },
        }]
        te = ToolExecutor(
            executor,
            context={"task_id": "task-1", "conversation_id": "conv-1"},
            resolved_tools=tools,
        )
        with patch.object(executor, "_post", new=AsyncMock(return_value={"memory": {}})) as mock:
            await te.execute(
                "save_agent_memory",
                {"category": "learning", "key": "k", "content": "Deck needs PNGs"},
            )
            assert mock.call_args.args[0] == "/api/agents/me/memories"
            assert mock.call_args.kwargs["extra_headers"] == {
                "X-Task-Id": "task-1",
                "X-Active-Conversation": "conv-1",
            }
