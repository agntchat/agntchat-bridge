"""Schema properties the SDK method does not take must not TypeError the call."""

from __future__ import annotations

from agentchat.tools.executor import _fit_kwargs_to_method


async def complete_task(task_id: str, *, response: str | None = None, result_data: dict | None = None) -> None:
    return None


async def send_message(conversation_id: str, content: str) -> None:
    return None


async def passthrough(**kwargs) -> None:
    return None


def test_complete_task_folds_unknown_args_into_result_data():
    fitted = _fit_kwargs_to_method(
        complete_task, "complete_task",
        {"task_id": "t1", "response": "done", "criteria_met": ["tests pass"]},
    )
    assert fitted == {"task_id": "t1", "response": "done", "result_data": {"criteria_met": ["tests pass"]}}


def test_complete_task_keeps_existing_result_data():
    fitted = _fit_kwargs_to_method(
        complete_task, "complete_task",
        {"task_id": "t1", "result_data": {"artifacts": ["x"]}, "criteria_met": ["a"]},
    )
    assert fitted["result_data"] == {"artifacts": ["x"], "criteria_met": ["a"]}


def test_other_methods_drop_unknown_args():
    fitted = _fit_kwargs_to_method(send_message, "send_message", {"conversation_id": "c", "content": "hi", "mood": "x"})
    assert fitted == {"conversation_id": "c", "content": "hi"}


def test_var_kwargs_methods_are_left_alone():
    args = {"anything": 1, "goes": 2}
    assert _fit_kwargs_to_method(passthrough, "passthrough", args) == args
