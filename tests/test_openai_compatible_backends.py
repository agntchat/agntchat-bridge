"""Request shaping for the OpenAI-compatible backends (openai, openrouter)."""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

from agentchat.backends import create_backend  # noqa: E402


def test_openai_direct_uses_max_completion_tokens():
    backend = create_backend("openai", model="gpt-6-sol", api_key="sk-test", max_tokens=1234)
    assert backend._limit_kwargs() == {"max_completion_tokens": 1234}


def test_gpt6_tool_calls_pin_reasoning_effort_none():
    # GPT-6 supports function calling on Chat Completions only at
    # reasoning_effort="none".
    backend = create_backend("openai", model="gpt-6-sol", api_key="sk-test", max_tokens=100)
    assert backend._limit_kwargs(with_tools=True) == {
        "max_completion_tokens": 100,
        "reasoning_effort": "none",
    }


def test_older_openai_models_keep_their_reasoning_default_with_tools():
    backend = create_backend("openai", model="gpt-5.5", api_key="sk-test", max_tokens=100)
    assert backend._limit_kwargs(with_tools=True) == {"max_completion_tokens": 100}


def test_compatible_provider_keeps_max_tokens():
    backend = create_backend(
        "openai", model="llama3", api_key="none", base_url="http://localhost:11434/v1", max_tokens=50
    )
    assert backend._limit_kwargs(with_tools=True) == {"max_tokens": 50}


def test_openrouter_points_at_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    backend = create_backend("openrouter", model="google/gemini-3.8-flash", api_key="sk-or-test")
    assert backend._base_url == "https://openrouter.ai/api/v1"
    assert str(backend._client.base_url).startswith("https://openrouter.ai/api/v1")
    assert backend._client.default_headers["X-Title"] == "agntchat"
    assert backend._request_model() == "google/gemini-3.8-flash"
    # OpenRouter takes max_tokens, and never gets OpenAI's reasoning pin.
    assert "max_tokens" in backend._limit_kwargs(with_tools=True)
    assert "reasoning_effort" not in backend._limit_kwargs(with_tools=True)


def test_openrouter_ignores_the_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    backend = create_backend("openrouter")
    assert backend._base_url == "https://openrouter.ai/api/v1"
    assert backend._client.api_key == "sk-or-env"
