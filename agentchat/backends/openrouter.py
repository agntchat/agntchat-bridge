"""OpenRouter model backend — every vendor OpenRouter routes to, on one key.

OpenRouter speaks the OpenAI Chat Completions API, so this is the ``openai``
backend pinned to https://openrouter.ai/api/v1 with its own key and model
env vars. Model ids are OpenRouter's ``vendor/model`` slugs
(``google/gemini-3.8-flash``); the server's catalog (``Agentchat.Models``)
decides which ones are offered. Tool calls use the OpenAI tool format.
"""

from __future__ import annotations

from typing import Any

from .openai import OpenAIBackend

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(OpenAIBackend):
    """``OpenAIBackend`` pointed at OpenRouter."""

    _default_base_url = _OPENROUTER_BASE_URL
    _default_model = "anthropic/claude-sonnet-5"
    _base_url_env = "OPENROUTER_BASE_URL"
    _api_key_env = "OPENROUTER_API_KEY"
    _model_env = "OPENROUTER_MODEL"
    # App attribution OpenRouter shows on its dashboards and rankings.
    _default_headers = {"HTTP-Referer": "https://agntchat.com", "X-Title": "agntchat"}


def create(**kwargs: Any) -> OpenRouterBackend:
    """Factory function called by create_backend()."""
    return OpenRouterBackend(**kwargs)
