"""A model process killed from outside is not a turn failure.

When the org-host restarts (a deploy, a systemd stop, the OOM killer) every
in-flight Claude CLI is signalled. The CLI exits 143 (128+SIGTERM) with an
empty stderr, so the failure text is the useless "unknown error" — and the
raw RuntimeError built from it used to be posted into the user's chat
verbatim as the agent's answer:

    RuntimeError: Claude CLI exited with code 143: unknown error

(prod, 2026-09-13 — one second after the host unit began stopping.)

These exits must classify as BackendInterruptedError, which — unlike the auth
and rate-limit categories — deliberately leaves backend health ALONE. The
agent is healthy; the machine under it went away.
"""

import signal

import pytest

from agentchat.backends import (
    BackendAuthError,
    BackendInterruptedError,
    BackendRateLimitError,
)
from agentchat.backends.claude_cli import ClaudeCliBackend


def _backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(
        cli_path="/bin/sh",
        api_url="http://localhost",
        agent_id="agent-test",
        api_key="key-test",
    )


@pytest.mark.parametrize(
    "returncode",
    [
        143,  # 128 + SIGTERM — the form prod actually produced
        137,  # 128 + SIGKILL — systemd escalating after its stop timeout
        -int(signal.SIGTERM),  # POSIX/asyncio form
        -int(signal.SIGKILL),
    ],
)
def test_signal_exits_classify_as_interrupted(returncode):
    err = _backend()._classify_failure("unknown error", returncode)

    assert isinstance(err, BackendInterruptedError)
    assert str(returncode) in str(err)


def test_interrupted_does_not_touch_health():
    """The whole point of the category: no blocker banner for a restart."""
    backend = _backend()
    before = backend.health.status

    backend._classify_failure("unknown error", 143)

    assert backend.health.status == before


def test_auth_failure_still_classifies_on_a_normal_exit():
    """The signal check runs first — it must not swallow the real categories."""
    err = _backend()._classify_failure("Failed to authenticate with Claude", 1)
    assert isinstance(err, BackendAuthError)


def test_signal_exit_wins_over_misleading_text():
    """Ordering is deliberate: the returncode is authoritative, text is not.

    A process killed mid-stream can carry any half-written text in stderr,
    including text that trips the (deliberately broad) auth/rate-limit
    regexes. Misreporting a host restart as "your credentials are broken"
    sends the user to re-authenticate something that was never wrong.
    """
    err = _backend()._classify_failure("Failed to authenticate with Claude", 143)
    assert isinstance(err, BackendInterruptedError)
    assert not isinstance(err, (BackendAuthError, BackendRateLimitError))


def test_ordinary_failures_are_unchanged():
    err = _backend()._classify_failure("some other problem", 1)
    assert type(err) is RuntimeError
