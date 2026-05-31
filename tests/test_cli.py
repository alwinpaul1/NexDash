"""Tests for :mod:`nexdash.cli` (the dispatcher command-line interface).

All tests are offline: the agent's ``ask`` method is monkeypatched so no
Anthropic client is constructed and no network call is made. They verify the
two contracts the CLI promises:

* ``--once "<question>"`` prints the agent's answer to stdout and exits 0.
  WHY: this is the scriptable smoke-test path; downstream tooling parses the
  printed answer, so it must reach stdout exactly and exit cleanly.
* A missing ``ANTHROPIC_API_KEY`` prints actionable setup guidance (to
  stderr) and exits non-zero *without* ever instantiating the agent. WHY: the
  user should get a clear fix, not a stack trace, and we must not attempt a
  network/credential-dependent call.
"""

from __future__ import annotations

from typing import Any

import pytest

from nexdash import cli
from nexdash.agent import DispatcherAgent


def test_once_prints_answer_and_exits_zero(monkeypatch, capsys):
    """``--once`` asks the agent and prints exactly its answer, exit code 0.

    WHY: the printed answer is the CLI's machine-readable output; any wrapper
    text or wrong exit code would break scripting around it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")

    captured_questions: list[str] = []

    def fake_ask(self: DispatcherAgent, question: str) -> str:
        captured_questions.append(question)
        return "Yes, the truck reaches it with a 35 kWh margin. (Caveat: model estimate.)"

    monkeypatch.setattr(DispatcherAgent, "ask", fake_ask)

    code = cli.main(["--once", "Can I reach 240 km at 60% SOC?"])

    assert code == 0
    assert captured_questions == ["Can I reach 240 km at 60% SOC?"]

    out = capsys.readouterr().out
    assert "Yes, the truck reaches it with a 35 kWh margin." in out


def test_missing_api_key_prints_guidance_and_no_agent(monkeypatch, capsys):
    """No API key -> guidance on stderr, non-zero exit, agent never built.

    WHY: the user must get an actionable message (how to set the key) instead
    of an opaque failure, and we must not construct the agent (which would try
    to reach the API).
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    # If the CLI tried to build/ask the agent we want a loud failure.
    def explode(*args: Any, **kwargs: Any):
        pytest.fail("DispatcherAgent must not be used without an API key")

    monkeypatch.setattr(cli, "DispatcherAgent", explode)

    code = cli.main(["--once", "anything"])

    assert code != 0  # contract: non-zero exit on missing key
    err = capsys.readouterr().err
    # Actionable guidance: mention both providers and how to set a key.
    assert "MINIMAX_API_KEY" in err
    assert "export MINIMAX_API_KEY" in err


def test_minimax_only_key_is_accepted(monkeypatch, capsys):
    """A MiniMax-only setup (the documented DEFAULT) must run the CLI.

    WHY: a regression where the CLI hard-checked ANTHROPIC_API_KEY refused the
    documented MiniMax default and printed a misleading "missing key" error even
    though the agent would have worked. The key pre-check must be provider-
    agnostic.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-not-real")

    def fake_ask(self: DispatcherAgent, question: str) -> str:
        return "Yes, it reaches."

    monkeypatch.setattr(DispatcherAgent, "ask", fake_ask)

    code = cli.main(["--once", "Does it reach?"])
    assert code == 0
    assert "Yes, it reaches." in capsys.readouterr().out


def test_blank_api_key_is_treated_as_missing(monkeypatch, capsys):
    """A whitespace-only key is rejected like a missing one.

    WHY: an accidentally blank env var must not slip through and trigger a
    confusing downstream auth error from the SDK.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(
        cli, "DispatcherAgent",
        lambda *a, **k: pytest.fail("agent built with blank key"),
    )

    code = cli.main(["--once", "anything"])

    assert code != 0
    assert "MINIMAX_API_KEY" in capsys.readouterr().err


def test_once_empty_question_returns_error(monkeypatch, capsys):
    """An empty ``--once`` question is rejected with a non-zero exit.

    WHY: dispatching an empty prompt wastes an API call and yields nothing
    useful; the CLI should fail fast instead.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")

    def fake_ask(self: DispatcherAgent, question: str) -> str:  # pragma: no cover
        pytest.fail("ask should not be called for an empty question")

    monkeypatch.setattr(DispatcherAgent, "ask", fake_ask)

    code = cli.main(["--once", "   "])

    assert code != 0
