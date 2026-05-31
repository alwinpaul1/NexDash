"""Tests for :mod:`nexdash.agent` (the Claude tool-use dispatcher).

These tests run entirely offline by injecting a *mock* Anthropic client into
:class:`~nexdash.agent.DispatcherAgent`. They verify the load-bearing
contract that the CLI and MCP layers depend on:

* The agent runs a genuine tool-use loop: when the model returns a
  ``tool_use`` block, the agent must execute it via
  :func:`nexdash.tools.dispatch` and feed a ``tool_result`` back on the next
  request. WHY: if the result were not fed back, the model could never ground
  its answer in real numbers — the whole point of the dispatcher is that
  every figure comes from the deterministic tools, not the LLM's guesswork.
* The agent injects its configured ``model_path`` into the tool call. WHY:
  predictions must come from the intended model artifact, not an accidental
  default, so a misconfigured path would silently serve wrong numbers.
* The final natural-language text from the follow-up response is returned
  verbatim. WHY: that string is exactly what the dispatcher reads.
* A missing ``ANTHROPIC_API_KEY`` (with no injected client) raises the
  explicit :class:`MissingAPIKeyError` rather than an opaque SDK error, and
  no real client is ever constructed when one is injected. WHY: tests and the
  CLI must never hit the network or require credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from nexdash import tools as nexdash_tools
from nexdash.agent import DispatcherAgent, MissingAPIKeyError, SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Lightweight Anthropic SDK stand-ins (attribute-style blocks/responses).
# The agent's block helpers read attributes via getattr, so these mirror the
# real SDK object shape without importing the anthropic package.
# --------------------------------------------------------------------------- #
@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _Response:
    content: list[Any]


@dataclass
class _Messages:
    """Records every ``create`` call and replays a scripted list of responses."""

    responses: list[_Response]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        # Pop in order so the first call gets the tool_use turn, the second the
        # final text turn. Reusing the last response would let a buggy agent
        # loop forever without us noticing, so we assert we don't overrun.
        assert self.calls, "create called with no scripted responses left"
        idx = len(self.calls) - 1
        assert idx < len(self.responses), (
            f"agent made {len(self.calls)} model calls but only "
            f"{len(self.responses)} were scripted (possible runaway loop)"
        )
        return self.responses[idx]


class _FakeClient:
    """Mimics ``anthropic.Anthropic`` exposing ``.messages.create``."""

    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _Messages(responses)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_ask_runs_tool_use_loop_and_returns_final_text(monkeypatch):
    """A tool_use turn is executed via dispatch and fed back; final text wins.

    WHY: this is the core dispatcher behavior — numbers must come from the
    tools, and the model's grounded final answer is what the user sees.
    """
    # Spy on dispatch so we prove the agent actually routed the tool call
    # rather than fabricating the result. Return a deterministic stub so the
    # test does not depend on a trained model existing on disk.
    dispatch_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
        dispatch_calls.append((name, dict(args)))
        return {"energy_kwh": 42.0, "inputs": dict(args)}

    monkeypatch.setattr(nexdash_tools, "dispatch", fake_dispatch)

    tool_use = _ToolUseBlock(
        id="toolu_001",
        name="predict_energy",
        input={
            "distance_km": 80,
            "payload_t": 18,
            "speed_kph": 75,
            "gradient_pct": 0,
            "temperature_c": 10,
        },
    )
    final_text = "About 42.0 kWh for that leg. Keep a safety cushion."
    client = _FakeClient(
        responses=[
            _Response(content=[tool_use]),               # turn 1: request tool
            _Response(content=[_TextBlock(final_text)]),  # turn 2: final answer
        ]
    )

    agent = DispatcherAgent(model_path="/tmp/custom_model.joblib", client=client)
    answer = agent.ask("How much energy to drive 80 km with 18 t at 75 kph?")

    # The agent returned exactly the model's final text.
    assert answer == final_text

    # dispatch was called once, for the requested tool, with the model's args
    # plus the agent's injected model_path.
    assert len(dispatch_calls) == 1
    name, args = dispatch_calls[0]
    assert name == "predict_energy"
    assert args["distance_km"] == 80
    assert args["model_path"] == "/tmp/custom_model.joblib"

    # The agent made exactly two model calls (tool round-trip + final answer).
    assert len(client.messages.calls) == 2

    # First request carried the system prompt and the tool specs.
    first_call = client.messages.calls[0]
    assert first_call["system"] == SYSTEM_PROMPT
    assert first_call["tools"] is nexdash_tools.TOOL_SPECS

    # Second request must include a tool_result that references the tool_use id
    # and carries the dispatched result — proof the result was fed back.
    second_messages = client.messages.calls[1]["messages"]
    tool_result_blocks = [
        block
        for msg in second_messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_result_blocks) == 1
    result_block = tool_result_blocks[0]
    assert result_block["tool_use_id"] == "toolu_001"
    assert "42.0" in result_block["content"]


def test_ask_returns_text_without_tool_use():
    """A response with only text returns immediately (no tool round-trip).

    WHY: when the model asks a clarifying question or answers directly, the
    agent must not invent a tool call or make a second API request.
    """
    client = _FakeClient(
        responses=[_Response(content=[_TextBlock("Which payload and speed?")])]
    )
    agent = DispatcherAgent(client=client)

    answer = agent.ask("Can my truck make it?")

    assert answer == "Which payload and speed?"
    assert len(client.messages.calls) == 1


def test_tool_error_is_reported_back_not_raised(monkeypatch):
    """A failing tool is caught and returned to the model as an error result.

    WHY: a tool exception must not crash the dispatch loop; the model needs a
    chance to recover or explain, so the error is surfaced as a tool_result.
    """
    def boom(name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("model file missing")

    monkeypatch.setattr(nexdash_tools, "dispatch", boom)

    tool_use = _ToolUseBlock(id="toolu_err", name="predict_energy", input={})
    client = _FakeClient(
        responses=[
            _Response(content=[tool_use]),
            _Response(content=[_TextBlock("I hit an error and cannot answer.")]),
        ]
    )
    agent = DispatcherAgent(client=client)

    answer = agent.ask("energy?")

    assert answer == "I hit an error and cannot answer."
    # The error text was fed back to the model.
    second_messages = client.messages.calls[1]["messages"]
    error_blocks = [
        block
        for msg in second_messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(error_blocks) == 1
    assert "ValueError" in error_blocks[0]["content"]
    assert "model file missing" in error_blocks[0]["content"]


def test_injected_client_never_constructs_real_client(monkeypatch):
    """With a client injected, no real Anthropic client is ever built.

    WHY: tests and offline use must not require ANTHROPIC_API_KEY or hit the
    network. We delete the key and make the real-client factory explode if
    touched, proving the injected client is used exclusively.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        DispatcherAgent,
        "_make_real_client",
        staticmethod(lambda: pytest.fail("real client must not be constructed")),
    )

    client = _FakeClient(responses=[_Response(content=[_TextBlock("hi")])])
    agent = DispatcherAgent(client=client)

    assert agent.ask("hello") == "hi"


def test_missing_api_key_raises_explicit_error(monkeypatch):
    """No injected client + no API key -> MissingAPIKeyError on first use.

    WHY: the CLI relies on this explicit, catchable error to print friendly
    setup guidance instead of leaking a raw SDK exception. With no provider key
    of either kind set, construction must raise.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    agent = DispatcherAgent()  # no client injected

    with pytest.raises(MissingAPIKeyError):
        # Touching .client triggers lazy construction, which checks the key.
        _ = agent.client
