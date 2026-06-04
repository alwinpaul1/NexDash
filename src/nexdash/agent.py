"""LLM dispatcher agent for the NexDash eActros 600 fleet.

This module wires the MiniMax (M3) LLM, via its OpenAI-compatible API, to the
deterministic energy / reachability / route-planning tools defined in
:mod:`nexdash.tools`. The agent runs a classic tool-use loop: it sends the
user's question together with the tool schemas, executes any requested tool
calls locally via :func:`nexdash.tools.dispatch`, feeds the results back, and
repeats until the model produces a final natural-language answer.

The :class:`DispatcherAgent` accepts an injected ``client`` so unit tests can
pass a mock and run entirely offline. A real MiniMax client is created lazily
(and only if no client was injected), which is the single point where a missing
``MINIMAX_API_KEY`` is surfaced as a clear error.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from .config import DEFAULT_MODEL_PATH
from . import tools as nexdash_tools

__all__ = ["DispatcherAgent", "SYSTEM_PROMPT", "MissingAPIKeyError", "AgentError"]

# The dispatcher runs on MiniMax (M3) by default, via its OpenAI-compatible API.
# The key is read from the environment — never hard-code it. NEXDASH_LLM_MODEL
# overrides the default model id.
DEFAULT_LLM_MODEL = os.environ.get("NEXDASH_LLM_MODEL", "MiniMax-M3")
MINIMAX_API_URL = "https://api.minimax.io/v1/chat/completions"

# Upper bound on tool-use round trips to guard against pathological loops.
MAX_TURNS = 8

SYSTEM_PROMPT = (
    "You are NexDash Dispatcher, an AI assistant for a fleet of "
    "Mercedes-Benz eActros 600 battery-electric trucks operating across "
    "Germany. Each truck has roughly 600 kWh of usable battery and about "
    "500 km of real-world range, carrying payloads from 0 to 22 tonnes.\n\n"
    "Your job is to help human fleet dispatchers answer operational "
    "questions about energy consumption and route reachability.\n\n"
    "Rules:\n"
    "1. ALWAYS use the provided tools to obtain any numeric estimate "
    "(energy needed, reachability, remaining range, state of charge). "
    "Never invent or guess numbers yourself.\n"
    "2. When a dispatcher describes a TRIP BETWEEN NAMED PLACES OR CITIES "
    "(e.g. 'Berlin to Munich, 12 tonnes, depart 9am') and wants the route, "
    "energy and charging plan, call plan_route with the origin, destination, "
    "payload_t and start_soc — it geocodes the places, computes "
    "the real truck road route, simulates state-of-charge with charging stops, "
    "and checks EU 561 driver hours.\n"
    "   WEATHER & TERRAIN ARE LIVE: plan_route fetches per-segment temperature, "
    "wind and elevation/gradient along the real route automatically from live "
    "data (Open-Meteo). NEVER ask the dispatcher for weather or temperature, and "
    "never treat it as a required input — leave temperature_c unset unless the "
    "dispatcher explicitly states an ambient override (e.g. 'assume -5 degC').\n"
    "   TIME HANDLING: If the dispatcher gives a DEPARTURE time (e.g. 'leaving "
    "Friday 9pm', 'depart tomorrow 06:00') pass it as 'departure'. If they give "
    "a DELIVERY DEADLINE (e.g. 'deliver by Friday 9pm', 'must arrive before "
    "noon Monday') pass it as 'deliver_by'. Both must be FULL ISO 8601 local "
    "datetimes like '2026-06-05T21:00'. Today's date is given to you in the "
    "conversation context — resolve natural language relative to it: 'tomorrow' "
    "= today+1; a bare weekday/time ('Friday 9pm') = the NEXT such datetime that "
    "is at or after the departure (or after today if no departure is set); 'pm' "
    "means 24h (9pm -> 21:00). If only a date is given with no time, assume "
    "08:00 for a departure and 23:59 for a deadline. Do NOT invent a deadline or "
    "departure the dispatcher did not state.\n"
    "   Then narrate the plan: distance, energy and "
    "kWh/100 km, arrival SOC, each charging stop (where, how much), driving and "
    "total time, and the ETA. When a 'deliver_by' deadline was given, the tool "
    "returns 'eta' / 'eta_iso', 'deliver_by' and 'on_time' (true=on time/early, "
    "false=late) — state CLEARLY whether the truck arrives on time, comfortably "
    "early, or late versus the deadline, and by roughly how long. Always state "
    "whether the schedule is EU 561 compliant (the 'eu561_ok' field). If plan_route "
    "returns an 'error' field, tell the dispatcher plainly what failed (e.g. a "
    "place couldn't be found) and ask them to clarify. When a dispatcher instead "
    "asks whether a single given segment is feasible, call check_reachability; "
    "when they only ask how much energy one leg of known distance needs, call "
    "predict_energy. Do NOT call plan_route for a single isolated segment or a "
    "pure reachability question, and do NOT re-run plan_route if the conversation "
    "already supplies the computed plan numbers — just summarize those.\n"
    "3. After the tools return, explain the result in plain, practical "
    "language a dispatcher can act on. State the bottom-line answer first "
    "(e.g. 'Yes, the truck reaches the destination'), then give the key "
    "numbers including the safety margin in kWh and the remaining state of "
    "charge or range where relevant.\n"
    "4. Always include a short caveat: these figures come from a "
    "machine-learning model with an expected error band, plus a reserve "
    "buffer, so dispatchers should keep a safety cushion and re-check if "
    "conditions (weather, load, traffic) change.\n"
    "5. If required inputs are missing, ask the dispatcher for them rather "
    "than assuming values.\n"
    "Be concise, concrete, and operationally focused."
)


class MissingAPIKeyError(RuntimeError):
    """Raised when a real LLM client is needed but no API key is set.

    This is a catchable, explicit error so callers (e.g. the CLI / API) can
    present a friendly message instead of an opaque SDK failure.
    """


class AgentError(RuntimeError):
    """Raised for recoverable provider failures (rate limits, transient errors).

    Catchable so callers can degrade gracefully (e.g. the API returns a friendly
    message) instead of surfacing a 500.
    """


class DispatcherAgent:
    """A tool-using MiniMax agent for eActros 600 fleet dispatch questions.

    Parameters
    ----------
    model_path:
        Filesystem path to the trained energy model. Passed through to the
        tool layer so predictions use the intended model artifact.
    client:
        An optional pre-constructed LLM client (any object exposing a
        compatible ``messages.create`` API). When ``None`` a real MiniMax
        client is created lazily on first use. Inject a mock here in tests to
        avoid any network access.
    model:
        LLM model id to call. Defaults to ``"MiniMax-M3"``.
    max_tokens:
        Maximum tokens per model response.
    """

    def __init__(
        self,
        model_path: Any = DEFAULT_MODEL_PATH,
        client: Optional[Any] = None,
        model: str = DEFAULT_LLM_MODEL,
        *,
        max_tokens: int = 2048,
    ) -> None:
        self.model_path = str(model_path)
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    # ------------------------------------------------------------------ #
    # Client management
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> Any:
        """Return the LLM client, creating a real one lazily if needed.

        Raises
        ------
        MissingAPIKeyError
            If no client was injected and ``MINIMAX_API_KEY`` is not set.
        """
        if self._client is None:
            self._client = self._make_real_client()
        return self._client

    def _make_real_client(self) -> Any:
        """Construct a real MiniMax client (OpenAI-compatible) from the API key.

        This is the only place a missing key is surfaced, so tests that inject a
        client never touch the network or require a key.
        """
        if os.environ.get("MINIMAX_API_KEY"):
            return _OpenAICompatClient(os.environ["MINIMAX_API_KEY"], MINIMAX_API_URL)

        raise MissingAPIKeyError(
            "No LLM API key set. Export MINIMAX_API_KEY (MiniMax-M3), then "
            "restart the server."
        )

    # ------------------------------------------------------------------ #
    # Date context (injected into the conversation, not the system prompt)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _date_context_turn() -> dict[str, Any]:
        """A leading user turn telling the model today's date/time.

        The agent needs the current date to resolve natural-language departure /
        delivery times (e.g. 'Friday 9pm') into the ISO datetimes plan_route
        expects. We inject it as a conversation turn (rather than mutating the
        system prompt) so SYSTEM_PROMPT stays stable and the date is always
        current at call time.
        """
        now = datetime.now()
        return {
            "role": "user",
            "content": (
                f"[context] Today is {now.strftime('%A, %Y-%m-%d')}; the current "
                f"local time is {now.strftime('%H:%M')}. Resolve any relative "
                "departure or delivery times against this when planning routes."
            ),
        }

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> str:
        """Answer a dispatcher question, using tools as needed.

        Runs the tool-use loop: the model may request one or more
        tool calls, which are executed locally via
        :func:`nexdash.tools.dispatch` (with ``model_path`` injected), and the
        results are fed back until the model returns a final text answer.

        Parameters
        ----------
        question:
            The dispatcher's natural-language question.

        Returns
        -------
        str
            The model's final natural-language answer (concatenated text
            blocks). Empty model output yields an empty string.
        """
        messages: list[dict[str, Any]] = [
            self._date_context_turn(),
            {"role": "user", "content": question},
        ]

        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=nexdash_tools.TOOL_SPECS,
                messages=messages,
            )

            content_blocks = list(response.content or [])
            tool_uses = [b for b in content_blocks if _block_type(b) == "tool_use"]

            # Record the assistant turn verbatim so tool_result blocks line up
            # with the tool_use ids the model produced.
            messages.append(
                {"role": "assistant", "content": _serialize_content(content_blocks)}
            )

            if not tool_uses:
                return _extract_text(content_blocks)

            # Execute every requested tool call and return the results.
            tool_results: list[dict[str, Any]] = []
            for block in tool_uses:
                result = self._run_tool(_block_name(block), _block_input(block))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": _block_id(block),
                        "content": _result_to_text(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        # Loop exhausted without a final text answer; surface what we have.
        return (
            "I was unable to reach a final answer within the allotted tool "
            "calls. Please simplify the request or provide the missing inputs."
        )

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Multi-turn chat variant of :meth:`ask` for the web UI.

        Parameters
        ----------
        messages:
            Conversation history as a list of ``{"role": "user"|"assistant",
            "content": str}`` plain-text turns (the last one being the new
            user message).

        Returns
        -------
        dict
            ``{"reply": str, "tools": [...], "planRequest": dict|None}``. The
            ``tools`` list lets the UI show which deterministic tools the agent
            invoked (e.g. ``check_reachability``). ``planRequest`` carries the
            resolved origin/destination (with coords) + planner params of the
            last ``plan_route`` call this turn, so the UI can fill the planner
            form and run the same Optimize pipeline; it is ``None`` when
            ``plan_route`` was not called (or did not resolve coordinates).
        """
        convo: list[dict[str, Any]] = []
        for m in messages or []:
            role = (m or {}).get("role")
            content = (m or {}).get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                convo.append({"role": role, "content": content})
        if not convo:
            return {"reply": "", "tools": [], "planRequest": None}

        # Clear any plan_route request from a previous turn so we only surface a
        # planRequest the agent actually produced this turn.
        nexdash_tools.reset_last_plan_request()

        # Prepend today's date so the agent can resolve 'Friday 9pm' etc.
        convo.insert(0, self._date_context_turn())

        tools_used: list[str] = []
        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=nexdash_tools.TOOL_SPECS,
                messages=convo,
            )
            blocks = list(response.content or [])
            tool_uses = [b for b in blocks if _block_type(b) == "tool_use"]
            convo.append({"role": "assistant", "content": _serialize_content(blocks)})

            if not tool_uses:
                return {
                    "reply": _extract_text(blocks),
                    "tools": tools_used,
                    "planRequest": nexdash_tools.get_last_plan_request(),
                }

            results: list[dict[str, Any]] = []
            for block in tool_uses:
                tools_used.append(_block_name(block))
                result = self._run_tool(_block_name(block), _block_input(block))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": _block_id(block),
                        "content": _result_to_text(result),
                    }
                )
            convo.append({"role": "user", "content": results})

        return {
            "reply": (
                "I couldn't finish that within the tool-call limit — try "
                "simplifying the question or giving the missing details."
            ),
            "tools": tools_used,
            "planRequest": nexdash_tools.get_last_plan_request(),
        }

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a single tool call, injecting the agent's model path.

        Errors are caught and returned as a structured ``{"error": ...}`` dict
        so the model can recover or explain the failure rather than crashing
        the loop.
        """
        call_args = dict(args or {})
        # Inject the configured model path unless the model explicitly set one.
        call_args.setdefault("model_path", self.model_path)
        try:
            return nexdash_tools.dispatch(name, call_args)
        except Exception as exc:  # noqa: BLE001 - report any tool failure to the model
            return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------- #
# Block helpers (tolerant of both SDK objects and plain dicts in tests)
# ---------------------------------------------------------------------- #
def _block_type(block: Any) -> Optional[str]:
    return _get(block, "type")


def _block_name(block: Any) -> str:
    return _get(block, "name") or ""


def _block_id(block: Any) -> str:
    return _get(block, "id") or ""


def _block_input(block: Any) -> dict[str, Any]:
    value = _get(block, "input")
    return dict(value) if isinstance(value, dict) else {}


def _block_text(block: Any) -> str:
    return _get(block, "text") or ""


def _get(block: Any, attr: str) -> Any:
    """Read ``attr`` from an SDK object (attribute) or a dict (key)."""
    if isinstance(block, dict):
        return block.get(attr)
    return getattr(block, attr, None)


def _strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning that models like MiniMax-M3 emit.

    The reasoning is kept verbatim in the conversation history (for the model's
    own continuity) but must not surface in the dispatcher-facing reply.
    """
    if not text:
        return text
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)  # truncated/unclosed
    cleaned = cleaned.strip()
    return cleaned or text.strip()


def _extract_text(blocks: list[Any]) -> str:
    """Concatenate all text blocks from a model response (sans reasoning)."""
    parts = [
        _block_text(b)
        for b in blocks
        if _block_type(b) == "text"
    ]
    return _strip_reasoning("".join(parts))


def _serialize_content(blocks: list[Any]) -> list[dict[str, Any]]:
    """Convert response content blocks into plain dicts for the next request.

    The tool-use API accepts the original block objects on round-trip, but
    normalizing to dicts keeps the loop robust against mock objects in tests
    and makes the conversation JSON-serializable.
    """
    serialized: list[dict[str, Any]] = []
    for block in blocks:
        btype = _block_type(block)
        if btype == "text":
            serialized.append({"type": "text", "text": _block_text(block)})
        elif btype == "tool_use":
            serialized.append(
                {
                    "type": "tool_use",
                    "id": _block_id(block),
                    "name": _block_name(block),
                    "input": _block_input(block),
                }
            )
        # Unknown block types are skipped; they are not part of our contract.
    return serialized


def _result_to_text(result: Any) -> str:
    """Render a tool result as a JSON string for the tool_result content."""
    import json

    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


# ---------------------------------------------------------------------- #
# OpenAI-compatible client adapter (used for MiniMax)
# ---------------------------------------------------------------------- #
# Exposes the same ``client.messages.create(...)`` surface the tool-use loop
# above expects, translating tool-use-style requests/responses to and from the
# OpenAI chat-completions schema. This lets the dispatcher run on any
# OpenAI-compatible endpoint (MiniMax-M3 here) with ZERO changes to the
# tool-use loop or the tool layer.
class _CompatResponse:
    """Minimal stand-in for an tool-use response: just a ``.content`` block list."""

    def __init__(self, content: list[dict[str, Any]]) -> None:
        self.content = content


class _OpenAICompatMessages:
    """Implements ``.create(...)`` against an OpenAI-compatible chat endpoint."""

    def __init__(self, api_key: str, url: str) -> None:
        self._key = api_key
        self._url = url

    def create(self, *, model, max_tokens, system, tools, messages, **_ignored):
        import httpx

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_openai_messages(system, messages),
            "tools": _to_openai_tools(tools),
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(self._url, json=payload, headers=headers, timeout=90.0)
        except Exception as exc:  # noqa: BLE001 - network failure -> recoverable
            raise AgentError(f"Could not reach the model provider: {exc}") from exc

        if resp.status_code == 429:
            raise AgentError(
                "The model is rate-limited right now (HTTP 429). Please retry in "
                "a moment."
            )
        if resp.status_code != 200:
            raise AgentError(f"Model provider error {resp.status_code}: {resp.text[:300]}")

        choices = (resp.json() or {}).get("choices") or []
        if not choices:
            raise AgentError("Model provider returned no choices.")
        return _CompatResponse(_to_tooluse_blocks(choices[0].get("message", {})))


class _OpenAICompatClient:
    """tool-use-shaped facade over an OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, url: str) -> None:
        self.messages = _OpenAICompatMessages(api_key, url)


def _to_openai_tools(tool_specs: list[Any]) -> list[dict[str, Any]]:
    """tool-use tool specs ({name, description, input_schema}) -> OpenAI funcs."""
    return [
        {
            "type": "function",
            "function": {
                "name": _get(s, "name"),
                "description": _get(s, "description") or "",
                "parameters": _get(s, "input_schema") or {},
            },
        }
        for s in (tool_specs or [])
    ]


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the tool-use-style conversation into OpenAI chat messages.

    Handles plain-string turns, assistant turns carrying text + tool_use blocks
    (-> ``tool_calls``), and user turns carrying tool_result blocks (-> ``tool``
    role messages keyed by the original tool-call id).
    """
    import json

    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = content or []
        if role == "assistant":
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            tool_calls = [
                {
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input") or {}),
                    },
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:  # user turn carrying tool_result (and/or text) blocks
            for b in blocks:
                if b.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": b.get("content", ""),
                        }
                    )
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b.get("text", "")})
    return out


def _to_tooluse_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate an OpenAI assistant message back into tool-use content blocks."""
    import json

    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": args if isinstance(args, dict) else {},
            }
        )
    return blocks
