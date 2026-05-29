"""Anthropic tool definitions and JSON-serializable dispatch layer.

This module exposes the NexDash energy-prediction capabilities as
Anthropic tool-use schemas (:data:`TOOL_SPECS`) together with thin Python
wrappers that the model-driven agents (and the MCP server) call when a
tool-use block is returned by Claude.

The wrappers are intentionally tolerant of *string* numeric inputs: tool
arguments arriving from an LLM frequently come through as strings (or as
``null``), so every numeric field is coerced via :func:`_to_float` before
being handed to the underlying physics/ML layer. All return values are
plain ``dict`` objects containing only JSON-serializable scalars so they
can be embedded directly in a ``tool_result`` content block.
"""

from __future__ import annotations

from typing import Any, Callable

from nexdash.config import DEFAULT_MODEL_PATH
from nexdash.model import predict_energy
from nexdash.range import check_reachability

__all__ = [
    "TOOL_SPECS",
    "predict_energy_tool",
    "check_reach_tool",
    "dispatch",
]


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _to_float(value: Any, *, default: float | None = None, field: str = "value") -> float:
    """Coerce ``value`` to ``float``, tolerating strings and ``None``.

    LLM-generated tool arguments are often strings (``"45"``) or omitted
    entirely. We accept ints/floats directly, strip and parse strings, and
    fall back to ``default`` when the value is missing/blank. A missing
    value with no default raises :class:`ValueError` so the failure is
    loud rather than silently wrong.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        if default is not None:
            return float(default)
        raise ValueError(f"Missing required numeric argument: {field!r}")
    if isinstance(value, bool):  # guard: bool is a subclass of int
        raise ValueError(f"Boolean is not a valid number for {field!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:  # pragma: no cover - message clarity
            raise ValueError(
                f"Could not parse numeric argument {field!r} from {value!r}"
            ) from exc
    raise ValueError(f"Unsupported type for {field!r}: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Anthropic tool schemas
# ---------------------------------------------------------------------------
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "predict_energy",
        "description": (
            "Predict the energy consumption (in kWh) for a single driving "
            "segment of a Mercedes-Benz eActros 600 electric truck using the "
            "trained ML model. Use this whenever a user asks how much energy / "
            "battery a trip or leg will consume. All numeric inputs may be "
            "provided as numbers or numeric strings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_km": {
                    "type": "number",
                    "description": "Segment distance in kilometres (e.g. 1-120).",
                },
                "payload_t": {
                    "type": "number",
                    "description": "Cargo payload in tonnes (0-22).",
                },
                "speed_kph": {
                    "type": "number",
                    "description": "Average travel speed in km/h (e.g. 30-90).",
                },
                "gradient_pct": {
                    "type": "number",
                    "description": (
                        "Average road gradient in percent; positive = uphill, "
                        "negative = downhill (typically -6 to +6)."
                    ),
                },
                "temperature_c": {
                    "type": "number",
                    "description": "Ambient temperature in degrees Celsius (-15 to 40).",
                },
                "wind_mps": {
                    "type": "number",
                    "description": (
                        "Headwind component in metres per second (0-12). "
                        "Defaults to 0 if unknown."
                    ),
                },
            },
            "required": [
                "distance_km",
                "payload_t",
                "speed_kph",
                "gradient_pct",
                "temperature_c",
            ],
        },
    },
    {
        "name": "check_reachability",
        "description": (
            "Determine whether a Mercedes-Benz eActros 600 can complete a "
            "segment given its current state of charge (SOC %), keeping a "
            "safety reserve. Returns energy needed vs. available, whether the "
            "destination is reachable, the kWh margin, and the estimated "
            "remaining SOC and range afterwards. Use this for any 'can it make "
            "it / will it reach' question. Numeric inputs may be strings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "soc_pct": {
                    "type": "number",
                    "description": "Current battery state of charge in percent (0-100).",
                },
                "distance_km": {
                    "type": "number",
                    "description": "Segment distance in kilometres.",
                },
                "payload_t": {
                    "type": "number",
                    "description": "Cargo payload in tonnes (0-22).",
                },
                "speed_kph": {
                    "type": "number",
                    "description": "Average travel speed in km/h.",
                },
                "gradient_pct": {
                    "type": "number",
                    "description": "Average road gradient in percent (positive = uphill).",
                },
                "temperature_c": {
                    "type": "number",
                    "description": "Ambient temperature in degrees Celsius.",
                },
                "wind_mps": {
                    "type": "number",
                    "description": "Headwind component in m/s (0-12). Defaults to 0.",
                },
                "reserve_pct": {
                    "type": "number",
                    "description": (
                        "Battery percentage to hold back as a safety reserve "
                        "(default 10)."
                    ),
                },
            },
            "required": [
                "soc_pct",
                "distance_km",
                "payload_t",
                "speed_kph",
                "gradient_pct",
                "temperature_c",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------
def predict_energy_tool(**kwargs: Any) -> dict[str, Any]:
    """Wrapper over :func:`nexdash.model.predict_energy`.

    Accepts the ``predict_energy`` tool arguments (numbers or numeric
    strings), coerces them, and returns a JSON-serializable result dict.
    """
    model_path = kwargs.get("model_path", DEFAULT_MODEL_PATH)
    features = {
        "distance_km": _to_float(kwargs.get("distance_km"), field="distance_km"),
        "payload_t": _to_float(kwargs.get("payload_t"), field="payload_t"),
        "speed_kph": _to_float(kwargs.get("speed_kph"), field="speed_kph"),
        "gradient_pct": _to_float(kwargs.get("gradient_pct"), field="gradient_pct"),
        "temperature_c": _to_float(kwargs.get("temperature_c"), field="temperature_c"),
        "wind_mps": _to_float(kwargs.get("wind_mps"), default=0.0, field="wind_mps"),
    }
    energy_kwh = float(predict_energy(features, model_path=model_path))
    return {
        "energy_kwh": round(energy_kwh, 3),
        "inputs": features,
    }


def check_reach_tool(**kwargs: Any) -> dict[str, Any]:
    """Wrapper over :func:`nexdash.range.check_reachability`.

    Coerces tool arguments and forwards them, returning the reachability
    dict produced by the range module (already JSON-serializable).
    """
    model_path = kwargs.get("model_path", DEFAULT_MODEL_PATH)
    result = check_reachability(
        soc_pct=_to_float(kwargs.get("soc_pct"), field="soc_pct"),
        distance_km=_to_float(kwargs.get("distance_km"), field="distance_km"),
        payload_t=_to_float(kwargs.get("payload_t"), field="payload_t"),
        speed_kph=_to_float(kwargs.get("speed_kph"), field="speed_kph"),
        gradient_pct=_to_float(kwargs.get("gradient_pct"), field="gradient_pct"),
        temperature_c=_to_float(kwargs.get("temperature_c"), field="temperature_c"),
        wind_mps=_to_float(kwargs.get("wind_mps"), default=0.0, field="wind_mps"),
        model_path=model_path,
        reserve_pct=_to_float(kwargs.get("reserve_pct"), default=10.0, field="reserve_pct"),
    )
    # Ensure the payload is a plain dict (defensive; range returns a dict).
    return dict(result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_DISPATCH_TABLE: dict[str, Callable[..., dict[str, Any]]] = {
    "predict_energy": predict_energy_tool,
    "check_reachability": check_reach_tool,
}


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Route a tool ``name`` to its wrapper, passing ``args`` as kwargs.

    Raises :class:`KeyError` for an unknown tool name so the caller's
    tool-use loop fails loudly rather than silently returning nothing.
    """
    try:
        func = _DISPATCH_TABLE[name]
    except KeyError:
        raise KeyError(
            f"Unknown tool {name!r}. Available tools: "
            f"{sorted(_DISPATCH_TABLE)}"
        ) from None
    return func(**(args or {}))
