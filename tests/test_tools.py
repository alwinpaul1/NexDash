"""Tests for :mod:`nexdash.tools`.

These tests verify the contract that the agent / MCP layers depend on:

* ``TOOL_SPECS`` are well-formed tool-use schemas. WHY: the agent
  ships these verbatim to the MiniMax API; a missing ``name`` /
  ``description`` / ``input_schema`` (or a non-``object`` schema) is
  rejected by the API at request time, breaking every tool-use call.
* The wrappers return *JSON-serializable* dicts and coerce numeric
  strings. WHY: LLM-generated tool arguments routinely arrive as strings
  and the results are embedded directly into a ``tool_result`` content
  block that must be JSON-encodable.
* ``dispatch`` routes by name and fails loudly on an unknown tool. WHY:
  silently returning nothing would make the agent loop hang or hallucinate.

A tiny real model is trained once and saved to a temp path so the
wrappers exercise the genuine model -> features -> physics stack rather
than a mock; the wrappers are pointed at it via the ``model_path`` kwarg.
"""

from __future__ import annotations

import json

import pytest

from nexdash.data_gen import generate_dataset
from nexdash.model import train_model
from nexdash.tools import (
    TOOL_SPECS,
    check_reach_tool,
    dispatch,
    predict_energy_tool,
)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train and persist a small deterministic model, returning its path."""
    df = generate_dataset(n_samples=400, seed=7)
    path = tmp_path_factory.mktemp("models") / "energy_model.joblib"
    train_model(df, save=True, path=path)
    return str(path)


# ---------------------------------------------------------------------------
# TOOL_SPECS schema validity
# ---------------------------------------------------------------------------
def test_tool_specs_is_nonempty_list() -> None:
    assert isinstance(TOOL_SPECS, list)
    assert len(TOOL_SPECS) >= 2


def test_tool_specs_cover_required_tools() -> None:
    names = {spec["name"] for spec in TOOL_SPECS}
    # WHY: the agent, MCP server and dispatch table all key off these
    # exact names; renaming one silently breaks routing.
    assert {"predict_energy", "check_reachability"} <= names


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s["name"])
def test_tool_spec_shape(spec: dict) -> None:
    # Required top-level fields for an tool-use definition.
    assert isinstance(spec["name"], str) and spec["name"]
    assert isinstance(spec["description"], str) and spec["description"].strip()

    schema = spec["input_schema"]
    assert isinstance(schema, dict)
    # The MiniMax API requires the input_schema root to be a JSON-Schema
    # object; anything else is rejected before the model ever runs.
    assert schema["type"] == "object"
    assert isinstance(schema["properties"], dict) and schema["properties"]

    # Every declared required field must be defined in properties.
    for field in schema.get("required", []):
        assert field in schema["properties"], f"{field} required but undefined"


def test_tool_specs_are_json_serializable() -> None:
    # WHY: TOOL_SPECS are serialized into the API request body verbatim.
    json.dumps(TOOL_SPECS)


# ---------------------------------------------------------------------------
# Wrapper behaviour
# ---------------------------------------------------------------------------
def _assert_json_serializable(obj: dict) -> None:
    """Round-trips through JSON; fails if any non-serializable scalar leaks."""
    restored = json.loads(json.dumps(obj))
    assert isinstance(restored, dict)


def test_predict_energy_tool_returns_serializable_dict(model_path: str) -> None:
    result = predict_energy_tool(
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.5,
        temperature_c=5,
        wind_mps=3,
        model_path=model_path,
    )
    assert isinstance(result, dict)
    assert "energy_kwh" in result
    assert isinstance(result["energy_kwh"], float)
    # A 50 km laden segment must consume a positive, sane amount of energy
    # (well under the 600 kWh pack); this fails if the model/units regress.
    assert 0 < result["energy_kwh"] < 600
    _assert_json_serializable(result)


def test_predict_energy_tool_coerces_string_numerics(model_path: str) -> None:
    """String args (as an LLM emits) must parse to the same result as floats."""
    numeric = predict_energy_tool(
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.5,
        temperature_c=5,
        model_path=model_path,
    )
    stringy = predict_energy_tool(
        distance_km="50",
        payload_t="10",
        speed_kph="70",
        gradient_pct="1.5",
        temperature_c="5",
        model_path=model_path,
    )
    assert stringy["energy_kwh"] == pytest.approx(numeric["energy_kwh"])


def test_predict_energy_tool_wind_defaults_to_zero(model_path: str) -> None:
    # Omitting the optional wind_mps must not raise and must default to 0.
    with_default = predict_energy_tool(
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.5,
        temperature_c=5,
        model_path=model_path,
    )
    explicit_zero = predict_energy_tool(
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.5,
        temperature_c=5,
        wind_mps=0,
        model_path=model_path,
    )
    assert with_default["energy_kwh"] == pytest.approx(explicit_zero["energy_kwh"])


def test_predict_energy_tool_missing_required_raises(model_path: str) -> None:
    # A missing required numeric must fail loudly, not silently default.
    with pytest.raises(ValueError):
        predict_energy_tool(
            payload_t=10,
            speed_kph=70,
            gradient_pct=1.5,
            temperature_c=5,
            model_path=model_path,
        )


def test_check_reach_tool_returns_serializable_dict(model_path: str) -> None:
    result = check_reach_tool(
        soc_pct=80,
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.5,
        temperature_c=5,
        wind_mps=2,
        model_path=model_path,
    )
    assert isinstance(result, dict)
    # Contract keys the dashboard / agent rely on.
    for key in (
        "energy_needed_kwh",
        "energy_available_kwh",
        "reaches",
        "margin_kwh",
    ):
        assert key in result
    assert isinstance(result["reaches"], bool)
    _assert_json_serializable(result)


def test_check_reach_tool_coerces_string_numerics(model_path: str) -> None:
    numeric = check_reach_tool(
        soc_pct=80,
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.5,
        temperature_c=5,
        model_path=model_path,
    )
    stringy = check_reach_tool(
        soc_pct="80",
        distance_km="50",
        payload_t="10",
        speed_kph="70",
        gradient_pct="1.5",
        temperature_c="5",
        model_path=model_path,
    )
    assert stringy["energy_needed_kwh"] == pytest.approx(numeric["energy_needed_kwh"])
    assert stringy["reaches"] == numeric["reaches"]


def test_check_reach_tool_low_soc_is_unreachable(model_path: str) -> None:
    # A 1% SOC cannot cover a long laden segment; this encodes the business
    # rule that reachability honours available charge minus reserve.
    result = check_reach_tool(
        soc_pct=1,
        distance_km=120,
        payload_t=22,
        speed_kph=85,
        gradient_pct=4,
        temperature_c=-10,
        model_path=model_path,
    )
    assert result["reaches"] is False
    assert result["margin_kwh"] < 0


def test_plan_route_tool_forwards_charger_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Min Charger Speed + Max Charging Detour + Max Charging Speed filters
    must reach the real-charger search (parity with the website sliders)."""
    import nexdash.route_planner as rp
    import nexdash.tomtom as tomtom
    from nexdash.tools import plan_route_tool

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tomtom, "geocode", lambda q: {"lat": 0.0, "lng": 0.0, "label": q}
    )
    monkeypatch.setattr(
        tomtom,
        "truck_route",
        lambda wps: {
            "distance_km": 100.0,
            "duration_s": 3600.0,
            "geometry": [[0.0, 0.0], [1.0, 1.0]],
            "leg_timings": [],
            "speed_limits": None,
            "traffic_delay_s": 0,
        },
    )
    monkeypatch.setattr(
        rp,
        "plan_route",
        lambda **kw: {
            "summary": {
                "distanceKm": 100,
                "energyKwh": 120,
                "chargingStops": 1,
                "etaIso": "2026-06-05T12:00",
                "etaLabel": "12:00",
                "totalTimeH": 2.0,
            },
            "chargingStops": [
                {
                    "arriveSoc": 20.0,
                    "departSoc": 80.0,
                    "durationMin": 40,
                    "kWh": 120,
                    "distKm": 50,
                    "lat": 0.0,
                    "lng": 0.0,
                    "name": "Hub 1",
                }
            ],
            "stops": [],
            "conditions": {},
        },
    )

    def fake_enrich(stops, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return [{**s, "station": {"effective_power_kw": 200}} for s in stops]

    monkeypatch.setattr(tomtom, "enrich_charging_stations", fake_enrich)
    monkeypatch.setattr(tomtom, "fetch_traffic_incidents", lambda g: [])

    plan_route_tool(
        origin="A",
        destination="B",
        min_charger_kw=250,
        max_detour_km=75,
        max_charge_kw=350,
    )
    assert captured["min_charger_kw"] == 250.0
    assert captured["radius_km"] == 75.0  # the "Max Charging Detour" slider
    assert captured["max_charge_kw"] == 350.0


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------
def test_dispatch_routes_predict_energy(model_path: str) -> None:
    args = {
        "distance_km": 40,
        "payload_t": 8,
        "speed_kph": 65,
        "gradient_pct": 0.0,
        "temperature_c": 15,
        "model_path": model_path,
    }
    direct = predict_energy_tool(**args)
    routed = dispatch("predict_energy", args)
    assert routed == direct


def test_dispatch_routes_check_reachability(model_path: str) -> None:
    args = {
        "soc_pct": 60,
        "distance_km": 40,
        "payload_t": 8,
        "speed_kph": 65,
        "gradient_pct": 0.0,
        "temperature_c": 15,
        "model_path": model_path,
    }
    routed = dispatch("check_reachability", args)
    assert routed["reaches"] == check_reach_tool(**args)["reaches"]


def test_dispatch_unknown_tool_raises() -> None:
    # WHY: an unknown tool name must fail loudly so the agent loop does not
    # silently feed an empty result back to the model.
    with pytest.raises(KeyError):
        dispatch("not_a_real_tool", {})


def test_dispatch_handles_none_args() -> None:
    # dispatch must tolerate a None args payload (defaulting to {}); a tool
    # with no provided args should still surface the missing-arg ValueError.
    with pytest.raises(ValueError):
        dispatch("predict_energy", None)  # type: ignore[arg-type]
