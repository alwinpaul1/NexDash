"""tool-use tool definitions and JSON-serializable dispatch layer.

This module exposes the NexDash energy-prediction capabilities as
tool-use tool-use schemas (:data:`TOOL_SPECS`) together with thin Python
wrappers that the model-driven agents (and the MCP server) call when a
tool-use block is returned by the model.

The wrappers are intentionally tolerant of *string* numeric inputs: tool
arguments arriving from an LLM frequently come through as strings (or as
``null``), so every numeric field is coerced via :func:`_to_float` before
being handed to the underlying physics/ML layer. All return values are
plain ``dict`` objects containing only JSON-serializable scalars so they
can be embedded directly in a ``tool_result`` content block.
"""

from __future__ import annotations

from typing import Any, Callable

from nexdash.config import DEFAULT_MODEL_PATH, TRUCK
from nexdash.model import predict_energy
from nexdash.range import check_reachability

__all__ = [
    "TOOL_SPECS",
    "predict_energy_tool",
    "check_reach_tool",
    "plan_route_tool",
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
# tool-use tool schemas
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
                        "Net road gradient in percent; positive = uphill, "
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
                    "description": "Net road gradient in percent; positive = uphill, negative = downhill.",
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
    {
        "name": "plan_route",
        "description": (
            "Plan a COMPLETE road trip for a Mercedes-Benz eActros 600 electric "
            "truck between two named places/cities. Geocodes the origin and "
            "destination, computes the real TomTom truck road route (40 t, 5-axle "
            "artic), then simulates state-of-charge drain, inserts DC fast-charging "
            "stops as needed, and checks EU 561 driver hours. Use this whenever a "
            "dispatcher describes a trip between places (e.g. 'Berlin to Munich, 12 "
            "tonnes, cold morning') and wants the full route + energy + charging "
            "plan. For a single isolated segment with a known distance, or a pure "
            "'will it reach' question, use predict_energy / check_reachability "
            "instead. Returns a compact JSON plan summary; on geocode/route failure "
            "it returns an 'error' field instead of throwing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Start location name (e.g. 'Berlin', 'Hamburg Hafen').",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination location name (e.g. 'Munich').",
                },
                "payload_t": {
                    "type": "number",
                    "description": "Cargo payload in tonnes (0-22). Defaults to 0.",
                },
                "start_soc": {
                    "type": "number",
                    "description": "Starting battery state of charge in percent (0-100). Defaults to 100.",
                },
                "temperature_c": {
                    "type": "number",
                    "description": "Ambient temperature in degrees Celsius. Defaults to 15.",
                },
                "departure": {
                    "type": "string",
                    "description": (
                        "Departure datetime as an ISO 8601 local string "
                        "(e.g. '2026-06-04T21:00'). Drives ETA, EU 561 breaks "
                        "and on-time checks. Defaults to now if omitted."
                    ),
                },
                "deliver_by": {
                    "type": "string",
                    "description": (
                        "Delivery deadline at the destination as an ISO 8601 "
                        "local datetime (e.g. '2026-06-05T12:00'). When set, the "
                        "plan reports whether arrival is on time / early / late."
                    ),
                },
                "min_soc": {
                    "type": "number",
                    "description": "SOC floor (%) never to dip below. Defaults to 15.",
                },
                "reserve_pct": {
                    "type": "number",
                    "description": "Safety-reserve buffer (%) above min SOC. Defaults to 10.",
                },
                "max_charge_kw": {
                    "type": "number",
                    "description": "Max charging power (kW) the truck accepts. Defaults to 400.",
                },
                "min_charger_kw": {
                    "type": "number",
                    "description": (
                        "Minimum charger power to consider (kW); slower chargers "
                        "are skipped. Defaults to 150."
                    ),
                },
                "max_detour_km": {
                    "type": "number",
                    "description": (
                        "Max detour off the route to reach a charger (km) -- the "
                        "charger search radius around each stop. Defaults to 30."
                    ),
                },
            },
            "required": ["origin", "destination"],
        },
    },
]


# The resolved parameters + coordinates of the most recent plan_route_tool call.
# The server surfaces this as a structured ``planRequest`` so the frontend can
# fill the planner form and run the same Optimize pipeline. It is best-effort
# (set on every successful geocode+route) and intentionally module-global so the
# agent/server can read the last call without threading state through the loop.
_PLAN_ROUTE_LAST: dict[str, Any] | None = None


def get_last_plan_request() -> dict[str, Any] | None:
    """Return the structured ``planRequest`` from the last successful plan_route.

    Shape (per the shared frontend contract)::

        {origin:{label,lat,lng}, destination:{label,lat,lng}, payloadKg,
         startSoc, temperatureC, departure, deliverBy, minSoc, reservePct,
         maxChargeKw}

    Returns ``None`` if no plan_route call has resolved coordinates yet.
    """
    return _PLAN_ROUTE_LAST


def reset_last_plan_request() -> None:
    """Clear the cached last plan_route request (call before a chat turn)."""
    global _PLAN_ROUTE_LAST
    _PLAN_ROUTE_LAST = None


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


def _reconcile_charge_times_to_real_power(
    stops: list[dict[str, Any]],
    *,
    max_charge_kw: float,
    battery_kwh: float,
) -> tuple[list[dict[str, Any]], float, int]:
    """Re-time each charging stop at the REAL matched station's power.

    The planner times every charge at the truck's flat max-accept rate
    (``max_charge_kw``, default 400 kW) because it has no station knowledge during
    the SOC walk. After TomTom enrichment each stop may carry a ``station`` with a
    real ``effective_power_kw`` (already capped at the truck intake). A real
    station can only equal or UNDER-cut the truck cap, so the recomputed charge
    time is always >= the planned time -- the ETA shifts later, never earlier
    (conservative; it can never make a trip look more reachable than it is).
    Charger power affects only TIME: SOC, kWh and which stop is taken are
    power-independent and left untouched.

    Mirrors the browser planner's ``reconcileChargeDurations`` so the agent/MCP
    path and the website agree. Like it, this folds the extra minutes into the
    ETA without re-running the EU 561 break/rest schedule (a conservative time
    shift, not a re-plan).

    Args:
        stops: The planner's charging stops AFTER enrichment (each has
            ``arriveSoc``/``departSoc``/``durationMin`` and an optional
            ``station`` dict).
        max_charge_kw: The truck's max charge-accept rate the plan was timed at.
        battery_kwh: Usable pack capacity (for the taper-aware time model).

    Returns:
        ``(stops_out, extra_minutes_total, n_unresolved)`` -- ``extra_minutes_total``
        is the summed (real - planned) charge minutes to add to the ETA / total
        time; ``n_unresolved`` counts NEEDED charges with no real station found
        (still timed at the truck cap and flagged ``stationResolved=False``).
    """
    from nexdash.route_planner import CHARGER_KW, _charge_minutes

    cap = max_charge_kw if (max_charge_kw and max_charge_kw > 0) else CHARGER_KW
    out: list[dict[str, Any]] = []
    extra_total = 0.0
    n_unresolved = 0
    for s in stops or []:
        new_stop = dict(s)
        station = s.get("station") if isinstance(s.get("station"), dict) else None
        arrive = s.get("arriveSoc")
        depart = s.get("departSoc")
        planned_min = s.get("durationMin")
        real_kw = None
        if station is not None:
            real_kw = station.get("effective_power_kw") or station.get("max_power_kw")

        if (
            real_kw
            and real_kw > 0
            and isinstance(arrive, (int, float))
            and isinstance(depart, (int, float))
            and depart > arrive
        ):
            eff_kw = min(float(cap), float(real_kw))
            real_min = _charge_minutes(float(arrive), float(depart), eff_kw, battery_kwh)
            base_min = (
                float(planned_min)
                if isinstance(planned_min, (int, float))
                else _charge_minutes(float(arrive), float(depart), float(cap), battery_kwh)
            )
            extra_total += max(0.0, real_min - base_min)
            new_stop["durationMin"] = round(real_min)
            new_stop["chargePowerKw"] = round(eff_kw)
            new_stop["stationResolved"] = True
        elif station is None:
            # A needed charge with no real charger resolved nearby: the time stays
            # at the truck-cap assumption. Flag it so the answer never presents a
            # phantom hub as if it were a confirmed real station.
            n_unresolved += 1
            new_stop["stationResolved"] = False
        out.append(new_stop)
    return out, extra_total, n_unresolved


def plan_route_tool(**kwargs: Any) -> dict[str, Any]:
    """Plan a full eActros 600 trip between two named places.

    Geocodes ``origin`` + ``destination`` (TomTom Search), routes the truck
    between them (TomTom calculateRoute), then runs the real SOC + charging
    simulation via :func:`nexdash.route_planner.plan_route`, and returns a
    compact JSON summary in the shared agent contract shape.

    Never raises: geocode/route failures (and any unexpected error) are caught
    and returned as ``{"error": "..."}`` so the agent's tool loop can narrate
    the failure rather than crash.
    """
    # Local imports keep TomTom/routing deps out of the import path for the
    # lightweight energy tools and the MCP server that only need the model.
    from nexdash import tomtom
    from nexdash.route_planner import plan_route as _plan_route

    origin = (kwargs.get("origin") or "").strip()
    destination = (kwargs.get("destination") or "").strip()
    if not origin or not destination:
        return {"error": "Both 'origin' and 'destination' are required."}

    payload_t = _to_float(kwargs.get("payload_t"), default=0.0, field="payload_t")
    payload_t = max(0.0, min(22.0, payload_t))
    start_soc = _to_float(kwargs.get("start_soc"), default=100.0, field="start_soc")
    start_soc = max(0.0, min(100.0, start_soc))
    temperature_c = _to_float(
        kwargs.get("temperature_c"), default=15.0, field="temperature_c"
    )
    min_soc = _to_float(kwargs.get("min_soc"), default=15.0, field="min_soc")
    min_soc = max(0.0, min(100.0, min_soc))
    reserve_pct = _to_float(kwargs.get("reserve_pct"), default=10.0, field="reserve_pct")
    reserve_pct = max(0.0, min(100.0, reserve_pct))
    max_charge_kw = _to_float(
        kwargs.get("max_charge_kw"), default=400.0, field="max_charge_kw"
    )
    if max_charge_kw <= 0:
        max_charge_kw = 400.0
    # Charger filters (mirror the website's "Min Charger Speed" + "Max Charging
    # Detour" sliders): skip chargers below `min_charger_kw`, and only consider
    # stations within `max_detour_km` of the route point (the POI search radius).
    min_charger_kw = _to_float(
        kwargs.get("min_charger_kw"), default=150.0, field="min_charger_kw"
    )
    if min_charger_kw <= 0:
        min_charger_kw = 150.0
    max_detour_km = _to_float(
        kwargs.get("max_detour_km"), default=30.0, field="max_detour_km"
    )
    if max_detour_km <= 0:
        max_detour_km = 30.0
    # Departure / deadline are optional ISO strings passed straight through; the
    # route planner parses them (and tolerates None -> "now").
    departure = kwargs.get("departure") or None
    if isinstance(departure, str):
        departure = departure.strip() or None
    deliver_by = kwargs.get("deliver_by") or None
    if isinstance(deliver_by, str):
        deliver_by = deliver_by.strip() or None
    model_path = kwargs.get("model_path", DEFAULT_MODEL_PATH)

    try:
        a = tomtom.geocode(origin)
        b = tomtom.geocode(destination)
        route = tomtom.truck_route(
            [
                {"lat": a["lat"], "lng": a["lng"]},
                {"lat": b["lat"], "lng": b["lng"]},
            ]
        )
    except tomtom.TomTomError as exc:
        # TomTomError messages are already scrubbed of the key in tomtom.py, but
        # redact again as defence-in-depth before returning to an MCP client.
        return {"error": tomtom._redact(str(exc))}
    except Exception as exc:  # noqa: BLE001 - never throw out of a tool wrapper
        # Report the exception TYPE only — a raw exception string can embed an
        # httpx URL-with-key or a filesystem path. Never interpolate ``exc``.
        return {"error": f"Route lookup failed ({type(exc).__name__})."}

    # The destination waypoint carries the delivery deadline so the planner can
    # flag on-time / late arrival.
    dest_waypoint: dict[str, Any] = {
        "lat": b["lat"],
        "lng": b["lng"],
        "label": b["label"],
    }
    if deliver_by:
        dest_waypoint["deliverBy"] = deliver_by

    try:
        plan = _plan_route(
            distance_km=route["distance_km"],
            duration_s=route["duration_s"],
            start_soc=start_soc,
            min_soc=min_soc,
            payload_kg=payload_t * 1000.0,
            reserve_pct=reserve_pct,
            max_charge_kw=max_charge_kw,
            departure=departure,
            temperature_c=temperature_c,
            geometry=route["geometry"],
            leg_timings=route["leg_timings"],
            speed_limits=route.get("speed_limits"),
            waypoints=[
                {"lat": a["lat"], "lng": a["lng"], "label": a["label"]},
                dest_waypoint,
            ],
            model_path=model_path,
        )
    except Exception as exc:  # noqa: BLE001 - simulation failure -> structured error
        # Type only: the simulation can raise with a model_path in the message.
        return {"error": f"Route simulation failed ({type(exc).__name__})."}

    # Record the resolved params + coords so the server can surface a structured
    # planRequest for the frontend (fills the planner + runs Optimize).
    global _PLAN_ROUTE_LAST
    _PLAN_ROUTE_LAST = {
        "origin": {"label": a["label"], "lat": a["lat"], "lng": a["lng"]},
        "destination": {"label": b["label"], "lat": b["lat"], "lng": b["lng"]},
        "payloadKg": payload_t * 1000.0,
        "startSoc": start_soc,
        "temperatureC": temperature_c,
        "departure": departure,
        "deliverBy": deliver_by,
        "minSoc": min_soc,
        "reservePct": reserve_pct,
        "maxChargeKw": max_charge_kw,
    }

    summary = plan.get("summary") or {}
    driver = summary.get("driver") or {}

    # Resolve each simulated charging stop to the ACTUAL time-optimal CCS station
    # near it (TomTom EV charging POIs) — operator name, power, live availability,
    # opening hours, price — exactly like the browser planner's enrichStations().
    # Best-effort: on any failure the stop keeps its synthetic name and station=None.
    raw_stops = plan.get("chargingStops") or []
    try:
        raw_stops = tomtom.enrich_charging_stations(
            raw_stops,
            radius_km=max_detour_km,
            min_charger_kw=min_charger_kw,
            max_charge_kw=max_charge_kw,
        )
    except Exception:  # noqa: BLE001 - never fail the plan over POI enrichment
        pass

    # Re-time each charge at the REAL matched station's power. The SOC walk timed
    # every charge at the truck's flat max-accept rate (no station knowledge mid-
    # walk); a real station can only charge the same or slower, so this shifts the
    # ETA later (never earlier) and surfaces the real per-stop charge minutes. SOC,
    # kWh and stop placement are power-independent and unchanged.
    extra_charge_min = 0.0
    n_unresolved_chargers = 0
    try:
        raw_stops, extra_charge_min, n_unresolved_chargers = (
            _reconcile_charge_times_to_real_power(
                raw_stops, max_charge_kw=max_charge_kw, battery_kwh=TRUCK.battery_kwh
            )
        )
    except Exception:  # noqa: BLE001 - never fail the plan over time reconciliation
        pass

    charging_stops = []
    for s in raw_stops:
        charging_stops.append(
            {
                "name": s.get("name"),
                "dist_km": s.get("distKm"),
                "arrive_soc": s.get("arriveSoc"),
                "depart_soc": s.get("departSoc"),
                "kwh": s.get("kWh"),
                # Charge time re-computed at the REAL station's power (taper-aware),
                # the power used (kW), and whether a real charger was resolved at
                # all (False -> still timed at the truck cap, surfaced honestly).
                "charge_min": s.get("durationMin"),
                "charge_power_kw": s.get("chargePowerKw"),
                "station_resolved": s.get("stationResolved"),
                # The real station it charges at (None if no charger could be
                # resolved): name, address, off_route_km, connectors, max/eff power,
                # live availability, opening hours, price_per_kwh.
                "station": s.get("station"),
            }
        )

    # on_time reflects the FINAL (destination) stop's onTime flag, where the
    # delivery deadline is checked. None when no deadline was supplied (or no
    # destination stop was emitted).
    stops_out = plan.get("stops") or []
    on_time = None
    if stops_out:
        on_time = stops_out[-1].get("onTime")

    # Fold the real-charger time correction into the headline ETA / total time.
    # extra_charge_min >= 0 (slower real chargers only add time), so the ETA can
    # only move later -- and a deadline that was met at the optimistic cap-rate
    # ETA is re-checked against the realistic one.
    eta_label = summary.get("etaLabel")
    eta_iso = summary.get("etaIso")
    total_time_h = summary.get("totalTimeH")
    if extra_charge_min and extra_charge_min > 0:
        from datetime import datetime, timedelta

        if isinstance(total_time_h, (int, float)):
            total_time_h = round(total_time_h + extra_charge_min / 60.0, 2)
        if isinstance(eta_iso, str):
            try:
                shifted = datetime.fromisoformat(eta_iso) + timedelta(
                    minutes=extra_charge_min
                )
                eta_iso = shifted.isoformat(timespec="minutes")
                eta_label = shifted.strftime("%H:%M")
                if deliver_by:
                    try:
                        on_time = shifted <= datetime.fromisoformat(deliver_by)
                    except Exception:  # noqa: BLE001 - deadline parse best-effort
                        pass
            except Exception:  # noqa: BLE001 - ETA shift best-effort
                pass

    # Surface the live route conditions the plan was optimised against — per-
    # segment wind + elevation/gradient + temperature come from Open-Meteo (when
    # a route geometry is available) and already shape the energy estimate, so the
    # MCP client can see they were taken into account (not just the headline kWh).
    cond = plan.get("conditions") or {}
    conditions = (
        {
            "avg_temp_c": cond.get("avgTempC"),
            "avg_wind_mps": cond.get("avgWindMps"),
            "wind_dir_deg": cond.get("windDirDeg"),
            "elevation_gain_m": cond.get("climbM"),
            "elevation_loss_m": cond.get("descentM"),
            "weather_source": cond.get("weatherSource"),
            "weather_degraded": cond.get("weatherDegraded"),
            "elevation_degraded": cond.get("elevationDegraded"),
            "source": "Open-Meteo (per-segment wind, elevation, temperature)",
        }
        if cond
        else None
    )

    # Live traffic the route was planned around: the delay already baked into the
    # ETA (routeType=fastest + traffic=true) plus the ETA-relevant incidents
    # (accidents / jams / closures / roadworks) on the corridor — the browser
    # planner's trafficDelayS + fetchIncidents, surfaced server-side. Best-effort.
    try:
        incidents = tomtom.fetch_traffic_incidents(route.get("geometry") or [])
    except Exception:  # noqa: BLE001 - incidents are advisory, never fatal
        incidents = []
    delay_s = int(route.get("traffic_delay_s") or 0)
    traffic = {
        "delay_s": delay_s,
        "delay_min": round(delay_s / 60.0, 1) if delay_s else 0.0,
        "incident_count": len(incidents),
        "incidents": incidents,
        "source": "TomTom (live traffic + incident details)",
    }

    return {
        "origin": {"label": a["label"], "lat": a["lat"], "lng": a["lng"]},
        "destination": {"label": b["label"], "lat": b["lat"], "lng": b["lng"]},
        "distance_km": summary.get("distanceKm"),
        "energy_kwh": summary.get("energyKwh"),
        "kwh_per_100": summary.get("kwhPer100"),
        "arrival_soc": summary.get("arrivalSoc"),
        "min_soc": summary.get("minSoc"),
        "charging_stops": charging_stops,
        "n_charging_stops": summary.get("chargingStops"),
        # Number of NEEDED charges with no real station resolvable nearby (timed at
        # the truck-cap assumption). 0 means every charge is at a confirmed station.
        "chargers_unresolved": n_unresolved_chargers,
        "driving_time_h": summary.get("drivingTimeH"),
        "total_time_h": total_time_h,
        "departure": departure,
        "eta": eta_label,
        "eta_iso": eta_iso,
        "deliver_by": deliver_by,
        "on_time": on_time,
        "eu561_ok": driver.get("eu561ok"),
        "conditions": conditions,
        "traffic": traffic,
        "assumptions": summary.get("assumptions"),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_DISPATCH_TABLE: dict[str, Callable[..., dict[str, Any]]] = {
    "predict_energy": predict_energy_tool,
    "check_reachability": check_reach_tool,
    "plan_route": plan_route_tool,
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
