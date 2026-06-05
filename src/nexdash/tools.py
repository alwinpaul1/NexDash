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

from nexdash.config import DEFAULT_MODEL_PATH
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

    plan_kwargs: dict[str, Any] = dict(
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
    try:
        # Pass 1: place the charging stops + walk SOC at the truck's max-accept
        # rate. ``return_enrichment`` hands back the fetched weather/elevation so a
        # second pass at real charger power needs no extra network call.
        plan = _plan_route(**plan_kwargs, return_enrichment=True)
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

    # Real-charger RE-SIMULATION. Pass 1 timed every charge at the truck cap and
    # ignored the off-route detour to each station; now that each stop is matched
    # to a real station, RE-RUN the simulation at (a) each station's real power so
    # the FULL EU 561 break/rest schedule reflects true charge times, and (b) each
    # station's real off-route distance so the round-trip spur's energy + time are
    # in the plan. Pass 1's enrichment is reused -> NO extra weather/elevation
    # fetch. SOC and stop placement are power-independent, so the stop sequence is
    # identical and the per-stop station mapping by index is exact. Both real power
    # (slower) and detours only ever lengthen the trip -> never optimistic.
    charger_kw_by_stop: list[float | None] = []
    detour_km_by_stop: list[float | None] = []
    n_unresolved_chargers = 0
    any_real_power = False
    for s in raw_stops:
        station = s.get("station") if isinstance(s.get("station"), dict) else None
        if station is None:
            n_unresolved_chargers += 1
            charger_kw_by_stop.append(None)
            detour_km_by_stop.append(None)
            continue
        kw = station.get("effective_power_kw") or station.get("max_power_kw")
        if kw and kw > 0:
            charger_kw_by_stop.append(float(kw))
            any_real_power = True
        else:
            charger_kw_by_stop.append(None)
        det = station.get("off_route_km")
        detour_km_by_stop.append(float(det) if (det and det > 0) else None)

    # True only once the re-simulation has actually replaced the cap-timed plan, so
    # the ETA / charge times / detour cost genuinely reflect the real chargers.
    # Stays False if nothing resolved OR the re-sim raised -- in which case the
    # times fall back to the truck-cap, detour-free assumption, and we say so.
    any_detour = any(d for d in detour_km_by_stop)
    real_power_applied = False
    if any_real_power or any_detour:
        try:
            plan2 = _plan_route(
                **plan_kwargs,
                precomputed_enrichment=plan.get("_enrichment"),
                charger_kw_by_stop=charger_kw_by_stop,
                detour_km_by_stop=detour_km_by_stop,
            )
            re_stops = plan2.get("chargingStops") or []
            # Placement is power- AND detour-independent (real power only changes
            # time; the detour model charges extra but leaves carried SOC unchanged),
            # so the re-sim MUST yield the same number of stops in the same order.
            # If the count ever diverges, the per-stop station mapping by index can no
            # longer be trusted -> bail out and keep the cap-timed plan rather than
            # ship a misattributed one.
            if len(re_stops) != len(charger_kw_by_stop):
                raise RuntimeError("re-sim charging-stop count diverged")
            # Re-attach the resolved stations (and their names) onto the re-timed
            # stops — same order, so the index mapping is exact.
            for i, rs in enumerate(re_stops):
                station = raw_stops[i].get("station") if i < len(raw_stops) else None
                rs["station"] = station
                if isinstance(station, dict) and station.get("name"):
                    rs["name"] = station["name"]
            raw_stops = re_stops
            plan = plan2
            summary = plan.get("summary") or {}
            real_power_applied = True
        except Exception:  # noqa: BLE001 - re-sim failure -> keep the cap-timed plan
            pass

    driver = summary.get("driver") or {}
    # Drop the internal enrichment handle so it never leaks into the response.
    plan.pop("_enrichment", None)

    charging_stops = []
    for s in raw_stops:
        station = s.get("station") if isinstance(s.get("station"), dict) else None
        eff_kw = (
            (station.get("effective_power_kw") or station.get("max_power_kw"))
            if station is not None
            else None
        )
        charging_stops.append(
            {
                "name": s.get("name"),
                "dist_km": s.get("distKm"),
                "arrive_soc": s.get("arriveSoc"),
                "depart_soc": s.get("departSoc"),
                "kwh": s.get("kWh"),
                # Charge time from the re-simulation at the real station's power
                # (taper- and EU-561-aware), the power used (kW), and whether a real
                # charger was resolved (False -> timed at the truck-cap assumption).
                "charge_min": s.get("durationMin"),
                # The power the charge was actually timed at: the real station's
                # effective power when known, else the truck cap (used by the
                # re-sim when a station resolved without a usable power rating, or
                # when no station resolved at all).
                "charge_power_kw": (
                    round(min(max_charge_kw, float(eff_kw)))
                    if eff_kw
                    else round(max_charge_kw)
                ),
                "station_resolved": station is not None,
                # The real station it charges at (None if no charger could be
                # resolved): name, address, off_route_km, connectors, max/eff power,
                # live availability, opening hours, price_per_kwh.
                "station": station,
            }
        )

    # on_time / ETA / totals come straight from the (re-simulated) plan — the EU 561
    # schedule already reflects the real charge times, so no post-hoc ETA shift.
    stops_out = plan.get("stops") or []
    on_time = None
    if stops_out:
        on_time = stops_out[-1].get("onTime")
    eta_label = summary.get("etaLabel")
    eta_iso = summary.get("etaIso")
    total_time_h = summary.get("totalTimeH")

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
        # True when the ETA / charge times reflect the matched real chargers' actual
        # power AND off-route detour (a full re-simulation). False -> they fall back
        # to the truck-cap, detour-free assumption (nothing resolved, or re-sim
        # could not run).
        "charge_times_real_power": real_power_applied,
        # Total round-trip distance (km) driven off-route to reach the matched real
        # chargers, already folded into energy_kwh / total_time_h / eta.
        "detour_km": summary.get("detourKm"),
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
