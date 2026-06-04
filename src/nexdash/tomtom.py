"""Server-side TomTom geocode + truck-routing helper.

Mirrors the frontend route pipeline (``frontend/src/lib/routePlanner.js``):
the same Search geocode endpoint and the same ``calculateRoute`` truck call
with the identical ``TRUCK_SPEC`` (40 t GCW, 5 axles, 16.5 m artic) so a route
planned through this module matches the one the browser planner produces and
the eActros 600 the backend simulates.

The TomTom API key is read from ``TOMTOM_API_KEY`` in the environment, falling
back to parsing ``VITE_TOMTOM_API_KEY`` out of ``frontend/.env`` (so the same
key the frontend uses works server-side with no extra setup).

Networking uses ``httpx`` (already a project dependency) with bounded timeouts.
Every public function fails GRACEFULLY: it raises :class:`TomTomError` with a
human-readable message rather than leaking a raw SDK/transport exception, so the
agent tool layer can turn it into ``{"error": ...}`` for the model.
"""

from __future__ import annotations

import contextvars
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

# httpx logs each request URL at INFO level, and our TomTom URLs carry the API
# key as a ``?key=`` query parameter — so silence httpx's request logger to keep
# the key out of local logs/stderr (defence-in-depth; it is already scrubbed from
# any client-facing error via _redact()).
logging.getLogger("httpx").setLevel(logging.WARNING)

# get_api_key is intentionally NOT exported: it returns the secret TomTom key,
# so it must not be reachable via ``from nexdash.tomtom import *``. Internal
# callers reference it by its qualified name (``tomtom.get_api_key``).
__all__ = [
    "TomTomError",
    "geocode",
    "truck_route",
    "enrich_charging_stations",
    "fetch_traffic_incidents",
    "rank_chargers_by_time",
    "_redact",
]

# Single source of truth for the routed vehicle — copied verbatim from the
# frontend TRUCK_SPEC so the server-routed truck never diverges from the browser
# one (kerb ~18 t + 22 t payload = 40 t GCW, 5-axle artic; routing capped at the
# German 80 km/h truck limit so ETAs stay realistic).
TRUCK_SPEC = {
    "weightKg": 40000,
    "axleWeightKg": 11500,
    "numberOfAxles": 5,
    "lengthM": 16.5,
    "widthM": 2.55,
    "heightM": 4.0,
    "maxSpeedKph": 80,
}

# Same country bias the frontend geocoder uses (Germany-centric EU corridor).
COUNTRY_SET = "DE,AT,CH,NL,BE,FR,PL,CZ,DK"

# Per-request HTTP timeout (seconds). Generous enough for a multi-stop truck
# route, bounded so the agent tool can't hang the chat request.
_TIMEOUT_S = 12.0

_ENV_FILE = Path(__file__).resolve().parents[2] / "frontend" / ".env"


class TomTomError(RuntimeError):
    """Raised for any TomTom geocode/routing failure (missing key, network, bad response).

    Catchable so callers (the ``plan_route`` tool) degrade to ``{"error": ...}``
    instead of crashing the agent's tool-use loop.
    """


# --------------------------------------------------------------------------- #
# API key resolution
# --------------------------------------------------------------------------- #
def _parse_env_key(path: Path) -> Optional[str]:
    """Pull ``VITE_TOMTOM_API_KEY`` out of a dotenv file, or ``None``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^\s*VITE_TOMTOM_API_KEY\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip().strip('"').strip("'")
    return val or None


# Per-request "bring your own key" override. A remote caller (e.g. the MCP
# server reading an X-TomTom-Key request header) sets this for the duration of
# one request so the routing uses THEIR TomTom key, never the host's. It is a
# ContextVar so concurrent requests can't see each other's key.
_request_api_key: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "tomtom_request_api_key", default=None
)


def set_request_api_key(key: Optional[str]):
    """Bind a per-request TomTom key; returns a token for :func:`reset_request_api_key`."""
    return _request_api_key.set(key.strip() if key and key.strip() else None)


def reset_request_api_key(token) -> None:
    """Restore the key bound before the matching :func:`set_request_api_key`."""
    _request_api_key.reset(token)


def get_api_key() -> str:
    """Return the TomTom key: a per-request override first, then env, then
    ``frontend/.env``.

    Raises
    ------
    TomTomError
        If no key can be found.
    """
    request_key = _request_api_key.get()
    if request_key and request_key.strip():
        return request_key.strip()
    key = os.environ.get("TOMTOM_API_KEY")
    if key and key.strip():
        return key.strip()
    key = _parse_env_key(_ENV_FILE)
    if key:
        return key
    raise TomTomError(
        "No TomTom API key found. Set TOMTOM_API_KEY in the environment or "
        "VITE_TOMTOM_API_KEY in frontend/.env."
    )


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _redact(text: str) -> str:
    """Strip the TomTom API key (and any ``key=...`` query param) from ``text``.

    Defence-in-depth: TomTom error bodies and httpx exception strings can echo
    the full request URL, which carries ``?key=<TOMTOM_API_KEY>``. Anything that
    might reach a log or an MCP client is passed through this first so the secret
    never leaks. Resolving the key here must itself never raise.
    """
    if not text:
        return text
    try:
        key = os.environ.get("TOMTOM_API_KEY") or _parse_env_key(_ENV_FILE)
    except Exception:  # noqa: BLE001 - redaction must never raise
        key = None
    if key:
        text = text.replace(key, "[REDACTED]")
    # Also mask any literal ``key=<value>`` even if the key resolves differently.
    text = re.sub(r"(?i)(key=)[^&\s\"']+", r"\1[REDACTED]", text)
    return text


def _get_json(url: str) -> dict[str, Any]:
    """GET ``url`` and return parsed JSON, mapping any failure to TomTomError.

    Built so the API key can never escape: the URL is never echoed, transport
    exceptions are reported by type only, and any upstream error body is redacted
    before it could reach a caller/log.
    """
    import httpx

    try:
        # follow_redirects=False: an upstream 3xx must not be used to bounce the
        # request (with its key) at an internal/metadata IP (SSRF hardening).
        resp = httpx.get(url, timeout=_TIMEOUT_S, follow_redirects=False)
    except Exception as exc:  # noqa: BLE001 - any transport failure -> graceful
        # Report only the exception TYPE: httpx exception strings embed the full
        # URL (which contains ?key=...). Never interpolate ``exc`` itself.
        raise TomTomError(
            f"Could not reach TomTom ({type(exc).__name__})."
        ) from exc
    if resp.status_code != 200:
        # Do NOT include resp.text: TomTom error bodies routinely echo the
        # request URL (with the key). Status code alone is enough to diagnose.
        raise TomTomError(f"TomTom request failed (HTTP {resp.status_code}).")
    try:
        return resp.json() or {}
    except ValueError as exc:
        raise TomTomError("TomTom returned a non-JSON response.") from exc


# --------------------------------------------------------------------------- #
# Geocoding (TomTom Search API) — mirrors frontend geocode()
# --------------------------------------------------------------------------- #
def geocode(query: str) -> dict[str, Any]:
    """Geocode a place name to ``{lat, lng, label}`` using the TomTom Search API.

    Returns the best (first) result. Raises :class:`TomTomError` when the query
    is blank or nothing matches.
    """
    q = (query or "").strip()
    if not q:
        raise TomTomError("Empty location query.")
    key = get_api_key()
    url = (
        f"https://api.tomtom.com/search/2/geocode/{quote(q)}.json"
        f"?key={key}&limit=6&countrySet={COUNTRY_SET}"
    )
    data = _get_json(url)
    for r in data.get("results", []) or []:
        pos = r.get("position") or {}
        lat = pos.get("lat")
        lng = pos.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            continue
        addr = r.get("address") or {}
        label = (
            (r.get("poi") or {}).get("name")
            or addr.get("municipality")
            or addr.get("localName")
            or (addr.get("freeformAddress") or q).split(",")[0].strip()
        )
        return {"lat": float(lat), "lng": float(lng), "label": label}
    raise TomTomError(f"No location found for {query!r}.")


def _haversine_km(a: list[float], b: list[float]) -> float:
    """Great-circle distance (km) between ``[lat, lng]`` points — same Earth
    radius the frontend ``haversineKm`` and ``geodata`` use, so the cumulative
    distance that maps speedLimit sections to km spans matches the browser."""
    r = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# --------------------------------------------------------------------------- #
# Routing (TomTom calculateRoute, truck profile) — mirrors frontend tomtomRoute()
# --------------------------------------------------------------------------- #
def truck_route(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Route a truck through ``points`` (``[{lat,lng}, ...]``, >= 2).

    Returns
    -------
    dict
        ``{geometry: [[lat,lng], ...], leg_timings: [{lengthM, travelTimeS}],
        speed_limits: [{fromKm, toKm, kmh}], distance_km, duration_s}`` — the exact
        shape ``route_planner.plan_route`` consumes for ``geometry`` +
        ``leg_timings`` + ``speed_limits``. ``speed_limits`` mirrors the browser
        planner so a server-routed trip shapes per-segment speed by the real posted
        limits (autobahn 80 / town 50 / village 30) exactly as the web dashboard.

    Raises
    ------
    TomTomError
        On fewer than 2 valid points or any routing failure.
    """
    pts = [
        p
        for p in (points or [])
        if isinstance((p or {}).get("lat"), (int, float))
        and isinstance((p or {}).get("lng"), (int, float))
    ]
    if len(pts) < 2:
        raise TomTomError("Need at least 2 valid waypoints to route.")

    key = get_api_key()
    locs = ":".join(f"{p['lat']},{p['lng']}" for p in pts)
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/{locs}/json"
        f"?key={key}"
        f"&travelMode=truck"
        f"&routeType=fastest"
        f"&traffic=true"
        f"&vehicleMaxSpeed={TRUCK_SPEC['maxSpeedKph']}"
        f"&vehicleWeight={TRUCK_SPEC['weightKg']}"
        f"&vehicleAxleWeight={TRUCK_SPEC['axleWeightKg']}"
        f"&vehicleNumberOfAxles={TRUCK_SPEC['numberOfAxles']}"
        f"&vehicleLength={TRUCK_SPEC['lengthM']}"
        f"&vehicleWidth={TRUCK_SPEC['widthM']}"
        f"&vehicleHeight={TRUCK_SPEC['heightM']}"
        f"&vehicleCommercial=true"
        # Posted speed-limit sections along the route so the energy model can shape
        # per-segment speed by the real road (autobahn 80 / town 50 / village 30)
        # instead of one flat average — identical to the browser planner.
        f"&sectionType=speedLimit"
    )
    data = _get_json(url)
    routes = data.get("routes") or []
    if not routes:
        raise TomTomError("TomTom returned no route for those waypoints.")
    route = routes[0]

    geometry: list[list[float]] = []
    # Cumulative distance (km) at each geometry point — parallel to ``geometry`` —
    # so the speedLimit sections (which index into the route's point array) can be
    # mapped to km spans. Mirrors the frontend's ``cumKm`` exactly.
    cum_km: list[float] = []
    _cum = 0.0
    leg_timings: list[dict[str, Any]] = []
    for leg in route.get("legs", []) or []:
        for pt in leg.get("points", []) or []:
            lat = pt.get("latitude")
            lng = pt.get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                p = [float(lat), float(lng)]
                if geometry:
                    _cum += _haversine_km(geometry[-1], p)
                geometry.append(p)
                cum_km.append(_cum)
        ls = leg.get("summary") or {}
        leg_timings.append(
            {
                "lengthM": ls.get("lengthInMeters", 0) or 0,
                "travelTimeS": ls.get("travelTimeInSeconds", 0) or 0,
            }
        )

    # Posted speed limits -> distance spans (km), capped at the truck's legal max.
    # route_planner uses these as the per-segment speed SHAPE, anchored to the
    # measured leg time so the total ETA is unchanged. Same parsing as the browser.
    cap = float(TRUCK_SPEC["maxSpeedKph"])
    speed_limits: list[dict[str, Any]] = []
    for sec in route.get("sections", []) or []:
        st = sec.get("sectionType")
        if st and st != "SPEED_LIMIT":
            continue
        try:
            kmh = float(sec.get("maxSpeedLimitInKmh"))
            a = cum_km[int(sec.get("startPointIndex"))]
            b = cum_km[int(sec.get("endPointIndex"))]
        except (TypeError, ValueError, IndexError):
            continue
        if kmh <= 0 or b <= a:
            continue
        speed_limits.append({"fromKm": a, "toKm": b, "kmh": min(kmh, cap)})

    summary = route.get("summary") or {}
    return {
        "geometry": geometry,
        "leg_timings": leg_timings,
        "speed_limits": speed_limits,
        "distance_km": (summary.get("lengthInMeters", 0) or 0) / 1000.0,
        "duration_s": summary.get("travelTimeInSeconds", 0) or 0,
        # Live-traffic delay already baked into the travel time (routeType=fastest
        # + traffic=true) — surfaced so the MCP client can report "incl. N min of
        # live traffic", exactly like the browser planner's trafficDelayS.
        "traffic_delay_s": summary.get("trafficDelayInSeconds", 0) or 0,
    }


# --------------------------------------------------------------------------- #
# Real EV charging-station lookup (TomTom EV charging POIs, category 7309)
# Server-side port of the browser planner's enrichStations() so an MCP trip names
# the ACTUAL station it charges at (operator, power, live availability, price)
# instead of a synthetic "DC Fast-Charge Hub N". Best-effort: any failure leaves
# the stop unchanged — a charging plan is never lost because a POI lookup failed.
# --------------------------------------------------------------------------- #

# TomTom connector enum -> short dispatcher-friendly label (raw enum kept on miss).
_CONNECTOR_LABEL = {
    "IEC62196Type2CCS": "CCS",
    "IEC62196Type2CableAttached": "Type 2",
    "IEC62196Type2Outlet": "Type 2",
    "Combo": "CCS",
    "Chademo": "CHAdeMO",
    "GBT20234Part2": "GB/T",
    "GBT20234Part3": "GB/T",
    "IEC62196Type1": "Type 1",
    "IEC62196Type1CCS": "CCS",
    "Tesla": "Tesla",
}


def _connector_label(t: Optional[str]) -> str:
    if not t:
        return "Unknown"
    if t in _CONNECTOR_LABEL:
        return _CONNECTOR_LABEL[t]
    if t.startswith("GBT"):
        return "GB/T"
    if "CCS" in t:
        return "CCS"
    if "Type2" in t:
        return "Type 2"
    return t


def _extract_price_per_kwh(r: dict[str, Any]) -> Optional[float]:
    """Per-kWh ENERGY tariff for a station POI, or None (mirrors the frontend)."""
    try:
        tariffs = (r.get("references") or {}).get("tariffs") or (
            r.get("chargingPark") or {}
        ).get("tariffs")
        if not isinstance(tariffs, list) or not tariffs:
            return None
        for t in tariffs:
            elements = t.get("elements")
            comps: list[dict[str, Any]] = []
            if isinstance(elements, list):
                for e in elements:
                    comps.extend((e or {}).get("priceComponents") or [])
            else:
                comps = t.get("priceComponents") or []
            for c in comps:
                if c.get("type") == "ENERGY" and isinstance(c.get("price"), (int, float)):
                    return float(c["price"])
        return None
    except Exception:  # noqa: BLE001 - pricing is best-effort metadata
        return None


def _connector_power_of(c: dict[str, Any]) -> float:
    """Max ratedPowerKW across a candidate's chargingPark connectors."""
    powers = [
        float(x.get("ratedPowerKW") or 0)
        for x in (c.get("chargingPark") or {}).get("connectors") or []
    ]
    return max(powers) if powers else 0.0


def rank_chargers_by_time(
    candidates: list[dict[str, Any]],
    energy_kwh: float = 400.0,
    max_charge_kw: float = 400.0,
    detour_kph: float = 60.0,
) -> list[dict[str, Any]]:
    """Rank candidates by TOTAL added time = charge time at the station's real power
    (capped at the truck's max) + a round-trip detour penalty. Faster-first. The
    energy to add is the same for every candidate, so this prefers a higher-power
    charger slightly off-route over a slow one on the line — the browser's logic."""
    cap = max_charge_kw if (max_charge_kw and max_charge_kw > 0) else 400.0
    e = energy_kwh if (energy_kwh and energy_kwh > 0) else 400.0

    def score(c: dict[str, Any]) -> float:
        eff = min(cap, _connector_power_of(c))
        charge_min = (e / (eff * 0.9)) * 60 if eff > 0 else float("inf")
        detour_min = ((float(c.get("dist") or 0) / 1000.0) / detour_kph) * 60 * 2
        return charge_min + detour_min

    ranked = [{"c": c, "score": score(c)} for c in (candidates or [])]
    ranked.sort(key=lambda x: x["score"])
    return ranked


def _fetch_availability(availability_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Live CCS availability ``{available, total}`` for one station, or None.

    Counts ONLY CCS connectors (the eActros charges on CCS DC). Treats TomTom's
    "unknown" status as UNKNOWN (returns None) rather than a misleading "0 free".
    """
    if not availability_id:
        return None
    try:
        url = (
            f"https://api.tomtom.com/search/2/chargingAvailability.json"
            f"?key={get_api_key()}&chargingAvailability={quote(str(availability_id))}"
        )
        data = _get_json(url)
    except TomTomError:
        return None
    available = 0
    total = 0
    definite = 0
    for c in data.get("connectors") or []:
        typ = str(c.get("type") or "")
        if not ("CCS" in typ or "Combo" in typ):
            continue
        total += int(c.get("total") or 0)
        cur = (c.get("availability") or {}).get("current") or {}
        a = int(cur.get("available") or 0)
        available += a
        definite += (
            a
            + int(cur.get("occupied") or 0)
            + int(cur.get("reserved") or 0)
            + int(cur.get("outOfService") or 0)
        )
    if total == 0 or definite == 0:
        return None
    return {"available": available, "total": total}


def enrich_charging_stations(
    stops: list[dict[str, Any]],
    radius_km: float = 30.0,
    min_charger_kw: float = 150.0,
    max_charge_kw: float = 400.0,
) -> list[dict[str, Any]]:
    """Attach the time-optimal REAL CCS HPC station to each charging stop.

    ``stops`` are the planner's charging stops (each needs ``lat``/``lng`` and the
    planned ``kWh``). Returns a new list: each stop gets a ``station`` dict (name,
    address, coords, off_route_km, connectors, max/eff power, availability, opening
    hours, price, station charge minutes) when a real charger is found, else the
    stop is returned unchanged with ``station=None``. Never raises.
    """
    out: list[dict[str, Any]] = []
    radius = max(1000, round((radius_km or 30) * 1000))
    truck_min_kw = min_charger_kw if (min_charger_kw and min_charger_kw > 0) else 150.0
    cap = max_charge_kw if (max_charge_kw and max_charge_kw > 0) else 400.0

    for s in stops or []:
        lat = s.get("lat")
        lng = s.get("lng")
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            out.append({**s, "station": None})
            continue
        try:
            base = (
                f"https://api.tomtom.com/search/2/categorySearch/EV%20charging.json"
                f"?key={get_api_key()}&lat={lat}&lon={lng}&radius={radius}"
                f"&categorySet=7309&limit=12&openingHours=nextSevenDays&relatedPois=child"
            )

            def _nearest(min_power: float = 0, connector_set: str = "") -> list[dict[str, Any]]:
                u = base
                if min_power:
                    u += f"&minPowerKW={min_power}"
                if connector_set:
                    u += f"&connectorSet={connector_set}"
                try:
                    return (_get_json(u).get("results")) or []
                except TomTomError:
                    return []

            # Preference: CCS HPC -> any DC fast -> nearest-any (never empty).
            candidates = _nearest(truck_min_kw, "IEC62196Type2CCS")
            if not candidates:
                candidates = _nearest(truck_min_kw)
            if not candidates:
                candidates = _nearest()
            if not candidates:
                out.append({**s, "station": None})
                continue

            kwh = s.get("kWh") or s.get("kwh") or 0
            ranked = rank_chargers_by_time(candidates, energy_kwh=kwh, max_charge_kw=cap)
            top_k = ranked[:6]
            avails = [
                _fetch_availability(
                    ((t["c"].get("dataSources") or {}).get("chargingAvailability") or {}).get("id")
                )
                for t in top_k
            ]
            r = ranked[0]["c"]
            availability = avails[0] if avails else None
            for i in range(len(top_k)):
                if avails[i] and avails[i]["available"] > 0:
                    r = top_k[i]["c"]
                    availability = avails[i]
                    break

            # Connectors: dedupe by label, keep highest power per label.
            raw_conns = (r.get("chargingPark") or {}).get("connectors") or []
            max_power = 0.0
            by_label: dict[str, dict[str, Any]] = {}
            for c in raw_conns:
                pk = float(c.get("ratedPowerKW") or 0)
                if pk > max_power:
                    max_power = pk
                label = _connector_label(c.get("connectorType"))
                prev = by_label.get(label)
                if not prev or pk > prev["power_kw"]:
                    by_label[label] = {"label": label, "power_kw": round(pk)}
            connectors = sorted(by_label.values(), key=lambda x: -x["power_kw"])

            # Opening hours -> short human string.
            opening_hours = None
            trs = ((r.get("poi") or {}).get("openingHours") or {}).get("timeRanges") or []
            if trs:
                t0 = trs[0]

                def _fmt(x: Optional[dict[str, Any]]) -> str:
                    if not x:
                        return ""
                    return f"{int(x.get('hour', 0)):02d}:{int(x.get('minute', 0)):02d}"

                op = _fmt(t0.get("startTime"))
                cl = _fmt(t0.get("endTime"))
                if op and cl:
                    opening_hours = (
                        "Open 24/7" if (op == "00:00" and cl == "00:00") else f"{op}-{cl}"
                    )

            power_kw = max_power if max_power > 0 else None
            eff = min(cap, power_kw) if power_kw else None
            station_charge_minutes = (
                round((kwh / (eff * 0.9)) * 60) if (eff and kwh and kwh > 0) else None
            )

            station = {
                "name": (r.get("poi") or {}).get("name") or s.get("name"),
                "address": (r.get("address") or {}).get("municipality")
                or (r.get("address") or {}).get("freeformAddress"),
                "lat": (r.get("position") or {}).get("lat", lat),
                "lng": (r.get("position") or {}).get("lon", lng),
                "off_route_km": round(float(r.get("dist") or 0) / 1000.0, 1),
                "connectors": connectors,
                "max_power_kw": round(power_kw) if power_kw else None,
                "effective_power_kw": round(eff) if eff else None,
                "station_charge_minutes": station_charge_minutes,
                "availability": availability,
                "opening_hours": opening_hours,
                "price_per_kwh": _extract_price_per_kwh(r),
            }
            enriched = {**s, "station": station}
            # Promote the real operator name onto the stop itself so a client that
            # only reads `name` still sees the actual station.
            if station["name"]:
                enriched["name"] = station["name"]
                enriched["lat"] = station["lat"]
                enriched["lng"] = station["lng"]
            out.append(enriched)
        except Exception:  # noqa: BLE001 - enrichment is best-effort, never fatal
            out.append({**s, "station": None})
    return out


# --------------------------------------------------------------------------- #
# Live traffic incidents (TomTom Traffic Incident Details v5) — server-side port
# of the browser planner's fetchIncidents(): ETA-relevant accidents / jams /
# closures / roadworks essentially ON the route. Best-effort, never raises.
# --------------------------------------------------------------------------- #

_INCIDENT_LABEL = {
    0: "Traffic incident",
    1: "Accident",
    2: "Fog",
    3: "Dangerous conditions",
    4: "Rain",
    5: "Ice",
    6: "Traffic jam",
    7: "Lane closed",
    8: "Road closed",
    9: "Road works",
    10: "Wind",
    11: "Flooding",
    14: "Broken-down vehicle",
}


def fetch_traffic_incidents(geometry: list[list[float]]) -> list[dict[str, Any]]:
    """ETA-relevant live traffic incidents along ``geometry`` ([[lat,lng], ...]).

    Samples small bboxes along the route (the v5 bbox has an area cap), keeps only
    flow-affecting incidents within ~2.5 km of the road, dedupes by id, and returns
    up to 8 ordered by delay then severity. Returns [] on any failure.
    """
    if not isinstance(geometry, list) or len(geometry) < 2:
        return []
    try:
        n = min(10, max(2, round(len(geometry) / 60)))
        step = max(1, len(geometry) // n)
        samples = [geometry[i] for i in range(0, len(geometry), step)]
        fields = quote(
            "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,"
            "magnitudeOfDelay,events{description,code},from,to,delay,roadNumbers}}}"
        )
        pad = 0.18  # ~25x40 km box, under TomTom's v5 area limit

        raw: list[dict[str, Any]] = []
        for pt in samples:
            lat, lng = pt[0], pt[1]
            bbox = (
                f"{lng - pad:.5f},{lat - pad:.5f},{lng + pad:.5f},{lat + pad:.5f}"
            )
            url = (
                f"https://api.tomtom.com/traffic/services/5/incidentDetails"
                f"?key={get_api_key()}&bbox={bbox}&fields={fields}&language=en-GB"
                f"&timeValidityFilter=present&categoryFilter=1,6,7,8,9,14"
            )
            try:
                raw.extend(_get_json(url).get("incidents") or [])
            except TomTomError:
                continue

        seen: set = set()
        out: list[dict[str, Any]] = []
        corridor_stride = max(1, len(geometry) // 400)
        for inc in raw:
            p = inc.get("properties") or {}
            iid = p.get("id")
            if iid:
                if iid in seen:
                    continue
                seen.add(iid)
            g = inc.get("geometry") or {}
            lat0 = lng0 = None
            coords = g.get("coordinates")
            if g.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
                lng0, lat0 = coords[0], coords[1]
            elif g.get("type") == "LineString" and isinstance(coords, list) and coords:
                mid = coords[len(coords) // 2]
                if isinstance(mid, list) and len(mid) >= 2:
                    lng0, lat0 = mid[0], mid[1]
            if not isinstance(lat0, (int, float)) or not isinstance(lng0, (int, float)):
                continue
            # Corridor filter: keep only incidents <= 2.5 km from the travelled line.
            nearest = float("inf")
            for i in range(0, len(geometry), corridor_stride):
                dd = _haversine_km(geometry[i], [lat0, lng0])
                if dd < nearest:
                    nearest = dd
            if nearest > 2.5:
                continue
            cat = p.get("iconCategory") or 0
            events = p.get("events") or []
            desc = (
                (events[0].get("description") if events else None)
                or _INCIDENT_LABEL.get(cat)
                or "Traffic incident"
            )
            roads = p.get("roadNumbers")
            out.append(
                {
                    "category": _INCIDENT_LABEL.get(cat, "Traffic incident"),
                    "magnitude": p.get("magnitudeOfDelay") or 0,
                    "description": desc,
                    "from": p.get("from") or "",
                    "to": p.get("to") or "",
                    "delay_s": p.get("delay") or 0,
                    "road": ", ".join(roads) if isinstance(roads, list) else "",
                }
            )

        # ETA-relevant only: measurable delay, major severity, closure or jam.
        relevant = [
            x
            for x in out
            if x["delay_s"] >= 30
            or x["magnitude"] >= 3
            or x["category"] == "Road closed"
            or x["category"] == "Traffic jam"
        ]
        relevant.sort(key=lambda x: (-x["delay_s"], -x["magnitude"]))
        return relevant[:8]
    except Exception:  # noqa: BLE001 - incidents are advisory, never fatal
        return []
