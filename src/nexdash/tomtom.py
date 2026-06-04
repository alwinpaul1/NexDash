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
__all__ = ["TomTomError", "geocode", "truck_route", "_redact"]

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


# --------------------------------------------------------------------------- #
# Routing (TomTom calculateRoute, truck profile) — mirrors frontend tomtomRoute()
# --------------------------------------------------------------------------- #
def truck_route(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Route a truck through ``points`` (``[{lat,lng}, ...]``, >= 2).

    Returns
    -------
    dict
        ``{geometry: [[lat,lng], ...], leg_timings: [{lengthM, travelTimeS}],
        distance_km, duration_s}`` — the exact shape ``route_planner.plan_route``
        consumes for ``geometry`` + ``leg_timings``.

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
    )
    data = _get_json(url)
    routes = data.get("routes") or []
    if not routes:
        raise TomTomError("TomTom returned no route for those waypoints.")
    route = routes[0]

    geometry: list[list[float]] = []
    leg_timings: list[dict[str, Any]] = []
    for leg in route.get("legs", []) or []:
        for pt in leg.get("points", []) or []:
            lat = pt.get("latitude")
            lng = pt.get("longitude")
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                geometry.append([float(lat), float(lng)])
        ls = leg.get("summary") or {}
        leg_timings.append(
            {
                "lengthM": ls.get("lengthInMeters", 0) or 0,
                "travelTimeS": ls.get("travelTimeInSeconds", 0) or 0,
            }
        )

    summary = route.get("summary") or {}
    return {
        "geometry": geometry,
        "leg_timings": leg_timings,
        "distance_km": (summary.get("lengthInMeters", 0) or 0) / 1000.0,
        "duration_s": summary.get("travelTimeInSeconds", 0) or 0,
    }
