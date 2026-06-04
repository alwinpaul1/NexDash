"""FastMCP server exposing NexDash EV-truck range intelligence as MCP tools.

This module wraps the deterministic/ML prediction layer of NexDash (the
Mercedes-Benz eActros 600 energy model) as a Model Context Protocol (MCP)
server so that ANY MCP-capable AI client (desktop assistants, IDEs, custom
agents) can call the tools directly -- WITHOUT the NexDash route-planner
frontend. A user can plug in this server and ask, in plain language,
"travelling with 18 t on 22 March 2026 at 09:00 from Berlin to Munich"; the
client calls :func:`plan_route` and gets the full plan back (route, energy,
charging stops, ETA, on-time vs deadline, EU 561 driver hours).

Exposed tools (all thin delegations to :mod:`nexdash.tools`, so the MCP
server, the in-process agent and the FastAPI dashboard share ONE source of
truth -- no logic drift):

* ``predict_energy``     -- kWh for a single known-distance segment.
* ``check_reachability`` -- "will it reach?" given current SOC + a reserve.
* ``plan_route``         -- FULL door-to-door trip plan from two place names.
* ``model_info``         -- the trained model's accuracy envelope (read-only).

Security posture (MCP best practices)
-------------------------------------
* Transport is STDIO (``FastMCP.run`` default). A local stdio server is only
  reachable by the client process that spawned it -- there is no network port,
  no ``0.0.0.0`` bind, and no DNS-rebinding surface. Do NOT switch to an HTTP
  transport unless you also bind 127.0.0.1, require an auth token, and enable
  ``TransportSecuritySettings`` Host/Origin validation.
* SECRETS NEVER LEAVE THE SERVER. The ONLY credential this server uses is
  ``TOMTOM_API_KEY`` (geocode + routing) -- resolved from the environment by the
  tool layer, or supplied per-request by the caller (BYO key / OAuth). It does
  NOT use ``MINIMAX_API_KEY``: that powers the dashboard's in-process chat agent
  only -- over MCP the *connecting* LLM is the agent, so no model key is needed.
  No tool returns, echoes or logs a key. Tool errors are mapped to short,
  generic, secret-free categories (raw exception text -- which can embed a
  filesystem path or an httpx URL-with-key -- is never returned); the full
  detail is logged to STDERR only (stdout is the MCP protocol channel).
* LEAST PRIVILEGE. The server exposes only energy/range/route/spec tools. It
  has no file-read, shell, or arbitrary-fetch tool. ``model_path`` is NOT a
  tool parameter, so a client cannot point the model loader at an arbitrary
  file -- every tool uses the pinned ``DEFAULT_MODEL_PATH``.
* BOUNDED EGRESS. The only outbound call (made by ``plan_route``) is HTTPS to
  fixed TomTom hosts (api.tomtom.com geocode + routing) with a 12 s timeout,
  ``follow_redirects=False`` and a country/limit cap. Origin/destination
  strings are URL-encoded into the path, never used to build a host.
* INPUT VALIDATION. Numeric arguments are range-checked at this boundary before
  reaching the model, so an out-of-domain value returns a clean ``{"error":...}``
  instead of a garbage prediction.

Running the server
------------------
The server speaks MCP over stdio (the default transport for ``FastMCP.run``)::

    python -m nexdash.mcp_server

Registering in an MCP client config
-----------------------------------
Add an entry to your client's MCP server configuration (for many desktop
clients this lives in ``claude_desktop_config.json``)::

    {
      "mcpServers": {
        "nexdash": {
          "command": "python",
          "args": ["-m", "nexdash.mcp_server"],
          "env": {
            "PYTHONPATH": "/absolute/path/to/NexDash/src",
            "TOMTOM_API_KEY": "<your-tomtom-key>"
          }
        }
      }
    }

If the ``nexdash`` package is installed into the active environment (e.g. via
``pip install -e .``) the ``PYTHONPATH`` block can be omitted and ``command``
can point at that environment's interpreter. The trained model must exist on
disk first; run ``python run_pipeline.py`` once to generate it. ``plan_route``
additionally needs ``TOMTOM_API_KEY`` (or ``VITE_TOMTOM_API_KEY`` in
``frontend/.env``) to reach the TomTom geocode/routing API.

Worked standalone example
-------------------------
Client prompt: "travelling with 18 t on 22 March 2026 at 09:00 from Berlin to
Munich" -> the client calls::

    plan_route(origin="Berlin", destination="Munich", payload_t=18,
               departure="2026-03-22T09:00")

and receives a JSON plan with ``distance_km``, ``energy_kwh``, ``charging_stops``,
``eta``, ``on_time`` and ``eu561_ok``.
"""

from __future__ import annotations

import logging
import os
import sys

from mcp.server.fastmcp import Context, FastMCP

from nexdash import tomtom
from nexdash import tools


def _tomtom_key_for_request(ctx: "Context | None") -> "str | None":
    """The caller's TomTom key for this request: from the verified OAuth token's
    claims (hosted OAuth mode) first, then an X-TomTom-Key / Bearer header."""
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token

        at = get_access_token()
        if at is not None and at.claims and at.claims.get("tomtom_key"):
            return at.claims["tomtom_key"]
    except Exception:  # noqa: BLE001 - no auth context (stdio / non-OAuth)
        pass
    return _tomtom_key_from_request(ctx)


def _tomtom_key_from_request(ctx: "Context | None") -> "str | None":
    """Pull a caller-supplied TomTom key from the HTTP request headers — either
    ``X-TomTom-Key`` or ``Authorization: Bearer <key>`` — so a remote user runs
    ``plan_route`` on THEIR key, never the host's. Returns ``None`` when there is
    no HTTP request (stdio transport) or no such header.
    """
    try:
        headers = ctx.request_context.request.headers  # Starlette Request (HTTP mode)
    except Exception:  # noqa: BLE001 - stdio has no HTTP request / headers
        return None
    try:
        key = headers.get("x-tomtom-key")
        if not key:
            auth = headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                key = auth[7:]
    except Exception:  # noqa: BLE001
        return None
    return key.strip() if key and key.strip() else None


def _require_api_key_asgi(inner):
    """Wrap the MCP Streamable-HTTP app so the WHOLE server is gated by the
    caller's API key: every HTTP request must carry ``X-TomTom-Key`` (or
    ``Authorization: Bearer <key>``) — connecting, listing tools and calling any
    tool all require it. The key is bound to the per-request context for the
    duration of the request so ``plan_route`` routes on it (never the host's).
    Non-HTTP scopes (the lifespan that starts the session manager) pass through.
    """
    import json as _json

    async def app(scope, receive, send):
        if scope.get("type") != "http":
            await inner(scope, receive, send)
            return
        raw = {k.lower(): v for k, v in (scope.get("headers") or [])}
        key = (raw.get(b"x-tomtom-key") or b"").decode().strip()
        if not key:
            auth = (raw.get(b"authorization") or b"").decode()
            if auth.lower().startswith("bearer "):
                key = auth[7:].strip()
        if not key:
            body = _json.dumps(
                {
                    "error": "api_key_required",
                    "detail": (
                        "This MCP server requires your TomTom API key. Add an "
                        "'X-TomTom-Key: <key>' header (or 'Authorization: Bearer <key>') "
                        "when you connect. Free key: https://developer.tomtom.com"
                    ),
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        token = tomtom.set_request_api_key(key)
        try:
            await inner(scope, receive, send)
        finally:
            tomtom.reset_request_api_key(token)

    return app

# Log to STDERR only. For a stdio MCP server, STDOUT is the JSON-RPC protocol
# channel -- any write to it corrupts message framing. Server-side diagnostics
# (including the full text of a caught exception) go here, never to the client.
logger = logging.getLogger("nexdash.mcp_server")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# OAuth mode (hosted): when MCP_OAUTH is set and a public URL is known, the
# server speaks OAuth 2.1 so non-technical users can connect via Claude Desktop's
# "Connect" button — a consent page collects their TomTom key. The SDK then gates
# every request behind the issued token. Otherwise it's a plain server (stdio, or
# HTTP guarded by the X-TomTom-Key gate in main()).
_PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL") or (
    "https://" + os.environ["RAILWAY_PUBLIC_DOMAIN"]
    if os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    else None
)
USE_OAUTH = bool(os.environ.get("MCP_OAUTH")) and bool(_PUBLIC_URL)

if USE_OAUTH:
    from nexdash.mcp_oauth import build_auth, register_consent_routes

    _provider, _auth_settings = build_auth(_PUBLIC_URL)
    mcp = FastMCP("nexdash", auth_server_provider=_provider, auth=_auth_settings)
    register_consent_routes(mcp, _provider)
else:
    mcp = FastMCP("nexdash")


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------
class _ToolInputError(ValueError):
    """A client-supplied argument is out of the tool's accepted domain."""


def _bounded(value: float, *, low: float, high: float, field: str) -> float:
    """Validate that ``low <= value <= high`` for a numeric tool argument.

    Raises :class:`_ToolInputError` with a short, secret-free message (it names
    only the field and the bound, never internal state) so an out-of-domain
    value becomes a clean structured error rather than a garbage prediction.
    """
    if value < low or value > high:
        raise _ToolInputError(
            f"{field} must be between {low} and {high} (got {value})."
        )
    return value


def _safe_error(exc: Exception) -> dict:
    """Map any exception to a short, generic, secret-free client error.

    The full exception (which may embed a filesystem path, an env-derived value
    or an httpx URL containing the API key) is logged to STDERR ONLY. The client
    receives a stable category string -- never a stack trace or raw message.
    """
    logger.warning("tool call failed: %s: %s", type(exc).__name__, exc)
    if isinstance(exc, _ToolInputError):
        # The bound message is self-authored above and contains no secret.
        return {"error": f"Invalid input: {exc}"}
    return {"error": "Tool call failed: invalid input or internal error."}


# ---------------------------------------------------------------------------
# Tool: predict_energy
# ---------------------------------------------------------------------------
@mcp.tool()
def predict_energy(
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    wind_mps: float = 0.0,
) -> dict:
    """Predict energy consumption (kWh) for one eActros 600 trip segment.

    Uses the trained NexDash energy model to estimate how many kilowatt-hours
    the truck will draw from its battery over a single segment of known length.
    For a door-to-door trip between two place names, use ``plan_route`` instead.

    Args:
        distance_km: Segment length in kilometres (1-2000).
        payload_t: Cargo payload in tonnes (0-22).
        speed_kph: Average travel speed in km/h (1-130).
        gradient_pct: Net road gradient in percent; negative is downhill
            (-30 to 30).
        temperature_c: Ambient temperature in degrees Celsius (-40 to 55).
        wind_mps: Headwind component in metres/second (default 0.0; -40 to 40).

    Returns:
        JSON-serializable dict with the predicted energy (kWh) and the echoed
        input features, or ``{"error": ...}`` for an out-of-range / failed call.
    """
    try:
        distance_km = _bounded(distance_km, low=0.1, high=2000.0, field="distance_km")
        payload_t = _bounded(payload_t, low=0.0, high=22.0, field="payload_t")
        speed_kph = _bounded(speed_kph, low=1.0, high=130.0, field="speed_kph")
        gradient_pct = _bounded(gradient_pct, low=-30.0, high=30.0, field="gradient_pct")
        temperature_c = _bounded(temperature_c, low=-40.0, high=55.0, field="temperature_c")
        wind_mps = _bounded(wind_mps, low=-40.0, high=40.0, field="wind_mps")
        return tools.predict_energy_tool(
            distance_km=distance_km,
            payload_t=payload_t,
            speed_kph=speed_kph,
            gradient_pct=gradient_pct,
            temperature_c=temperature_c,
            wind_mps=wind_mps,
        )
    except Exception as exc:  # noqa: BLE001 - MCP boundary: never crash the call.
        return _safe_error(exc)


# ---------------------------------------------------------------------------
# Tool: check_reachability
# ---------------------------------------------------------------------------
@mcp.tool()
def check_reachability(
    soc_pct: float,
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    wind_mps: float = 0.0,
    reserve_pct: float = 10.0,
) -> dict:
    """Decide whether an eActros 600 can reach a destination on current charge.

    Combines the energy prediction with the current state of charge and a safety
    reserve to determine reachability, then reports the energy margin and the
    estimated remaining state of charge / range on arrival. Use this for any
    'can it make it / will it reach' question over a single known-distance leg.

    Args:
        soc_pct: Current battery state of charge in percent (0-100).
        distance_km: Trip distance in kilometres (1-2000).
        payload_t: Cargo payload in tonnes (0-22).
        speed_kph: Average travel speed in km/h (1-130).
        gradient_pct: Net road gradient in percent; negative is downhill
            (-30 to 30).
        temperature_c: Ambient temperature in degrees Celsius (-40 to 55).
        wind_mps: Headwind component in metres/second (default 0.0; -40 to 40).
        reserve_pct: Battery reserve to keep untouched, in percent (0-100,
            default 10.0).

    Returns:
        JSON-serializable dict with ``reaches`` (bool), energy needed/available,
        usable energy after reserve, margin, remaining SOC/range estimates, and
        a confidence note referencing the model's error band -- or
        ``{"error": ...}`` for an out-of-range / failed call.
    """
    try:
        soc_pct = _bounded(soc_pct, low=0.0, high=100.0, field="soc_pct")
        distance_km = _bounded(distance_km, low=0.1, high=2000.0, field="distance_km")
        payload_t = _bounded(payload_t, low=0.0, high=22.0, field="payload_t")
        speed_kph = _bounded(speed_kph, low=1.0, high=130.0, field="speed_kph")
        gradient_pct = _bounded(gradient_pct, low=-30.0, high=30.0, field="gradient_pct")
        temperature_c = _bounded(temperature_c, low=-40.0, high=55.0, field="temperature_c")
        wind_mps = _bounded(wind_mps, low=-40.0, high=40.0, field="wind_mps")
        reserve_pct = _bounded(reserve_pct, low=0.0, high=100.0, field="reserve_pct")
        return tools.check_reach_tool(
            soc_pct=soc_pct,
            distance_km=distance_km,
            payload_t=payload_t,
            speed_kph=speed_kph,
            gradient_pct=gradient_pct,
            temperature_c=temperature_c,
            wind_mps=wind_mps,
            reserve_pct=reserve_pct,
        )
    except Exception as exc:  # noqa: BLE001 - MCP boundary: never crash the call.
        return _safe_error(exc)


# ---------------------------------------------------------------------------
# Tool: plan_route  (the standalone, no-frontend trip planner)
# ---------------------------------------------------------------------------
@mcp.tool()
def plan_route(
    origin: str,
    destination: str,
    payload_t: float = 0.0,
    start_soc: float = 100.0,
    temperature_c: float = 15.0,
    departure: str | None = None,
    deliver_by: str | None = None,
    min_soc: float = 15.0,
    reserve_pct: float = 10.0,
    max_charge_kw: float = 400.0,
    min_charger_kw: float = 150.0,
    max_detour_km: float = 30.0,
    ctx: Context | None = None,
) -> dict:
    """Plan a COMPLETE door-to-door trip for a Mercedes-Benz eActros 600 truck.

    This is the standalone trip planner: it works for any MCP client with NO
    NexDash frontend. Give it two place names and it (1) geocodes both via
    TomTom, (2) routes a 40 t / 5-axle artic eActros 600 over the real TomTom
    truck road network, (3) enriches that route with live per-segment wind,
    elevation/gradient and temperature from Open-Meteo and factors them into the
    energy estimate, (4) simulates state-of-charge drain, (5) inserts DC
    fast-charging stops where needed -- resolving each to the actual time-optimal
    CCS station nearby (operator name, power, live availability, opening hours,
    price), (6) computes ETA and -- if ``deliver_by`` is set -- whether arrival is
    on time, (7) checks EU 561 driver-hours (a >=45-min charge counts as the
    required break), and (8) reports the live traffic delay + ETA-relevant
    incidents (accidents / jams / closures) on the corridor.

    The plan is therefore optimised against real headwind/tailwind and the
    climbs/descents along the actual roads -- not just flat distance. Those live
    conditions are echoed back in the ``conditions`` field so the caller can see
    what was taken into account.

    Use this whenever a user describes a trip between places (e.g. "Berlin to
    Munich, 18 tonnes, depart 09:00, deliver by noon"). For a single isolated
    segment with a known distance, or a pure "will it reach" question, prefer
    ``predict_energy`` / ``check_reachability``.

    Example (the prompt "travelling with 18 t on 22 March 2026 at 09:00 from
    Berlin to Munich")::

        plan_route(origin="Berlin", destination="Munich", payload_t=18,
                   departure="2026-03-22T09:00")

    Args:
        origin: Start location name (e.g. "Berlin", "Hamburg Hafen"). Required.
        destination: Destination location name (e.g. "Munich"). Required.
        payload_t: Cargo payload in tonnes (0-22, clamped). Default 0.
        start_soc: Starting battery state of charge in percent (0-100, clamped).
            Default 100.
        temperature_c: Ambient temperature in degrees Celsius. Default 15.
        departure: Departure datetime as an ISO 8601 local string
            (e.g. "2026-03-22T09:00"). Drives ETA, EU 561 breaks and on-time
            checks. Defaults to now if omitted.
        deliver_by: Delivery deadline at the destination as an ISO 8601 local
            datetime (e.g. "2026-03-22T18:00"). When set, the plan reports
            whether arrival is on time / early / late via ``on_time``.
        min_soc: SOC floor (%) never to dip below (0-100). Default 15.
        reserve_pct: Safety-reserve buffer (%) above min SOC (0-100). Default 10.
        max_charge_kw: Max charging power (kW) the truck accepts. Default 400.
        min_charger_kw: Minimum charger power to consider (kW); slower chargers
            are skipped when resolving a real station. Default 150.
        max_detour_km: Max detour off the route to reach a charger (km) -- the
            charger search radius around each stop. Default 30.

    Returns:
        JSON-serializable plan summary: ``origin``/``destination`` (label+coords),
        ``distance_km``, ``energy_kwh``, ``kwh_per_100``, ``arrival_soc``,
        ``min_soc``, ``charging_stops`` (with arrive/depart SOC + kWh),
        ``n_charging_stops``, ``chargers_unresolved``, ``driving_time_h``,
        ``total_time_h``, ``departure``, ``eta``/``eta_iso``, ``deliver_by``,
        ``on_time``, ``eu561_ok``, ``conditions`` (the live Open-Meteo wind /
        elevation-gain+loss / temperature the trip was optimised against),
        ``traffic`` (live delay + ETA-relevant incidents on the route) and
        ``assumptions``. Each ``charging_stops`` entry carries a ``station``
        object naming the real CCS charger (operator, power, live availability,
        opening hours, price) or ``None``, plus ``charge_min`` and
        ``charge_power_kw`` -- the charge time re-computed at that real station's
        power -- and ``station_resolved`` (``False`` when no real charger was
        found nearby, so the stop is still timed at the truck-cap assumption;
        ``chargers_unresolved`` counts these). ``total_time_h``/``eta`` already
        include the real-charger time, so a slower station pushes the ETA later.
        On geocode/route/simulation failure it returns ``{"error": ...}``
        (secret-free) instead of throwing.
    """
    # Bring-your-own-key: use the caller's TomTom key (from the request header)
    # for this trip so the host's key is never spent. If none is supplied AND no
    # host/env key is configured, return a clear, actionable message instead of
    # routing — the other three tools need no key and keep working.
    user_key = _tomtom_key_for_request(ctx)
    key_token = None
    if user_key:
        key_token = tomtom.set_request_api_key(user_key)
    else:
        try:
            tomtom.get_api_key()  # raises if no per-request / env key is available
        except Exception:  # noqa: BLE001
            return {
                "error": "tomtom_key_required",
                "detail": (
                    "plan_route needs a TomTom API key. Supply your own by adding an "
                    "'X-TomTom-Key: <key>' header (or 'Authorization: Bearer <key>') "
                    "when you connect this MCP server, so the trip routes on YOUR key. "
                    "Get a free key at https://developer.tomtom.com. The other tools "
                    "(predict_energy, check_reachability, model_info) need no key."
                ),
            }
    try:
        if not isinstance(origin, str) or not origin.strip():
            raise _ToolInputError("origin is required.")
        if not isinstance(destination, str) or not destination.strip():
            raise _ToolInputError("destination is required.")
        # plan_route_tool already clamps payload_t/start_soc/min_soc/reserve_pct
        # and tolerates string/None numerics. We forward typed values straight
        # through; do NOT expose model_path here (least privilege).
        return tools.plan_route_tool(
            origin=origin,
            destination=destination,
            payload_t=payload_t,
            start_soc=start_soc,
            temperature_c=temperature_c,
            departure=departure,
            deliver_by=deliver_by,
            min_soc=min_soc,
            reserve_pct=reserve_pct,
            max_charge_kw=max_charge_kw,
            min_charger_kw=min_charger_kw,
            max_detour_km=max_detour_km,
        )
    except Exception as exc:  # noqa: BLE001 - MCP boundary: never crash the call.
        return _safe_error(exc)
    finally:
        if key_token is not None:
            tomtom.reset_request_api_key(key_token)


# ---------------------------------------------------------------------------
# Tool: model_info  (read-only trust/calibration signal -- no secrets, no net)
# ---------------------------------------------------------------------------
@mcp.tool()
def model_info() -> dict:
    """Return the trained energy model's accuracy metrics (no inputs).

    Lets a client gauge prediction confidence BEFORE acting on a
    ``predict_energy`` / ``check_reachability`` / ``plan_route`` result. Reads
    only local model metadata -- it makes no external call and exposes no key.

    Returns:
        JSON-serializable dict ``{mae_kwh, rmse_kwh, mape_pct, r2,
        pct_range_error, model_version}`` (values are ``None`` where unknown),
        or ``{"error": ...}`` if the metrics cannot be resolved.
    """
    try:
        from nexdash.model_info import model_info as _model_info

        info = dict(_model_info())
        info["truck_model"] = "Mercedes-Benz eActros 600"
        return info
    except Exception as exc:  # noqa: BLE001 - MCP boundary: never crash the call.
        return _safe_error(exc)


def main() -> None:
    """Run the server.

    * **stdio** (default) — local only, reachable just by the client that spawns
      it; no network port is opened.
    * **Streamable HTTP** — set ``MCP_HTTP=1`` (or ``MCP_TRANSPORT=http``). Binds
      ``0.0.0.0:$PORT`` and serves the MCP endpoint at ``/mcp`` so REMOTE clients
      (Claude custom connectors, remote agents) can connect over the public
      internet. Runs stateless so it deploys cleanly as a normal web service.
    """
    import os

    want_http = bool(os.environ.get("MCP_HTTP")) or os.environ.get(
        "MCP_TRANSPORT", ""
    ).lower() in ("http", "streamable-http")

    if want_http:
        from mcp.server.transport_security import TransportSecuritySettings

        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("PORT") or os.environ.get("MCP_PORT") or "8000")
        mcp.settings.stateless_http = True
        # DNS-rebinding protection defaults to a localhost-only Host allowlist,
        # which 421s a PUBLIC server reached via its real domain (Railway, etc.).
        # This server is meant to be reached from anywhere (incl. Anthropic's
        # cloud for Claude custom connectors), so lift the host check.
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )
        if USE_OAUTH:
            # OAuth mode: the SDK auth layer already gates every request behind a
            # valid token (and serves the discovery/registration/authorize/token
            # endpoints + the consent page), so run it directly.
            mcp.run(transport="streamable-http")
        else:
            # No OAuth: gate the ENTIRE server behind the caller's key header —
            # every request must carry X-TomTom-Key / Bearer, used by plan_route.
            import uvicorn

            uvicorn.run(
                _require_api_key_asgi(mcp.streamable_http_app()),
                host=mcp.settings.host,
                port=mcp.settings.port,
            )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
