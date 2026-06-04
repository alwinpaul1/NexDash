"""Geospatial enrichment data layer for route planning.

Pure data layer built on the Python standard library only (``urllib.request``
and ``json``) -- no third-party HTTP dependencies. It turns a road polyline
into per-segment physical conditions (elevation gradient, temperature, wind)
that the energy model can consume.

Data sources (free, no API key):
  * Elevation -- Open-Meteo Elevation API
    https://api.open-meteo.com/v1/elevation
  * Weather (temperature / wind) -- Open-Meteo Forecast API
    https://api.open-meteo.com/v1/forecast

Design rules honoured here:
  * Everything fails *soft*. A network outage, malformed response, or bad
    input never raises to the caller -- we degrade to sane defaults
    (elevation 0.0 m, gradient 0 %, temperature 15 C, wind 3 m/s).
  * Elevation lookups are cached in-process keyed on rounded coordinates so
    repeated planning of the same corridor does not re-hit the network.
  * All returned values are plain floats / ints -- JSON-serialisable.

Public API:
  * ``elevations(points) -> list[float]``
  * ``sample_polyline(geometry, max_points=80) -> list[(lat, lon)]``
  * ``enrich_route(geometry, departure_iso=None) -> dict``
"""

from __future__ import annotations

import json
import math
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Fail-soft defaults
# ---------------------------------------------------------------------------
DEFAULT_ELEV_M = 0.0
DEFAULT_TEMP_C = 15.0
DEFAULT_WIND_MPS = 3.0
DEFAULT_WIND_DIR_DEG = 0.0

# Open-Meteo accepts up to ~100 coordinates per request; stay a little under.
_ELEV_BATCH = 90
# Forecast is comparatively expensive, so sample weather at fewer points and
# interpolate the rest along the route. NOTE (honest limitation): a long route
# gets temperature/wind from only this many points (each segment reuses the
# nearest), so on a 600 km route weather samples are ~100 km apart and can miss a
# frontal passage or a mountain-pass wind shift — see `conditions.weatherSamples`.
_WEATHER_SAMPLES = 6
_HTTP_TIMEOUT_S = 8

# Bounded retry/backoff for transient failures (429 / 5xx / connection / timeout).
# A single dropped packet otherwise collapses an entire elevation batch to 0.0 m,
# silently flattening terrain for the energy model — worse than a brief stall.
_HTTP_RETRIES = 2  # extra attempts after the first (so up to 3 total)
_HTTP_BACKOFF_S = 0.5  # base backoff, doubled each retry
# 429 (Too Many Requests) is deliberately NOT retried: a rate-limit persists for
# the whole window, so retrying with sub-second backoff just stalls the request
# (and the request count keeps the limit tripped). Fail FAST to defaults instead
# — the conditions.weatherDegraded / elevationDegraded flags then tell the UI.
# Genuinely transient 5xx / connection / timeout errors are still retried.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

# Dynamic rate-limit handling — NO fixed cooldown timer. A route plan makes ~6
# Open-Meteo calls (one elevation batch + several weather samples); when the
# provider is rate-limited or down, retrying each of them stacks up to a 60s+
# stall. So the FIRST hard failure (a 429, or an exhausted 5xx/timeout) opens a
# PER-REQUEST circuit: the remaining calls in that plan bail instantly and it
# degrades to seasonal/flat defaults (the UI shows the degraded flags).
# ``enrich_route`` RE-ARMS the circuit at the start of every plan, so the next
# request re-probes the provider and self-heals the moment the limit clears — no
# hardcoded wait. If a 429 carries a ``Retry-After``, we honour exactly that
# (the server's own dynamic value), waiting it out across requests.
_circuit_open = False  # per-request: set on a hard failure, re-armed by enrich_route()
_retry_after_until = 0.0  # monotonic deadline from a 429 Retry-After header (if any)


def _parse_retry_after(exc) -> "float | None":
    """Seconds to wait from a 429's ``Retry-After`` header, or ``None``.

    Supports the numeric (delta-seconds) form; capped at 5 min so a bogus value
    can't wedge enrichment. The HTTP-date form is ignored (degrades to the
    per-request circuit, which re-probes next plan anyway).
    """
    try:
        val = (exc.headers.get("Retry-After") or "").strip()
    except Exception:  # noqa: BLE001 - header access must never raise here
        return None
    if val.isdigit():
        return min(float(val), 300.0)
    return None

_ELEV_API = "https://api.open-meteo.com/v1/elevation"
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
# Recent-past departures (older than the forecast window below) are most
# accurately served by the Historical Forecast API, which mirrors the forecast
# response shape. Avoids silently quoting today's weather for a past trip.
_HIST_FORECAST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# The forecast endpoint only accepts roughly [today-92d, today+15d]; outside that
# it returns {"error": true, "reason": "...out of allowed range..."}.
_FORECAST_PAST_LIMIT_DAYS = 92
_FORECAST_FUTURE_LIMIT_DAYS = 15

# In-process caches. Keyed on rounded coords to fold near-identical lookups.
_elev_cache: dict[tuple[float, float], float] = {}
_weather_cache: dict[tuple[float, float, str], tuple[float, float, float, str]] = {}


# ---------------------------------------------------------------------------
# Low-level HTTP helper (stdlib only, always fail-soft)
# ---------------------------------------------------------------------------
def _get_json(url: str):
    """GET ``url`` and parse JSON, with bounded retry. Returns ``None`` on failure.

    Retries only *transient* conditions — HTTP 429/5xx and connection/timeout
    errors — with exponential backoff; a permanent 4xx or a JSON-decode error is
    not retried (it would never succeed). Always fail-soft: after the retries are
    exhausted (or on a non-retryable error) it returns ``None`` and the caller
    degrades to defaults, so the planner still runs offline.
    """
    global _circuit_open, _retry_after_until
    # Circuit OPEN (this plan already saw a hard failure) or a server-issued
    # Retry-After is still in effect: skip the network and degrade instantly —
    # no per-call timeout to absorb.
    if _circuit_open or (_retry_after_until and time.monotonic() < _retry_after_until):
        return None

    for attempt in range(_HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexDash/1.0"})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRYABLE_STATUS and attempt < _HTTP_RETRIES:
                time.sleep(_HTTP_BACKOFF_S * (2 ** attempt))
                continue
            # No more retries. A 429 (rate-limit) or an exhausted 5xx means the
            # provider is overloaded -> open the per-request circuit so sibling
            # calls bail fast. A 429 may also tell us EXACTLY how long to wait
            # (Retry-After) — honour that across requests. A permanent 4xx
            # (e.g. 400 out-of-range) does NOT trip the circuit.
            if exc.code == 429 or exc.code in _RETRYABLE_STATUS:
                _circuit_open = True
                if exc.code == 429:
                    ra = _parse_retry_after(exc)
                    if ra is not None:
                        _retry_after_until = time.monotonic() + ra
            return None
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            # Transient connection / timeout: back off and retry.
            if attempt < _HTTP_RETRIES:
                time.sleep(_HTTP_BACKOFF_S * (2 ** attempt))
                continue
            # Sustained outage -> open the per-request circuit.
            _circuit_open = True
            return None
        except Exception:
            # Bad JSON or anything unexpected: don't retry, degrade. The provider
            # is reachable, so don't trip the circuit.
            return None
    return None


def _key(lat: float, lon: float) -> tuple[float, float]:
    """Cache key: round to ~11 m grid so near-identical points share a result."""
    return (round(float(lat), 4), round(float(lon), 4))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in kilometres."""
    r = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial compass bearing (deg, 0=N, clockwise) for travel from a to b."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _coerce_points(geometry) -> list[tuple[float, float]]:
    """Normalise a [[lat, lng], ...] polyline into clean (lat, lon) tuples.

    Silently drops malformed entries; returns [] if nothing is usable.
    """
    out: list[tuple[float, float]] = []
    if not geometry:
        return out
    try:
        for pt in geometry:
            try:
                lat = float(pt[0])
                lon = float(pt[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                out.append((lat, lon))
    except TypeError:
        return []
    return out


# ---------------------------------------------------------------------------
# Public: elevations
# ---------------------------------------------------------------------------
def elevations(points) -> list[float]:
    """Return ground elevation (m) for each ``(lat, lon)`` in ``points``.

    Batched against Open-Meteo (<=90 coords/request), cached in-process, and
    fail-soft: any point we cannot resolve becomes ``0.0`` (DEFAULT_ELEV_M).
    Output length always matches the cleaned input length.
    """
    return _fetch_elevations(points)[0]


def _fetch_elevations(points) -> tuple[list[float], bool]:
    """Like :func:`elevations` but also reports whether every fetch succeeded.

    The ``ok`` flag is ``False`` if any batch we had to fetch over the network
    returned no usable elevation list, so :func:`enrich_route` can flag that the
    terrain is a degraded default (0.0 m) rather than genuine sea level.
    """
    pts = _coerce_points(points)
    if not pts:
        return [], True

    result: list[float | None] = [None] * len(pts)
    missing_idx: list[int] = []
    ok = True

    # Serve from cache first.
    for i, p in enumerate(pts):
        cached = _elev_cache.get(_key(*p))
        if cached is not None:
            result[i] = cached
        else:
            missing_idx.append(i)

    # Fetch the remainder in batches.
    for start in range(0, len(missing_idx), _ELEV_BATCH):
        batch_idx = missing_idx[start:start + _ELEV_BATCH]
        lats = ",".join(f"{pts[i][0]:.5f}" for i in batch_idx)
        lons = ",".join(f"{pts[i][1]:.5f}" for i in batch_idx)
        url = f"{_ELEV_API}?latitude={lats}&longitude={lons}"
        data = _get_json(url)
        elev_list = data.get("elevation") if isinstance(data, dict) else None
        if not isinstance(elev_list, list):
            ok = False  # this batch fell back to defaults

        for j, i in enumerate(batch_idx):
            val = DEFAULT_ELEV_M
            if isinstance(elev_list, list) and j < len(elev_list):
                try:
                    v = float(elev_list[j])
                    if math.isfinite(v):
                        val = v
                except (TypeError, ValueError):
                    val = DEFAULT_ELEV_M
            result[i] = val
            # Only cache real responses, not the failure default, so a later
            # call can retry when the network recovers.
            if isinstance(elev_list, list) and j < len(elev_list):
                _elev_cache[_key(*pts[i])] = val

    return [DEFAULT_ELEV_M if v is None else float(v) for v in result], ok


# ---------------------------------------------------------------------------
# Public: sample_polyline
# ---------------------------------------------------------------------------
def sample_polyline(geometry, max_points: int = 80) -> list[tuple[float, float]]:
    """Downsample a dense polyline to <= ``max_points`` points by arc-length.

    Picks points at roughly equal cumulative-distance intervals so the sample
    is geometrically representative regardless of how the source vertices are
    spaced. Always keeps the first and last point. Returns (lat, lon) tuples.
    """
    pts = _coerce_points(geometry)
    if max_points < 2:
        max_points = 2
    if len(pts) <= max_points:
        return pts

    # Cumulative arc-length at each vertex.
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + _haversine_km(pts[i - 1], pts[i]))
    total = cum[-1]

    if total <= 0:  # Degenerate (all points coincide).
        return [pts[0], pts[-1]]

    # Target arc-length positions, evenly spaced including both ends.
    step = total / (max_points - 1)
    sampled: list[tuple[float, float]] = []
    seen: set[int] = set()
    j = 0
    for k in range(max_points):
        target = k * step
        while j < len(cum) - 1 and cum[j] < target:
            j += 1
        if j not in seen:
            sampled.append(pts[j])
            seen.add(j)

    # Guarantee the endpoint is present.
    if sampled[-1] != pts[-1]:
        sampled.append(pts[-1])
    return sampled


# ---------------------------------------------------------------------------
# Weather (temperature / wind) at a point for a departure hour
# ---------------------------------------------------------------------------
def _parse_departure(departure_iso):
    """Return a timezone-aware UTC datetime for the departure, or now (UTC)."""
    if departure_iso:
        try:
            s = str(departure_iso).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _weather_at(lat: float, lon: float, dep: datetime):
    """Return ``(temperatureC, windMps, windDirDeg, source)`` for a point & hour.

    Picks the right Open-Meteo endpoint for the departure date — the standard
    forecast API within roughly ``[today-92d, today+15d]``, the Historical
    Forecast API for older trips — and reads the hourly value nearest ``dep``.

    Crucially it does **not** quietly pass off today's live ``current=`` weather
    as the departure-time weather: that ``current=`` fallback is used only when
    the departure is genuinely "now-ish" (within a day). For an out-of-window
    date the forecast API returns ``{"error": true}`` and we degrade to documented
    defaults with ``source="default"`` so the caller can flag the data as
    unavailable rather than silently wrong. ``source`` is one of
    ``forecast | historical | current | default``. Cached per (lat, lon, hour).
    """
    date_str = dep.strftime("%Y-%m-%d")
    ckey = (round(lat, 3), round(lon, 3), dep.strftime("%Y-%m-%dT%H"))
    cached = _weather_cache.get(ckey)
    if cached is not None:
        return cached

    common = (
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        "&wind_speed_unit=ms&temperature_unit=celsius&timezone=UTC"
    )

    # Choose the endpoint by how far the departure is from today.
    days_off = (dep.date() - datetime.now(timezone.utc).date()).days
    if days_off < -_FORECAST_PAST_LIMIT_DAYS:
        api, source = _HIST_FORECAST_API, "historical"
    else:
        api, source = _FORECAST_API, "forecast"

    url = (
        f"{api}?{common}"
        "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m"
        f"&start_date={date_str}&end_date={date_str}"
    )
    data = _get_json(url)
    # Out-of-window dates come back as {"error": true, "reason": "..."}; treat
    # that as "no usable hourly data", NOT as a reason to substitute live weather.
    api_error = bool(isinstance(data, dict) and data.get("error"))
    hourly = data.get("hourly") if (isinstance(data, dict) and not api_error) else None

    temp = wind = wdir = None
    if isinstance(hourly, dict):
        times = hourly.get("time") or []
        target = dep.strftime("%Y-%m-%dT%H:00")
        idx = None
        if isinstance(times, list) and times:
            if target in times:
                idx = times.index(target)
            else:
                prefix = dep.strftime("%Y-%m-%dT%H")
                for n, t in enumerate(times):
                    if isinstance(t, str) and t.startswith(prefix):
                        idx = n
                        break
                if idx is None:
                    idx = min(dep.hour, len(times) - 1)
        if idx is not None:
            temp = _pick(hourly.get("temperature_2m"), idx)
            wind = _pick(hourly.get("wind_speed_10m"), idx)
            wdir = _pick(hourly.get("wind_direction_10m"), idx)

    # Live ``current=`` is a legitimate proxy ONLY for a near-now departure.
    if (temp is None or wind is None) and abs(days_off) <= 1:
        cur = _get_json(
            f"{_FORECAST_API}?{common}"
            "&current=temperature_2m,wind_speed_10m,wind_direction_10m"
        )
        block = cur.get("current") if isinstance(cur, dict) else None
        if isinstance(block, dict):
            c_temp = _num(block.get("temperature_2m"))
            c_wind = _num(block.get("wind_speed_10m"))
            c_wdir = _num(block.get("wind_direction_10m"))
            if temp is None:
                temp = c_temp
            if wind is None:
                wind = c_wind
            if wdir is None:
                wdir = c_wdir
            if c_temp is not None and c_wind is not None:
                source = "current"

    if temp is None or wind is None:
        source = "default"

    out = (
        temp if temp is not None else DEFAULT_TEMP_C,
        wind if wind is not None else DEFAULT_WIND_MPS,
        wdir if wdir is not None else DEFAULT_WIND_DIR_DEG,
        source,
    )
    # Cache only genuine readings so a transient failure can retry later.
    if source != "default":
        _weather_cache[ckey] = out
    return out


def _pick(seq, idx):
    if isinstance(seq, list) and 0 <= idx < len(seq):
        return _num(seq[idx])
    return None


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public: enrich_route
# ---------------------------------------------------------------------------
def enrich_route(geometry, departure_iso=None, leg_timings=None) -> dict:
    """Enrich a road polyline with per-segment physical conditions.

    When ``leg_timings`` (the routing engine's per-leg ``{lengthM, travelTimeS}``)
    is supplied, each segment is additionally stamped with a ``measuredSpeedKph``
    — the REAL traffic-aware speed of the leg covering that segment (mapped by
    cumulative-distance fraction, so it is robust to polyline downsampling). When
    absent, segments carry no speed field and the planner uses its heuristic.

    Returns a dict (all values JSON-serialisable, numeric):

        {
          "segments": [
            {distKm, cumKm, gradientPct, elevM, temperatureC, windMps}, ...
          ],
          "elevationProfile": [{distKm, elevM}, ...],
          "conditions": {avgTempC, avgWindMps, windDirDeg,
                         maxGradientPct, climbM, descentM}
        }

    One segment is produced per span between consecutive sampled points.
    ``gradientPct = 100 * deltaElevM / (spanKm * 1000)``.

    Weather is sampled at a handful of points along the route and the nearest
    sample is reused for each segment, keeping forecast calls bounded.

    Never raises: on any failure (empty/degenerate geometry, network down) it
    returns a coherent default-filled structure instead.
    """
    # Re-arm the per-request circuit so this plan re-probes the provider; if it's
    # recovered, live data flows again with no fixed wait. A still-pending
    # Retry-After (server-issued) is left intact and respected.
    global _circuit_open
    _circuit_open = False

    empty = {
        "segments": [],
        "elevationProfile": [],
        "conditions": {
            "avgTempC": DEFAULT_TEMP_C,
            "avgWindMps": DEFAULT_WIND_MPS,
            "windDirDeg": DEFAULT_WIND_DIR_DEG,
            "maxGradientPct": 0.0,
            "climbM": 0.0,
            "descentM": 0.0,
            # Honesty flags: no real data here, so the values above are defaults.
            "weatherSource": "default",
            "weatherDegraded": True,
            "elevationDegraded": True,
            "weatherSamples": 0,
        },
    }

    try:
        sampled = sample_polyline(geometry, max_points=80)
        if len(sampled) < 2:
            return empty

        dep = _parse_departure(departure_iso)
        elev, elev_ok = _fetch_elevations(sampled)
        if len(elev) != len(sampled):
            elev = (elev + [DEFAULT_ELEV_M] * len(sampled))[:len(sampled)]

        # Pick a small set of weather sample indices spread across the route.
        n = len(sampled)
        if n <= _WEATHER_SAMPLES:
            w_idx = list(range(n))
        else:
            w_idx = sorted({round(i * (n - 1) / (_WEATHER_SAMPLES - 1)) for i in range(_WEATHER_SAMPLES)})
        weather = {i: _weather_at(sampled[i][0], sampled[i][1], dep) for i in w_idx}

        def nearest_weather(i: int):
            j = min(w_idx, key=lambda k: abs(k - i))
            return weather[j]

        # Provenance of the weather: did real readings come back, and from where?
        weather_sources = [v[3] for v in weather.values()]
        weather_degraded = any(src == "default" for src in weather_sources)
        real_sources = [s for s in weather_sources if s != "default"]
        weather_source = max(set(real_sources), key=real_sources.count) if real_sources else "default"

        segments = []
        elevation_profile = [{"distKm": 0.0, "elevM": round(float(elev[0]), 1)}]
        cum_km = 0.0
        climb = 0.0
        descent = 0.0
        max_grad = 0.0
        temp_acc = 0.0
        wind_acc = 0.0
        dist_acc = 0.0
        # windDir taken from the first sampled point (representative heading wind).
        wind_dir = weather[w_idx[0]][2]

        for i in range(1, n):
            span_km = _haversine_km(sampled[i - 1], sampled[i])
            if span_km <= 0:
                continue
            cum_km += span_km
            d_elev = float(elev[i]) - float(elev[i - 1])
            grad = 100.0 * d_elev / (span_km * 1000.0)
            # Clamp absurd gradients from elevation noise on tiny spans.
            if grad > 30:
                grad = 30.0
            elif grad < -30:
                grad = -30.0

            if d_elev > 0:
                climb += d_elev
            else:
                descent += -d_elev
            if abs(grad) > abs(max_grad):
                max_grad = grad

            temp_i, wind_speed_i, wind_dir_i, _src_i = nearest_weather(i)
            # Signed headwind for the MODEL. Open-Meteo wind_direction_10m is the
            # WMO convention (compass bearing the wind blows FROM). Project it onto
            # the segment's travel bearing so a head-on wind is a full +headwind
            # and a following wind is a -tailwind:
            #   headwind = speed * cos(fromDir - travelBearing)
            travel_bearing = _bearing_deg(sampled[i - 1], sampled[i])
            headwind_i = wind_speed_i * math.cos(math.radians(wind_dir_i - travel_bearing))
            temp_acc += temp_i * span_km
            wind_acc += wind_speed_i * span_km  # magnitude, for the display tile
            dist_acc += span_km

            segments.append({
                "distKm": round(span_km, 3),
                "cumKm": round(cum_km, 3),
                "gradientPct": round(grad, 3),
                "elevM": round(float(elev[i]), 1),
                "temperatureC": round(temp_i, 1),
                "windMps": round(headwind_i, 2),  # signed headwind fed to the model
            })
            elevation_profile.append({"distKm": round(cum_km, 3), "elevM": round(float(elev[i]), 1)})

        if not segments:
            return empty

        # Stamp REAL per-segment speed from the routing engine's per-leg travel time
        # when supplied. Map each segment to its leg by cumulative-distance FRACTION
        # (robust to the great-circle-vs-road scale gap and to downsampling): a leg's
        # measured speed = lengthM/1000 / (travelTimeS/3600). Purely additive — when
        # leg_timings is absent, segments carry no speed field.
        if leg_timings:
            total_road_m = sum(max(0.0, float(t.get("lengthM", 0) or 0)) for t in leg_timings)
            final_cum = float(segments[-1]["cumKm"]) or 0.0
            if total_road_m > 0 and final_cum > 0:
                bounds = []  # (cumulative_end_fraction, measured_kph_or_None)
                acc_m = 0.0
                for t in leg_timings:
                    length_m = max(0.0, float(t.get("lengthM", 0) or 0))
                    travel_s = float(t.get("travelTimeS", 0) or 0)
                    acc_m += length_m
                    spd = (length_m / 1000.0) / (travel_s / 3600.0) if travel_s > 0 else None
                    bounds.append((acc_m / total_road_m, spd))
                for seg in segments:
                    frac = float(seg["cumKm"]) / final_cum
                    spd = next((s for f, s in bounds if frac <= f + 1e-9), bounds[-1][1])
                    if spd and spd > 0:
                        seg["measuredSpeedKph"] = round(spd, 2)

        avg_temp = temp_acc / dist_acc if dist_acc > 0 else DEFAULT_TEMP_C
        avg_wind = wind_acc / dist_acc if dist_acc > 0 else DEFAULT_WIND_MPS

        return {
            "segments": segments,
            "elevationProfile": elevation_profile,
            "conditions": {
                "avgTempC": round(avg_temp, 1),
                "avgWindMps": round(avg_wind, 2),
                "windDirDeg": round(float(wind_dir), 1),
                "maxGradientPct": round(max_grad, 2),
                "climbM": round(climb, 1),
                "descentM": round(descent, 1),
                # Honesty flags so the planner/UI can tell real data from a
                # fail-soft default and disclose the weather sampling coarseness.
                "weatherSource": weather_source,
                "weatherDegraded": bool(weather_degraded),
                "elevationDegraded": not bool(elev_ok),
                "weatherSamples": len(w_idx),
            },
        }
    except Exception:
        # Absolute last-resort guard -- the contract says we never raise.
        return empty
