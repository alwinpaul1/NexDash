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
# interpolate the rest along the route.
_WEATHER_SAMPLES = 6
_HTTP_TIMEOUT_S = 8

_ELEV_API = "https://api.open-meteo.com/v1/elevation"
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# In-process caches. Keyed on rounded coords to fold near-identical lookups.
_elev_cache: dict[tuple[float, float], float] = {}
_weather_cache: dict[tuple[float, float, str], tuple[float, float, float]] = {}


# ---------------------------------------------------------------------------
# Low-level HTTP helper (stdlib only, always fail-soft)
# ---------------------------------------------------------------------------
def _get_json(url: str):
    """GET ``url`` and parse JSON. Returns ``None`` on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NexDash/1.0"})
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        # Network down, timeout, HTTP error, bad JSON -- caller degrades.
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
    pts = _coerce_points(points)
    if not pts:
        return []

    result: list[float | None] = [None] * len(pts)
    missing_idx: list[int] = []

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

    return [DEFAULT_ELEV_M if v is None else float(v) for v in result]


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
    """Return (temperatureC, windMps, windDirDeg) for a point & departure hour.

    Tries the forecast hourly series at the nearest hour to ``dep``; falls back
    to ``current=`` if the date is out of forecast range; finally to defaults.
    Cached per (lat, lon, date-hour).
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
    url = (
        f"{_FORECAST_API}?{common}"
        "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m"
        f"&start_date={date_str}&end_date={date_str}"
    )
    data = _get_json(url)

    temp = wind = wdir = None
    hourly = data.get("hourly") if isinstance(data, dict) else None
    if isinstance(hourly, dict):
        times = hourly.get("time") or []
        target = dep.strftime("%Y-%m-%dT%H:00")
        idx = None
        if isinstance(times, list) and times:
            if target in times:
                idx = times.index(target)
            else:
                # Nearest by hour prefix.
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

    # Fallback: current conditions (e.g. departure date outside forecast window).
    if temp is None or wind is None:
        url_cur = (
            f"{_FORECAST_API}?{common}"
            "&current=temperature_2m,wind_speed_10m,wind_direction_10m"
        )
        cur = _get_json(url_cur)
        block = cur.get("current") if isinstance(cur, dict) else None
        if isinstance(block, dict):
            temp = temp if temp is not None else _num(block.get("temperature_2m"))
            wind = wind if wind is not None else _num(block.get("wind_speed_10m"))
            wdir = wdir if wdir is not None else _num(block.get("wind_direction_10m"))

    out = (
        temp if temp is not None else DEFAULT_TEMP_C,
        wind if wind is not None else DEFAULT_WIND_MPS,
        wdir if wdir is not None else DEFAULT_WIND_DIR_DEG,
    )
    # Cache only genuine network successes so failures can retry later.
    if isinstance(data, dict) or (temp is not None and wind is not None):
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
def enrich_route(geometry, departure_iso=None) -> dict:
    """Enrich a road polyline with per-segment physical conditions.

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
        },
    }

    try:
        sampled = sample_polyline(geometry, max_points=80)
        if len(sampled) < 2:
            return empty

        dep = _parse_departure(departure_iso)
        elev = elevations(sampled)
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

            temp_i, wind_speed_i, wind_dir_i = nearest_weather(i)
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
            },
        }
    except Exception:
        # Absolute last-resort guard -- the contract says we never raise.
        return empty
