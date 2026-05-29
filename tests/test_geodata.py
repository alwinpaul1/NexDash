"""Tests for :mod:`nexdash.geodata`, the route-enrichment data layer.

These verify the *intent* of the data layer, not just its mechanics:

* The layer is contractually **fail-soft**: a network outage (the HTTP call
  raising) must never propagate to the caller. ``enrich_route`` must still
  return a coherent, fully-defaulted structure so the planner keeps working
  offline. We assert this by monkeypatching the single stdlib HTTP entry
  point (:func:`nexdash.geodata._get_json`) to raise.
* ``sample_polyline`` exists to bound the number of expensive Open-Meteo
  lookups; it must actually *downsample* a dense polyline (and keep the
  endpoints) — otherwise a long route would blow the API's per-request coord
  budget.
* ``enrich_route`` must surface the real physical context the energy model
  consumes: per-segment ``gradientPct`` / ``temperatureC`` / ``windMps``, an
  ``elevationProfile``, and aggregate ``conditions``. A steep elevation series
  must yield a non-zero gradient — a flat one must not — because the whole
  point of the layer is to make gradient *vary* with the terrain.

No network is ever touched: every HTTP call is intercepted at ``_get_json``.
"""

from __future__ import annotations

import math

import pytest

from nexdash import geodata


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset the in-process caches so each test sees its own stubbed HTTP."""
    geodata._elev_cache.clear()
    geodata._weather_cache.clear()
    yield
    geodata._elev_cache.clear()
    geodata._weather_cache.clear()


# A short west->east polyline across a few densely-spaced vertices.
DENSE_GEOMETRY = [
    [52.0 + i * 0.01, 13.0 + i * 0.01] for i in range(40)
]


def _install_fake_http(monkeypatch, *, elevations, temp=10.0, wind=5.0, wdir=180.0):
    """Patch ``geodata._get_json`` to answer Open-Meteo calls deterministically.

    ``elevations`` is a callable ``(lat, lon) -> metres`` so a test can shape an
    arbitrary terrain (flat, ramp, etc.). Weather is constant. No real request
    is ever issued.
    """

    def fake_get_json(url: str):
        if url.startswith(geodata._ELEV_API):
            # Parse the lat/lon CSVs back out of the query string and answer
            # one elevation per coordinate, in order.
            qs = url.split("?", 1)[1]
            params = dict(p.split("=", 1) for p in qs.split("&"))
            lats = [float(x) for x in params["latitude"].split(",")]
            lons = [float(x) for x in params["longitude"].split(",")]
            return {"elevation": [elevations(la, lo) for la, lo in zip(lats, lons)]}
        if url.startswith(geodata._FORECAST_API):
            if "current=" in url:
                return {
                    "current": {
                        "temperature_2m": temp,
                        "wind_speed_10m": wind,
                        "wind_direction_10m": wdir,
                    }
                }
            # Hourly series: 24 identical hours so any departure index resolves.
            times = [f"2026-05-30T{h:02d}:00" for h in range(24)]
            return {
                "hourly": {
                    "time": times,
                    "temperature_2m": [temp] * 24,
                    "wind_speed_10m": [wind] * 24,
                    "wind_direction_10m": [wdir] * 24,
                }
            }
        return None

    monkeypatch.setattr(geodata, "_get_json", fake_get_json)


# --------------------------------------------------------------------------- #
# sample_polyline
# --------------------------------------------------------------------------- #
def test_sample_polyline_downsamples_and_keeps_endpoints():
    """A dense polyline must be reduced to <= max_points, endpoints preserved.

    The cap exists so we never exceed Open-Meteo's per-request coordinate
    budget; dropping the endpoints would mis-locate the route's start/finish.
    """
    sampled = geodata.sample_polyline(DENSE_GEOMETRY, max_points=10)

    assert 2 <= len(sampled) <= 11  # bounded by max_points (+ endpoint guard)
    assert len(sampled) < len(DENSE_GEOMETRY)  # genuinely downsampled
    assert sampled[0] == tuple(DENSE_GEOMETRY[0])
    assert sampled[-1] == tuple(DENSE_GEOMETRY[-1])


def test_sample_polyline_passthrough_when_short():
    """If already under the cap, every (cleaned) point is returned unchanged."""
    short = [[52.0, 13.0], [52.1, 13.1], [52.2, 13.2]]
    sampled = geodata.sample_polyline(short, max_points=80)
    assert sampled == [(52.0, 13.0), (52.1, 13.1), (52.2, 13.2)]


# --------------------------------------------------------------------------- #
# enrich_route — happy path
# --------------------------------------------------------------------------- #
def test_enrich_route_returns_segments_profile_and_conditions(monkeypatch):
    """enrich_route surfaces the model-relevant fields from a climbing route.

    A monotonic elevation ramp must produce positive gradients and a non-zero
    climb; the stubbed weather must flow through to each segment. This is the
    contract the planner relies on to make gradient/temperature/wind *real*.
    """
    # Elevation rises with longitude -> a steady uphill ramp eastbound.
    _install_fake_http(
        monkeypatch,
        elevations=lambda la, lo: (lo - 13.0) * 2000.0,  # ~20 m per 0.01 deg lon
        temp=8.0,
        wind=6.5,
        wdir=200.0,
    )

    out = geodata.enrich_route(DENSE_GEOMETRY, departure_iso="2026-05-30T09:00")

    # Structure.
    assert out["segments"], "expected at least one enriched segment"
    assert out["elevationProfile"], "expected an elevation profile"
    assert "conditions" in out

    seg = out["segments"][0]
    for field in ("distKm", "cumKm", "gradientPct", "elevM", "temperatureC", "windMps"):
        assert field in seg
        assert isinstance(seg[field], (int, float))

    # Weather stub flowed through to the segments.
    assert all(s["temperatureC"] == pytest.approx(8.0, abs=0.2) for s in out["segments"])
    assert all(s["windMps"] == pytest.approx(6.5, abs=0.2) for s in out["segments"])

    # An uphill ramp -> positive gradients and real climb, ~zero descent.
    assert all(s["gradientPct"] > 0 for s in out["segments"])
    cond = out["conditions"]
    assert cond["maxGradientPct"] > 0
    assert cond["climbM"] > 0
    assert cond["descentM"] == pytest.approx(0.0, abs=1.0)
    assert cond["avgTempC"] == pytest.approx(8.0, abs=0.2)
    assert cond["avgWindMps"] == pytest.approx(6.5, abs=0.2)

    # elevationProfile is monotonic in distance (cumulative) and JSON-numeric.
    dists = [p["distKm"] for p in out["elevationProfile"]]
    assert dists == sorted(dists)
    assert all(isinstance(p["elevM"], (int, float)) for p in out["elevationProfile"])


def test_enrich_route_flat_terrain_has_zero_gradient(monkeypatch):
    """Flat elevation -> ~zero gradient everywhere, contrasting the ramp case.

    This is the discriminating control: if gradient came out non-zero on flat
    ground the layer would inject phantom climbs into the energy model.
    """
    _install_fake_http(monkeypatch, elevations=lambda la, lo: 100.0)

    out = geodata.enrich_route(DENSE_GEOMETRY)

    assert out["segments"]
    assert all(s["gradientPct"] == pytest.approx(0.0, abs=1e-6) for s in out["segments"])
    assert out["conditions"]["climbM"] == pytest.approx(0.0, abs=1e-6)
    assert out["conditions"]["descentM"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Fail-soft behaviour
# --------------------------------------------------------------------------- #
def test_elevations_fail_soft_when_network_down(monkeypatch):
    """A network outage degrades elevations to the 0.0 default, never raises.

    The fail-soft guarantee lives in the stdlib HTTP primitive: ``_get_json``
    swallows the ``urlopen`` error and returns ``None``, and ``elevations``
    degrades each unresolved point to ``DEFAULT_ELEV_M``. We patch ``urlopen``
    itself (the real network boundary) to prove the whole path is soft, and
    that output length still matches the cleaned input.
    """
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(geodata.urllib.request, "urlopen", boom)

    pts = [(52.0, 13.0), (52.1, 13.1), (52.2, 13.2)]
    elev = geodata.elevations(pts)
    assert elev == [geodata.DEFAULT_ELEV_M] * len(pts)


def test_enrich_route_fail_soft_when_http_raises(monkeypatch):
    """A total network outage still yields a coherent, defaulted enrichment.

    The planner must be able to run offline: enrich_route returns segments with
    zero gradient, default temperature/wind, and a structurally valid
    conditions block — never an exception.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(geodata.urllib.request, "urlopen", boom)

    out = geodata.enrich_route(DENSE_GEOMETRY, departure_iso="2026-05-30T09:00")

    # Still structurally complete.
    assert "segments" in out and "elevationProfile" in out and "conditions" in out
    cond = out["conditions"]
    assert cond["maxGradientPct"] == pytest.approx(0.0, abs=1e-6)
    assert cond["climbM"] == pytest.approx(0.0, abs=1e-6)
    assert cond["descentM"] == pytest.approx(0.0, abs=1e-6)
    assert cond["avgTempC"] == pytest.approx(geodata.DEFAULT_TEMP_C)
    assert cond["avgWindMps"] == pytest.approx(geodata.DEFAULT_WIND_MPS)

    # Segments, if produced, carry the fail-soft defaults (flat, default wx).
    for s in out["segments"]:
        assert s["gradientPct"] == pytest.approx(0.0, abs=1e-6)
        assert s["temperatureC"] == pytest.approx(geodata.DEFAULT_TEMP_C)
        assert s["windMps"] == pytest.approx(geodata.DEFAULT_WIND_MPS)


def test_enrich_route_empty_geometry_fail_soft():
    """Empty / garbage geometry returns the default structure, never raises."""
    for bad in (None, [], [["x", "y"]], "not-a-polyline"):
        out = geodata.enrich_route(bad)
        assert out["segments"] == []
        assert out["elevationProfile"] == []
        assert out["conditions"]["avgTempC"] == pytest.approx(geodata.DEFAULT_TEMP_C)
