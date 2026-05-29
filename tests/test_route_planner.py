"""Tests for :mod:`nexdash.route_planner`, the SOC-drain trip planner.

These verify the *intent* of the planner, not just that it returns a dict:

* The planner must produce the ``PlanResult`` shape the frontend depends on
  (``socProfile`` / ``segments`` / ``summary``); a missing key silently breaks
  the dashboard's route view.
* Gradient must *actually reach the model*. The headline feature of geometry
  mode is real terrain: a steep-uphill enriched route must consume MORE energy
  than an otherwise-identical flat route. If it didn't, the whole enrichment
  pipeline would be cosmetic — so this is the test that can fail when the
  business logic (per-segment gradient wiring) regresses.
* Charging stops must be inserted when a long route would otherwise dip below
  ``min_soc`` on the fixed 600 kWh battery — that's the operational decision
  the planner exists to make.

The real trained model artifact is used for inference (it must already exist,
as every other model-backed test in this suite assumes). ``geodata.enrich_route``
is monkeypatched so terrain is deterministic and no network is touched.
"""

from __future__ import annotations

import pytest

from nexdash import route_planner
from nexdash.config import DEFAULT_MODEL_PATH


# A trivial 4-vertex polyline; its real coordinates are irrelevant because we
# stub enrich_route, but plan_route requires a truthy geometry to enter
# geometry mode.
GEOMETRY = [[52.0, 13.0], [52.3, 13.3], [52.6, 13.6], [52.9, 13.9]]


def _fake_enrichment(*, gradient_pct, n_segments=8, dist_km_each=50.0, temp=15.0, wind=3.0):
    """Build a deterministic enrich_route() return with a uniform gradient.

    Mirrors the real :func:`nexdash.geodata.enrich_route` contract closely
    enough for the planner: per-segment ``distKm`` / ``gradientPct`` /
    ``temperatureC`` / ``windMps``, plus ``elevationProfile`` and
    ``conditions`` (with ``climbM`` driving ``summary.elevationGainM``).
    """
    segments = []
    cum = 0.0
    climb = 0.0
    elev = 0.0
    profile = [{"distKm": 0.0, "elevM": 0.0}]
    for _ in range(n_segments):
        cum += dist_km_each
        d_elev = gradient_pct / 100.0 * dist_km_each * 1000.0
        elev += d_elev
        if d_elev > 0:
            climb += d_elev
        segments.append(
            {
                "distKm": dist_km_each,
                "cumKm": round(cum, 3),
                "gradientPct": gradient_pct,
                "elevM": round(elev, 1),
                "temperatureC": temp,
                "windMps": wind,
            }
        )
        profile.append({"distKm": round(cum, 3), "elevM": round(elev, 1)})
    return {
        "segments": segments,
        "elevationProfile": profile,
        "conditions": {
            "avgTempC": temp,
            "avgWindMps": wind,
            "windDirDeg": 0.0,
            "maxGradientPct": gradient_pct,
            "climbM": round(climb, 1),
            "descentM": 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #
def test_plan_route_returns_expected_shape(monkeypatch):
    """A simple geometry-mode plan returns the full PlanResult contract."""
    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(gradient_pct=0.0, dist_km_each=10.0, n_segments=4),
    )

    result = route_planner.plan_route(
        distance_km=40.0,
        duration_s=40.0 / 70.0 * 3600.0,
        start_soc=90.0,
        min_soc=15.0,
        payload_kg=10000.0,
        departure="2026-05-30T08:00",
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )

    # Core PlanResult keys.
    for key in ("socProfile", "segments", "chargingStops", "summary"):
        assert key in result
    # Geometry mode also surfaces enrichment.
    assert "elevationProfile" in result
    assert "conditions" in result

    assert result["socProfile"][0] == {"distKm": 0.0, "soc": 90.0}
    assert result["socProfile"][-1]["distKm"] == pytest.approx(40.0, abs=0.5)

    summary = result["summary"]
    for key in (
        "distanceKm", "drivingTimeH", "chargingTimeMin", "totalTimeH",
        "startSoc", "arrivalSoc", "minSoc", "energyKwh", "kwhPer100",
        "chargingCostEur", "chargingStops", "elevationGainM", "driver",
    ):
        assert key in summary
    assert summary["distanceKm"] == pytest.approx(40.0, abs=0.1)
    assert summary["startSoc"] == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# Gradient actually drives the model
# --------------------------------------------------------------------------- #
def test_steep_uphill_uses_more_energy_than_flat(monkeypatch):
    """Identical route, steep climb vs. flat -> the climb must cost more energy.

    This is the load-bearing test: it can only pass if the per-segment
    ``gradient_pct`` from the enrichment actually reaches
    ``predict_energy``. Same distance, payload, speed, temperature and wind in
    both runs — only the terrain differs.
    """
    common = dict(
        distance_km=400.0,
        duration_s=400.0 / 70.0 * 3600.0,
        start_soc=100.0,
        min_soc=0.0,            # disable charging so we compare raw energy
        payload_kg=12000.0,
        departure="2026-05-30T08:00",
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )

    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(gradient_pct=0.0),
    )
    flat = route_planner.plan_route(**common)

    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(gradient_pct=6.0),
    )
    steep = route_planner.plan_route(**common)

    assert steep["summary"]["energyKwh"] > flat["summary"]["energyKwh"] * 1.2, (
        f"steep climb ({steep['summary']['energyKwh']} kWh) should clearly exceed "
        f"flat ({flat['summary']['energyKwh']} kWh)"
    )
    # The climb must be reported as elevation gain; the flat run gains nothing.
    assert steep["summary"]["elevationGainM"] > 0
    assert flat["summary"]["elevationGainM"] == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Charging insertion
# --------------------------------------------------------------------------- #
def test_charging_stops_inserted_on_long_low_soc_route(monkeypatch):
    """A long route starting low must trigger >=1 charging stop above the floor.

    With a modest start SOC and a high min-SOC floor over a long climb, raw
    drain would breach the floor; the planner must insert charging stops,
    recharge toward the target, and the final SOC must respect the floor.
    """
    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(
            gradient_pct=3.0, n_segments=16, dist_km_each=50.0
        ),
    )

    result = route_planner.plan_route(
        distance_km=800.0,
        duration_s=800.0 / 70.0 * 3600.0,
        start_soc=60.0,
        min_soc=15.0,
        payload_kg=15000.0,
        departure="2026-05-30T06:00",
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )

    assert len(result["chargingStops"]) >= 1, "expected at least one charging stop"
    # Never dip below the operator's floor.
    assert min(p["soc"] for p in result["socProfile"]) >= 15.0 - 1e-6
    # Each stop recharges meaningfully toward the target and is costed.
    for stop in result["chargingStops"]:
        assert stop["departSoc"] > stop["arriveSoc"]
        assert stop["kWh"] > 0
        assert stop["costEur"] > 0
    # Charging time and cost roll up into the summary.
    assert result["summary"]["chargingTimeMin"] > 0
    assert result["summary"]["chargingCostEur"] > 0
    assert result["summary"]["chargingStops"] == len(result["chargingStops"])


# --------------------------------------------------------------------------- #
# Flat fallback (no geometry)
# --------------------------------------------------------------------------- #
def test_flat_fallback_without_geometry_omits_enrichment():
    """Without geometry the planner keeps the old shape (no enrichment keys).

    This guards the documented backward-compatible path: callers that don't
    send a polyline must still get socProfile/segments/summary, and must NOT
    get elevationProfile/conditions (which would imply real terrain we never
    computed).
    """
    result = route_planner.plan_route(
        distance_km=120.0,
        duration_s=120.0 / 70.0 * 3600.0,
        start_soc=90.0,
        min_soc=15.0,
        payload_kg=8000.0,
        model_path=DEFAULT_MODEL_PATH,
    )

    assert "socProfile" in result and "segments" in result and "summary" in result
    assert "elevationProfile" not in result
    assert "conditions" not in result
    assert result["summary"]["elevationGainM"] == pytest.approx(0.0)
    assert result["summary"]["energyKwh"] > 0
