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
from nexdash.config import DEFAULT_MODEL_PATH, TRUCK


# --------------------------------------------------------------------------- #
# Charging power-vs-SOC taper
# --------------------------------------------------------------------------- #
def test_charge_taper_costs_more_per_percent_in_the_tail():
    """A percent of charge in the tapered tail must take longer than below the knee.

    WHY: heavy-BEV DC charging holds near-peak power only to ~80% SOC then
    derates steeply; a flat-power model that ignores this materially understates
    session time for a top-to-95% stop. The taper helper must reflect that the
    94->95% percent is slower than the 40->41% percent, and that below the knee
    there is no taper (equals the flat rate).
    """
    rated, batt = 400.0, TRUCK.battery_kwh
    below_knee = route_planner._charge_minutes(40.0, 41.0, rated, batt)
    in_tail = route_planner._charge_minutes(94.0, 95.0, rated, batt)
    assert in_tail > below_knee, "tail percent must be slower than below-knee percent"

    # Below the knee there is no taper: it equals the flat-rate time.
    flat_below = (batt / 100.0) / rated * 60.0  # minutes for 1% at full power
    assert below_knee == pytest.approx(flat_below, rel=1e-6)

    # Charging into the tail (80->100) must exceed the naive flat estimate.
    taper_tail = route_planner._charge_minutes(80.0, 100.0, rated, batt)
    flat_tail = (batt * 0.20) / rated * 60.0
    assert taper_tail > flat_tail * 1.3


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


# --------------------------------------------------------------------------- #
# Geometry-mode energy must track physics, not polyline density
# --------------------------------------------------------------------------- #
def test_geometry_energy_invariant_to_polyline_density(monkeypatch):
    """The SAME uniform-condition route must cost ~the same energy regardless of
    how finely its polyline is sampled.

    WHY: the energy model over-predicts on very short (<~5 km) legs, so the old
    "predict per enriched sub-chunk and sum" inflated total route energy with
    polyline DENSITY — a real downsampled TomTom polyline is ~2-10 km/segment,
    well inside the inflated regime, producing absurd >600 kWh/100km totals and
    phantom charging stops. The planner now aggregates enriched segments into
    ~CHUNK_KM windows (distance-weighted-average conditions) before predicting,
    so energy depends on physics, not segment count. This is the load-bearing
    guard against reintroducing per-sub-chunk prediction.
    """
    common = dict(
        distance_km=200.0,
        duration_s=200.0 / 70.0 * 3600.0,
        start_soc=100.0,
        min_soc=0.0,            # disable charging so we compare raw energy
        payload_kg=22000.0,
        temperature_c=-10.0,
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )

    energies = []
    for n, d in ((4, 50.0), (40, 5.0), (80, 2.5)):  # 200 km as coarse / fine / finer
        monkeypatch.setattr(
            route_planner.geodata,
            "enrich_route",
            lambda geometry, departure_iso=None, _n=n, _d=d: _fake_enrichment(
                gradient_pct=0.0, n_segments=_n, dist_km_each=_d, wind=8.0
            ),
        )
        energies.append(route_planner.plan_route(**common)["summary"]["energyKwh"])

    # Identical physics, identical net terrain -> energy must not drift with the
    # number of polyline segments (it varied ~4x before the fix).
    assert max(energies) - min(energies) < 5.0, (
        f"geometry-mode energy still varies with polyline density: {energies}"
    )


# --------------------------------------------------------------------------- #
# Charging look-ahead: no spurious charge on a reachable trip
# --------------------------------------------------------------------------- #
def test_short_reachable_trip_inserts_no_charge():
    """A short trip that finishes above the hard floor must NOT force a charge.

    WHY: the trigger used floor = min_soc + reserve and fired whenever any chunk
    merely dipped into the reserve band, so a 30 km hop from 20% SOC got a full
    ~76 min recharge AT THE ORIGIN even though it arrives ~15% — comfortably above
    the 10% hard floor — directly contradicting check_reachability, which calls
    the identical trip reachable. The planner now only charges if continuing would
    breach the HARD min_soc before the destination. Fails if the look-ahead is
    removed (the reserve band must be a soft cushion, not a hard origin trigger).
    """
    result = route_planner.plan_route(
        distance_km=30.0,
        duration_s=30.0 / 70.0 * 3600.0,
        start_soc=20.0,
        min_soc=10.0,
        payload_kg=0.0,
        reserve_pct=10.0,
        temperature_c=15.0,
        model_path=DEFAULT_MODEL_PATH,
    )
    assert len(result["chargingStops"]) == 0, "reachable short trip must not charge"
    assert all(s["type"] != "charge" for s in result["segments"])
    # Arrives above the hard floor (the reserve cushion may be dipped into).
    assert result["summary"]["arrivalSoc"] >= 10.0


# --------------------------------------------------------------------------- #
# Per-leg simulation + per-stop wiring (payload decay / unload / deliver-by)
# --------------------------------------------------------------------------- #
# Berlin -> Leipzig -> Munich-ish; coords only set the leg-distance fractions.
_ORIGIN = {"lat": 52.52, "lng": 13.40, "label": "Berlin"}
_MID = {"lat": 51.34, "lng": 12.37, "label": "Leipzig"}
_FINAL = {"lat": 48.14, "lng": 11.58, "label": "Munich"}


def _multistop_plan(mid_extra=None, final_extra=None, **overrides):
    mid = {**_MID, **(mid_extra or {})}
    final = {**_FINAL, **(final_extra or {})}
    kwargs = dict(
        distance_km=500.0,
        duration_s=500.0 / 70.0 * 3600.0,
        start_soc=100.0,
        min_soc=0.0,          # disable charging so we isolate the leg effects
        payload_kg=20000.0,
        departure="2026-05-30T08:00",
        model_path=DEFAULT_MODEL_PATH,
    )
    kwargs.update(overrides)
    return route_planner.plan_route(waypoints=[_ORIGIN, mid, final], **kwargs)


def test_payload_decay_makes_constant_payload_the_conservative_estimate():
    """Dropping cargo at a stop must lower later-leg energy vs holding it.

    WHY (encodes the honesty claim): the planner held payload constant for the
    whole trip, which it documents as 'conservative — over-estimates later legs'.
    Wiring per-stop drops must make that literally true: the same route with a
    10 t drop at the mid-stop must consume strictly LESS energy than with no drop,
    so constant-payload is the upper bound it claims to be.
    """
    decayed = _multistop_plan(mid_extra={"dropWeightKg": 10000})
    constant = _multistop_plan(mid_extra={"dropWeightKg": 0})
    assert decayed["summary"]["energyKwh"] < constant["summary"]["energyKwh"]
    # The mid-stop must report the reduced post-drop payload.
    mid_stop = decayed["stops"][0]
    assert mid_stop["payloadAfterT"] == pytest.approx(10.0, abs=0.01)


def test_per_stop_arrival_soc_and_eta_reported():
    """Each destination must carry its own arrival SOC + ETA; SOC falls along route."""
    plan = _multistop_plan()
    stops = plan["stops"]
    assert len(stops) == 2  # two destinations (origin excluded)
    assert all("etaIso" in s and "arriveSoc" in s for s in stops)
    # No charging here, so arrival SOC must be non-increasing leg over leg.
    assert stops[0]["arriveSoc"] >= stops[1]["arriveSoc"]
    assert stops[-1]["isFinal"] is True and stops[0]["isFinal"] is False


def test_deliver_by_feasibility_flag():
    """A deliver-by deadline must drive an on-time flag (True / False / None)."""
    impossible = _multistop_plan(final_extra={"deliverBy": "2026-05-30T08:30"})
    assert impossible["stops"][-1]["onTime"] is False  # 30 min for ~500 km: no
    generous = _multistop_plan(final_extra={"deliverBy": "2026-05-31T20:00"})
    assert generous["stops"][-1]["onTime"] is True
    none_set = _multistop_plan()
    assert none_set["stops"][-1]["onTime"] is None  # no deadline -> unknown


def test_unload_dwell_extends_eta_and_appears_in_timeline():
    """Unload time at an intermediate stop must push the ETA and show in segments."""
    no_dwell = _multistop_plan(mid_extra={"unloadMin": 0})
    dwell = _multistop_plan(mid_extra={"unloadMin": 60})
    assert dwell["summary"]["totalTimeH"] > no_dwell["summary"]["totalTimeH"]
    assert dwell["summary"]["unloadTimeMin"] == 60
    assert any(seg["type"] == "unload" for seg in dwell["segments"])


def test_summary_carries_machine_readable_assumptions():
    """The response must expose honest-limitations as a machine-readable list."""
    plan = _multistop_plan(mid_extra={"dropWeightKg": 8000})
    assumptions = plan["summary"]["assumptions"]
    assert isinstance(assumptions, list) and assumptions
    blob = " ".join(assumptions).lower()
    assert "eu 561" in blob and "taper" in blob
    assert any("payload decays" in a.lower() for a in assumptions)
