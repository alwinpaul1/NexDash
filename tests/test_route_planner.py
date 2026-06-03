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


def test_descent_route_clamps_soc_and_floors_displayed_energy(monkeypatch):
    """Regression for the two regen blockers.

    A sustained steep descent credits regen (negative per-chunk energy), which
    (1) must NOT push SOC above 100% — the pack cannot charge past full — and
    (2) must NOT surface a negative energy headline: the DISPLAYED total is
    floored at 0 for a net-downhill trip, while the signed value still drove the
    conservative SOC walk. Before the fix this exact -6% route reported
    socProfile max ~122% and kwhPer100 ~-108.
    """
    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(
            gradient_pct=-6.0, n_segments=16, dist_km_each=50.0
        ),
    )

    result = route_planner.plan_route(
        distance_km=800.0,
        duration_s=800.0 / 70.0 * 3600.0,
        start_soc=95.0,  # high: descent regen would overflow 100% without the clamp
        min_soc=15.0,
        payload_kg=15000.0,
        departure="2026-05-30T06:00",
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )

    socs = [p["soc"] for p in result["socProfile"]]
    assert max(socs) <= 100.0 + 1e-6, f"SOC must never exceed 100%; saw {max(socs):.2f}"
    assert result["summary"]["arrivalSoc"] <= 100.0 + 1e-6
    assert result["summary"]["energyKwh"] >= 0.0, "displayed energy must be floored at >= 0"
    assert result["summary"]["kwhPer100"] >= 0.0, "displayed kWh/100km must be floored at >= 0"


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


def test_field_calibration_scales_displayed_energy_only(monkeypatch):
    """The field-calibration factor lowers the DISPLAYED energy headline but must
    NOT change the SOC walk or any charging decision.

    Safety contract: charging/reachability run on the conservative (un-discounted)
    estimate, so a lower displayed figure can never delay a charge or strand the
    truck. Verifies (1) energyKwh/kwhPer100 scale ~linearly with the factor, and
    (2) on a charge-requiring route the chargingStops, arrival SOC, per-stop charge
    kWh and the FULL SOC profile are byte-identical at factor 0.85 vs 1.0.
    """
    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(
            gradient_pct=3.0, n_segments=16, dist_km_each=50.0
        ),
    )
    common = dict(
        distance_km=800.0,
        duration_s=800.0 / 70.0 * 3600.0,
        start_soc=60.0,
        min_soc=15.0,
        payload_kg=15000.0,
        departure="2026-05-30T06:00",
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )
    full = route_planner.plan_route(field_calibration=1.0, **common)
    cal = route_planner.plan_route(field_calibration=0.85, **common)

    # (1) Displayed energy scales by the factor (small tolerance for rounding).
    assert cal["summary"]["energyKwh"] == pytest.approx(
        0.85 * full["summary"]["energyKwh"], rel=0.01
    )
    assert cal["summary"]["kwhPer100"] == pytest.approx(
        0.85 * full["summary"]["kwhPer100"], rel=0.01
    )
    # (2) SAFETY: the charging plan and SOC trajectory are untouched by the factor.
    assert cal["summary"]["chargingStops"] == full["summary"]["chargingStops"]
    assert cal["summary"]["arrivalSoc"] == full["summary"]["arrivalSoc"]
    assert [c["kWh"] for c in cal["chargingStops"]] == [
        c["kWh"] for c in full["chargingStops"]
    ]
    assert cal["socProfile"] == full["socProfile"]


def test_field_calibration_is_clamped_to_unit_interval(monkeypatch):
    """The factor can only LOWER energy: values >1 clamp to 1.0 (no inflation),
    and the default (None-equivalent) leaves the headline at the conservative
    figure when set to 1.0."""
    monkeypatch.setattr(
        route_planner.geodata,
        "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(gradient_pct=0.0),
    )
    common = dict(
        distance_km=200.0,
        duration_s=200.0 / 70.0 * 3600.0,
        start_soc=100.0,
        min_soc=0.0,
        payload_kg=12000.0,
        departure="2026-05-30T08:00",
        geometry=GEOMETRY,
        model_path=DEFAULT_MODEL_PATH,
    )
    base = route_planner.plan_route(field_calibration=1.0, **common)
    inflated = route_planner.plan_route(field_calibration=1.5, **common)
    assert inflated["summary"]["energyKwh"] == base["summary"]["energyKwh"]
    # Lower clamp: a positive value below 0.5 floors at 0.5 (no absurd headline).
    floored = route_planner.plan_route(field_calibration=0.3, **common)
    half = route_planner.plan_route(field_calibration=0.5, **common)
    assert floored["summary"]["energyKwh"] == half["summary"]["energyKwh"]
    # Module default: omitting the kwarg matches passing the configured default,
    # so a drift in FIELD_CALIBRATION_FACTOR can't silently change behaviour.
    default = route_planner.plan_route(**common)
    explicit = route_planner.plan_route(
        field_calibration=route_planner.FIELD_CALIBRATION_FACTOR, **common
    )
    assert default["summary"]["energyKwh"] == explicit["summary"]["energyKwh"]


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

    WHY: the en-route floor is max(min_soc, reserve) -- here max(10, 20) = 20% --
    and a chunk dipping into the reserve band would naively trigger a charge. But a
    30 km hop from 20% SOC arrives ~15%: BELOW the 20% reserve cushion yet ABOVE the
    10% HARD min_soc, so it must NOT force a ~76 min recharge AT THE ORIGIN -- that
    would contradict check_reachability, which calls the identical trip reachable.
    The planner only charges if continuing would breach the HARD min_soc before the
    destination. Fails if the look-ahead is removed (the reserve band is a soft
    cushion, not a hard origin trigger) -- and pins that the floor is max(), not the
    old min_soc + reserve sum (which would have reserved 30% here).
    """
    result = route_planner.plan_route(
        distance_km=30.0,
        duration_s=30.0 / 70.0 * 3600.0,
        start_soc=20.0,
        min_soc=10.0,
        payload_kg=0.0,
        reserve_pct=20.0,   # reserve (20) > min_soc (10): a real soft band to dip into
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
    """Each destination must carry its own arrival SOC + ETA; SOC falls along route.

    Uses a 300 km route (~390 kWh < the 600 kWh battery) so it genuinely finishes
    WITHOUT a charge -- only then is "arrival SOC non-increasing leg over leg" a
    valid invariant. (The 500 km default needs >1 battery, so it charges mid-route,
    which legitimately raises a later stop's SOC above an earlier one -- that path
    is covered by the charging tests, not this leg-monotonicity check.)
    """
    plan = _multistop_plan(distance_km=300.0, duration_s=300.0 / 70.0 * 3600.0)
    stops = plan["stops"]
    assert len(stops) == 2  # two destinations (origin excluded)
    assert all("etaIso" in s and "arriveSoc" in s for s in stops)
    assert not plan["chargingStops"], "300 km must finish without charging for this invariant"
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


# --------------------------------------------------------------------------- #
# Physics cross-check: out-of-envelope terrain must be flagged, not trusted
# --------------------------------------------------------------------------- #
def test_out_of_envelope_grade_is_flagged_low_confidence(monkeypatch):
    """A sustained steep/cold grade must trip the planner's physics cross-check.

    WHY (the dangerous direction): the data-driven model under-predicts energy on
    terrain outside its training envelope. Training caps gradient at +/-6% for a
    ~25 km chunk (the 1500 m net-climb ceiling), so a sustained +10% grade is
    beyond what the model has seen and it saturates below physics. Trusting that
    optimistic number would delay a charge and risk stranding the truck. The
    planner must mirror range.check_reachability: when physics exceeds the model
    beyond the divergence band, use the conservative value AND surface a LOW
    CONFIDENCE assumption. A normal flat route must NOT be flagged (no false
    alarms).
    """
    monkeypatch.setattr(
        route_planner.geodata, "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(
            gradient_pct=10.0, n_segments=6, dist_km_each=25.0, temp=-15.0, wind=9.0
        ),
    )
    steep = route_planner.plan_route(
        distance_km=150.0, duration_s=150.0 / 70.0 * 3600.0, start_soc=100.0,
        min_soc=10.0, payload_kg=22000.0, temperature_c=-15.0,
        geometry=GEOMETRY, model_path=DEFAULT_MODEL_PATH,
    )
    assert any("low confidence" in a.lower() for a in steep["summary"]["assumptions"]), \
        steep["summary"]["assumptions"]

    monkeypatch.setattr(
        route_planner.geodata, "enrich_route",
        lambda geometry, departure_iso=None: _fake_enrichment(gradient_pct=0.0, n_segments=6),
    )
    flat = route_planner.plan_route(
        distance_km=150.0, duration_s=150.0 / 70.0 * 3600.0, start_soc=100.0,
        min_soc=10.0, payload_kg=11000.0, temperature_c=15.0,
        geometry=GEOMETRY, model_path=DEFAULT_MODEL_PATH,
    )
    assert not any("low confidence" in a.lower() for a in flat["summary"]["assumptions"])


# --------------------------------------------------------------------------- #
# Per-segment speed (Tier B): gradient-shaped, ETA-anchored
# --------------------------------------------------------------------------- #
def _mixed_enrichment(legs, *, temp=15.0, wind=3.0):
    """enrich_route() stub with a per-segment gradient: ``legs`` = [(km, grad), ...]."""
    segments, profile = [], [{"distKm": 0.0, "elevM": 0.0}]
    cum = elev = climb = 0.0
    for dist_km, grad in legs:
        cum += dist_km
        d_elev = grad / 100.0 * dist_km * 1000.0
        elev += d_elev
        if d_elev > 0:
            climb += d_elev
        segments.append(
            {"distKm": dist_km, "cumKm": round(cum, 3), "gradientPct": grad,
             "elevM": round(elev, 1), "temperatureC": temp, "windMps": wind}
        )
        profile.append({"distKm": round(cum, 3), "elevM": round(elev, 1)})
    return {
        "segments": segments, "elevationProfile": profile,
        "conditions": {"avgTempC": temp, "avgWindMps": wind, "windDirDeg": 0.0,
                       "maxGradientPct": max(g for _, g in legs), "climbM": round(climb, 1),
                       "descentM": 0.0},
    }


def test_segment_speed_shape_is_monotone_in_gradient():
    """WHY: a loaded truck must be slower the steeper the climb, and never freewheel
    fast downhill (governed). A non-monotone shape would mis-rank where speed — and
    thus the speed-sensitive aero/rolling energy — concentrates along the route."""
    assert route_planner._segment_speed_shape(0.0) == 1.0
    assert route_planner._segment_speed_shape(2.0) > route_planner._segment_speed_shape(6.0)
    assert route_planner._segment_speed_shape(6.0) < 1.0
    assert 1.0 < route_planner._segment_speed_shape(-6.0) <= route_planner.SPEED_DESC_CAP_FRAC


def test_per_segment_speed_preserves_total_eta(monkeypatch):
    """WHY: per-segment speed only REDISTRIBUTES time across segments; it must not
    change the total drive time the routing engine measured. The anchor guarantees
    Σ leg_dist/leg_speed == the route-average duration — a broken anchor would
    silently move the ETA the dispatcher was promised."""
    legs = [(50.0, 6.0), (50.0, -4.0), (50.0, 0.0), (50.0, 5.0), (50.0, -2.0), (50.0, 0.0)]
    monkeypatch.setattr(route_planner.geodata, "enrich_route",
                        lambda geometry, departure_iso=None: _mixed_enrichment(legs))
    dur_s = 300.0 / 65.0 * 3600.0
    result = route_planner.plan_route(
        distance_km=300.0, duration_s=dur_s, start_soc=100.0, min_soc=0.0,
        payload_kg=10000.0, geometry=GEOMETRY, model_path=DEFAULT_MODEL_PATH,
    )
    assert result["summary"]["drivingTimeH"] == pytest.approx(dur_s / 3600.0, rel=0.02)


def test_adaptive_target_soc_charges_to_target_or_into_tail():
    """WHY: the charge policy is 'charge UP TO the target' (the long-haul 'charge it
    up' behaviour), NOT 'just enough'. A stop must (a) top to the soft-ceiling TARGET
    even when the remaining route would need less (so it leaves a wide buffer), (b)
    reach ABOVE the target into the slow 80->100% tail when one charge needs more to
    finish the trip, and (c) when even 100% can't finish in one stop, charge to the
    target and stop again later — never exceeding 100% nor charging below arrival."""
    batt = 600.0
    short = route_planner._adaptive_target_soc(
        remaining_energy_kwh=0.12 * batt, arrive_soc=15.0, battery_kwh=batt,
        charge_floor=20.0, soft_ceiling_soc=80.0)
    assert short == pytest.approx(80.0)              # (a) tops to the TARGET, not just-enough
    tail = route_planner._adaptive_target_soc(
        remaining_energy_kwh=0.70 * batt, arrive_soc=15.0, battery_kwh=batt,
        charge_floor=20.0, soft_ceiling_soc=80.0)
    assert 80.0 < tail <= 100.0                      # (b) into the tail when one stop needs it
    multi = route_planner._adaptive_target_soc(
        remaining_energy_kwh=1.50 * batt, arrive_soc=15.0, battery_kwh=batt,
        charge_floor=20.0, soft_ceiling_soc=80.0)
    assert multi == pytest.approx(80.0)              # (c) capped at target, finish later
    assert short >= 15.5 and tail >= 15.5            # never below arrival


def test_eu561_splits_long_trip_with_daily_rest(monkeypatch):
    """WHY: a >9 h-driving trip is LEGAL once split with an 11 h daily rest — the old
    single-shift model falsely flagged it a 561 'Violation' and reported dailyH as the
    whole-trip total. The machine must insert the rest, keep each day <=9 h driving,
    report Compliant, and give a per-day breakdown that sums back to total driving."""
    monkeypatch.setattr(
        route_planner.geodata, "enrich_route",
        lambda geometry, departure_iso=None:
        _fake_enrichment(gradient_pct=0.0, n_segments=18, dist_km_each=50.0),
    )
    # 900 km at 70 km/h ~ 12.9 h of driving -> must split across 2 calendar days.
    result = route_planner.plan_route(
        distance_km=900.0, duration_s=900.0 / 70.0 * 3600.0, start_soc=100.0,
        min_soc=10.0, payload_kg=10000.0, departure="2026-05-30T06:00",
        geometry=GEOMETRY, model_path=DEFAULT_MODEL_PATH,
    )
    rests = [s for s in result["segments"] if s.get("type") == "daily_rest"]
    assert rests, "a >9 h trip must insert at least one 11 h daily rest"
    assert all(s["durationMin"] == round(route_planner.EU561_DAILY_REST_H * 60.0) for s in rests)
    d = result["summary"]["driver"]
    assert d["dailyH"] <= route_planner.EU561_DAILY_MAX_DRIVE_H + 1e-6   # heaviest day <= 9 h
    assert d["eu561ok"] is True                                         # legal once split
    assert d["days"] == len(d["perDay"]) >= 2
    assert sum(p["drivingH"] for p in d["perDay"]) == pytest.approx(d["drivingH"], abs=0.05)


def test_assumptions_do_not_leak_internal_field_names(monkeypatch):
    """WHY: user-facing caveats must read as plain language. The old EU 561 string
    leaked the raw JSON field names 'drivingH/dailyH/weeklyH' into the dispatcher's
    Modelling-assumptions panel — this guards against that regressing."""
    monkeypatch.setattr(route_planner.geodata, "enrich_route",
                        lambda geometry, departure_iso=None: _fake_enrichment(gradient_pct=2.0))
    result = route_planner.plan_route(
        distance_km=200.0, duration_s=200.0 / 70.0 * 3600.0, start_soc=90.0,
        min_soc=10.0, payload_kg=10000.0, geometry=GEOMETRY, model_path=DEFAULT_MODEL_PATH,
    )
    blob = " ".join(result["summary"]["assumptions"])
    for leaked in ("drivingH", "dailyH", "weeklyH"):
        assert leaked not in blob, f"internal field name {leaked!r} leaked into assumptions"


def test_check_reachability_rejects_nonpositive_speed():
    """#18 (fail-loud): a moving segment needs speed>0. check_reachability must raise
    a clear ValueError on speed<=0 — not crash deep in the physics layer and not let
    predict_energy silently extrapolate a stationary segment."""
    from nexdash.range import check_reachability

    with pytest.raises(ValueError, match="speed_kph"):
        check_reachability(
            soc_pct=80, distance_km=50, payload_t=10, speed_kph=0,
            gradient_pct=0, temperature_c=20,
        )


def test_adaptive_target_soc_uncertainty_cushion_raises_depart():
    """#13: the forecast-uncertainty cushion must push the depart SOC strictly above
    the no-cushion target, so a charge does not aim to arrive exactly at the floor on
    the model's own (possibly optimistic) energy number."""
    # Use a remaining-energy in the TAIL region (need_depart between the target and
    # 100%), where the charge is sized by the route, so the cushion visibly raises
    # it; below the target both pin to the ceiling and the cushion is (safely) moot.
    kw = dict(arrive_soc=15.0, battery_kwh=600.0, charge_floor=20.0, soft_ceiling_soc=80.0)
    base = route_planner._adaptive_target_soc(0.65 * 600.0, **kw)
    cushioned = route_planner._adaptive_target_soc(0.65 * 600.0, uncertainty_kwh=30.0, **kw)
    assert cushioned > base


def test_measured_per_leg_speed_is_used_when_available(monkeypatch):
    """A1(1) 'use real data where available': when the enrichment carries a MEASURED
    per-leg speed (from the routing engine's travel time), the planner drives on it
    directly — not the gradient heuristic — and the panel says the speed is measured.
    Absent that field, the existing tests prove the heuristic path is unchanged."""
    def _enrich_with_measured(geometry, departure_iso=None, leg_timings=None):
        enr = _fake_enrichment(gradient_pct=2.0, n_segments=8, dist_km_each=25.0)
        for s in enr["segments"]:
            s["measuredSpeedKph"] = 50.0  # real measured leg speed
        return enr

    monkeypatch.setattr(route_planner.geodata, "enrich_route", _enrich_with_measured)
    result = route_planner.plan_route(
        distance_km=200.0, duration_s=200.0 / 70.0 * 3600.0, start_soc=100.0, min_soc=0.0,
        payload_kg=10000.0, geometry=GEOMETRY, model_path=DEFAULT_MODEL_PATH,
    )
    # Driving time reflects the MEASURED 50 km/h (200/50 = 4.0 h), NOT the supplied
    # 70 km/h route average (~2.86 h) — proving the real per-leg speed drives the sim.
    assert result["summary"]["drivingTimeH"] == pytest.approx(200.0 / 50.0, rel=0.03)
    blob = " ".join(result["summary"]["assumptions"]).lower()
    assert "measured" in blob and "gradient heuristic" not in blob


# --------------------------------------------------------------------------- #
# A2-1 — EU 561 extended 10 h driving day (opt-in, max 2x/week)
# --------------------------------------------------------------------------- #
# A 9.5 h flat trip (665 km @ 70 km/h, no geometry so drive time == distance/avg).
_EXT_TRIP = dict(
    distance_km=665.0, duration_s=665.0 / 70.0 * 3600.0, start_soc=100.0,
    min_soc=0.0, payload_kg=5000.0, departure="2026-05-30T06:00",
    model_path=DEFAULT_MODEL_PATH,
)


def test_extended_day_default_is_byte_identical():
    """WHY (regression-safe): with the default allow_extended_days=0 the planner
    must behave exactly as before this option existed — the 9 h cap, the 11 h rest
    that splits a 9.5 h trip across two days, and no `extended` day. If the default
    drifted, every existing plan would silently change."""
    base = route_planner.plan_route(**_EXT_TRIP)
    d = base["summary"]["driver"]
    assert d["days"] == 2                                   # 9.5 h split across 2 days
    assert d["dailyH"] <= route_planner.EU561_DAILY_MAX_DRIVE_H + 1e-6
    assert sum(1 for s in base["segments"] if s.get("type") == "daily_rest") == 1
    assert all(p["extended"] is False for p in d["perDay"])  # no day used the 10 h cap


def test_extended_day_keeps_9_5h_trip_in_one_day():
    """WHY: the headline of A2-1 — allowing one 10 h day must let a 9.5 h trip finish
    in a single shift (no overnight rest), where the 9 h default forces a second day.
    The day must be flagged `extended`, its driving must sit in (9, 10] h, and the
    plan must still read EU 561-compliant (a 10 h day is legal up to 2x/week)."""
    ext = route_planner.plan_route(allow_extended_days=1, **_EXT_TRIP)
    d = ext["summary"]["driver"]
    assert d["days"] == 1                                   # the 10 h cap absorbs it
    assert sum(1 for s in ext["segments"] if s.get("type") == "daily_rest") == 0
    assert 9.0 < d["dailyH"] <= route_planner.EU561_EXT_DAILY_MAX_DRIVE_H + 1e-6
    assert d["perDay"][0]["extended"] is True              # the slot was consumed
    assert d["eu561ok"] is True                            # 10 h day is legal


def test_extended_days_allowance_is_clamped_and_spent_in_order():
    """WHY: the allowance is a scarce, clamped resource — once spent, later days must
    revert to the 9 h cap (EU 561 permits the 10 h day at most 2x/week). Over a trip
    needing two long days, allow=1 must extend ONLY the first (the second reverts to
    9 h and inserts its rest early); allow=2 extends both. A clamp above 2 must not
    grant a third extended day."""
    long_trip = dict(
        distance_km=1330.0, duration_s=1330.0 / 70.0 * 3600.0, start_soc=100.0,
        min_soc=0.0, payload_kg=5000.0, departure="2026-05-30T06:00",
        model_path=DEFAULT_MODEL_PATH,
    )
    one = route_planner.plan_route(allow_extended_days=1, **long_trip)["summary"]["driver"]
    ext_flags_one = [p["extended"] for p in one["perDay"]]
    assert ext_flags_one[0] is True                        # first long day extended
    assert ext_flags_one[1] is False                       # allowance spent -> 9 h cap
    two = route_planner.plan_route(allow_extended_days=2, **long_trip)["summary"]["driver"]
    assert sum(1 for p in two["perDay"] if p["extended"]) == 2  # both long days extended
    # Clamp: asking for 5 cannot extend more days than the 2/week statutory ceiling.
    clamped = route_planner.plan_route(allow_extended_days=5, **long_trip)["summary"]["driver"]
    assert sum(1 for p in clamped["perDay"] if p["extended"]) <= route_planner.EU561_MAX_EXT_DAYS_PER_WEEK


def test_extended_day_constants_match_eu561():
    """WHY: downstream code and the assumptions panel key off these constants; they
    must encode the statutory values (10 h extended cap, max 2 per week)."""
    assert route_planner.EU561_EXT_DAILY_MAX_DRIVE_H == 10.0
    assert route_planner.EU561_MAX_EXT_DAYS_PER_WEEK == 2


# --------------------------------------------------------------------------- #
# A2-4 — prior-week duty seed (mid-week driver closer to the 56 h weekly cap)
# --------------------------------------------------------------------------- #
# ~50 h of fresh driving (3500 km @ 70 km/h) — compliant on a fresh week.
_WEEK_TRIP = dict(
    distance_km=3500.0, duration_s=3500.0 / 70.0 * 3600.0, start_soc=100.0,
    min_soc=0.0, payload_kg=5000.0, departure="2026-05-30T06:00",
    model_path=DEFAULT_MODEL_PATH,
)


def test_prior_week_seed_default_is_unchanged():
    """WHY (regression-safe): default hours_already_driven_this_week=0.0 must assume a
    fresh week, leaving weeklyH and eu561ok exactly as before. A ~50 h trip is legal
    on a fresh week and must stay so."""
    fresh = route_planner.plan_route(**_WEEK_TRIP)["summary"]["driver"]
    assert fresh["weeklyH"] <= route_planner.EU561_WEEKLY_MAX_DRIVE_H + 1e-6
    assert fresh["eu561ok"] is True


def test_prior_week_seed_pushes_driver_over_weekly_cap():
    """WHY: the headline of A2-4 — a mid-week driver who already drove earlier this
    week is closer to the 56 h cap. Seeding 20 h of prior duty into the heaviest
    7-day window must push a ~50 h trip (legal fresh) over 56 h, flipping eu561ok to
    False. Without the seed the same trip is compliant — proving the seed is what
    moves the weekly judgement, not the trip alone."""
    fresh = route_planner.plan_route(**_WEEK_TRIP)["summary"]["driver"]
    seeded = route_planner.plan_route(
        hours_already_driven_this_week=20.0, **_WEEK_TRIP
    )["summary"]["driver"]
    assert seeded["weeklyH"] > fresh["weeklyH"]                          # seed raises the window
    assert seeded["weeklyH"] > route_planner.EU561_WEEKLY_MAX_DRIVE_H    # over the cap
    assert seeded["eu561ok"] is False                                   # and now non-compliant
    assert fresh["eu561ok"] is True                                     # vs legal fresh


def test_prior_week_seed_is_clamped_non_negative():
    """WHY (fail-safe): a negative prior-hours value is nonsensical and must not
    SUBTRACT from the weekly window (which would optimistically hide a real breach).
    A negative seed is clamped to 0 -> identical to a fresh week."""
    fresh = route_planner.plan_route(**_WEEK_TRIP)["summary"]["driver"]
    negative = route_planner.plan_route(
        hours_already_driven_this_week=-50.0, **_WEEK_TRIP
    )["summary"]["driver"]
    assert negative["weeklyH"] == pytest.approx(fresh["weeklyH"])
    assert negative["eu561ok"] == fresh["eu561ok"]


# --------------------------------------------------------------------------- #
# A2-5 — payload-drop placement on true ROAD distance (geometry mode)
# --------------------------------------------------------------------------- #
# A polyline that DETOURS far north between origin and the mid waypoint, then runs
# straight to the final. Straight-line legs put the mid stop at ~50% of the route;
# the actual road (the detour) only reaches it at ~88% of the driven distance.
_DETOUR_GEO = [[52.0, 13.0], [53.0, 13.1], [53.0, 13.4], [52.0, 13.5], [52.0, 14.0]]
_DETOUR_WPS = [
    {"lat": 52.0, "lng": 13.0, "label": "O"},
    {"lat": 52.0, "lng": 13.5, "label": "M", "dropWeightKg": 10000},
    {"lat": 52.0, "lng": 14.0, "label": "F"},
]


def _flat_enrichment_for(total_km, n=6):
    """A flat enrich_route() stub matching total_km, so plan_route enters geometry
    mode deterministically (the snap logic reads the raw polyline, not this)."""
    d = total_km / n
    segs, cum = [], 0.0
    for _ in range(n):
        cum += d
        segs.append({"distKm": d, "cumKm": round(cum, 3), "gradientPct": 0.0,
                     "elevM": 0.0, "temperatureC": 15.0, "windMps": 3.0})
    return {"segments": segs, "elevationProfile": [{"distKm": 0.0, "elevM": 0.0}],
            "conditions": {"avgTempC": 15.0, "avgWindMps": 3.0, "windDirDeg": 0.0,
                           "maxGradientPct": 0.0, "climbM": 0.0, "descentM": 0.0}}


def test_snap_km_on_geometry_uses_along_polyline_arc_length():
    """WHY (unit): the helper must return the ALONG-polyline arc length to the snapped
    waypoint, not the straight-line distance. The mid waypoint sits at the straight
    midpoint (great-circle frac 0.5) but, because the road detours north before
    reaching it, its road arc length is a much larger fraction of the polyline. If
    the snap returned the chord it would land the drop on the wrong half of the
    route."""
    M = (52.0, 13.5)
    arc, total = route_planner._snap_km_on_geometry(M, _DETOUR_GEO)
    assert total > 0
    # The mid stop is reached LATE along the road (>75%), not at the chord's 50%.
    assert arc / total > 0.75
    # Final waypoint snaps to the polyline end (arc == total).
    arc_f, total_f = route_planner._snap_km_on_geometry((52.0, 14.0), _DETOUR_GEO)
    assert arc_f == pytest.approx(total_f)


def test_payload_drop_placed_on_road_distance_with_geometry(monkeypatch):
    """WHY (the A2-5 headline): a payload drop must be placed where the truck actually
    REACHES the stop on the road, not where the straight origin->stop line guesses.
    On a detouring route the road reaches the mid stop near the route's end; the
    great-circle fallback places it at the midpoint. Same waypoints, same total km —
    only the presence of geometry differs — must move the drop's distKm
    substantially (here ~150 km -> ~263 km). This is the load-bearing guard that the
    snap-to-road arc length actually reaches _build_stops."""
    total_km = 300.0
    monkeypatch.setattr(
        route_planner.geodata, "enrich_route",
        lambda geometry, departure_iso=None, leg_timings=None: _flat_enrichment_for(total_km),
    )
    common = dict(distance_km=total_km, duration_s=total_km / 70.0 * 3600.0,
                  start_soc=100.0, min_soc=0.0, payload_kg=20000.0,
                  waypoints=_DETOUR_WPS, model_path=DEFAULT_MODEL_PATH)
    with_geo = route_planner.plan_route(geometry=_DETOUR_GEO, **common)
    without_geo = route_planner.plan_route(**common)  # great-circle fallback

    geo_mid = with_geo["stops"][0]["distKm"]
    chord_mid = without_geo["stops"][0]["distKm"]
    # Great-circle fallback unchanged: mid at ~half the route.
    assert chord_mid == pytest.approx(150.0, abs=2.0)
    # Road distance moves the drop far down-route (detour reached late).
    assert geo_mid > chord_mid + 50.0
    assert geo_mid > 0.75 * total_km
    # Final stop still lands exactly at route end in both modes.
    assert with_geo["stops"][-1]["distKm"] == pytest.approx(total_km, abs=0.1)


def test_payload_drop_no_geometry_is_byte_identical():
    """WHY (regression-safe): without geometry, stop placement must be the legacy
    scaled-great-circle estimate, untouched by A2-5. A symmetric O-M-F line must put
    the mid stop at exactly half the route, as before."""
    total_km = 400.0
    plan = route_planner.plan_route(
        distance_km=total_km, duration_s=total_km / 70.0 * 3600.0, start_soc=100.0,
        min_soc=0.0, payload_kg=20000.0, waypoints=_DETOUR_WPS,
        model_path=DEFAULT_MODEL_PATH,
    )
    assert plan["stops"][0]["distKm"] == pytest.approx(200.0, abs=2.0)
    assert plan["stops"][-1]["distKm"] == pytest.approx(total_km, abs=0.1)
