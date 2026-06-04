"""Tests for :mod:`nexdash.range` reachability reasoning.

These tests verify operational *intent*, not just numeric plumbing:

* A short hop on a nearly full battery must be flagged as reachable with
  energy to spare -- a dispatcher would never strand such a truck.
* A near-empty truck asked to drive a very long leg must be flagged as
  *not* reachable, because that is precisely the failure the tool exists
  to prevent.
* The safety reserve must genuinely hold energy back: raising
  ``reserve_pct`` must reduce usable energy (and never increase the margin).
* The returned contract must be complete and self-consistent so the
  dashboard / MCP layers can rely on it (all keys present, SOC bounded to
  a physically meaningful 0-100%).

A tiny model is trained and saved to a temporary path once per module so
the tests exercise the real predict -> reachability pipeline without
depending on a pre-built artifact on disk.
"""

from __future__ import annotations

import pytest

from nexdash.data_gen import generate_dataset
from nexdash.model import train_model, predict_energy
from nexdash.range import check_reachability

# Keys the reachability contract promises to every caller.
EXPECTED_KEYS = {
    "energy_needed_kwh",
    "energy_available_kwh",
    "usable_after_reserve_kwh",
    "reaches",
    "margin_kwh",
    "remaining_soc_pct",
    "remaining_range_km",
    "confidence",
    "model_kwh",
    "physics_kwh",
    "confidence_note",
}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a small deterministic model and persist it to a temp file.

    A reduced sample count keeps the fixture fast while still producing a
    fitted, physics-grounded model whose predictions are realistic enough
    for the reachability assertions below.
    """
    path = tmp_path_factory.mktemp("models") / "energy_model.joblib"
    df = generate_dataset(n_samples=800, seed=42)
    train_model(df, save=True, path=path)
    # Reset any cached default-path model so predictions resolve to *this*
    # artifact (predict_energy caches by resolved path, so this is mostly
    # defensive but keeps the test hermetic).
    return path


def test_short_trip_high_soc_reaches(model_path):
    """A short, easy leg at ~90% SOC must be reachable with positive margin.

    This is the canonical "obviously fine" case; if the tool flagged it as
    unreachable a dispatcher would lose all trust in it.
    """
    result = check_reachability(
        soc_pct=90.0,
        distance_km=20.0,
        payload_t=5.0,
        speed_kph=60.0,
        gradient_pct=0.0,
        temperature_c=20.0,
        model_path=model_path,
    )
    assert result["reaches"] is True
    assert result["margin_kwh"] > 0.0
    # The truck started near-full and barely used energy, so plenty remains.
    assert result["remaining_soc_pct"] > 80.0


def test_long_trip_low_soc_does_not_reach(model_path):
    """A very long, heavy, fast leg at low SOC must be flagged unreachable.

    This is the failure mode the tool exists to catch: predicted demand far
    exceeds the small amount of usable charge on board.
    """
    result = check_reachability(
        soc_pct=5.0,
        distance_km=120.0,
        payload_t=22.0,
        speed_kph=90.0,
        gradient_pct=5.0,
        temperature_c=-15.0,
        model_path=model_path,
    )
    assert result["reaches"] is False
    assert result["margin_kwh"] < 0.0
    # Cannot finish the leg, so no spare range should be promised.
    assert result["remaining_range_km"] == 0.0


def test_reserve_reduces_usable_energy(model_path):
    """A larger reserve must hold back more energy and never help the margin.

    Energy available on board is identical between the two calls (same SOC),
    so the only difference is how much the reserve withholds. The larger
    reserve must therefore yield strictly less usable energy and a margin
    that is no better (and here strictly worse).
    """
    common = dict(
        soc_pct=60.0,
        distance_km=50.0,
        payload_t=10.0,
        speed_kph=70.0,
        gradient_pct=1.0,
        temperature_c=10.0,
        model_path=model_path,
    )
    low_reserve = check_reachability(**common, reserve_pct=5.0)
    high_reserve = check_reachability(**common, reserve_pct=25.0)

    # Same SOC -> same energy on board.
    assert low_reserve["energy_available_kwh"] == high_reserve["energy_available_kwh"]
    # Bigger reserve withholds more, so less usable energy remains.
    assert (
        high_reserve["usable_after_reserve_kwh"]
        < low_reserve["usable_after_reserve_kwh"]
    )
    # A bigger reserve can only shrink (never grow) the safety margin.
    assert high_reserve["margin_kwh"] < low_reserve["margin_kwh"]


def test_output_contract_keys_and_bounds(model_path):
    """The result must expose every contract key with sane types/bounds.

    Downstream consumers (FastAPI endpoint, MCP tools) depend on this exact
    shape, and remaining SOC must always be a physically valid percentage.
    """
    result = check_reachability(
        soc_pct=50.0,
        distance_km=40.0,
        payload_t=12.0,
        speed_kph=65.0,
        gradient_pct=2.0,
        temperature_c=15.0,
        wind_mps=4.0,
        model_path=model_path,
    )

    assert set(result.keys()) == EXPECTED_KEYS
    assert isinstance(result["reaches"], bool)
    assert isinstance(result["confidence_note"], str) and result["confidence_note"]
    assert 0.0 <= result["remaining_soc_pct"] <= 100.0
    # Energy figures are non-negative quantities of charge on board / needed.
    assert result["energy_available_kwh"] >= 0.0


def test_out_of_envelope_segment_uses_conservative_max(model_path):
    """An out-of-envelope steep climb must never quote LESS than physics.

    A +4.5% grade sustained over 110 km implies a ~5 km net climb — outside the
    (physically-coupled) training envelope. The OLD raw-kWh model *saturated* and
    badly under-predicted here, so the cross-check had to flag low confidence and
    fall back to physics. The energy model now learns the PHYSICS RESIDUAL
    (`nexdash.model`), so it TRACKS physics on this leg instead of saturating —
    model and physics agree to within a few percent, the divergence guard
    correctly does NOT fire, and confidence is high. That is the fix working.

    The safety contract is unchanged and still asserted here: the decision uses
    the conservative ``max(model, physics)`` and can never quote less than
    physics. We do NOT assert the model under-predicts anymore — asserting the old
    saturation would lock in the very bug this change removed.
    """
    result = check_reachability(
        soc_pct=80.0,
        distance_km=110.0,
        payload_t=22.0,
        speed_kph=70.0,
        gradient_pct=4.5,
        temperature_c=-12.0,
        model_path=model_path,
    )
    # The conservative max guard is intact: energy_needed is never below physics.
    assert result["energy_needed_kwh"] == pytest.approx(
        max(result["model_kwh"], result["physics_kwh"]), rel=1e-6
    )
    assert result["energy_needed_kwh"] >= result["physics_kwh"] - 1e-6
    # The residual model now tracks physics here (no saturation): the two agree
    # well inside the divergence band, so confidence is high.
    assert result["confidence"] == "high"
    assert abs(result["model_kwh"] - result["physics_kwh"]) <= 0.10 * abs(
        result["physics_kwh"]
    )


def test_in_envelope_segment_is_high_confidence(model_path):
    """A normal segment inside the envelope: model and physics agree, so
    confidence is high and the model's own data-driven prediction is used."""
    result = check_reachability(
        soc_pct=80.0,
        distance_km=120.0,
        payload_t=12.0,
        speed_kph=80.0,
        gradient_pct=0.5,
        temperature_c=2.0,
        model_path=model_path,
    )
    assert result["confidence"] == "high"
    assert result["energy_needed_kwh"] == pytest.approx(result["model_kwh"], rel=1e-6)
    assert result["energy_needed_kwh"] >= 0.0
    assert result["remaining_range_km"] >= 0.0
    # Margin must be internally consistent with the reported sub-values.
    assert result["margin_kwh"] == pytest.approx(
        result["usable_after_reserve_kwh"] - result["energy_needed_kwh"], abs=1e-2
    )


def test_out_of_range_soc_and_reserve_are_clamped(model_path):
    """SOC > 100 and a negative reserve must be clamped, not trusted.

    WHY: a bad sensor reading (SOC 150) or an LLM-supplied negative reserve would
    otherwise INFLATE the usable energy and produce an unsafely optimistic
    "reaches" verdict. SOC clamps to 100 (available == full battery) and a
    negative reserve clamps to 0 (usable never exceeds what's on board).
    """
    from nexdash.config import TRUCK

    r = check_reachability(
        soc_pct=150.0,
        distance_km=50.0,
        payload_t=10.0,
        speed_kph=70.0,
        gradient_pct=1.0,
        temperature_c=10.0,
        reserve_pct=-20.0,
        model_path=model_path,
    )
    # SOC clamped to 100 -> available is exactly the usable battery, not 1.5x it.
    assert r["energy_available_kwh"] == pytest.approx(TRUCK.battery_kwh)
    # Negative reserve clamped to 0 -> usable == available (never more).
    assert r["usable_after_reserve_kwh"] == pytest.approx(r["energy_available_kwh"])
    assert r["usable_after_reserve_kwh"] <= TRUCK.battery_kwh + 1e-9


def test_confidence_note_names_the_estimate_actually_used(model_path):
    """The note must name whichever estimate actually drove the decision.

    WHY: a previous note always claimed "the conservative physics value is used",
    which is false when the model value drove the decision. The note tag must
    always match the value in ``energy_needed_kwh`` (which is the conservative
    ``max(model, physics)``). Since the physics-residual model now tracks physics
    on this formerly out-of-envelope steep climb, confidence is high and the model
    value is used — so the note must NOT claim physics drove it. We assert the tag
    matches the value used regardless of which branch is taken, so the test stays
    honest whether or not the guard fires.
    """
    r = check_reachability(
        soc_pct=80.0,
        distance_km=110.0,
        payload_t=22.0,
        speed_kph=70.0,
        gradient_pct=4.5,
        temperature_c=-12.0,
        model_path=model_path,
    )
    # The decision must use the conservative (higher) of the two estimates.
    assert r["energy_needed_kwh"] == pytest.approx(max(r["model_kwh"], r["physics_kwh"]))
    # The note tag must match the estimate actually used (no dishonest claim).
    used_physics = r["energy_needed_kwh"] == pytest.approx(r["physics_kwh"])
    if r["confidence"] == "low":
        expected_tag = "(physics)" if used_physics else "(the model)"
        assert expected_tag in r["confidence_note"]
    else:
        # High confidence: the model value is used; the note must not claim the
        # conservative physics fallback drove the number.
        assert "(physics)" not in r["confidence_note"]


def test_routine_descent_is_not_falsely_flagged_out_of_envelope(model_path):
    """A normal downhill leg must stay HIGH confidence, not be called OOD.

    WHY: the physics cross-check exists to catch the model UNDER-predicting on
    steep climbs (the dangerous, optimistic direction). On a descent the
    first-principles estimate can go *negative* (net regen), and a symmetric
    abs() test — with a ``0.15 * physics`` band that collapses when physics is
    negative — used to flag every routine descent as "outside the envelope the
    model was trained on" and tell the dispatcher to keep a wide reserve. But a
    -4% grade is squarely inside the -6..+6% training range. The gate must be
    directional: model predicting MORE than physics (the safe direction) keeps
    high confidence and quotes the model, and the note must not lie about the
    envelope. This fails if the symmetric/absolute gate is reintroduced.
    """
    r = check_reachability(
        soc_pct=70.0,
        distance_km=80.0,
        payload_t=10.0,
        speed_kph=75.0,
        gradient_pct=-4.0,
        temperature_c=8.0,
        model_path=model_path,
    )
    assert r["confidence"] == "high"
    assert "outside the envelope" not in r["confidence_note"]
    # On a descent the model (realistic) sits above the negative physics estimate;
    # the decision quotes the model, not the physics floor.
    assert r["energy_needed_kwh"] == pytest.approx(r["model_kwh"], rel=1e-6)


def test_descent_remaining_range_is_physically_bounded(model_path):
    """``remaining_range_km`` must never exceed the truck's nominal range.

    WHY: it used to extrapolate the *segment's* kWh/km forward, but a descent's
    near-zero (or net-regen negative) rate cannot be sustained — a truck cannot
    descend forever — so the figure exploded to 1,000-1,800 km for a 600 kWh /
    ~500 km truck, an obviously impossible operator-facing number. Flooring the
    consumption rate at the nominal flat rate keeps it physical. This fails if
    the consumption floor is removed.
    """
    from nexdash.config import TRUCK

    r = check_reachability(
        soc_pct=70.0,
        distance_km=80.0,
        payload_t=10.0,
        speed_kph=75.0,
        gradient_pct=-4.0,
        temperature_c=8.0,
        model_path=model_path,
    )
    assert 0.0 <= r["remaining_range_km"] <= TRUCK.nominal_range_km
