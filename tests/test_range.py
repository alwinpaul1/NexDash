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


def test_out_of_envelope_segment_flags_low_confidence(model_path):
    """A physically implausible segment must trip the physics sanity-clamp.

    A +4.5% grade sustained over 110 km implies a ~5 km net climb — it cannot
    occur in the (physically-coupled) training data, so the model extrapolates
    and *under*-predicts. The decision must NOT use that optimistic number: the
    cross-check has to flag low confidence and fall back to the conservative
    physics estimate, or the tool would green-light a trip that strands a truck.
    This test fails if the clamp is removed or quietly trusts the model.
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
    assert result["confidence"] == "low"
    # Model under-predicts vs first-principles here, and the decision uses the
    # conservative (higher) value — never the optimistic one.
    assert result["physics_kwh"] > result["model_kwh"]
    assert result["energy_needed_kwh"] == pytest.approx(result["physics_kwh"], rel=1e-6)


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
