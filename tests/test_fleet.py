"""Tests for :mod:`nexdash.fleet` -- the model-driven mock fleet roster.

These tests verify operational *intent*, not just plumbing:

* The roster must be deterministic and expose the exact contract the
  dispatcher console reads (id/name/lat/lng/soc/status/nextStop), so the UI
  can rely on its shape across renders.
* The model-derived verdict (``atRisk`` / ``marginKwh`` / ``remainingSocPct``)
  must genuinely come from :func:`nexdash.range.check_reachability`: a near-
  empty heavy truck on a long leg must be flagged at risk, and an obviously
  comfortable leg must not be -- because that flag is the whole point of the
  console.
* The fleet must fail soft when the model artifact is missing (verdict fields
  become ``None``), since the roster still has to render for the operator.
* ``model_info`` must surface numeric headline metrics, because the console
  shows them to justify trusting the model.

A small but real model is trained to a temp artifact once per module so the
tests exercise the genuine predict -> reachability pipeline.
"""

from __future__ import annotations

import pytest

from nexdash import fleet as fleet_module
from nexdash.data_gen import generate_dataset
from nexdash.model import train_model

# The roster contract every truck must satisfy for the console.
_BASE_KEYS = {"id", "name", "lat", "lng", "soc", "status", "nextStop"}
_NEXTSTOP_KEYS = {"label", "distanceKm", "payloadT", "temperatureC"}
_VERDICT_KEYS = {"reachable", "marginKwh", "remainingSocPct", "atRisk"}
_VALID_STATUSES = {"in_transit", "available", "charging", "maintenance"}
_DRIVING = {"in_transit", "available"}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a small deterministic model and persist it to a temp file."""
    path = tmp_path_factory.mktemp("models") / "energy_model.joblib"
    df = generate_dataset(n_samples=800, seed=42)
    train_model(df, save=True, path=path)
    return path


def test_roster_contract_and_statuses(model_path):
    """Every truck must expose the documented fields with valid status/nextStop.

    The console maps each of these fields directly onto the map and table; a
    missing or mistyped field would silently break a card.
    """
    trucks = fleet_module.fleet_status(model_path=model_path)
    assert len(trucks) >= 10

    ids = [t["id"] for t in trucks]
    assert len(ids) == len(set(ids)), "truck ids must be unique"

    for t in trucks:
        assert _BASE_KEYS.issubset(t.keys())
        assert _VERDICT_KEYS.issubset(t.keys())
        assert t["status"] in _VALID_STATUSES
        assert isinstance(t["lat"], (int, float))
        assert isinstance(t["lng"], (int, float))
        assert 0.0 <= t["soc"] <= 100.0
        assert _NEXTSTOP_KEYS.issubset(t["nextStop"].keys())


def test_deterministic(model_path):
    """Two calls must return identical rosters -- the console must not flicker."""
    a = fleet_module.fleet_status(model_path=model_path)
    b = fleet_module.fleet_status(model_path=model_path)
    assert a == b


def test_driving_trucks_have_model_verdict(model_path):
    """Driving trucks must carry real model output; parked trucks must not.

    The reachability verdict only makes sense for a truck actually on a leg, so
    charging/maintenance trucks expose ``None`` verdicts and are never at risk.
    """
    trucks = fleet_module.fleet_status(model_path=model_path)
    for t in trucks:
        if t["status"] in _DRIVING:
            assert isinstance(t["reachable"], bool)
            assert isinstance(t["marginKwh"], (int, float))
            assert isinstance(t["remainingSocPct"], (int, float))
            assert isinstance(t["atRisk"], bool)
        else:
            assert t["reachable"] is None
            assert t["marginKwh"] is None
            assert t["remainingSocPct"] is None
            assert t["atRisk"] is False


def test_at_risk_matches_model_rule(model_path):
    """``atRisk`` must equal the documented model rule for every driving truck.

    at_risk == (not reachable) OR (margin < RISK_MARGIN_KWH). If this drifted
    from the model output the console's red flag would lie to the dispatcher.
    """
    trucks = fleet_module.fleet_status(model_path=model_path)
    driving = [t for t in trucks if t["status"] in _DRIVING]
    assert driving, "fixture must contain at least one driving truck"

    for t in driving:
        expected = (not t["reachable"]) or (
            t["marginKwh"] < fleet_module.RISK_MARGIN_KWH
        )
        assert t["atRisk"] is expected

    # The roster is designed to surface at least one genuinely at-risk truck
    # (low SOC + heavy + long leg) so the console has something to alert on.
    assert any(t["atRisk"] for t in driving)
    # ...and at least one comfortable truck that is NOT at risk.
    assert any(not t["atRisk"] for t in driving)


def test_counts_consistent_with_roster(model_path):
    """Status + at-risk counts derived in the API must match the roster itself."""
    trucks = fleet_module.fleet_status(model_path=model_path)
    counts = {
        "inTransit": sum(1 for t in trucks if t["status"] == "in_transit"),
        "available": sum(1 for t in trucks if t["status"] == "available"),
        "charging": sum(1 for t in trucks if t["status"] == "charging"),
        "maintenance": sum(1 for t in trucks if t["status"] == "maintenance"),
        "atRisk": sum(1 for t in trucks if t["atRisk"] is True),
    }
    assert (
        counts["inTransit"]
        + counts["available"]
        + counts["charging"]
        + counts["maintenance"]
        == len(trucks)
    )
    # at-risk is a subset of the driving trucks.
    n_driving = counts["inTransit"] + counts["available"]
    assert 0 <= counts["atRisk"] <= n_driving


def test_fails_soft_without_model(tmp_path):
    """A missing model artifact must null the verdicts, not raise.

    The roster still has to render for the dispatcher even before the pipeline
    has been run, so the model-derived fields collapse to ``None``.
    """
    missing = tmp_path / "does_not_exist.joblib"
    trucks = fleet_module.fleet_status(model_path=missing)
    assert len(trucks) >= 10
    for t in trucks:
        if t["status"] in _DRIVING:
            assert t["reachable"] is None
            assert t["marginKwh"] is None
            assert t["remainingSocPct"] is None
            assert t["atRisk"] is None
        else:
            assert t["atRisk"] is False


def test_model_info_numeric_metrics(model_path):
    """``model_info`` must return numeric headline metrics from the artifact.

    The console displays these to justify trusting the model, so they must be
    real numbers, not nulls, when an artifact is present.
    """
    info = fleet_module.model_info(model_path=model_path)
    assert set(info.keys()) == {
        "mae_kwh",
        "rmse_kwh",
        "mape_pct",
        "r2",
        "pct_range_error",
    }
    for key in ("mae_kwh", "rmse_kwh", "mape_pct", "r2", "pct_range_error"):
        assert isinstance(info[key], float), f"{key} must be numeric"
        assert info[key] >= 0.0

    # pct_range_error is MAE as a fraction of a full charge -- must be consistent.
    from nexdash.config import TRUCK

    expected = info["mae_kwh"] / TRUCK.battery_kwh * 100.0
    assert info["pct_range_error"] == pytest.approx(expected, abs=1e-2)


def test_model_info_fails_soft_to_nulls(tmp_path, monkeypatch):
    """With no artifact AND no report, metrics must degrade to nulls, not raise."""
    missing = tmp_path / "does_not_exist.joblib"
    # Point the report lookup at an empty temp dir so the fallback also misses.
    monkeypatch.setattr(fleet_module, "REPORTS_DIR", tmp_path)
    info = fleet_module.model_info(model_path=missing)
    assert info == {
        "mae_kwh": None,
        "rmse_kwh": None,
        "mape_pct": None,
        "r2": None,
        "pct_range_error": None,
    }
