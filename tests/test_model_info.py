"""Tests for :mod:`nexdash.model_info` (headline metrics for the API).

* ``model_info`` must surface numeric headline metrics when a trained artifact
  exists, because the API exposes them so a dispatcher can judge how much to
  trust the model.
* It must degrade to nulls (never raise) when neither the artifact nor the
  report is available, so the endpoint stays fail-soft.
"""

from __future__ import annotations

import pytest

from nexdash import model_info as model_info_module
from nexdash.data_gen import generate_dataset
from nexdash.model import train_model


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a small deterministic model and persist it to a temp file."""
    path = tmp_path_factory.mktemp("models") / "energy_model.joblib"
    df = generate_dataset(n_samples=800, seed=42)
    train_model(df, save=True, path=path)
    return path


def test_model_info_numeric_metrics(model_path):
    """``model_info`` must return numeric headline metrics from the artifact.

    The API displays these to justify trusting the model, so they must be real
    numbers, not nulls, when an artifact is present.
    """
    info = model_info_module.model_info(model_path=model_path)
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
    monkeypatch.setattr(model_info_module, "REPORTS_DIR", tmp_path)
    info = model_info_module.model_info(model_path=missing)
    assert info == {
        "mae_kwh": None,
        "rmse_kwh": None,
        "mape_pct": None,
        "r2": None,
        "pct_range_error": None,
    }
