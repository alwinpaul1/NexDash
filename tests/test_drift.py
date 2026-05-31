"""Tests for :mod:`nexdash.drift` — data/concept drift detection.

These encode WHY drift detection exists, not just that it returns numbers:

* The same distribution against itself must read ``stable`` with PSI ~ 0 — a
  monitor that cried wolf on identical data would be useless.
* A deliberately cold-shifted German-winter batch must trip ``drift`` on the
  temperature feature — the exact real-world shift (a harsh winter) the brief's
  long-term section says we must notice.
* When the batch carries true labels, the residual monitor must report realized
  error, since concept drift can hide behind stable-looking inputs.
"""

from __future__ import annotations

import numpy as np

from nexdash.data_gen import generate_dataset
from nexdash.drift import drift_report, feature_drift, psi
from nexdash.model import train_model


def test_same_distribution_is_stable():
    """Reference vs itself: PSI ~ 0 on every feature, overall stable."""
    df = generate_dataset(n_samples=3000, seed=42)
    report = drift_report(df, df)
    assert report["overall"] == "stable"
    for feat, info in report["features"].items():
        assert info["psi"] < 0.1, f"{feat} PSI should be ~0 on identical data"
    assert report["drifted_features"] == []


def test_psi_zero_on_identical_arrays():
    """The PSI primitive must be ~0 when the two samples are identical."""
    x = generate_dataset(n_samples=1000, seed=1)["temperature_c"].to_numpy()
    assert psi(x, x) < 1e-6


def test_cold_shifted_batch_trips_temperature_drift():
    """A 25 C cold shift (a harsh winter) must register as drift on temperature.

    WHY: the long-term deliverable is *noticing when reality has moved*. A fleet
    operating through a colder-than-trained winter is the canonical input-drift
    case; the monitor must flag the temperature feature, not silently keep
    serving a model trained on milder data.
    """
    ref = generate_dataset(n_samples=3000, seed=42)
    cur = ref.copy()
    cur["temperature_c"] = cur["temperature_c"] - 25.0  # whole-fleet cold shift

    report = drift_report(ref, cur)
    assert report["overall"] == "drift"
    assert "temperature_c" in report["drifted_features"]
    assert report["features"]["temperature_c"]["psi"] > 0.25
    # KS, when scipy is present, should also see the shift (tiny p-value).
    ks_p = report["features"]["temperature_c"]["ks_pvalue"]
    assert ks_p is None or ks_p < 0.01


def test_residual_monitor_present_for_labeled_batch():
    """A labelled batch must yield a realized-error residual monitor."""
    df = generate_dataset(n_samples=1200, seed=42)
    model = train_model(df, save=False)
    report = drift_report(df, df, model=model)
    assert report["residuals"] is not None
    assert report["residuals"]["n"] == len(df)
    assert report["residuals"]["realized_mae_kwh"] >= 0.0
    assert 0.0 <= report["residuals"]["optimistic_rate"] <= 1.0


def test_unlabeled_batch_has_no_residual_monitor():
    """Without the target column the residual monitor is absent (not faked)."""
    df = generate_dataset(n_samples=500, seed=3)
    model = train_model(df, save=False)
    batch = df.drop(columns=["energy_kwh"])
    report = drift_report(df, batch, model=model)
    assert report["residuals"] is None
    # Feature drift still computable on the inputs alone.
    fd = feature_drift(df, batch)
    assert "distance_km" in fd
