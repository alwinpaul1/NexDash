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
import pandas as pd

import pytest

from nexdash.data_gen import generate_dataset
from nexdash.drift import (
    KS_D_MIN,
    drift_report,
    feature_drift,
    psi,
    _escalate,
)
from nexdash.model import train_model

scipy = pytest.importorskip("scipy")  # KS escalation tests need scipy present


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


# --- KS effect-size escalation (B5) -----------------------------------------
#
# WHY these exist: a two-sample KS p-value collapses toward 0 at large n even on
# operationally trivial shifts. Wiring that p-value straight into the verdict is a
# known over-alert pathology. The contract is that KS escalates the tier ONLY via
# *effect size* (statistic D), by at most one bounded step, and can never originate
# an alert. These tests pin that contract so it can't silently regress to p-value
# gating.


def test_ks_over_alert_guard_large_n_tiny_shift_stays_stable():
    """Huge n + a microscopic mean shift => significant p but tiny D => no escalation.

    This is THE pathology the design exists to prevent: with n in the hundreds of
    thousands, ks_2samp returns p well below 0.01 for a shift far too small to act
    on. PSI reads the feature stable; because the effect size D stays under KS_D_MIN,
    the bounded escalation must refuse to bump it, leaving the tier stable. If this
    ever flips to 'watch'/'drift', the verdict has regressed to trusting the
    n-inflated p-value.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    ref = pd.DataFrame({"distance_km": rng.normal(100.0, 10.0, n)})
    # Shift the mean by 0.2 of a unit on a std of 10 — a ~2% nudge, far below any
    # operationally meaningful move (D ~ 0.009), yet at n=200k it is wildly
    # "significant" (p ~ 4e-8). The classic large-n KS over-alert.
    cur = pd.DataFrame({"distance_km": rng.normal(100.2, 10.0, n)})

    fd = feature_drift(ref, cur, feature_columns=["distance_km"])
    info = fd["distance_km"]

    assert info["psi_tier"] == "stable"
    assert info["ks_pvalue"] is not None and info["ks_pvalue"] < 0.01, (
        "precondition: large-n shift is 'statistically significant'"
    )
    assert info["ks_stat"] < KS_D_MIN, (
        "precondition: the effect size is trivially small"
    )
    assert info["tier"] == "stable", "tiny-but-significant shift must NOT escalate"


def test_ks_shape_drift_same_mean_widened_variance_escalates_one_step():
    """Same mean, widened std => D clears KS_D_MIN => PSI tier bumped one step up.

    WHY: a variance-only change is a genuine distribution-shape shift (the marginal
    mean is unchanged, so a mean/PSI-light glance can under-rate it) and KS's CDF-gap
    statistic D catches it. The contract under test is that when D > KS_D_MIN on a
    real shift, the verdict is lifted by EXACTLY ONE bounded step over what PSI alone
    said — KS *confirming and sharpening* a forming alert, never an unbounded jump.

    On this monitor PSI is sensitive enough that a fully doubled variance already
    reads 'drift' on its own (KS would then have nothing to add — it is bounded at the
    top). To isolate a genuine KS-driven escalation we widen the std by 1.5x at a
    large n: PSI lands at 'watch', D crosses the effect-size floor, and the bounded
    escalation must carry it exactly one step to 'drift'. If the escalation ever
    skipped a step or failed to fire on this real shift, this test breaks.
    """
    rng = np.random.default_rng(1)
    n = 50_000
    ref = pd.DataFrame({"distance_km": rng.normal(100.0, 10.0, n)})
    cur = pd.DataFrame({"distance_km": rng.normal(100.0, 15.0, n)})  # 1.5x std, same mean

    fd = feature_drift(ref, cur, feature_columns=["distance_km"])
    info = fd["distance_km"]

    assert info["ks_stat"] is not None and info["ks_stat"] > KS_D_MIN, (
        "precondition: the shape shift is a real, large effect size"
    )
    assert info["ks_pvalue"] < 0.01
    order = ["stable", "watch", "drift"]
    base = info["psi_tier"]
    final = info["tier"]
    # PSI alone must be strictly below the top tier so the escalation is observable.
    assert order.index(base) < order.index("drift"), (
        "precondition: PSI base tier leaves room for a +1 KS escalation"
    )
    # Exactly one bounded step up from the PSI base tier (and never past 'drift').
    assert order.index(final) == min(order.index(base) + 1, len(order) - 1)
    assert order.index(final) - order.index(base) == 1


def test_ks_identical_samples_never_escalate():
    """Identical reference/current => D == 0 => escalation is a no-op at every tier."""
    df = generate_dataset(n_samples=2000, seed=7)
    fd = feature_drift(df, df)
    for feat, info in fd.items():
        assert info["tier"] == info["psi_tier"], (
            f"{feat}: identical data must not trigger any KS escalation"
        )

    # And the primitive itself: D=0, p=1 cannot move any base tier.
    assert _escalate("stable", 0.0, 1.0, n=10_000) == "stable"
    assert _escalate("watch", 0.0, 1.0, n=10_000) == "watch"
    assert _escalate("drift", 0.0, 1.0, n=10_000) == "drift"
    # Bounded at the top: a real shift on an already-'drift' feature stays 'drift'.
    assert _escalate("drift", 0.9, 0.0001, n=10_000) == "drift"
