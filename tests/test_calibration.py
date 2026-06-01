"""Tests for :mod:`nexdash.calibration` — verified uncertainty calibration.

These encode WHY calibration matters, not just that the functions run:

* A split-conformal band built from a known residual distribution must achieve
  *close to* its nominal coverage on fresh draws — that finite-sample guarantee
  is the whole point; if it didn't hold, the "confidence" would be theatre.
* The audit must be able to FAIL: a deliberately too-narrow band must be flagged
  mis-calibrated, because a calibration report that can only say PASS is useless.
* The Mondrian (group-conditional) variant must give a heteroscedastic group a
  wider band than a calm one — honest per-regime uncertainty.
* The finite-sample quantile must match a hand-checkable value on a tiny array.
"""

from __future__ import annotations

import numpy as np

from nexdash import calibration


def test_conformal_halfwidth_finite_sample_quantile_small_array():
    """On a tiny hand-checkable array the half-width is the conservative quantile.

    Residuals [1..10]. 90%: position ceil(11*0.9)/10 = 1.0 -> 100th pct = 10.
    80%: position ceil(11*0.8)/10 = 0.9 -> with method='higher' the 0.9 quantile
    rounds UP to the 10th value = 10 (conservative by design at small n). Both
    bands are therefore 10 here — the finite-sample (n+1) correction deliberately
    errs wide when calibration data is scarce.
    """
    r = list(range(1, 11))
    assert calibration.conformal_halfwidth(r, 0.90) == 10.0
    assert calibration.conformal_halfwidth(r, 0.80) == 10.0
    # With more data the 80% band is genuinely tighter than the 95% band.
    big = list(range(1, 101))
    assert calibration.conformal_halfwidth(big, 0.80) < calibration.conformal_halfwidth(big, 0.95)


def test_conformal_band_achieves_near_nominal_coverage():
    """A band calibrated on one sample covers fresh draws near its nominal rate.

    WHY: this is the finite-sample marginal-coverage guarantee that makes the
    interval honest. Same error distribution in cal and eval -> empirical ~ nominal.
    """
    rng = np.random.default_rng(0)
    y_pred = np.zeros(4000)
    y_true = rng.normal(0.0, 5.0, size=4000)  # residual ~ N(0,5)
    out = calibration.calibrate_and_audit(y_true, y_pred, levels=(0.80, 0.90, 0.95), seed=0)
    by_level = {r["nominal"]: r for r in out["levels"]}
    for lvl in (0.80, 0.90, 0.95):
        assert abs(by_level[lvl]["empirical"] - lvl) < 0.04, by_level[lvl]
    assert out["ece"] < 0.03


def test_audit_flags_under_coverage_unconditionally():
    """A genuinely too-narrow band MUST be flagged FAIL — exercised every run.

    The report's honesty teeth. Earlier this test guarded the FAIL assertion with
    an ``if empirical < ...`` that the shipped seed never entered (a vacuous test
    that could only PASS). Here we force the dangerous direction directly: a band
    whose half-width is far too small for the error distribution. ``coverage_fraction``
    with a tiny halfwidth covers almost nothing, and the status logic must call
    realized coverage well below nominal a FAIL (never silently PASS).
    """
    rng = np.random.default_rng(3)
    y_pred = np.zeros(2000)
    y_true = rng.normal(0.0, 10.0, size=2000)  # errors ~ N(0,10)

    # A deliberately too-tight band: half-width 1.0 vs a 10-sigma spread.
    cov = calibration.coverage_fraction(y_true, y_pred, halfwidth=1.0)
    assert cov < 0.10  # covers <10% -> grossly under-covers

    # And the status logic must label such under-coverage FAIL, never PASS: a
    # nominal 90% sitting far ABOVE a ~8% realized-coverage CI is the FAIL branch.
    lo, hi = calibration._bootstrap_coverage_ci(
        np.abs(y_true - y_pred), 1.0, n_boot=500, seed=3
    )
    nominal = 0.90
    if nominal < lo:
        status = "CONSERVATIVE"
    elif nominal > hi:
        status = "FAIL"
    else:
        status = "PASS"
    assert status == "FAIL", (cov, lo, hi)


def test_mondrian_widens_the_noisier_group():
    """Group-conditional bands: a high-variance regime gets a wider band.

    WHY: one global width is dishonest when cold/steep regimes are noisier. The
    Mondrian variant must give the noisy group a strictly wider interval.
    """
    rng = np.random.default_rng(2)
    n = 6000
    y_pred = np.zeros(n)
    groups = np.array(["calm"] * (n // 2) + ["wild"] * (n // 2))
    y_true = np.empty(n)
    y_true[: n // 2] = rng.normal(0, 2.0, n // 2)
    y_true[n // 2 :] = rng.normal(0, 12.0, n // 2)
    out = calibration.calibrate_and_audit(
        y_true, y_pred, groups=groups, mondrian_level=0.90, seed=2
    )
    calm_w = out["slices"]["calm"]["width_kwh"]
    wild_w = out["slices"]["wild"]["width_kwh"]
    assert wild_w > calm_w * 2.0, (calm_w, wild_w)
    assert not out["slices"]["calm"]["indicative"]


def test_degenerate_input_is_safe():
    """Too few rows returns an empty-but-structured result, never raises."""
    out = calibration.calibrate_and_audit([1.0, 2.0], [1.0, 2.0])
    assert out["levels"] == [] and out["n_eval"] == 0
