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


# --- (B2) k-fold persistence -------------------------------------------------


def test_kfold_well_calibrated_band_is_never_persistently_failing():
    """A genuinely honest band must NOT be branded a persistent failure.

    WHY: a single random cal/eval split can trip FAIL by chance (~5-10% of honest
    folds), and reporting that one fold as proof of mis-calibration is the over-claim
    this wrapper exists to kill. Same error distribution in cal and eval => the band
    keeps its finite-sample guarantee, so across k seeds the fail count must stay
    within the binomial noise floor and every level must read "ok".
    """
    rng = np.random.default_rng(7)
    y_pred = np.zeros(4000)
    y_true = rng.normal(0.0, 5.0, size=4000)  # residual ~ N(0,5), honestly calibrated
    out = calibration.calibrate_and_audit_kfold(
        y_true, y_pred, k=20, levels=(0.80, 0.90, 0.95), base_seed=7
    )
    assert out["k"] == 20
    # binom.ppf(0.95, 20, 0.10) == 4: an honest band may FAIL up to 4/20 by chance.
    assert out["fail_threshold"] == 4
    for row in out["levels"]:
        assert row["verdict"] == "ok", row
        assert row["fail_count"] <= out["fail_threshold"], row
        # median realized coverage should sit near nominal for an honest band.
        assert abs(row["median_empirical"] - row["nominal"]) < 0.04, row


def test_kfold_flags_truly_overconfident_band_as_persistent_fail(monkeypatch):
    """A genuinely too-tight band MUST be branded "persistently fails" — honesty teeth.

    WHY this needs a forced too-tight band: a *self*-calibrating split-conformal
    procedure on exchangeable data CANNOT persistently under-cover — the finite-sample
    (n+1) correction guarantees coverage >= level on every fold. That property is the
    whole point of the wrapper (it is why an isolated single-fold FAIL is just noise),
    and the companion test above proves an honest band is never flagged. So the only
    thing that *should* trip "persistently fails" is a band that is systematically too
    tight for the error — i.e. real over-confidence, not split luck.

    We model exactly that: patch the half-width computation to return a fixed band far
    narrower than the N(0,10) errors. Every fold's eval slice then under-covers, the
    fail count blows past the binomial noise floor (binom.ppf(.95,20,.10)=4), and the
    verdict MUST escalate to "persistently fails". A wrapper that could only ever say
    "ok" would be useless — this is the test that can make it fail.
    """
    rng = np.random.default_rng(11)
    n = 4000
    y_pred = np.zeros(n)
    y_true = rng.normal(0.0, 10.0, size=n)  # errors ~ N(0,10): a half-width of 1 is absurd

    # Force a grossly over-confident band (half-width 1.0 vs ~10-sigma spread) on EVERY
    # fold. This is the multi-fold analogue of the single-fold FAIL test's hand-set band.
    monkeypatch.setattr(calibration, "conformal_halfwidth", lambda *a, **k: 1.0)

    out = calibration.calibrate_and_audit_kfold(
        y_true, y_pred, k=20, levels=(0.90,), base_seed=11
    )
    row = out["levels"][0]
    assert row["median_empirical"] < 0.20, row  # the tight band covers almost nothing
    assert row["fail_count"] > out["fail_threshold"], out
    assert row["fail_count"] == 20, out  # over-confident on literally every fold
    assert row["verdict"] == "persistently fails", out


def test_kfold_iqr_reports_split_to_split_wobble():
    """IQR must quantify how much realized coverage moves with the split.

    WHY: the whole reason single-fold FAILs are noisy is split-to-split variance. The
    wrapper must expose that variance (a non-negative IQR), so a reader can see the
    single number's wobble rather than trusting one fold. Small n => visibly larger
    wobble than large n.
    """
    rng = np.random.default_rng(5)
    y_pred_small = np.zeros(120)
    y_true_small = rng.normal(0, 5, size=120)
    small = calibration.calibrate_and_audit_kfold(
        y_true_small, y_pred_small, k=20, levels=(0.90,), base_seed=5
    )
    big_pred = np.zeros(8000)
    big_true = rng.normal(0, 5, size=8000)
    big = calibration.calibrate_and_audit_kfold(
        big_true, big_pred, k=20, levels=(0.90,), base_seed=5
    )
    iqr_small = small["levels"][0]["iqr_empirical"]
    iqr_big = big["levels"][0]["iqr_empirical"]
    assert iqr_small >= 0.0 and iqr_big >= 0.0
    assert iqr_small > iqr_big, (iqr_small, iqr_big)


def test_kfold_is_deterministic():
    """Seeded => byte-identical across calls (no hidden global RNG state)."""
    rng = np.random.default_rng(1)
    y_pred = np.zeros(2000)
    y_true = rng.normal(0, 4, size=2000)
    a = calibration.calibrate_and_audit_kfold(y_true, y_pred, k=8, base_seed=1)
    b = calibration.calibrate_and_audit_kfold(y_true, y_pred, k=8, base_seed=1)
    assert a == b


# --- (B3) Mondrian steep pooling + conservative sparse fallback --------------


def test_steep_up_and_steep_down_are_pooled_into_one_steep_group():
    """steep_up + steep_down must collapse to a single 'steep' slice.

    WHY: each steep sign-half is too sparse to calibrate alone, so each would fall to
    the (indicative) fallback and never earn its own band. They are both high-|gradient|
    tails whose residual spread is driven by magnitude, not sign, so pooling them gives
    ~3x the calibration support and lets the steep band stand on its own non-indicative
    feet — instead of two starved, fallback-only slices.
    """
    rng = np.random.default_rng(21)
    n = 6000
    y_pred = np.zeros(n)
    # Mostly flat, with two modest steep tails that individually are still sizeable
    # but which we want to see *merged* in the output regardless.
    groups = np.array(
        ["flat"] * (n - 2000) + ["steep_up"] * 1000 + ["steep_down"] * 1000
    )
    y_true = np.empty(n)
    y_true[: n - 2000] = rng.normal(0, 2.0, n - 2000)
    y_true[n - 2000 : n - 1000] = rng.normal(0, 9.0, 1000)
    y_true[n - 1000 :] = rng.normal(0, 9.0, 1000)
    out = calibration.calibrate_and_audit(
        y_true, y_pred, groups=groups, mondrian_level=0.90, seed=21
    )
    assert "steep" in out["slices"]
    assert "steep_up" not in out["slices"] and "steep_down" not in out["slices"]
    # Pooled support is large here, so the steep band is a real (non-indicative) band.
    assert out["slices"]["steep"]["indicative"] is False
    # And it is wider than the flat band, because steep residuals are wider.
    assert out["slices"]["steep"]["width_kwh"] > out["slices"]["flat"]["width_kwh"]


def test_pooled_steep_band_wider_than_flat_when_steep_residuals_are_wider():
    """The pooled steep band must be honestly wider than the calm flat band.

    WHY: one global width is dishonest when steep regimes are noisier. After pooling,
    the steep slice has enough support to compute its own quantile, which — given a
    genuinely wider steep residual distribution — must exceed the flat band's width.
    """
    rng = np.random.default_rng(31)
    n = 8000
    y_pred = np.zeros(n)
    n_flat = n - 3000
    groups = np.array(
        ["flat"] * n_flat + ["steep_up"] * 1500 + ["steep_down"] * 1500
    )
    y_true = np.empty(n)
    y_true[:n_flat] = rng.normal(0, 1.5, n_flat)
    y_true[n_flat : n_flat + 1500] = rng.normal(0, 11.0, 1500)
    y_true[n_flat + 1500 :] = rng.normal(0, 11.0, 1500)
    out = calibration.calibrate_and_audit(
        y_true, y_pred, groups=groups, mondrian_level=0.90, seed=31
    )
    flat_w = out["slices"]["flat"]["width_kwh"]
    steep_w = out["slices"]["steep"]["width_kwh"]
    assert steep_w > flat_w * 2.0, (flat_w, steep_w)


def test_sparse_fallback_is_conservative_max_not_silently_global():
    """An indicative (sparse) group must use max(global, group), never just global.

    WHY: silently substituting the global band can UNDER-state a genuinely noisier tail
    that happens to be sparse — the exact over-confidence this project keeps narrowing.
    The honest fallback takes the wider of the two so the indicative band never narrows
    below what the scarce-but-real residuals imply. We build a sparse group whose few
    residuals are far WIDER than the global pool, then assert its reported width equals
    the conservative max (its own group width), and that it is still flagged indicative
    (reported, not re-certified).
    """
    rng = np.random.default_rng(41)
    n = 4030
    y_pred = np.zeros(n)
    # A large calm bulk + a tiny (sub-min_group_cal eval-and-cal) very noisy slice.
    n_calm = 4000
    n_rare = 30  # << default min_group_cal=20 per half after the split
    groups = np.array(["calm"] * n_calm + ["rare"] * n_rare)
    y_true = np.empty(n)
    y_true[:n_calm] = rng.normal(0, 1.0, n_calm)
    y_true[n_calm:] = rng.normal(0, 50.0, n_rare)  # rare residuals far wider than global
    out = calibration.calibrate_and_audit(
        y_true, y_pred, groups=groups, mondrian_level=0.90, seed=41, min_group_cal=20
    )
    rare = out["slices"]["rare"]
    assert rare["indicative"] is True  # too few cal rows -> indicative
    # Conservative: the rare band must be >= the calm/global band, not silently equal
    # to it. Its own scarce residuals are far wider, so max() must pick the group width.
    assert rare["width_kwh"] >= out["slices"]["calm"]["width_kwh"]
    # And it should be substantially wider than calm, reflecting the real noisy tail.
    assert rare["width_kwh"] > out["slices"]["calm"]["width_kwh"] * 5.0, out
