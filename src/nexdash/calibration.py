"""Verified uncertainty calibration — does our stated confidence hold up?

The NexDash range tool *claims* an uncertainty band (``range.confidence_note``
quotes the held-out MAE) and ``docs/LONG_TERM.md`` promises to "validate that band
against measured error". This module stops asserting and starts **measuring**: it
builds distribution-free **split-conformal** prediction intervals and then audits
whether they actually cover the held-out truth at their nominal rate — globally and
per failure-slice — reporting interval width (sharpness), an Expected Calibration
Error, and a bootstrap PASS/FAIL flag per level.

Why split conformal: given a held-out *calibration* set of absolute residuals
``|y - y_hat|``, the half-width at the ``ceil((n+1)*level)/n`` empirical quantile
gives a finite-sample marginal coverage guarantee ``P(y in y_hat +/- h) >= level``
with **no distributional assumptions** — far more honest than assuming Gaussian
errors. The audit then checks the *realized* coverage on a disjoint evaluation
slice, so a "90% interval" that really covers 84% is exposed, not hidden.

Honest scope: this proves coverage of the **synthetic held-out labels** (our own
noisy physics), i.e. coverage-of-physics, not coverage-of-reality — the same
circular-evaluation caveat the report already states. Pure numpy, deterministic
(seeded), offline.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import binom

__all__ = [
    "conformal_halfwidth",
    "coverage_fraction",
    "calibrate_and_audit",
    "calibrate_and_audit_kfold",
    "DEFAULT_LEVELS",
]

#: Nominal coverage levels audited by default.
DEFAULT_LEVELS: tuple[float, ...] = (0.80, 0.90, 0.95)


def _finite_sample_quantile(n: int, level: float) -> float:
    """The split-conformal quantile position for ``n`` calibration points.

    Returns ``ceil((n+1)*level)/n`` clamped to ``[0, 1]``. Using ``(n+1)`` (not
    ``n``) is the standard finite-sample correction that makes the resulting
    interval's marginal coverage provably ``>= level``.
    """
    if n <= 0:
        return 1.0
    return min(1.0, math.ceil((n + 1) * level) / n)


#: Mondrian group labels pooled into a single 'steep' bucket. Both steep tails are
#: high-|gradient| regimes whose residual spread is driven by magnitude not sign, so
#: pooling them roughly triples the calibration support for the steep band.
_STEEP_POOL: frozenset[str] = frozenset({"steep_up", "steep_down"})


def _pool_group(label: object) -> str:
    """Map ``steep_up``/``steep_down`` to one ``steep`` group; pass others through."""
    s = str(label)
    return "steep" if s in _STEEP_POOL else s


def conformal_halfwidth(cal_abs_residuals: Sequence[float], level: float) -> float:
    """Split-conformal interval half-width at ``level`` from calibration residuals.

    ``cal_abs_residuals`` are the absolute errors ``|y - y_hat|`` on a calibration
    set the model never trained on. Returns the (conservative, ``method="higher"``)
    finite-sample quantile so the band's coverage is guaranteed ``>= level``.
    """
    r = np.asarray(cal_abs_residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("inf")  # no calibration data -> uninformative (covers all)
    if math.ceil((r.size + 1) * level) > r.size:
        # No finite-sample conformal quantile exists at this (n, level): the required
        # rank ceil((n+1)*level) exceeds n, so no band can guarantee >= level. The
        # honest answer is an uninformative (+inf) band, NOT the max residual — which
        # would silently under-cover (e.g. n=10 @ 0.95 covers only n/(n+1) ~ 91%).
        return float("inf")
    q = _finite_sample_quantile(r.size, level)
    return float(np.quantile(r, q, method="higher"))


def coverage_fraction(
    y_true: np.ndarray, y_pred: np.ndarray, halfwidth: float
) -> float:
    """Fraction of rows whose truth lies within ``y_hat +/- halfwidth``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) <= halfwidth))


def _bootstrap_coverage_ci(
    abs_err_eval: np.ndarray,
    halfwidth: float,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the realized coverage of a fixed band."""
    n = abs_err_eval.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    covered = (abs_err_eval <= halfwidth).astype(float)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = covered[idx].mean(axis=1)
    return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


def calibrate_and_audit(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    levels: Sequence[float] = DEFAULT_LEVELS,
    groups: Optional[Sequence[str]] = None,
    cal_frac: float = 0.5,
    seed: int = 42,
    n_boot: int = 1000,
    mondrian_level: float = 0.90,
    min_group_cal: int = 20,
) -> dict[str, Any]:
    """Calibrate split-conformal intervals and audit their realized coverage.

    The held-out rows are split (seeded) into a *calibration* half — used only to
    set each band's half-width — and a disjoint *evaluation* half, on which the
    realized coverage is measured. For each nominal ``level`` we report empirical
    coverage, mean interval width (sharpness), a bootstrap CI on the coverage, and
    a one-sided PASS / FAIL / CONSERVATIVE status (FAIL only when realized coverage
    is significantly BELOW nominal — the over-confident direction). This is a
    single-fold indicator: realized coverage fluctuates around nominal, so an
    isolated FAIL at small n can be sampling noise (~5-10% of honest folds) — read
    it as a flag to investigate, not proof of mis-calibration. ``ece`` is the mean
    ``|empirical - nominal|`` across levels.

    When ``groups`` is given (one label per row, e.g. a gradient regime), a
    **Mondrian / group-conditional** audit at ``mondrian_level`` computes a
    *separate* half-width per group from that group's own calibration residuals,
    so a well-supported regime gets its own (often wider) band. The two rare steep
    regimes ``steep_up`` and ``steep_down`` are **pooled** into one ``steep`` group
    before the per-group loop: they are both "high-|gradient|" tails whose residual
    spread is dominated by magnitude, not sign, so pooling roughly triples their
    calibration support and lets the steep band stand on its own rather than always
    collapsing to the sparse fallback. (This is a regrouping of inputs, not a change
    to ``min_group_cal``.)

    A group that *still* has too few calibration rows falls back to a half-width of
    ``max(global_hw, group_hw)`` and is flagged ``indicative``. This is honestly
    *conservative*: rather than silently substituting the global band (which could be
    narrower than the group's own scarce-but-real residuals suggest, hiding a noisier
    tail), we take the wider of the two so the indicative band never under-states the
    sparse regime's uncertainty. It is still flagged ``indicative`` because the
    group-level quantile cannot be trusted at this n — reported, not re-certified.

    All inputs are arrays; this module is decoupled from the model. Deterministic.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.size
    if n < 4:
        return {"n_cal": 0, "n_eval": 0, "levels": [], "ece": float("nan"), "slices": {}}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = max(1, int(round(n * cal_frac)))
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]

    abs_res_cal = np.abs(y_true[cal_idx] - y_pred[cal_idx])
    abs_err_eval = np.abs(y_true[eval_idx] - y_pred[eval_idx])

    level_rows: list[dict[str, Any]] = []
    abs_diffs: list[float] = []
    for i, level in enumerate(levels):
        hw = conformal_halfwidth(abs_res_cal, level)
        emp = coverage_fraction(y_true[eval_idx], y_pred[eval_idx], hw)
        lo, hi = _bootstrap_coverage_ci(abs_err_eval, hw, n_boot=n_boot, seed=seed + i)
        # Split-conformal only *guarantees* coverage >= level (a one-sided
        # property), so under-coverage is the only true failure: that is the
        # dangerous, over-confident direction. Over-coverage (the band is wider
        # than it strictly needs) honours the guarantee — it is conservative, not
        # mis-calibrated — so we label it CONSERVATIVE, never FAIL.
        #   FAIL  -> realized coverage is significantly BELOW nominal (band too tight)
        #   PASS  -> nominal sits inside the realized-coverage CI
        #   CONSERVATIVE -> realized coverage significantly ABOVE nominal
        if level < lo:
            status = "CONSERVATIVE"
        elif level > hi:
            status = "FAIL"  # nominal above the CI => the band under-covers
        else:
            status = "PASS"
        level_rows.append(
            {
                "nominal": round(float(level), 4),
                "empirical": round(float(emp), 4),
                "width_kwh": round(float(hw), 3),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "status": status,
            }
        )
        abs_diffs.append(abs(emp - level))

    ece = round(float(np.mean(abs_diffs)), 4) if abs_diffs else float("nan")

    slices: dict[str, Any] = {}
    if groups is not None:
        # Pool the two sparse steep tails (steep_up + steep_down) into one 'steep'
        # group BEFORE the per-group loop, so the steep band draws on ~3x the
        # calibration support instead of each sign-half collapsing to the fallback.
        groups = np.array([_pool_group(g) for g in np.asarray(groups).tolist()])
        global_hw = conformal_halfwidth(abs_res_cal, mondrian_level)
        for g in sorted(set(groups.tolist())):
            g_cal = cal_idx[groups[cal_idx] == g]
            g_eval = eval_idx[groups[eval_idx] == g]
            if g_eval.size == 0:
                continue
            indicative = g_cal.size < min_group_cal
            group_hw = conformal_halfwidth(np.abs(y_true[g_cal] - y_pred[g_cal]), mondrian_level)
            if indicative:
                # Too few group cal rows -> the group quantile is untrustworthy, but
                # silently using global_hw could UNDER-state a genuinely noisier tail.
                # Be honestly conservative: take the wider of (global, group) so the
                # indicative band never narrows below what the scarce residuals imply.
                hw = max(global_hw, group_hw)
            else:
                hw = group_hw
            emp = coverage_fraction(y_true[g_eval], y_pred[g_eval], hw)
            slices[str(g)] = {
                "nominal": round(float(mondrian_level), 4),
                "empirical": round(float(emp), 4),
                "width_kwh": round(float(hw), 3),
                "n_eval": int(g_eval.size),
                "indicative": bool(indicative),
            }

    return {
        "n_cal": int(n_cal),
        "n_eval": int(n - n_cal),
        "levels": level_rows,
        "ece": ece,
        "slices": slices,
    }


#: Per-fold one-sided false-flag rate of a single ``calibrate_and_audit`` FAIL.
#: Realized coverage fluctuates around nominal, so even a perfectly-calibrated band
#: trips the FAIL branch on a minority of random cal/eval splits. ~0.10 is the rate
#: the single-fold docstring already warns about ("~5-10% of honest folds"); using
#: the upper end keeps the persistence test conservative (harder to false-flag).
_FOLD_FALSE_FLAG_RATE: float = 0.10


def calibrate_and_audit_kfold(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    k: int = 20,
    levels: Sequence[float] = DEFAULT_LEVELS,
    groups: Optional[Sequence[str]] = None,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Repeat the single-fold audit over ``k`` reshuffled splits and test *persistence*.

    WHY this exists: ``calibrate_and_audit`` is a single random cal/eval split, and
    its own docstring admits an isolated FAIL "can be sampling noise (~5-10% of honest
    folds)". Reporting one fold's FAIL as if it were proof of mis-calibration is
    exactly the over-claim this project keeps narrowing. This wrapper runs the audit
    over ``k`` seeds (``base_seed + fold``, so each fold reshuffles the split) and asks
    a sharper question: does a level *persistently* FAIL, more often than an honest
    band would by chance?

    For each nominal level we collect the ``k`` realized empirical coverages and the
    ``k`` PASS/FAIL/CONSERVATIVE statuses, then report:

    * ``median_empirical`` — median realized coverage across folds (robust central
      estimate, unaffected by one unlucky split);
    * ``iqr_empirical`` — across-fold inter-quartile range (Q3-Q1) of realized
      coverage, i.e. how much the single-fold number wobbles with the split;
    * ``fail_count`` / ``k`` — how many folds tripped FAIL;
    * ``fail_threshold`` — the 95th-percentile of ``Binomial(k, alpha)`` with
      ``alpha`` = the per-fold false-flag rate (~0.10): the largest fail count an
      honest, well-calibrated band would plausibly produce by chance alone;
    * ``verdict`` — ``"persistently fails"`` only when ``fail_count > fail_threshold``
      (genuine, beyond-chance under-coverage), else ``"ok"``.

    So a level that FAILs on 1-3 of 20 folds is reported ``ok`` (within the binomial
    noise floor of an honest band), while one that FAILs on, say, 12 of 20 is called
    out as ``persistently fails``. The threshold is one-sided (we only flag the
    over-confident direction), matching the single-fold FAIL semantics.

    Deterministic: fold ``i`` uses ``seed=base_seed + i`` throughout (including the
    bootstrap), so repeated calls return byte-identical results. ``groups`` is passed
    through unchanged to each fold (steep pooling and the conservative sparse fallback
    apply there as usual). Inherits every honest-scope caveat of the single-fold audit
    — this proves *persistence of calibration against the synthetic held-out labels*,
    not against reality.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    # Per-level accumulators, keyed by the rounded nominal (matching level_rows keys).
    emp_by_level: dict[float, list[float]] = {}
    fail_by_level: dict[float, int] = {}
    order: list[float] = []

    n_eval = 0
    for i in range(k):
        fold = calibrate_and_audit(
            y_true, y_pred, levels=levels, groups=groups, seed=base_seed + i
        )
        n_eval = fold["n_eval"]
        for row in fold["levels"]:
            nominal = row["nominal"]
            if nominal not in emp_by_level:
                emp_by_level[nominal] = []
                fail_by_level[nominal] = 0
                order.append(nominal)
            emp = row["empirical"]
            if math.isfinite(emp):
                emp_by_level[nominal].append(emp)
            if row["status"] == "FAIL":
                fail_by_level[nominal] += 1

    # Binomial noise floor: the most FAILs an honest band would produce ~95% of the
    # time. binom.ppf(0.95, k, alpha) is an integer count; flag only strictly above it.
    fail_threshold = int(binom.ppf(0.95, k, _FOLD_FALSE_FLAG_RATE))

    level_rows: list[dict[str, Any]] = []
    for nominal in order:
        emps = np.asarray(emp_by_level[nominal], dtype=float)
        if emps.size:
            median_emp = float(np.median(emps))
            iqr = float(np.quantile(emps, 0.75) - np.quantile(emps, 0.25))
        else:
            median_emp = float("nan")
            iqr = float("nan")
        fail_count = fail_by_level[nominal]
        verdict = "persistently fails" if fail_count > fail_threshold else "ok"
        level_rows.append(
            {
                "nominal": round(float(nominal), 4),
                "median_empirical": round(median_emp, 4),
                "iqr_empirical": round(iqr, 4),
                "fail_count": int(fail_count),
                "fail_threshold": int(fail_threshold),
                "verdict": verdict,
            }
        )

    return {
        "k": k,
        "n_eval": int(n_eval),
        "fail_threshold": int(fail_threshold),
        "false_flag_rate": _FOLD_FALSE_FLAG_RATE,
        "levels": level_rows,
    }
