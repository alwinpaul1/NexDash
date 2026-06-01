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

__all__ = [
    "conformal_halfwidth",
    "coverage_fraction",
    "calibrate_and_audit",
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
    a PASS/FAIL status (FAIL when the nominal level sits outside the CI, i.e. the
    band is mis-calibrated). ``ece`` is the mean ``|empirical - nominal|`` across
    levels.

    When ``groups`` is given (one label per row, e.g. a gradient regime), a
    **Mondrian / group-conditional** audit at ``mondrian_level`` computes a
    *separate* half-width per group from that group's own calibration residuals —
    so cold/steep regimes honestly get wider bands — and reports per-group
    realized coverage. Groups with too few calibration rows fall back to the
    global half-width and are flagged ``indicative``.

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
        groups = np.asarray(groups)
        global_hw = conformal_halfwidth(abs_res_cal, mondrian_level)
        for g in sorted(set(groups.tolist())):
            g_cal = cal_idx[groups[cal_idx] == g]
            g_eval = eval_idx[groups[eval_idx] == g]
            if g_eval.size == 0:
                continue
            indicative = g_cal.size < min_group_cal
            if indicative:
                hw = global_hw  # too few group cal rows -> reuse global band
            else:
                hw = conformal_halfwidth(np.abs(y_true[g_cal] - y_pred[g_cal]), mondrian_level)
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
