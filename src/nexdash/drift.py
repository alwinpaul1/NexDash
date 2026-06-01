"""Data / concept drift detection — "how would you notice it drifted from reality?".

The runnable form of ``docs/LONG_TERM.md`` section 4: the third long-term
deliverable the brief asks for. Given a new batch of operating data (e.g. a
month of real telematics) and the training reference, it answers three questions:

* **Did the inputs move?** Per-feature Population Stability Index (PSI) on fixed
  quantile bins taken from the training reference. **PSI is primary: it sets the
  base tier and is the only thing that can *originate* an alert** (industry-standard
  tiers: < 0.1 stable, 0.1-0.25 watch, > 0.25 significant drift). A two-sample
  Kolmogorov-Smirnov test is reported per feature (both the statistic ``D`` and the
  p-value) and is wired into the verdict through *effect size, not significance*:
  it can bump a feature's tier by AT MOST one step (e.g. stable -> watch) and ONLY
  when the shift is both statistically significant (p < 0.01) AND large enough to
  matter (D > ``KS_D_MIN``). The raw p-value is deliberately NOT trusted on its own:
  two-sample KS p-values collapse toward 0 at large n even on operationally trivial
  shifts, so a tiny-but-significant large-n shift must NOT escalate (the documented
  over-alert pathology). KS can thus *confirm and sharpen* an alert, never create
  one from nothing.
* **Is the relationship still holding?** When the new batch carries the true
  energy labels, a realized-residual monitor compares live MAE / mean bias
  against the model's training-time error — the only signal grounded in truth,
  which catches *concept* drift that input-only tests miss.
* **What's the verdict?** A single tiered ``stable | watch | drift`` rollup so a
  retrain can be triggered automatically.

Honest scope: this is *marginal* (per-feature) drift only. Multivariate / novel
feature-interaction drift (LONG_TERM 4.2) is left as documented future work
rather than overbuilt. Pure numpy; the KS test uses scipy if available and
degrades gracefully if not. Offline and deterministic.

Run as ``python -m nexdash.drift <reference.csv> <current.csv> [--model <path>]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from .features import FEATURE_COLUMNS, TARGET

__all__ = ["psi", "feature_drift", "residual_monitor", "drift_report", "main"]

#: PSI tier cut-offs (standard population-stability convention).
PSI_WATCH: float = 0.10
PSI_DRIFT: float = 0.25

#: Ordered tiers, worst last — used to bump a tier by one bounded step.
_TIER_ORDER: tuple[str, ...] = ("stable", "watch", "drift")

#: Minimum KS statistic D for an *effect-size* escalation. D is the max gap
#: between the two empirical CDFs (0 = identical, 1 = disjoint support), so it is
#: independent of sample size — unlike the p-value, it does not shrink as n grows.
#: 0.1 = a 10-percentage-point CDF gap, the floor below which a "significant"
#: difference is operationally trivial and must not raise the verdict.
KS_D_MIN: float = 0.10

#: KS p-value below which a shift is treated as statistically real (not noise).
KS_P_MAX: float = 0.01


def psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Bin edges are quantiles of the *reference* (expected) distribution, so PSI
    measures how the current sample's mass redistributes across the reference's
    own bins. Returns 0.0 for a degenerate (near-constant) reference feature,
    where PSI is undefined and "no drift" is the honest answer.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0  # constant feature -> no meaningful bins
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    eps = 1e-6
    e_pct = np.clip(e_counts / e_counts.sum(), eps, None)
    a_pct = np.clip(a_counts / a_counts.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _ks_stat_pvalue(
    expected: np.ndarray, actual: np.ndarray
) -> tuple[Optional[float], Optional[float]]:
    """Two-sample KS ``(statistic D, p-value)``, or ``(None, None)`` if scipy absent.

    ``D`` is the maximum gap between the two empirical CDFs (effect size; sample-size
    independent); the p-value is its significance (collapses toward 0 at large n).
    """
    try:
        from scipy.stats import ks_2samp
    except Exception:  # pragma: no cover - scipy is a declared dep but stay soft
        return None, None
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size < 2 or actual.size < 2:
        return None, None
    res = ks_2samp(expected, actual)
    return float(res.statistic), float(res.pvalue)


def _tier(psi_value: float) -> str:
    if psi_value < PSI_WATCH:
        return "stable"
    if psi_value < PSI_DRIFT:
        return "watch"
    return "drift"


def _escalate(
    psi_tier: str,
    ks_stat: Optional[float],
    ks_p: Optional[float],
    n: int,
) -> str:
    """Bump the PSI tier by AT MOST one step on a genuine, large-enough KS shift.

    The escalation fires only when the KS shift is both statistically real
    (``ks_p < KS_P_MAX``) AND large in effect size (``ks_stat > KS_D_MIN``). Using
    ``D`` as the gate — not the p-value alone — is what stops the over-alert
    pathology: a trivially small shift that becomes "significant" purely because n
    is huge has ``D`` near 0, fails the effect-size gate, and does NOT escalate.

    KS can therefore only *confirm* an alert PSI already sees forming (e.g. lift a
    borderline ``stable`` to ``watch``); it can never originate one beyond a single
    bounded step, and ``drift`` (the top tier) is never exceeded. ``n`` is accepted
    for signature clarity / future logging but intentionally does not relax the
    gates — the whole point is that the verdict is n-independent.
    """
    if ks_stat is None or ks_p is None:
        return psi_tier
    if not (ks_p < KS_P_MAX and ks_stat > KS_D_MIN):
        return psi_tier
    idx = _TIER_ORDER.index(psi_tier)
    return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_columns: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Per-feature PSI + KS (statistic D & p-value) + tier for every input feature.

    PSI sets the base tier; the KS statistic/p-value may then bump it by one bounded
    step via :func:`_escalate` (effect size, never the n-inflated p-value alone).
    """
    cols = feature_columns or list(FEATURE_COLUMNS)
    out: dict[str, dict[str, Any]] = {}
    n = int(len(current))
    for col in cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = reference[col].to_numpy(dtype=float)
        cur = current[col].to_numpy(dtype=float)
        p = psi(ref, cur)
        ks_stat, ks_p = _ks_stat_pvalue(ref, cur)
        psi_tier = _tier(p)
        out[col] = {
            "psi": round(p, 4),
            "ks_stat": (round(ks_stat, 6) if ks_stat is not None else None),
            "ks_pvalue": (round(ks_p, 6) if ks_p is not None else None),
            "psi_tier": psi_tier,
            "tier": _escalate(psi_tier, ks_stat, ks_p, n),
        }
    return out


def residual_monitor(
    model: Any, current: pd.DataFrame
) -> Optional[dict[str, Any]]:
    """Compare live error against truth when the new batch carries labels.

    Returns mean signed residual (pred - true; positive = conservative
    over-prediction) and realized MAE, or ``None`` if the batch is unlabelled.
    Concept drift shows up here even when the input distributions look stable.
    """
    if TARGET not in current.columns:
        return None
    y_true = current[TARGET].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(current[list(FEATURE_COLUMNS)]), dtype=float)
    resid = y_pred - y_true
    return {
        "realized_mae_kwh": round(float(np.mean(np.abs(resid))), 4),
        "mean_signed_residual_kwh": round(float(np.mean(resid)), 4),
        "optimistic_rate": round(float(np.mean(y_pred < y_true)), 4),
        "n": int(y_true.size),
    }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    model: Any = None,
) -> dict[str, Any]:
    """Aggregate per-feature drift (+ optional residual monitor) into one verdict.

    The overall verdict is the worst per-feature tier; ``stable`` only if every
    feature is stable. Deterministic and offline.
    """
    features = feature_drift(reference, current)
    tiers = [f["tier"] for f in features.values()]
    if "drift" in tiers:
        overall = "drift"
    elif "watch" in tiers:
        overall = "watch"
    else:
        overall = "stable"

    report: dict[str, Any] = {
        "overall": overall,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "features": features,
        "drifted_features": [c for c, f in features.items() if f["tier"] == "drift"],
    }
    if model is not None:
        report["residuals"] = residual_monitor(model, current)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: ``python -m nexdash.drift <reference.csv> <current.csv> [--model <path>]``."""
    parser = argparse.ArgumentParser(
        description="Detect data/concept drift of a new batch against the training reference."
    )
    parser.add_argument(
        "reference", nargs="?", default=str(DEFAULT_DATASET_PATH),
        help="Reference (training) dataset CSV (default: the project dataset).",
    )
    parser.add_argument("current", help="New-batch CSV to test for drift.")
    parser.add_argument(
        "--model", default=None,
        help="Optional model artifact; if the batch has labels, adds a residual monitor.",
    )
    args = parser.parse_args(argv)

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    model = None
    if args.model:
        from .model import EnergyModel

        model = EnergyModel.load(args.model)

    report = drift_report(reference, current, model=model)
    print(json.dumps(report, indent=2))
    # Exit non-zero when significant drift is detected (useful as a retrain trigger).
    return 1 if report["overall"] == "drift" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
