"""Offline model-promotion gate — "is the new version genuinely better?".

This implements the single thing the case-study brief most explicitly asks to
*see*: how you would decide that a retrained model should replace the incumbent
**before** deploying it. It is the runnable form of ``docs/LONG_TERM.md``
sections 3.1-3.3.

A challenger is promoted over the champion only if ALL hold on one frozen
held-out set:

1. **It actually wins, beyond noise.** A *paired bootstrap* 95% confidence
   interval on ``MAE(champion) - MAE(challenger)`` must lie entirely above 0.
   Comparing two point MAEs is not enough — a 0.05 kWh "win" can be pure
   resampling noise. The paired design (same test rows, same resample indices)
   cancels row-difficulty variance so the CI is tight and honest.
2. **No regime regresses.** No failure-mode slice (cold / steep / heavy, etc.)
   may get worse by more than a small tolerance — an aggregate win must not hide
   a cold-weather or steep-grade regression, which is exactly the tail a
   dispatcher cares about. The steep-grade bins are *rare* (~1.6% of rows, so
   ~18-20 in a held-out fold) and would slip under a naive 30-row support floor,
   so this gate is two-tier: well-supported slices are vetoed at the strict
   tolerance, sparse-but-safety-relevant slices (the steep tail) are still vetoed
   at a wider tolerance and flagged ``indicative``, and slices too sparse to
   judge at all are *disclosed* as ``unguarded`` rather than silently skipped —
   the blind spot must never pass with a clean verdict.
3. **It is not more dangerous.** The *optimistic-error rate* (fraction of rows
   where the model predicts LESS energy than truth — the direction that strands
   a truck) must not rise. A challenger that lowers mean MAE while becoming more
   optimistic overall is rejected.

Everything is deterministic (fixed bootstrap seed), offline, and dependency-light
(numpy only). Run as ``python -m nexdash.promote <champion.joblib> <challenger.joblib>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET_PATH
from .evaluate import failure_mode_report
from .features import FEATURE_COLUMNS, TARGET
from .model import EnergyModel

__all__ = ["compare", "held_out_split", "main"]

#: A slice may not get worse by more than this many kWh of MAE (absolute).
SLICE_REGRESSION_TOL_KWH: float = 0.5
#: Minimum rows for a slice to be *strictly* vetoed at the tight tolerance above.
SLICE_MIN_N: int = 30
#: Lower support floor for a *sparse* slice to still be vetoed (at the wider
#: tolerance below) and flagged ``indicative``. The steep-grade safety bins land
#: here (~18-20 rows in a real held-out fold) — below ``SLICE_MIN_N`` they used to
#: be silently skipped, defeating the very protection gate #2 promises.
SLICE_INDICATIVE_MIN_N: int = 8
#: Wider MAE-regression tolerance applied to sparse (indicative) slices, so
#: small-sample noise does not trip the veto but a genuine regression still does.
SLICE_INDICATIVE_REGRESSION_TOL_KWH: float = 1.0
#: The optimistic-error rate may not rise by more than this fraction.
OPTIMISTIC_RATE_TOL: float = 0.01


def held_out_split(
    dataset_path: Union[str, Path] = DEFAULT_DATASET_PATH,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the frozen held-out test fold both models are judged on.

    Uses the same ``train_test_split`` seed/fraction as ``run_pipeline`` so the
    comparison set is exactly the data neither model should have been tuned on.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset_path)
    _, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    return df_test.reset_index(drop=True)


def _abs_errors(model: EnergyModel, df_test: pd.DataFrame) -> np.ndarray:
    """Per-row absolute error (kWh) of ``model`` on ``df_test``."""
    y_true = df_test[TARGET].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(df_test[FEATURE_COLUMNS]), dtype=float)
    return np.abs(y_pred - y_true)


def _optimistic_rate(model: EnergyModel, df_test: pd.DataFrame) -> float:
    """Fraction of rows where the model predicts LESS energy than truth.

    This is the dangerous direction: under-predicting energy demand makes a trip
    look reachable when it is not, so a rising rate must block promotion.
    """
    y_true = df_test[TARGET].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(df_test[FEATURE_COLUMNS]), dtype=float)
    return float(np.mean(y_pred < y_true))


def _paired_bootstrap_ci(
    err_champ: np.ndarray,
    err_chal: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired-bootstrap CI on ``MAE(champion) - MAE(challenger)``.

    Resamples the shared test rows with replacement; for each resample takes the
    mean of the per-row error difference (champ - chal). Positive => challenger
    has lower error. Returns ``(point_estimate, ci_low, ci_high)`` at the
    ``1 - alpha`` level. Deterministic given ``seed``.
    """
    diff = err_champ - err_chal  # >0 => challenger better on that row
    n = diff.size
    point = float(diff.mean())
    if n == 0:
        return (point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return (point, lo, hi)


def _slice_regressions(
    champ: EnergyModel, chal: EnergyModel, df_test: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Failure-mode slices where the challenger's MAE regresses beyond tolerance.

    Returns ``(regressions, unguarded)``:

    * ``regressions`` — slices that **block** promotion. Well-supported slices
      (``n >= SLICE_MIN_N``) are vetoed at ``SLICE_REGRESSION_TOL_KWH`` and
      flagged ``indicative=False``; sparse-but-safety-relevant slices
      (``SLICE_INDICATIVE_MIN_N <= n < SLICE_MIN_N`` — the steep-grade tail) are
      still vetoed but at the wider ``SLICE_INDICATIVE_REGRESSION_TOL_KWH`` and
      flagged ``indicative=True`` so the small-sample caveat travels with the
      verdict.
    * ``unguarded`` — slices too sparse to veto at all (``0 < n <
      SLICE_INDICATIVE_MIN_N``) that nonetheless *appear* to regress. These do
      NOT block (the sample is too small to trust a veto) but are **disclosed**
      so the blind spot is never silent — this is precisely the gap that used to
      let a steep-grade regression pass with a clean verdict.
    """
    champ_fm = failure_mode_report(champ, df_test)
    chal_fm = failure_mode_report(chal, df_test)
    regressions: list[dict[str, Any]] = []
    unguarded: list[dict[str, Any]] = []
    for dim, bins in champ_fm.items():
        for label, cmetrics in bins.items():
            chmetrics = chal_fm.get(dim, {}).get(label, {})
            n = int(chmetrics.get("n", 0) or 0)
            c_mae = cmetrics.get("mae_kwh")
            x_mae = chmetrics.get("mae_kwh")
            if n <= 0 or c_mae is None or x_mae is None:
                continue
            if np.isnan(c_mae) or np.isnan(x_mae):
                continue
            regression = float(x_mae - c_mae)
            row = {
                "slice": f"{dim}:{label}",
                "n": n,
                "champion_mae": round(float(c_mae), 4),
                "challenger_mae": round(float(x_mae), 4),
                "regression_kwh": round(regression, 4),
            }
            if n >= SLICE_MIN_N:
                if regression > SLICE_REGRESSION_TOL_KWH:
                    regressions.append({**row, "indicative": False})
            elif n >= SLICE_INDICATIVE_MIN_N:
                # Sparse but safety-relevant (steep tail): veto at the wider
                # tolerance so a real regression still blocks, flagged indicative.
                if regression > SLICE_INDICATIVE_REGRESSION_TOL_KWH:
                    regressions.append({**row, "indicative": True})
            else:
                # Too sparse to veto, but disclose an apparent regression rather
                # than silently dropping the safety slice.
                if regression > SLICE_REGRESSION_TOL_KWH:
                    unguarded.append(row)
    return regressions, unguarded


def compare(
    champion_path: Union[str, Path],
    challenger_path: Union[str, Path],
    *,
    dataset_path: Union[str, Path] = DEFAULT_DATASET_PATH,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Decide whether the challenger should replace the champion.

    Returns a structured verdict dict with ``promote`` (bool), the bootstrap CI
    on the MAE improvement, any blocking ``slice_regressions`` (incl. sparse
    steep-tail slices vetoed at the wider tolerance and flagged ``indicative``),
    any ``unguarded_slices`` (too sparse to veto but disclosed so the steep-grade
    blind spot is never silent), the two optimistic-error rates, and
    human-readable ``reasons``. Deterministic and offline.
    """
    champ = EnergyModel.load(champion_path)
    chal = EnergyModel.load(challenger_path)
    df_test = held_out_split(dataset_path, seed=seed)

    err_champ = _abs_errors(champ, df_test)
    err_chal = _abs_errors(chal, df_test)
    mae_champ = float(err_champ.mean())
    mae_chal = float(err_chal.mean())

    point, ci_lo, ci_hi = _paired_bootstrap_ci(err_champ, err_chal, n_boot=n_boot)
    wins_significantly = ci_lo > 0.0  # whole CI above 0 => challenger truly better

    regressions, unguarded = _slice_regressions(champ, chal, df_test)
    no_slice_regression = len(regressions) == 0

    opt_champ = _optimistic_rate(champ, df_test)
    opt_chal = _optimistic_rate(chal, df_test)
    not_more_dangerous = opt_chal <= opt_champ + OPTIMISTIC_RATE_TOL

    promote = bool(wins_significantly and no_slice_regression and not_more_dangerous)

    reasons: list[str] = []
    reasons.append(
        f"MAE improvement (champion - challenger) = {point:.4f} kWh, "
        f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}] -> "
        + ("significant win" if wins_significantly else "NOT a significant win (CI includes/below 0)")
    )
    if regressions:
        worst = max(regressions, key=lambda r: r["regression_kwh"])
        n_ind = sum(1 for r in regressions if r.get("indicative"))
        suffix = f", incl. {n_ind} sparse/indicative" if n_ind else ""
        reasons.append(
            f"{len(regressions)} slice(s) regressed beyond tolerance{suffix} "
            f"(worst {worst['slice']}: +{worst['regression_kwh']} kWh, n={worst['n']})"
        )
    else:
        reasons.append("no failure-mode slice regressed beyond tolerance")
    if unguarded:
        worst_u = max(unguarded, key=lambda r: r["regression_kwh"])
        reasons.append(
            f"{len(unguarded)} safety-critical slice(s) too sparse to veto "
            f"(n<{SLICE_INDICATIVE_MIN_N}; worst {worst_u['slice']}: "
            f"+{worst_u['regression_kwh']} kWh, n={worst_u['n']}) — disclosed, inspect manually"
        )
    reasons.append(
        f"optimistic-error rate champion {opt_champ:.3f} -> challenger {opt_chal:.3f} "
        + ("(not more dangerous)" if not_more_dangerous else "(MORE optimistic -> blocked)")
    )

    return {
        "promote": promote,
        "champion_mae_kwh": round(mae_champ, 4),
        "challenger_mae_kwh": round(mae_chal, 4),
        "mae_improvement_kwh": round(point, 4),
        "improvement_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "wins_significantly": bool(wins_significantly),
        "slice_regressions": regressions,
        "unguarded_slices": unguarded,
        "optimistic_rate_champion": round(opt_champ, 4),
        "optimistic_rate_challenger": round(opt_chal, 4),
        "not_more_dangerous": bool(not_more_dangerous),
        "n_test": int(len(df_test)),
        "reasons": reasons,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: ``python -m nexdash.promote <champion.joblib> <challenger.joblib>``."""
    parser = argparse.ArgumentParser(
        description="Decide whether a challenger energy model should replace the champion."
    )
    parser.add_argument("champion", help="Path to the incumbent model artifact (.joblib).")
    parser.add_argument("challenger", help="Path to the candidate model artifact (.joblib).")
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET_PATH), help="Dataset CSV for the frozen held-out split."
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples (default 2000).")
    args = parser.parse_args(argv)

    verdict = compare(args.champion, args.challenger, dataset_path=args.dataset, n_boot=args.n_boot)

    print("=" * 64)
    print(f"PROMOTION GATE: {'PROMOTE ✅' if verdict['promote'] else 'REJECT ❌'}")
    print("=" * 64)
    print(
        f"champion MAE   {verdict['champion_mae_kwh']:.3f} kWh   "
        f"challenger MAE {verdict['challenger_mae_kwh']:.3f} kWh   (n={verdict['n_test']})"
    )
    for r in verdict["reasons"]:
        print(f"  - {r}")
    return 0 if verdict["promote"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
