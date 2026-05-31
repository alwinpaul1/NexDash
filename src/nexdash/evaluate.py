"""Honest evaluation of the NexDash energy model.

This module is the integrity backbone of the project: it answers *how good is
the model, and where does it break?* rather than merely *what number does it
produce?*. It provides three things:

* :func:`evaluate` -- headline regression metrics (MAE, RMSE, MAPE, R^2)
  computed on a held-out test set.
* :func:`failure_mode_report` -- the same error metrics sliced by operating
  regime (temperature, road gradient, payload) so we can see *where* error
  concentrates instead of hiding it behind a single average.
* :func:`make_plots` -- diagnostic figures (predicted-vs-actual, residual-vs-
  temperature, error-by-payload) saved as PNGs.

Design notes / honesty caveats
------------------------------
* All metrics are computed on data the model did **not** train on. A low
  in-sample error would tell us nothing useful.
* MAPE is undefined / explosive for near-zero targets. Strong-downhill
  segments can produce tiny or slightly negative energy_kwh values, so MAPE is
  computed only over rows whose actual energy exceeds a small floor; the count
  of excluded rows is reported transparently as ``mape_n``.
* ``pct_range_error`` expresses the MAE as a percentage of the energy required
  for a *nominal full-range trip* (~500 km at typical motorway load). This puts
  the error on an intuitive "how much of a charge could we be wrong by" scale;
  it is **not** a per-trip percentage error (that is MAPE).
* A single average metric is necessarily optimistic for the hard cases. The
  failure-mode report exists precisely because a fleet dispatcher cares far
  more about the cold/steep/heavy tail than about the easy mid-range mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MAPE_FLOOR_KWH, REPORTS_DIR, TRUCK
from .features import FEATURE_COLUMNS, TARGET

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Nominal full-range trip distance (km) used to scale MAE into a fleet-
#: intuitive "percentage of a full charge" figure.
NOMINAL_TRIP_KM: float = 500.0

#: Energy denominator for :data:`pct_range_error`: the usable battery capacity, so
#: the figure reads as "MAE as a fraction of one full charge". Uses
#: ``TRUCK.battery_kwh`` so the report text, this divisor, ``run_pipeline`` and
#: ``model_info`` all reference ONE value (no 600-vs-570 mismatch).
_NOMINAL_TRIP_KWH: float = TRUCK.battery_kwh

#: Minimum |actual energy| (kWh) for a row to participate in MAPE — shared with
#: the model's comparison metrics so every MAPE in the report uses one floor.
_MAPE_FLOOR_KWH: float = MAPE_FLOOR_KWH


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _predict(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Run the model on a feature DataFrame and return a 1-D float array.

    The model consumes the raw :data:`FEATURE_COLUMNS` (it applies feature
    engineering internally via :func:`nexdash.features.transform`), so we hand
    it exactly those columns to avoid leaking the target.
    """
    preds = model.predict(df[FEATURE_COLUMNS])
    return np.asarray(preds, dtype=float).ravel()


def _core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE / RMSE / MAPE / R^2 / n for one (sub)set of rows.

    Returns a flat dict. MAPE is restricted to rows whose actual energy exceeds
    :data:`_MAPE_FLOOR_KWH`; the participating count is returned as ``mape_n``.
    Empty input yields NaN metrics with ``n == 0`` rather than raising, so that
    sparse failure-mode slices degrade gracefully.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = int(y_true.size)

    if n == 0:
        return {
            "mae_kwh": float("nan"),
            "rmse_kwh": float("nan"),
            "mape_pct": float("nan"),
            "r2": float("nan"),
            "n": 0,
            "mape_n": 0,
        }

    mae = float(mean_absolute_error(y_true, y_pred))
    # squared=False is deprecated/removed in newer sklearn; sqrt is portable.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE only where the denominator is meaningful (same floor/comparator the
    # model's comparison metrics use, so the table and headline MAPE agree).
    mask = np.abs(y_true) >= _MAPE_FLOOR_KWH
    if mask.any():
        mape = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
        )
    else:
        mape = float("nan")

    # R^2 is undefined for a single point (zero variance); guard it.
    r2 = float(r2_score(y_true, y_pred)) if n >= 2 else float("nan")

    return {
        "mae_kwh": round(mae, 4),
        "rmse_kwh": round(rmse, 4),
        "mape_pct": round(mape, 4) if not np.isnan(mape) else float("nan"),
        "r2": round(r2, 4) if not np.isnan(r2) else float("nan"),
        "n": n,
        "mape_n": int(mask.sum()),
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def evaluate(model: Any, df_test: pd.DataFrame) -> dict[str, float]:
    """Compute headline regression metrics on a held-out test set.

    Args:
        model: A trained model exposing ``predict(rows)`` where ``rows`` is a
            DataFrame of raw :data:`FEATURE_COLUMNS`. (Typically an
            :class:`nexdash.model.EnergyModel`.)
        df_test: Test DataFrame containing :data:`FEATURE_COLUMNS` and the
            :data:`TARGET` column. Must not have been seen during training.

    Returns:
        Dict with keys:

        * ``mae_kwh`` -- mean absolute error in kWh. *Tells us*: typical
          absolute miss per segment. *Does not tell us*: whether large errors
          cluster in a dangerous regime (see :func:`failure_mode_report`).
        * ``rmse_kwh`` -- root mean squared error (kWh); penalises large
          misses more than MAE, so RMSE >> MAE signals heavy-tailed errors.
        * ``mape_pct`` -- mean absolute percentage error over rows above the
          MAPE floor. *Does not tell us* anything about near-zero downhill
          segments, which are excluded.
        * ``pct_range_error`` -- MAE as a percentage of a nominal full-range
          (~500 km) trip's energy; a fleet-intuitive "fraction of a charge we
          might be off by". Documented as nominal-trip based, **not** per-trip.
        * ``r2`` -- coefficient of determination; fraction of target variance
          explained. *Does not tell us* the magnitude of error in kWh.
        * ``n`` -- number of test rows scored.

    Raises:
        KeyError: if required columns are missing from ``df_test``.
    """
    y_true = df_test[TARGET].to_numpy(dtype=float)
    y_pred = _predict(model, df_test)

    core = _core_metrics(y_true, y_pred)

    pct_range_error = round(core["mae_kwh"] / _NOMINAL_TRIP_KWH * 100.0, 4)

    return {
        "mae_kwh": core["mae_kwh"],
        "rmse_kwh": core["rmse_kwh"],
        "mape_pct": core["mape_pct"],
        "pct_range_error": pct_range_error,
        "r2": core["r2"],
        "n": core["n"],
    }


def failure_mode_report(model: Any, df_test: pd.DataFrame) -> dict[str, Any]:
    """Slice error metrics by operating regime to expose where the model fails.

    A single MAE hides the cases a dispatcher actually worries about: freezing
    weather (HVAC + battery penalty), steep climbs (peak power, no regen), and
    near-max payload. This report computes per-slice MAE/RMSE/MAPE/R^2 so those
    tails are visible instead of averaged away.

    Bins:
        * temperature_c: ``cold`` (< 0 C), ``mild`` (0-30 C), ``hot`` (> 30 C).
        * gradient_pct: ``steep_down`` (< -4 %), ``flat`` (-4..+4 %),
          ``steep_up`` (> +4 %).
        * payload_t: ``light`` (< 7 t), ``mid`` (7-15 t), ``heavy`` (> 15 t).

    Args:
        model: Trained model with a ``predict`` method (see :func:`evaluate`).
        df_test: Held-out test DataFrame with features and target.

    Returns:
        Nested dict ``{dimension: {bin_label: metrics_dict}}`` where each
        ``metrics_dict`` is the output of the internal core-metrics routine
        (``mae_kwh``, ``rmse_kwh``, ``mape_pct``, ``r2``, ``n``, ``mape_n``).
        Empty bins are still present with ``n == 0`` and NaN metrics so the
        report's structure is stable across datasets.

    Caveat:
        Sparse bins (small ``n``) have noisy metrics; always read them
        alongside ``n``. The bins intentionally overlap nothing and cover the
        full feature ranges used by :mod:`nexdash.data_gen`.
    """
    df = df_test.copy()
    df["_y_true"] = df[TARGET].to_numpy(dtype=float)
    df["_y_pred"] = _predict(model, df)

    def _slice(mask: pd.Series) -> dict[str, float]:
        sub = df.loc[mask]
        return _core_metrics(sub["_y_true"].to_numpy(), sub["_y_pred"].to_numpy())

    temperature = {
        "cold (<0C)": _slice(df["temperature_c"] < 0),
        "mild (0-30C)": _slice((df["temperature_c"] >= 0) & (df["temperature_c"] <= 30)),
        "hot (>30C)": _slice(df["temperature_c"] > 30),
    }

    gradient = {
        "steep_down (<-4%)": _slice(df["gradient_pct"] < -4),
        "flat (-4..+4%)": _slice((df["gradient_pct"] >= -4) & (df["gradient_pct"] <= 4)),
        "steep_up (>+4%)": _slice(df["gradient_pct"] > 4),
    }

    payload = {
        "light (<7t)": _slice(df["payload_t"] < 7),
        "mid (7-15t)": _slice((df["payload_t"] >= 7) & (df["payload_t"] <= 15)),
        "heavy (>15t)": _slice(df["payload_t"] > 15),
    }

    return {
        "temperature": temperature,
        "gradient": gradient,
        "payload": payload,
    }


def make_plots(
    model: Any,
    df_test: pd.DataFrame,
    out_dir: str | Path = REPORTS_DIR / "figures",
) -> list[str]:
    """Render diagnostic figures and return the saved PNG paths.

    Uses the non-interactive ``Agg`` matplotlib backend so it runs headless
    (CI / servers). Three plots are produced:

    1. **Predicted vs actual** -- points should hug the 45-degree line; spread
       reveals overall calibration.
    2. **Residual vs temperature** -- residual = actual - predicted; a slope or
       fanning at the cold/hot extremes exposes HVAC-regime bias.
    3. **Mean absolute error by payload bin** -- bar chart showing whether the
       model degrades under heavy load.

    Args:
        model: Trained model with a ``predict`` method.
        df_test: Held-out test DataFrame with features and target.
        out_dir: Directory to write PNGs into; created if missing.

    Returns:
        List of absolute file paths (as strings) for the saved figures, in the
        order described above.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend; must precede pyplot import
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_true = df_test[TARGET].to_numpy(dtype=float)
    y_pred = _predict(model, df_test)
    residual = y_true - y_pred
    temperature = df_test["temperature_c"].to_numpy(dtype=float)
    payload = df_test["payload_t"].to_numpy(dtype=float)

    # EV-green accents to match the product theme.
    primary = "#006d32"
    accent = "#00d166"
    error_col = "#ba1a1a"

    saved: list[str] = []

    # 1) Predicted vs actual ------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4, color=primary, edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=error_col, linewidth=1.5, label="perfect")
    ax.set_xlabel("Actual energy (kWh)")
    ax.set_ylabel("Predicted energy (kWh)")
    ax.set_title("Predicted vs actual segment energy")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p1 = out_path / "predicted_vs_actual.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    saved.append(str(p1.resolve()))

    # 2) Residual vs temperature ------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(temperature, residual, s=10, alpha=0.4, color=accent, edgecolors="none")
    ax.axhline(0.0, color=error_col, linewidth=1.2)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Residual = actual - predicted (kWh)")
    ax.set_title("Residual vs temperature (HVAC-regime bias check)")
    fig.tight_layout()
    p2 = out_path / "residual_vs_temperature.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    saved.append(str(p2.resolve()))

    # 3) MAE by payload bin ------------------------------------------------- #
    bins = [
        ("light\n(<7t)", payload < 7),
        ("mid\n(7-15t)", (payload >= 7) & (payload <= 15)),
        ("heavy\n(>15t)", payload > 15),
    ]
    labels: list[str] = []
    maes: list[float] = []
    for label, mask in bins:
        labels.append(label)
        if mask.any():
            maes.append(float(np.mean(np.abs(residual[mask]))))
        else:
            maes.append(0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, maes, color=primary)
    ax.set_ylabel("Mean absolute error (kWh)")
    ax.set_title("Error by payload bin")
    for i, value in enumerate(maes):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p3 = out_path / "error_by_payload.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    saved.append(str(p3.resolve()))

    return saved


__all__ = ["evaluate", "failure_mode_report", "make_plots", "NOMINAL_TRIP_KM"]
