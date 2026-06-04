"""Energy-consumption model for the NexDash EV Truck Range Intelligence system.

This module wraps a scikit-learn regression pipeline that predicts the energy
(kWh) required for a route segment of the Mercedes-Benz eActros 600 given raw
operating features (distance, payload, speed, gradient, temperature, wind).

Design notes
------------
* The primary estimator is a :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
  trained on the **physics residual** ``r = energy_kwh - segment_energy_kwh(...)``
  rather than on raw kWh. At inference the prediction is reconstructed as
  ``physics_baseline + model.predict(residual)``. This makes the deterministic
  physics the structural backbone: physics carries the dominant gradient/distance
  work term *analytically* (and so extrapolates linearly past the training
  envelope), while the tree only learns the bounded, near-zero-mean correction
  (label noise, speed/driver effects) where data is dense. A tree cannot
  extrapolate above the largest label it saw, so a raw-kWh target SATURATES on
  out-of-envelope steep climbs / long distances (the dangerous under-prediction
  direction); the residual reparametrisation removes that failure by construction
  because physics — not the tree — supplies the linear tail.
* The :class:`~sklearn.linear_model.LinearRegression` baseline is still trained on
  the RAW kWh target so the comparison table honestly contrasts the
  physics-residual HGB against a plain linear fit of energy.
* A :class:`~sklearn.linear_model.LinearRegression` baseline is *also* trained on
  the same engineered features so we can report how much the gradient-boosted
  model improves over a simple linear fit. Both sets of metrics are kept on
  :attr:`EnergyModel.metrics` under the ``"hgb"`` and ``"linear"`` keys.
* All feature engineering lives in :mod:`nexdash.features`. This module never
  touches raw columns directly for inference: it always routes through
  :func:`nexdash.features.transform`, so callers may pass raw feature dicts.
* Persistence uses ``joblib``. :func:`predict_energy` caches the loaded default
  model in a module global to avoid re-reading the artifact on every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features
from .config import DEFAULT_MODEL_PATH, MAPE_FLOOR_KWH, TRUCK
from .physics import segment_energy_kwh

__all__ = [
    "EnergyModel",
    "train_model",
    "predict_energy",
]

# A "row" of input may be a full DataFrame, a single feature dict, or a list of
# feature dicts. Feature engineering / column ordering is handled by
# ``features.transform``.
Rows = Union[pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _build_pipeline() -> Pipeline:
    """Construct the primary gradient-boosted pipeline.

    A :class:`StandardScaler` is harmless for tree ensembles (monotonic per
    feature) but keeps the pipeline shape consistent and aids any future
    swap-in of a scale-sensitive estimator.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.08,
                    max_iter=400,
                    max_leaf_nodes=31,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_baseline() -> Pipeline:
    """Construct the linear baseline pipeline (scaled features + OLS)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE / RMSE / R^2 / MAPE for a prediction set.

    MAPE uses the project-wide :data:`nexdash.config.MAPE_FLOOR_KWH` floor (the
    same one :mod:`nexdash.evaluate` uses), so the comparison-table MAPE and the
    report headline MAPE share one definition. ``mape_n`` reports how many rows
    participated (the rest are excluded near-zero-denominator downhill rows).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    mask = np.abs(y_true) >= MAPE_FLOOR_KWH
    if mask.any():
        mape = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
        )
    else:
        mape = float("nan")

    return {
        "mae_kwh": mae,
        "rmse_kwh": rmse,
        "r2": r2,
        "mape_pct": mape,
        "mape_n": int(mask.sum()),
    }


def _physics_baseline(rows: pd.DataFrame) -> np.ndarray:
    """Vectorise :func:`nexdash.physics.segment_energy_kwh` over raw feature rows.

    Computes the deterministic physics energy (kWh) for each row using ONLY the
    canonical raw :data:`nexdash.features.FEATURE_COLUMNS` (so the train-time and
    inference-time baselines are byte-identical and there is no train/serve skew).
    This is the structural backbone of the residual model: the tree learns
    ``energy_kwh - physics_baseline`` and inference adds the baseline back, so the
    prediction follows physics (which extrapolates linearly) on out-of-envelope
    steep/long inputs instead of saturating at the tree's largest seen label.

    Args:
        rows: A DataFrame containing every :data:`FEATURE_COLUMNS` column.

    Returns:
        A 1-D numpy array of physics kWh, one per row, aligned to ``rows``.
    """
    cols = features.FEATURE_COLUMNS
    arr = rows[cols].to_numpy(dtype=float)
    out = np.empty(len(arr), dtype=float)
    for i, (d, p, s, g, t, w) in enumerate(arr):
        out[i] = segment_energy_kwh(
            distance_km=float(d),
            payload_t=float(p),
            speed_kph=float(s),
            gradient_pct=float(g),
            temperature_c=float(t),
            wind_mps=float(w),
            truck=TRUCK,
        )
    return out


class EnergyModel:
    """Trained energy-consumption model with a gradient-boosted primary fit.

    Attributes:
        pipeline: The fitted primary scikit-learn :class:`Pipeline`
            (``StandardScaler`` -> ``HistGradientBoostingRegressor``).
        baseline: The fitted :class:`LinearRegression` baseline pipeline.
        metrics: Held-out metrics for both estimators, keyed ``"hgb"`` and
            ``"linear"``; each value is a dict with ``mae_kwh``, ``rmse_kwh``,
            ``r2`` and ``mape_pct``. Also contains ``n_train`` / ``n_test``.
        feature_columns: Engineered feature names the pipeline expects, in order.
    """

    def __init__(self) -> None:
        self.pipeline: Pipeline = _build_pipeline()
        self.baseline: Pipeline = _build_baseline()
        self.metrics: dict[str, Any] = {}
        self.feature_columns: list[str] = list(features.ENGINEERED_COLUMNS)
        #: When True, the primary pipeline predicts the physics RESIDUAL and
        #: :meth:`predict` reconstructs energy as ``physics_baseline + residual``.
        #: A freshly-trained model sets this True; old artifacts that stored raw
        #: kWh load with the default False (backward-compatible: they predict raw
        #: energy and never add a phantom physics term).
        self.residual_target: bool = True

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        df: pd.DataFrame,
        df_eval: pd.DataFrame | None = None,
        *,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Fit the primary and baseline models and report held-out metrics.

        Two modes:

        * **Explicit hold-out** (``df_eval`` given): fit on ALL rows of ``df``
            and report ``hgb``/``linear`` metrics on ``df_eval``. This is what
            :mod:`run_pipeline` uses with its outer split, so the *served* model,
            the comparison table, the stored artifact metrics, and the report
            headline all describe ONE model on ONE test set (no hidden split).
        * **Internal split** (``df_eval`` is ``None``): split ``df`` and report
            metrics on the inner test fold. Convenient for quick/standalone fits.

        Returns the :attr:`metrics` dict (also stored on the instance).
        """
        X, y = features.build_features(df)
        self.feature_columns = list(X.columns)
        self.residual_target = True

        if df_eval is not None:
            X_eval, y_eval = features.build_features(df_eval)
            X_eval = X_eval.reindex(columns=self.feature_columns)
            X_train, y_train = X, y
            X_test, y_test = X_eval, y_eval
            df_train_raw, df_test_raw = df, df_eval
        else:
            # Split the engineered matrix, target AND the raw frame together so
            # the physics baseline is computed on exactly the test rows.
            (
                X_train,
                X_test,
                y_train,
                y_test,
                df_train_raw,
                df_test_raw,
            ) = train_test_split(X, y, df, test_size=test_size, random_state=seed)

        # Physics baseline per row (raw features only) — the residual backbone.
        p_train = _physics_baseline(df_train_raw)
        p_test = _physics_baseline(df_test_raw)

        # Primary HGB learns the RESIDUAL (energy - physics); the linear baseline
        # stays on raw kWh so the comparison table contrasts the residual-physics
        # hybrid against a plain linear fit of energy, honestly.
        resid_train = y_train.to_numpy() - p_train
        self.pipeline.fit(X_train, resid_train)
        self.baseline.fit(X_train, y_train)

        # Reconstruct kWh-scale predictions so MAE/RMSE/R2/MAPE stay on the kWh
        # scale the report and range.py consume (NOT the residual scale).
        hgb_pred = p_test + self.pipeline.predict(X_test)
        lin_pred = self.baseline.predict(X_test)

        self.metrics = {
            "hgb": _regression_metrics(y_test.to_numpy(), hgb_pred),
            "linear": _regression_metrics(y_test.to_numpy(), lin_pred),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "residual_target": True,
        }
        return self.metrics

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, rows: Rows) -> np.ndarray:
        """Predict energy (kWh) for one or more raw feature rows.

        Args:
            rows: A DataFrame of raw features, a single feature ``dict``, or a
                list of feature dicts. Engineering / column ordering is handled
                by :func:`nexdash.features.transform`.

        Returns:
            A 1-D numpy array of predicted kWh (one entry per input row).
        """
        X = features.transform(rows)
        # Guard column order/identity against the trained pipeline.
        X = X.reindex(columns=self.feature_columns)
        pred = np.asarray(self.pipeline.predict(X), dtype=float)

        # Residual reconstruction: add the deterministic physics baseline back so
        # the prediction tracks physics on out-of-envelope inputs (no saturation).
        # ``X`` already carries the raw FEATURE_COLUMNS as its leading columns
        # (ENGINEERED_COLUMNS == FEATURE_COLUMNS + derived), so the baseline uses
        # the SAME features handed to the tree — no train/serve skew.
        if getattr(self, "residual_target", False):
            pred = _physics_baseline(X) + pred
        return pred

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
        """Serialize this model (both estimators + metrics) to ``path`` via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "baseline": self.baseline,
            "metrics": self.metrics,
            "feature_columns": self.feature_columns,
            "residual_target": getattr(self, "residual_target", True),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_MODEL_PATH) -> "EnergyModel":
        """Load a previously saved model from ``path``.

        Raises:
            FileNotFoundError: If no artifact exists at ``path`` (with a clear
                hint to run the training pipeline).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained energy model found at '{path}'. "
                "Run `python run_pipeline.py` (or `nexdash.model.train_model(...)`) "
                "to generate it first."
            )
        payload = joblib.load(path)
        obj = cls.__new__(cls)  # bypass __init__ to restore fitted state
        obj.pipeline = payload["pipeline"]
        obj.baseline = payload["baseline"]
        obj.metrics = payload.get("metrics", {})
        obj.feature_columns = payload.get(
            "feature_columns", list(features.ENGINEERED_COLUMNS)
        )
        # Default False for backward-compat: a pre-residual artifact stored raw
        # kWh, so it must NOT add a physics term at inference. The retrained
        # artifact persists True and reconstructs energy = physics + residual.
        obj.residual_target = payload.get("residual_target", False)
        return obj


def train_model(
    df: pd.DataFrame,
    df_eval: pd.DataFrame | None = None,
    save: bool = True,
    path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> EnergyModel:
    """Train an :class:`EnergyModel` on ``df`` and optionally persist it.

    Args:
        df: Raw training dataset (features + target column).
        df_eval: Optional explicit hold-out set. When given, the model is fit on
            ALL of ``df`` and metrics are reported on ``df_eval`` (consistent with
            :mod:`run_pipeline`'s outer split). When ``None``, an internal split
            of ``df`` is used for the reported metrics.
        save: If ``True``, save the fitted model to ``path``.
        path: Destination artifact path.

    Returns:
        The trained :class:`EnergyModel`.
    """
    model = EnergyModel()
    model.train(df, df_eval)
    if save:
        model.save(path)
    return model


# Module-level cache of the default model, keyed by (resolved path, mtime). The
# mtime component means an in-process retrain that OVERWRITES the same path is
# picked up automatically — a plain path key would serve the stale model.
_MODEL_CACHE: dict[tuple[str, int], EnergyModel] = {}


def predict_energy(
    features: dict[str, Any],  # noqa: A002 - matches public interface contract
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> float:
    """Predict energy (kWh) for a single raw feature dict using the saved model.

    The model is cached by ``(resolved path, file mtime)`` so repeated calls are
    fast, yet a retrain to the same path is never served stale.

    Args:
        features: A single raw feature dict (e.g. ``{"distance_km": 50,
            "payload_t": 10, "speed_kph": 70, "gradient_pct": 1.5,
            "temperature_c": 5, "wind_mps": 3}``).
        model_path: Path to the saved model artifact.

    Returns:
        Predicted energy for the segment in kWh.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """
    resolved = Path(model_path).resolve()
    key = (str(resolved), resolved.stat().st_mtime_ns)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = EnergyModel.load(model_path)
        _MODEL_CACHE[key] = model
    return float(model.predict(features)[0])
