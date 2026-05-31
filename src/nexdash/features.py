"""Feature engineering for the NexDash energy model.

This module is the single source of truth for the feature schema used by the
energy-prediction model. It defines the raw feature columns the rest of the
pipeline produces (see :mod:`nexdash.data_gen`), the prediction target, and the
*engineered* feature matrix consumed by :mod:`nexdash.model`.

Two entry points are exposed:

* :func:`build_features` — split a full dataset (with target) into an engineered
  feature matrix ``X`` and the target series ``y``. Used during training.
* :func:`transform` — turn one or many raw feature rows (a ``dict`` for a single
  sample or a :class:`pandas.DataFrame`) into the engineered matrix, with columns
  in the canonical :data:`ENGINEERED_COLUMNS` order. Used at inference time.

Keeping feature engineering centralised here guarantees that training and
inference apply exactly the same transformations, avoiding train/serve skew.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #

#: Raw input features, in the order produced by ``nexdash.data_gen``.
FEATURE_COLUMNS: list[str] = [
    "distance_km",
    "payload_t",
    "speed_kph",
    "gradient_pct",
    "temperature_c",
    "wind_mps",
]

#: Name of the regression target column (energy consumed over the segment).
TARGET: str = "energy_kwh"

#: Engineered (derived) feature names, appended after the raw features.
#: Each captures a non-linear or interaction effect that the underlying physics
#: exhibits, helping the (especially linear baseline) model fit better:
#:
#: * ``abs_gradient``        — magnitude of slope; both up- and down-hill change
#:   energy use, so the *absolute* gradient carries signal beyond the signed one.
#: * ``temp_dev_from_20``    — |temperature - 20 C|; HVAC/auxiliary load grows at
#:   both cold and hot extremes, a V-shape that a raw temperature cannot express.
#: * ``payload_x_gradient``  — payload x gradient; gravity work on a slope scales
#:   with total mass, so heavy loads amplify gradient cost (a true interaction).
#: * ``speed_sq``            — speed squared; aerodynamic drag power grows with
#:   the square of speed, the dominant non-linearity at highway speeds.
#: * ``payload_x_distance``  — payload x distance; rolling-resistance energy is
#:   proportional to mass times distance travelled.
_DERIVED_COLUMNS: list[str] = [
    "abs_gradient",
    "temp_dev_from_20",
    "payload_x_gradient",
    "speed_sq",
    "payload_x_distance",
]

#: Full engineered feature matrix column order: raw features then derived ones.
ENGINEERED_COLUMNS: list[str] = FEATURE_COLUMNS + _DERIVED_COLUMNS


# --------------------------------------------------------------------------- #
# Engineering helpers
# --------------------------------------------------------------------------- #

def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame holding the engineered features in canonical order.

    Args:
        df: A frame containing (at least) every column in :data:`FEATURE_COLUMNS`.

    Returns:
        A DataFrame whose columns are exactly :data:`ENGINEERED_COLUMNS`, in
        order, with the same index as ``df``.

    Raises:
        KeyError: If any required raw feature column is missing.
        ValueError: If any raw feature value is non-numeric or non-finite
            (NaN/inf), which would otherwise propagate silently into the feature
            matrix and yield a garbage prediction.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature column(s): {missing}")

    out = df[FEATURE_COLUMNS].copy()

    # Fail loud on non-numeric / non-finite inputs rather than producing a
    # silently-wrong prediction (e.g. a NaN temperature or a string speed).
    try:
        numeric = out.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric feature value(s): {exc}") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        bad = [c for c in FEATURE_COLUMNS if not np.isfinite(numeric[c].to_numpy()).all()]
        raise ValueError(f"Non-finite (NaN/inf) feature value(s) in: {bad}")
    out = numeric

    # |slope|: both directions affect energy (climb costs, descent regens).
    out["abs_gradient"] = out["gradient_pct"].abs()
    # |T - 20 C|: V-shaped auxiliary/HVAC load away from a ~20 C comfort point.
    out["temp_dev_from_20"] = (out["temperature_c"] - 20.0).abs()
    # mass x slope interaction: gravity work on a grade scales with load.
    out["payload_x_gradient"] = out["payload_t"] * out["gradient_pct"]
    # v^2: aerodynamic drag power is quadratic in speed.
    out["speed_sq"] = out["speed_kph"] ** 2
    # mass x distance: rolling-resistance energy proxy.
    out["payload_x_distance"] = out["payload_t"] * out["distance_km"]

    # Guarantee canonical column order regardless of insertion order.
    return out[ENGINEERED_COLUMNS]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled dataset into an engineered feature matrix and target.

    Args:
        df: A dataset containing every :data:`FEATURE_COLUMNS` column and the
            :data:`TARGET` column.

    Returns:
        A ``(X, y)`` tuple where ``X`` is the engineered feature matrix with
        columns in :data:`ENGINEERED_COLUMNS` order and ``y`` is the target
        :class:`pandas.Series`. Both share ``df``'s index.

    Raises:
        KeyError: If a required feature or the target column is missing.
    """
    if TARGET not in df.columns:
        raise KeyError(f"Missing target column: {TARGET!r}")
    X = _add_engineered(df)
    y = df[TARGET].copy()
    return X, y


def transform(
    df_or_dict: Union[pd.DataFrame, Mapping[str, object]],
) -> pd.DataFrame:
    """Engineer features for one or many raw samples.

    This is the inference-time counterpart of :func:`build_features`: it never
    touches the target and accepts either a single sample as a mapping or a
    batch as a DataFrame.

    Args:
        df_or_dict: Either a mapping of ``{feature_name: value}`` for a single
            sample, or a :class:`pandas.DataFrame` of one or more samples. In
            both cases every :data:`FEATURE_COLUMNS` entry must be present.

    Returns:
        A DataFrame with columns in :data:`ENGINEERED_COLUMNS` order. A mapping
        input yields a single-row frame; a DataFrame input preserves its index.

    Raises:
        KeyError: If any required raw feature column is missing.
        TypeError: If ``df_or_dict`` is neither a mapping nor a DataFrame.
    """
    if isinstance(df_or_dict, pd.DataFrame):
        df = df_or_dict
    elif isinstance(df_or_dict, Mapping):
        # Wrap a single sample into a one-row frame.
        df = pd.DataFrame([dict(df_or_dict)])
    elif isinstance(df_or_dict, Sequence) and not isinstance(df_or_dict, (str, bytes)):
        # A sequence of mappings (list[dict]) becomes a multi-row frame.
        rows = list(df_or_dict)
        if rows and not all(isinstance(r, Mapping) for r in rows):
            raise TypeError(
                "transform() expects a sequence of mappings (list[dict]); "
                "found a non-mapping element"
            )
        df = pd.DataFrame([dict(r) for r in rows])
    else:
        raise TypeError(
            "transform() expects a pandas.DataFrame, a mapping (dict), or a "
            f"sequence of mappings, got {type(df_or_dict).__name__}"
        )
    return _add_engineered(df)
