"""Tests for :mod:`nexdash.features`.

These tests verify the *intent* of the feature layer: training and inference
must apply identical engineering (no train/serve skew), the canonical column
order/schema must be stable (downstream model unpickling depends on it), and
the derived features must encode the physics they claim to (V-shaped HVAC load,
direction-agnostic gradient, mass-on-slope interaction).
"""

from __future__ import annotations

import pandas as pd
import pytest

from nexdash.features import (
    ENGINEERED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET,
    build_features,
    transform,
)


# A single, fully-specified raw sample reused across tests. Values are chosen
# so every engineered quantity is non-trivial (non-zero, sign-carrying).
SAMPLE = {
    "distance_km": 50.0,
    "payload_t": 10.0,
    "speed_kph": 80.0,
    "gradient_pct": -3.0,
    "temperature_c": 5.0,
    "wind_mps": 4.0,
}


def _sample_frame(rows: int = 3) -> pd.DataFrame:
    """Build a small labelled DataFrame with distinct, varied rows."""
    data = {
        "distance_km": [50.0, 10.0, 120.0],
        "payload_t": [10.0, 0.0, 22.0],
        "speed_kph": [80.0, 30.0, 60.0],
        "gradient_pct": [-3.0, 0.0, 5.0],
        "temperature_c": [5.0, 20.0, 38.0],
        "wind_mps": [4.0, 0.0, 12.0],
        TARGET: [40.0, 5.0, 95.0],
    }
    return pd.DataFrame({k: v[:rows] for k, v in data.items()})


# --------------------------------------------------------------------------- #
# Schema / contract
# --------------------------------------------------------------------------- #

def test_engineered_columns_extend_raw_features():
    """ENGINEERED_COLUMNS must start with the raw features in canonical order.

    The model persists the column order; reordering would silently corrupt
    inference, so this contract is load-bearing.
    """
    assert ENGINEERED_COLUMNS[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    # No duplicate names — duplicates would collide as DataFrame columns.
    assert len(ENGINEERED_COLUMNS) == len(set(ENGINEERED_COLUMNS))
    # Derived features actually add signal beyond the raw inputs.
    assert len(ENGINEERED_COLUMNS) > len(FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# build_features
# --------------------------------------------------------------------------- #

def test_build_features_returns_X_and_y_with_right_shape_and_columns():
    df = _sample_frame()
    X, y = build_features(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    # X carries exactly the engineered schema in order.
    assert list(X.columns) == ENGINEERED_COLUMNS
    # y is the target, untouched.
    assert y.name == TARGET
    pd.testing.assert_series_equal(y, df[TARGET])
    # Rows preserved and index aligned with input.
    assert len(X) == len(df)
    assert list(X.index) == list(df.index)


def test_build_features_does_not_leak_target_into_X():
    """The target must never appear as a feature (would be data leakage)."""
    df = _sample_frame()
    X, _ = build_features(df)
    assert TARGET not in X.columns


def test_build_features_missing_target_raises():
    df = _sample_frame().drop(columns=[TARGET])
    with pytest.raises(KeyError):
        build_features(df)


def test_build_features_missing_raw_feature_raises():
    df = _sample_frame().drop(columns=["speed_kph"])
    with pytest.raises(KeyError):
        build_features(df)


# --------------------------------------------------------------------------- #
# transform — single dict
# --------------------------------------------------------------------------- #

def test_transform_dict_returns_single_row_in_canonical_order():
    out = transform(SAMPLE)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert list(out.columns) == ENGINEERED_COLUMNS


def test_transform_dataframe_preserves_index_and_columns():
    df = _sample_frame().drop(columns=[TARGET])
    df.index = [100, 200, 300]
    out = transform(df)
    assert list(out.columns) == ENGINEERED_COLUMNS
    assert list(out.index) == [100, 200, 300]


def test_transform_rejects_unsupported_type():
    with pytest.raises(TypeError):
        transform(["not", "a", "frame"])


def test_transform_missing_feature_raises():
    incomplete = dict(SAMPLE)
    del incomplete["wind_mps"]
    # wind_mps is a raw feature column, so its absence must be rejected.
    with pytest.raises(KeyError):
        transform(incomplete)


# --------------------------------------------------------------------------- #
# Engineered values are computed correctly (the physics they encode)
# --------------------------------------------------------------------------- #

def test_abs_gradient_is_magnitude_regardless_of_sign():
    """Down-hill (-3%) and up-hill (+3%) yield identical abs_gradient."""
    out = transform(SAMPLE)
    assert out["abs_gradient"].iloc[0] == abs(SAMPLE["gradient_pct"]) == 3.0

    uphill = dict(SAMPLE, gradient_pct=3.0)
    assert transform(uphill)["abs_gradient"].iloc[0] == 3.0


def test_temp_dev_from_20_is_v_shaped_distance_from_comfort():
    """|T - 20| must be symmetric about 20 C and zero at the comfort point."""
    # Cold: 5 C -> 15 away from 20.
    assert transform(SAMPLE)["temp_dev_from_20"].iloc[0] == 15.0
    # Comfort point -> zero auxiliary deviation.
    assert transform(dict(SAMPLE, temperature_c=20.0))["temp_dev_from_20"].iloc[0] == 0.0
    # Hot 35 C is the same distance as cold 5 C from 20 C.
    assert transform(dict(SAMPLE, temperature_c=35.0))["temp_dev_from_20"].iloc[0] == 15.0


def test_interaction_and_power_features_are_exact():
    out = transform(SAMPLE).iloc[0]
    # mass x slope (sign-carrying interaction).
    assert out["payload_x_gradient"] == pytest.approx(
        SAMPLE["payload_t"] * SAMPLE["gradient_pct"]
    )
    # v^2 aerodynamic-drag proxy.
    assert out["speed_sq"] == pytest.approx(SAMPLE["speed_kph"] ** 2)
    # mass x distance rolling-resistance proxy.
    assert out["payload_x_distance"] == pytest.approx(
        SAMPLE["payload_t"] * SAMPLE["distance_km"]
    )


def test_raw_features_passed_through_unchanged():
    out = transform(SAMPLE).iloc[0]
    for col in FEATURE_COLUMNS:
        assert out[col] == SAMPLE[col]


def test_build_features_and_transform_agree():
    """No train/serve skew: build_features (drop target) == transform on the same rows."""
    df = _sample_frame()
    X_train, _ = build_features(df)
    X_infer = transform(df.drop(columns=[TARGET]))
    pd.testing.assert_frame_equal(X_train, X_infer)
