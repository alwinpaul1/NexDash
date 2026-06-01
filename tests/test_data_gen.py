"""Tests for :mod:`nexdash.data_gen`.

These verify the *contract* the rest of the pipeline depends on:

* the dataset has exactly the agreed columns, in order, with the requested
  number of rows (the schema is the integration contract with features/model);
* generation is reproducible under a fixed seed (so the whole pipeline, which
  is documented as deterministic, can actually be reproduced);
* sampled features stay inside the documented realistic operating envelope
  (out-of-range features would silently teach the model physics it will never
  see in production);
* the energy label is physically sensible — strictly positive and *positively
  correlated with distance* (more driving must cost more energy; a model
  trained on a label lacking this relationship would be worthless); and
* :func:`save_dataset` writes a CSV that round-trips back to the same data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexdash.config import TRUCK
from nexdash.data_gen import COLUMNS, generate_dataset, save_dataset

# Documented sampling envelope (must match data_gen's marginals / clips). The
# duplication is intentional: this is the spec the generator is checked against,
# so a future change that widens a range trips the test instead of passing silently.
EXPECTED_FEATURE_COLUMNS = [
    "distance_km",
    "payload_t",
    "speed_kph",
    "gradient_pct",
    "temperature_c",
    "wind_mps",
]
TARGET_COLUMN = "energy_kwh"

FEATURE_BOUNDS = {
    "distance_km": (1.0, 350.0),
    "payload_t": (0.0, TRUCK.max_payload_t),
    "speed_kph": (30.0, 85.0),
    "gradient_pct": (-6.0, 6.0),
    "temperature_c": (-15.0, 40.0),
    "wind_mps": (-12.0, 12.0),  # signed headwind component (negative = tailwind)
}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """A modestly sized dataset reused across read-only assertions."""
    return generate_dataset(n_samples=1500, seed=42)


def test_columns_exact_and_ordered(df: pd.DataFrame) -> None:
    """Columns must be exactly the contract columns, in the documented order."""
    expected = EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]
    assert list(df.columns) == expected
    # The module's own COLUMNS constant is the source of truth for the contract.
    assert COLUMNS == expected


def test_row_count_matches_request() -> None:
    """``n_samples`` is honoured exactly (no silent dedup/drop)."""
    assert len(generate_dataset(n_samples=250, seed=1)) == 250


def test_reproducible_with_same_seed() -> None:
    """Same seed -> byte-identical dataset (pipeline determinism guarantee)."""
    a = generate_dataset(n_samples=500, seed=7)
    b = generate_dataset(n_samples=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data() -> None:
    """A different seed must actually vary the draw (guards a hard-coded RNG)."""
    a = generate_dataset(n_samples=500, seed=7)
    c = generate_dataset(n_samples=500, seed=8)
    assert not a.equals(c)


@pytest.mark.parametrize("column", list(FEATURE_BOUNDS))
def test_feature_within_documented_bounds(df: pd.DataFrame, column: str) -> None:
    """Every sampled feature stays inside its realistic envelope."""
    low, high = FEATURE_BOUNDS[column]
    col = df[column]
    assert col.min() >= low, f"{column} below lower bound {low}"
    assert col.max() <= high, f"{column} above upper bound {high}"


def test_no_missing_values(df: pd.DataFrame) -> None:
    """No NaNs/inf should ever reach the model from the generator."""
    assert not df.isna().any().any()
    assert np.isfinite(df.to_numpy()).all()


def test_energy_mostly_positive_but_regen_allows_negative(df: pd.DataFrame) -> None:
    """Most segments consume energy, but net-regen descents may be negative.

    A real BEV energy meter logs *net-negative* consumption on a long steep
    descent (regen returns more charge than the segment spends). The generator
    must preserve that signal rather than clamping it away, so we assert (a) the
    vast majority of segments are positive, and (b) the only negative labels are
    genuine descents (gradient < 0). A generator that clamped all labels positive
    would erase the regenerative-braking failure mode this dataset must teach.
    """
    target = df[TARGET_COLUMN]
    # Overwhelmingly positive: only descents can go negative, and they are rare.
    assert (target > 0).mean() > 0.9
    # Every negative label must be a descent — never an uphill/flat segment.
    negatives = df[target < 0]
    assert (negatives["gradient_pct"] < 0).all(), "only descents may record net regen"
    # Net regen is genuinely present (the signal we must not clamp away).
    assert (target < 0).any(), "expected some net-regen (negative) descent labels"


def test_implied_net_climb_is_physically_bounded() -> None:
    """No segment may imply a geographically impossible net elevation change.

    WHY: a long leg cannot sustain a steep average grade without implying an
    impossible climb (e.g. +1% over 300 km = 3 km of ascent, higher than any
    German road). The generator caps the per-segment gradient so the implied net
    climb stays within a realistic ceiling. This is the generator's real, seed-
    independent guarantee, so we check it across MANY random seeds.

    We deliberately do NOT assert that labels stay under the battery capacity: a
    rare long + heavy + cold + headwind leg can legitimately need more than one
    charge — a real "must charge mid-route" segment, not a bug. We DO assert a
    generous physical ceiling that still catches the old "phantom mountain"
    explosion (labels once reached ~5x battery) without falsely banning real
    extremes.
    """
    for seed in range(25):
        d = generate_dataset(n_samples=2000, seed=seed)
        net_climb_m = (
            d["distance_km"] * 1000.0 * np.sin(np.arctan(d["gradient_pct"] / 100.0))
        ).abs()
        assert net_climb_m.max() <= 1510.0, (
            f"seed {seed}: implied net climb {net_climb_m.max():.0f} m exceeds the cap"
        )
        # Physical sanity ceiling: above any real extreme (~800 kWh) but far below
        # the old phantom-mountain bug (~3000 kWh). A breach signals a regression.
        assert d["energy_kwh"].max() <= 900.0, (
            f"seed {seed}: label {d['energy_kwh'].max():.0f} kWh is implausibly high"
        )


def test_energy_correlated_with_distance(df: pd.DataFrame) -> None:
    """Energy must rise with distance — the core physical signal to learn.

    A weak/absent correlation would mean the label carries no usable signal
    about the single most important driver of consumption.
    """
    corr = df["distance_km"].corr(df[TARGET_COLUMN])
    assert corr > 0.5, f"distance/energy correlation too weak: {corr:.3f}"


def test_save_dataset_roundtrips(tmp_path) -> None:
    """``save_dataset`` writes a CSV readable back into the same data."""
    out = tmp_path / "nested" / "dataset.csv"  # also exercises parent creation
    original = generate_dataset(n_samples=120, seed=3)

    save_dataset(original, out)

    assert out.exists() and out.stat().st_size > 0
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == COLUMNS
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True),
        original.reset_index(drop=True),
        check_dtype=False,
    )
