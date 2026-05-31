"""Tests for :mod:`nexdash.evaluate`.

These tests verify *intent*, not just mechanics:

* ``evaluate`` must report the documented headline keys, and on a model that
  actually learned the (mostly deterministic, lightly-noised) physics target,
  those numbers must land in defensible ranges -- a positive MAE (no model is
  perfect on noisy labels) and an R^2 that is high but not impossibly equal to
  1.0. A test that merely checks "keys exist" could pass for a broken model, so
  we also assert the metrics are sane and self-consistent (RMSE >= MAE).
* ``failure_mode_report`` exists so a dispatcher can see *where* error
  concentrates. The contract is the bin structure (temp / gradient / payload
  with three labels each); we assert that structure and that populated bins
  carry the per-slice metric keys.
* ``make_plots`` must run headless and actually write the three diagnostic PNGs
  to the requested directory; we assert files exist on disk and are non-empty.

A tiny model is trained inline (small dataset, default hyper-params) so the
suite stays fast while still exercising the real ``EnergyModel`` -> ``evaluate``
integration rather than a stand-in stub.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# Skip the whole module cleanly if scientific deps are unavailable, rather than
# erroring at import time (keeps the rest of the suite runnable).
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from nexdash.data_gen import generate_dataset  # noqa: E402
from nexdash.evaluate import evaluate, failure_mode_report, make_plots  # noqa: E402
from nexdash.features import FEATURE_COLUMNS, TARGET  # noqa: E402
from nexdash.model import EnergyModel  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: one shared tiny trained model + held-out test split.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained() -> tuple[EnergyModel, pd.DataFrame]:
    """Train a small model and return it with an UNSEEN test split.

    We split first, then train only on the train portion, so ``df_test`` is
    genuinely held out -- evaluating on it measures generalisation, which is the
    whole point of :func:`evaluate`.
    """
    df = generate_dataset(n_samples=1200, seed=7)
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=7)

    model = EnergyModel()
    model.train(df_train.reset_index(drop=True))
    return model, df_test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_returns_documented_keys(trained) -> None:
    model, df_test = trained
    result = evaluate(model, df_test)

    expected_keys = {
        "mae_kwh",
        "rmse_kwh",
        "mape_pct",
        "pct_range_error",
        "r2",
        "n",
        "mape_n",
    }
    assert expected_keys.issubset(result.keys()), (
        f"evaluate must return the documented headline keys; "
        f"got {sorted(result.keys())}"
    )
    # mape_n is the count of rows above the MAPE floor; it is surfaced so the
    # report can disclose how many near-zero rows the headline MAPE excludes.
    # It must be a subset of the scored rows.
    assert 0 <= result["mape_n"] <= result["n"]


def test_evaluate_metrics_are_sane(trained) -> None:
    """Metrics must be self-consistent and in defensible ranges.

    These bounds encode *why* the metrics matter: a real model on noisy labels
    has positive error (MAE > 0), large misses are penalised at least as hard by
    RMSE (RMSE >= MAE), R^2 cannot exceed 1, and a model that learned the
    physics signal should explain most variance (R^2 not catastrophically low).
    """
    model, df_test = trained
    result = evaluate(model, df_test)

    # Error is real and positive on noisy data.
    assert result["mae_kwh"] > 0.0
    # RMSE penalises tail errors at least as much as MAE.
    assert result["rmse_kwh"] >= result["mae_kwh"] - 1e-9
    # R^2 is bounded above by 1 and should be high for a learnable target,
    # but the lower bound stays generous so the test isn't flaky on a tiny fit.
    assert result["r2"] <= 1.0
    assert result["r2"] > 0.5, f"model should explain real variance, r2={result['r2']}"
    # pct_range_error is MAE scaled to a nominal full charge: a small positive %.
    assert 0.0 < result["pct_range_error"] < 100.0
    # MAPE, when defined, is a positive percentage.
    assert math.isnan(result["mape_pct"]) or result["mape_pct"] > 0.0
    # n reflects the rows actually scored.
    assert result["n"] == len(df_test)


def test_evaluate_missing_target_raises(trained) -> None:
    """A test frame without the target column is a programming error: fail loud."""
    model, df_test = trained
    with pytest.raises(KeyError):
        evaluate(model, df_test.drop(columns=[TARGET]))


# --------------------------------------------------------------------------- #
# failure_mode_report
# --------------------------------------------------------------------------- #
def test_failure_mode_report_structure(trained) -> None:
    """The report's value is its slicing; assert all three dimensions/labels."""
    model, df_test = trained
    report = failure_mode_report(model, df_test)

    assert set(report.keys()) == {"temperature", "gradient", "payload"}

    assert set(report["temperature"].keys()) == {
        "cold (<0C)",
        "mild (0-30C)",
        "hot (>30C)",
    }
    assert set(report["gradient"].keys()) == {
        "steep_down (<-4%)",
        "flat (-4..+4%)",
        "steep_up (>+4%)",
    }
    assert set(report["payload"].keys()) == {
        "light (<7t)",
        "mid (7-15t)",
        "heavy (>15t)",
    }


def test_failure_mode_report_slices_carry_metrics(trained) -> None:
    """Each slice must expose the per-slice metric keys, and counts must add up.

    Summing ``n`` across a dimension's (mutually exclusive, exhaustive) bins must
    equal the test-set size -- this is what makes the report a faithful
    partition of the data rather than an arbitrary subset.
    """
    model, df_test = trained
    report = failure_mode_report(model, df_test)

    metric_keys = {"mae_kwh", "rmse_kwh", "mape_pct", "r2", "n", "mape_n"}
    for dimension, bins in report.items():
        total_n = 0
        for label, metrics in bins.items():
            assert metric_keys.issubset(metrics.keys()), (
                f"{dimension}/{label} missing metric keys: "
                f"{metric_keys - set(metrics.keys())}"
            )
            assert metrics["n"] >= 0
            total_n += metrics["n"]
            # Populated slices must report a positive MAE; empty ones are NaN.
            if metrics["n"] > 0:
                assert metrics["mae_kwh"] >= 0.0
        assert total_n == len(df_test), (
            f"{dimension} bins must partition the test set "
            f"(sum n={total_n} != {len(df_test)})"
        )


# --------------------------------------------------------------------------- #
# make_plots
# --------------------------------------------------------------------------- #
def test_make_plots_writes_png_files(trained, tmp_path: Path) -> None:
    """make_plots must produce three non-empty PNGs in the requested directory."""
    model, df_test = trained
    out_dir = tmp_path / "figures"

    paths = make_plots(model, df_test, out_dir=out_dir)

    assert isinstance(paths, list)
    assert len(paths) == 3, f"expected 3 diagnostic figures, got {len(paths)}"

    for p in paths:
        fp = Path(p)
        assert fp.exists(), f"figure not written: {fp}"
        assert fp.suffix == ".png"
        assert fp.stat().st_size > 0, f"figure is empty: {fp}"
        # Files must land inside the requested output directory.
        assert fp.parent.resolve() == out_dir.resolve()


def test_make_plots_uses_feature_columns(trained, tmp_path: Path) -> None:
    """Sanity: plotting relies on the contracted raw feature columns being present.

    If the model/evaluate path silently changed which columns it reads, this
    guards the assumption that ``FEATURE_COLUMNS`` + ``TARGET`` are sufficient.
    """
    model, df_test = trained
    minimal = df_test[FEATURE_COLUMNS + [TARGET]].copy()
    paths = make_plots(model, minimal, out_dir=tmp_path / "fig2")
    assert len(paths) == 3
