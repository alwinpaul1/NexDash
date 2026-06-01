"""Tests for :mod:`nexdash.promote` — the offline model-promotion gate.

These verify the *decision intent*, not just plumbing:

* A challenger that is genuinely better (trained on far more data) must be
  PROMOTED — the bootstrap CI on the MAE improvement excludes zero.
* Comparing a model against ITSELF must NOT promote — a zero improvement is not
  a significant win, so the gate must refuse (guards against a gate that rubber-
  stamps any challenger).
* The verdict must carry the auditable evidence (CI, slice regressions,
  optimistic-error rates) the long-term story relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexdash.data_gen import generate_dataset, save_dataset
from nexdash.features import FEATURE_COLUMNS, TARGET
from nexdash.model import train_model
from nexdash.promote import (
    SLICE_INDICATIVE_MIN_N,
    SLICE_MIN_N,
    _slice_regressions,
    compare,
)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A frozen dataset plus a weak champion and a strong challenger.

    The champion is trained on a tiny subset (under-fit, higher held-out error);
    the challenger on a large dataset. Both are scored on the same frozen split,
    so the challenger should win with a CI clear of zero.
    """
    d = tmp_path_factory.mktemp("promote")
    dataset = d / "dataset.csv"
    save_dataset(generate_dataset(n_samples=2000, seed=42), dataset)

    champion = d / "champion.joblib"
    challenger = d / "challenger.joblib"
    train_model(generate_dataset(n_samples=120, seed=7), save=True, path=champion)
    train_model(generate_dataset(n_samples=2000, seed=42), save=True, path=challenger)
    return dataset, champion, challenger


def test_better_challenger_is_promoted(models):
    """A clearly stronger challenger must pass the gate with a CI clear of zero."""
    dataset, champion, challenger = models
    verdict = compare(champion, challenger, dataset_path=dataset, n_boot=800)

    assert verdict["promote"] is True
    assert verdict["challenger_mae_kwh"] < verdict["champion_mae_kwh"]
    # The whole 95% CI on the MAE improvement must sit above zero (a real win).
    assert verdict["improvement_ci_95"][0] > 0.0
    assert verdict["wins_significantly"] is True


def test_identical_model_is_not_promoted(models):
    """Comparing a model to itself yields a zero improvement -> must REJECT.

    WHY: a gate that promoted a tie (or any challenger) would defeat its purpose.
    The paired bootstrap on identical predictions gives a degenerate CI at 0, so
    ``wins_significantly`` is False and the model is not promoted.
    """
    dataset, _champion, challenger = models
    verdict = compare(challenger, challenger, dataset_path=dataset, n_boot=800)

    assert verdict["promote"] is False
    assert verdict["wins_significantly"] is False
    assert verdict["mae_improvement_kwh"] == pytest.approx(0.0, abs=1e-9)


def test_verdict_carries_auditable_evidence(models):
    """The verdict must expose the evidence the long-term story claims to use."""
    dataset, champion, challenger = models
    verdict = compare(champion, challenger, dataset_path=dataset, n_boot=400)

    for key in (
        "improvement_ci_95",
        "slice_regressions",
        "optimistic_rate_champion",
        "optimistic_rate_challenger",
        "reasons",
        "n_test",
    ):
        assert key in verdict
    assert isinstance(verdict["reasons"], list) and verdict["reasons"]
    assert 0.0 <= verdict["optimistic_rate_challenger"] <= 1.0


# --------------------------------------------------------------------------- #
# Steep-grade blind-spot regression guard (the slice-veto support tiers)
#
# WHY: the steep-grade safety bins are ~1.6% of rows, so a held-out fold holds
# only ~18-20 steep rows — below the old SLICE_MIN_N=30 floor. The veto used to
# `continue` past them, so a challenger that regressed *only* on the steep tail
# (the most dangerous regime) passed with a clean verdict. These tests pin the
# two-tier fix: strict veto when well-supported, indicative veto on the sparse
# steep tail, and explicit disclosure when too sparse to veto at all.
# --------------------------------------------------------------------------- #

_TRUTH_KWH = 100.0
_STEEP_LABEL = "gradient:steep_up (>+4%)"


class _ConstModel:
    """Predict a constant truth, plus a fixed error on steep-up rows only.

    The champion (``steep_error=0``) is exact everywhere, so every slice MAE is
    the challenger's own error — isolating the steep-grade regression we inject.
    """

    def __init__(self, steep_error: float = 0.0) -> None:
        self._steep_error = float(steep_error)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        grad = np.asarray(X["gradient_pct"], dtype=float)
        return _TRUTH_KWH + np.where(grad > 4, self._steep_error, 0.0)


def _df_with_steep(n_steep: int, *, n_flat: int = 300) -> pd.DataFrame:
    """``n_flat`` flat rows + ``n_steep`` steep-up rows, all mild temp / mid load.

    Truth is constant, so the only slice that can regress is ``gradient:steep_up``;
    the many flat rows dilute the steep error inside the well-supported
    temperature/payload bins below their tolerance, keeping the test isolated.
    """
    base = dict(distance_km=100.0, payload_t=10.0, speed_kph=70.0,
                gradient_pct=0.0, temperature_c=15.0, wind_mps=0.0)
    rows = [dict(base) for _ in range(n_flat)]
    rows += [dict(base, gradient_pct=6.0) for _ in range(n_steep)]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df[TARGET] = _TRUTH_KWH
    return df


def test_well_supported_slice_still_blocks_strictly():
    """A well-supported (n >= 30) steep regression still blocks at the tight tol."""
    df = _df_with_steep(40)
    regressions, unguarded = _slice_regressions(_ConstModel(0.0), _ConstModel(3.0), df)
    steep = [r for r in regressions if r["slice"] == _STEEP_LABEL]
    assert steep and steep[0]["indicative"] is False and steep[0]["n"] >= SLICE_MIN_N
    assert not unguarded


def test_sparse_steep_slice_is_now_vetoed_indicatively():
    """THE FIX: a ~15-row steep regression (below the old 30-row floor) must now
    BLOCK, flagged indicative — not silently skipped as it was before."""
    df = _df_with_steep(15)
    regressions, _unguarded = _slice_regressions(_ConstModel(0.0), _ConstModel(3.0), df)
    steep = [r for r in regressions if r["slice"] == _STEEP_LABEL]
    assert steep, "sparse steep regression must be vetoed, not silently skipped"
    assert steep[0]["indicative"] is True
    assert SLICE_INDICATIVE_MIN_N <= steep[0]["n"] < SLICE_MIN_N


def test_ultra_sparse_steep_slice_is_disclosed_not_silent():
    """Too few rows to trust a veto (n < 8) must still be DISCLOSED as unguarded
    rather than dropped with a clean verdict — the silent-failure being fixed."""
    df = _df_with_steep(5)
    regressions, unguarded = _slice_regressions(_ConstModel(0.0), _ConstModel(3.0), df)
    assert not any(r["slice"] == _STEEP_LABEL for r in regressions)
    flagged = [r for r in unguarded if r["slice"] == _STEEP_LABEL]
    assert flagged and flagged[0]["n"] < SLICE_INDICATIVE_MIN_N
