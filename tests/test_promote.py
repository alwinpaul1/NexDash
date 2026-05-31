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

import pytest

from nexdash.data_gen import generate_dataset, save_dataset
from nexdash.model import train_model
from nexdash.promote import compare


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
