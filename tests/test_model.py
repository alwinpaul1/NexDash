"""Tests for :mod:`nexdash.model`.

These verify the *intent* of the energy model, not merely its mechanics:

* Training must produce finite, sensible metrics on a held-out split and must
  beat a trivial mean-predictor on MAE — otherwise the learned model adds no
  value over guessing the dataset average and should not ship.
* The inference API must accept the raw-feature shapes callers actually use
  (a single ``dict`` and a ``list[dict]``), because every downstream consumer
  (range checker, tools, dashboard) passes raw feature dicts, never engineered
  matrices.
* ``save``/``load`` must round-trip to *identical* predictions; a persisted
  model that drifts from the in-memory one would silently break the served
  predictions that the API loads at startup.

A small ``n_samples`` keeps the suite fast while remaining large enough for a
gradient-boosted fit to be meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexdash.data_gen import generate_dataset
from nexdash.features import FEATURE_COLUMNS, TARGET
from nexdash.model import EnergyModel, predict_energy, train_model


# Small but non-trivial dataset; deterministic via the generator's seed.
N_SAMPLES = 800
SEED = 42


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """A small, deterministic synthetic dataset for model tests."""
    return generate_dataset(n_samples=N_SAMPLES, seed=SEED)


@pytest.fixture(scope="module")
def trained(dataset: pd.DataFrame) -> EnergyModel:
    """An EnergyModel trained once (no disk write) for read-only assertions."""
    model = EnergyModel()
    model.train(dataset)
    return model


def _sample_row(dataset: pd.DataFrame) -> dict:
    """One raw feature dict (target excluded) drawn from the dataset."""
    row = dataset.iloc[0]
    return {col: float(row[col]) for col in FEATURE_COLUMNS}


# --------------------------------------------------------------------------- #
# Training quality
# --------------------------------------------------------------------------- #


def test_train_returns_finite_metrics(trained: EnergyModel) -> None:
    """train() must report finite MAE/RMSE/R^2 for both estimators.

    Non-finite metrics signal a broken fit (NaNs propagating, empty split,
    etc.) which would make every downstream confidence claim meaningless.
    """
    metrics = trained.metrics
    assert set(metrics) >= {"hgb", "linear", "n_train", "n_test"}

    for key in ("hgb", "linear"):
        m = metrics[key]
        assert np.isfinite(m["mae_kwh"]), f"{key} MAE not finite"
        assert np.isfinite(m["rmse_kwh"]), f"{key} RMSE not finite"
        assert np.isfinite(m["r2"]), f"{key} R^2 not finite"
        assert m["mae_kwh"] >= 0.0
        assert m["rmse_kwh"] >= m["mae_kwh"] - 1e-9  # RMSE >= MAE always

    assert metrics["n_train"] > 0 and metrics["n_test"] > 0


def test_model_beats_mean_predictor(dataset: pd.DataFrame) -> None:
    """The fitted model must beat predicting the (train) mean on test MAE.

    A model that cannot outperform the dataset average has learned nothing
    useful about the physics; this is the minimum bar for shipping it. We
    recompute the same split the model uses internally so the comparison is
    apples-to-apples on identical held-out rows.
    """
    from sklearn.model_selection import train_test_split

    from nexdash import features

    X, y = features.build_features(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = EnergyModel()
    model.train(dataset)  # uses the same test_size/seed defaults internally

    # Trivial baseline: always predict the training-set mean.
    mean_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
    mean_mae = float(np.mean(np.abs(y_test.to_numpy() - mean_pred)))

    model_mae = model.metrics["hgb"]["mae_kwh"]
    assert model_mae < mean_mae, (
        f"HGB MAE {model_mae:.3f} did not beat mean-predictor MAE {mean_mae:.3f}"
    )
    # The gradient-boosted model should also be at least as good as the
    # linear baseline on this strongly non-linear target.
    assert model.metrics["hgb"]["mae_kwh"] <= model.metrics["linear"]["mae_kwh"]


def test_predictions_track_actuals(trained: EnergyModel, dataset: pd.DataFrame) -> None:
    """In-sample predictions should correlate strongly with true energy.

    This guards against a model that returns a near-constant value (which can
    pass MAE checks on a low-variance slice but is useless operationally).
    """
    preds = trained.predict(dataset[FEATURE_COLUMNS])
    actual = dataset[TARGET].to_numpy()
    corr = float(np.corrcoef(preds, actual)[0, 1])
    assert corr > 0.9, f"prediction/actual correlation too low: {corr:.3f}"


# --------------------------------------------------------------------------- #
# Inference input shapes
# --------------------------------------------------------------------------- #


def test_predict_accepts_single_dict(trained: EnergyModel, dataset: pd.DataFrame) -> None:
    """predict() accepts one raw feature dict and returns a length-1 array."""
    out = trained.predict(_sample_row(dataset))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)
    assert np.isfinite(out[0])


def test_predict_accepts_list_of_dicts(trained: EnergyModel, dataset: pd.DataFrame) -> None:
    """predict() accepts a list[dict] and returns one prediction per row."""
    rows = [
        {col: float(dataset.iloc[i][col]) for col in FEATURE_COLUMNS}
        for i in range(3)
    ]
    out = trained.predict(rows)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_predict_dict_matches_dataframe(trained: EnergyModel, dataset: pd.DataFrame) -> None:
    """A dict and the equivalent one-row DataFrame must predict identically.

    Confirms the dict convenience path routes through the same engineering as
    the DataFrame path (no train/serve skew from input typing).
    """
    row = _sample_row(dataset)
    dict_pred = trained.predict(row)[0]
    df_pred = trained.predict(pd.DataFrame([row]))[0]
    assert dict_pred == pytest.approx(df_pred)


# --------------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip_predicts_identically(
    dataset: pd.DataFrame, tmp_path
) -> None:
    """save() then load() must reproduce byte-for-byte identical predictions.

    The served API loads the persisted artifact; if a reloaded model diverged
    from the trained one, production predictions would silently differ from
    what was evaluated.
    """
    path = tmp_path / "energy_model.joblib"
    model = train_model(dataset, save=True, path=path)
    assert path.exists()

    reloaded = EnergyModel.load(path)
    assert reloaded.feature_columns == model.feature_columns

    rows = dataset[FEATURE_COLUMNS]
    np.testing.assert_array_equal(model.predict(rows), reloaded.predict(rows))


def test_load_missing_path_raises(tmp_path) -> None:
    """Loading a non-existent artifact fails loudly (not silently empty)."""
    with pytest.raises(FileNotFoundError):
        EnergyModel.load(tmp_path / "does_not_exist.joblib")


def test_predict_energy_helper_uses_saved_model(
    dataset: pd.DataFrame, tmp_path
) -> None:
    """predict_energy() returns a float matching the model's own prediction.

    Also exercises the module-level model cache keyed by resolved path.
    """
    path = tmp_path / "energy_model.joblib"
    model = train_model(dataset, save=True, path=path)

    row = _sample_row(dataset)
    helper_val = predict_energy(row, model_path=path)
    assert isinstance(helper_val, float)
    assert np.isfinite(helper_val)
    assert helper_val == pytest.approx(float(model.predict(row)[0]))


def test_predict_cache_invalidated_on_retrain(tmp_path) -> None:
    """predict_energy must NOT serve a stale model after a retrain to the same path.

    WHY: the cache is keyed by (path, file mtime); overwriting the artifact in the
    same process must be picked up. A plain path key would return the first model's
    prediction forever, silently serving a stale model after a hot retrain.
    """
    import time

    path = tmp_path / "m.joblib"
    row = {col: float(v) for col, v in zip(FEATURE_COLUMNS, (80, 12, 75, 1.0, 5.0, 2.0))}

    a = train_model(generate_dataset(n_samples=500, seed=1), save=True, path=path)
    v1 = predict_energy(row, model_path=path)
    assert v1 == pytest.approx(float(a.predict(row)[0]))

    time.sleep(0.01)  # guarantee a distinct file mtime for the overwrite
    b = train_model(generate_dataset(n_samples=500, seed=123), save=True, path=path)
    v2 = predict_energy(row, model_path=path)
    # The fresh model must drive the prediction (not the cached first model).
    assert v2 == pytest.approx(float(b.predict(row)[0])), "stale model served after retrain"
