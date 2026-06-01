"""Integration tests for the FastAPI server (``dashboard/server.py``).

These exercise the HTTP API surface end-to-end with FastAPI's ``TestClient``:

* ``POST /api/predict`` must run a real range check and return a
  :func:`nexdash.range.check_reachability` dict (notably with a boolean
  ``reaches`` field) once a trained model artifact is present.
* ``POST /api/predict`` must degrade gracefully (HTTP 503 + clear message)
  when no model artifact exists, since that is the operator-facing contract.

WHY these assertions matter (not just WHAT they check):

* ``reaches`` is the single most important field a dispatcher acts on; a
  response that omits it or returns a non-bool would silently break the UI's
  green/red verdict, so we assert its presence and type explicitly.
* The 503 path is the documented "you forgot to run the pipeline" guardrail;
  if it regressed to a 500 or a stack trace the operator would be misled.

A tiny model (small ``n``, default estimator) is trained and saved to a
``tmp_path`` artifact; ``server.DEFAULT_MODEL_PATH`` is monkeypatched so both
the startup loader and the request handler use that artifact instead of the
repository's real ``models/energy_model.joblib``. The prediction cache in
``nexdash.model`` is cleared around each test so a stale path can never leak
between tests.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make the dashboard package importable (it lives at repo-root/dashboard, not
# under src/). The repo root is two levels up from this test file.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DASHBOARD_DIR = _REPO_ROOT / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import server as server_module  # noqa: E402  (dashboard/server.py)
from nexdash import data_gen, model as model_module  # noqa: E402


# A representative, comfortably-reachable trip used by the happy-path test.
_SAMPLE_TRIP = {
    "soc_pct": 80.0,
    "distance_km": 40.0,
    "payload_t": 8.0,
    "speed_kph": 65.0,
    "gradient_pct": 1.0,
    "temperature_c": 10.0,
    "wind_mps": 2.0,
}


def _train_tiny_model(path: Path) -> None:
    """Train and persist a small but real EnergyModel artifact at ``path``."""
    df = data_gen.generate_dataset(n_samples=400, seed=7)
    model_module.train_model(df, save=True, path=path)


@pytest.fixture
def client_with_model(tmp_path, monkeypatch):
    """A TestClient backed by a freshly trained tiny model at a temp path.

    The artifact path is injected into ``server`` so neither startup nor the
    request handler touches the repository's real model file.
    """
    model_path = tmp_path / "tiny_energy_model.joblib"
    _train_tiny_model(model_path)

    # The route passes ``server.DEFAULT_MODEL_PATH`` into check_reachability and
    # the lifespan loader reads the same name, so patching it here covers both.
    monkeypatch.setattr(server_module, "DEFAULT_MODEL_PATH", model_path, raising=True)

    # Ensure predict_energy's module-level cache cannot serve a stale model.
    model_module._MODEL_CACHE.clear()

    # ``with TestClient(...)`` triggers the lifespan startup, loading the model.
    with TestClient(server_module.app) as client:
        yield client

    model_module._MODEL_CACHE.clear()


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    """A TestClient whose configured model artifact does NOT exist."""
    missing_path = tmp_path / "does_not_exist.joblib"
    assert not missing_path.exists()

    monkeypatch.setattr(server_module, "DEFAULT_MODEL_PATH", missing_path, raising=True)
    model_module._MODEL_CACHE.clear()

    with TestClient(server_module.app) as client:
        yield client

    model_module._MODEL_CACHE.clear()


def test_health_reports_model_available(client_with_model):
    """The health endpoint must reflect that the model loaded at startup."""
    resp = client_with_model.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["model_available"] is True


def test_predict_returns_reachability_dict(client_with_model):
    """POST /api/predict returns a full reachability dict with a bool verdict."""
    resp = client_with_model.post("/api/predict", json=_SAMPLE_TRIP)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # The contract keys the dashboard renders must all be present.
    expected_keys = {
        "energy_needed_kwh",
        "energy_available_kwh",
        "usable_after_reserve_kwh",
        "reaches",
        "margin_kwh",
        "remaining_soc_pct",
        "remaining_range_km",
        "confidence_note",
    }
    assert expected_keys.issubset(data.keys())

    # ``reaches`` is the load-bearing verdict: it MUST be a real bool.
    assert isinstance(data["reaches"], bool)
    # Energy needed for a real moving segment must be a positive number.
    assert isinstance(data["energy_needed_kwh"], (int, float))
    assert data["energy_needed_kwh"] > 0
    # Internal consistency: margin == usable_after_reserve - needed (rounded).
    assert data["margin_kwh"] == pytest.approx(
        data["usable_after_reserve_kwh"] - data["energy_needed_kwh"], abs=0.01
    )


def test_predict_short_trip_reaches_long_trip_does_not(client_with_model):
    """The verdict must track physics: a tiny trip reaches, an impossible one does not.

    This guards against a model/route that returns a constant ``reaches`` value
    regardless of input -- the verdict has to actually depend on the trip.
    """
    short = dict(_SAMPLE_TRIP, soc_pct=90.0, distance_km=5.0)
    impossible = dict(
        _SAMPLE_TRIP, soc_pct=12.0, distance_km=120.0, payload_t=22.0,
        gradient_pct=6.0, temperature_c=-15.0,
    )

    short_resp = client_with_model.post("/api/predict", json=short)
    impossible_resp = client_with_model.post("/api/predict", json=impossible)
    assert short_resp.status_code == 200
    assert impossible_resp.status_code == 200

    assert short_resp.json()["reaches"] is True
    assert impossible_resp.json()["reaches"] is False


def test_predict_without_model_returns_503(client_without_model):
    """When no model artifact exists, /api/predict must 503 with a clear message."""
    resp = client_without_model.post("/api/predict", json=_SAMPLE_TRIP)
    assert resp.status_code == 503
    payload = resp.json()
    assert payload.get("error") == "model_unavailable"
    # The operator must be told how to fix it.
    assert "run_pipeline.py" in payload.get("detail", "")


def test_optimize_reorders_and_reports_savings(client_with_model):
    """POST /api/optimize returns a permutation of the input stops + the saving vs
    the typed order (never worse). This is what the "Optimize Route" button calls to
    actually REORDER the stops before routing them."""
    body = {
        "origin": {"lat": 52.52, "lng": 13.40, "label": "Berlin"},
        "destinations": [
            {"lat": 53.55, "lng": 10.00, "label": "Hamburg"},
            {"lat": 51.34, "lng": 12.37, "label": "Leipzig"},
            {"lat": 51.05, "lng": 13.74, "label": "Dresden"},
        ],
        "startSoc": 95, "minSoc": 15, "payloadKg": 12000,
    }
    resp = client_with_model.post("/api/optimize", json=body)
    assert resp.status_code == 200
    out = resp.json()
    assert sorted(out["optimizedOrder"]) == [0, 1, 2]   # a permutation of the 3 stops
    assert out["savingsEur"] >= -1e-6                    # never worse than the typed order
    assert "assumptions" in out


def test_optimize_without_model_returns_503(client_without_model):
    """Optimisation costs orders via plan_route, so it needs the model -> 503 when absent."""
    resp = client_without_model.post(
        "/api/optimize",
        json={
            "origin": {"lat": 52.5, "lng": 13.4},
            "destinations": [{"lat": 53.5, "lng": 10.0}, {"lat": 51.3, "lng": 12.4}],
            "startSoc": 90, "minSoc": 15, "payloadKg": 8000,
        },
    )
    assert resp.status_code == 503
    assert resp.json().get("error") == "model_unavailable"
