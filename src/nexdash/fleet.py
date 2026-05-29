"""Deterministic mock fleet, evaluated through the REAL energy model.

The dispatcher's console needs a roster of trucks to look at, but the case
study ships no live telematics feed. This module fabricates a small, *fixed*
fleet of Mercedes-Benz eActros 600 trucks scattered across German cities and
then runs each driving truck's next leg through the genuine reachability check
(:func:`nexdash.range.check_reachability`). Nothing here is a vanity number:

* the roster itself is hand-seeded and deterministic (same output every call),
* ``reachable`` / ``marginKwh`` / ``remainingSocPct`` / ``atRisk`` come straight
  out of the trained model, so the console reflects real model behaviour.

Everything is fail-soft: if the model artifact is missing, the model-derived
fields collapse to ``None`` (``atRisk = None``) rather than raising, so the API
can still render the roster. :func:`model_info` likewise degrades to nulls.

All returned values are plain Python scalars/bools/None so the result is
directly JSON-serializable for the FastAPI ``/api/fleet`` endpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_MODEL_PATH, REPORTS_DIR, TRUCK
from . import range as range_module

__all__ = ["fleet_status", "model_info"]

#: A leg whose model margin is below this many kWh is treated as "at risk" even
#: when technically reachable -- it sits inside the model's own error band, so a
#: dispatcher should not trust it blindly. Tuned to roughly the reported MAE.
RISK_MARGIN_KWH: float = 10.0

#: Statuses that imply the truck is (or is about to be) driving a leg, so its
#: reachability is meaningful. ``charging`` / ``maintenance`` trucks are parked.
_DRIVING_STATUSES = {"in_transit", "available"}

# --------------------------------------------------------------------------- #
# The fixed roster.
#
# Each entry is fully specified and deterministic -- no RNG, no clock. Coords
# are real German city centres. ``nextStop`` describes the leg used for the
# reachability check (distance/payload/temperature feed the model; speed and
# gradient use sensible defaults below).
# --------------------------------------------------------------------------- #
_FLEET: list[dict[str, Any]] = [
    {
        "id": "EA-01",
        "name": "eActros 600 #01",
        "lat": 52.5200, "lng": 13.4050,  # Berlin
        "soc": 82.0, "status": "in_transit",
        "nextStop": {"label": "Leipzig DC", "distanceKm": 190.0, "payloadT": 14.0, "temperatureC": 9.0},
    },
    {
        "id": "EA-02",
        "name": "eActros 600 #02",
        "lat": 48.1351, "lng": 11.5820,  # Munich
        "soc": 24.0, "status": "in_transit",
        "nextStop": {"label": "Nuremberg Hub", "distanceKm": 170.0, "payloadT": 21.0, "temperatureC": 3.0},
    },
    {
        "id": "EA-03",
        "name": "eActros 600 #03",
        "lat": 50.1109, "lng": 8.6821,  # Frankfurt
        "soc": 95.0, "status": "available",
        "nextStop": {"label": "Cologne RDC", "distanceKm": 190.0, "payloadT": 8.0, "temperatureC": 12.0},
    },
    {
        "id": "EA-04",
        "name": "eActros 600 #04",
        "lat": 53.5511, "lng": 9.9937,  # Hamburg
        "soc": 41.0, "status": "in_transit",
        "nextStop": {"label": "Bremen Port", "distanceKm": 120.0, "payloadT": 18.0, "temperatureC": 6.0},
    },
    {
        "id": "EA-05",
        "name": "eActros 600 #05",
        "lat": 51.2277, "lng": 6.7735,  # Düsseldorf
        "soc": 12.0, "status": "charging",
        "nextStop": {"label": "Dortmund XD", "distanceKm": 70.0, "payloadT": 10.0, "temperatureC": 8.0},
    },
    {
        "id": "EA-06",
        "name": "eActros 600 #06",
        "lat": 51.0504, "lng": 13.7373,  # Dresden
        "soc": 18.0, "status": "in_transit",
        "nextStop": {"label": "Chemnitz Cross-dock", "distanceKm": 80.0, "payloadT": 16.0, "temperatureC": -4.0},
    },
    {
        "id": "EA-07",
        "name": "eActros 600 #07",
        "lat": 48.7758, "lng": 9.1829,  # Stuttgart
        "soc": 67.0, "status": "available",
        "nextStop": {"label": "Karlsruhe DC", "distanceKm": 80.0, "payloadT": 12.0, "temperatureC": 14.0},
    },
    {
        "id": "EA-08",
        "name": "eActros 600 #08",
        "lat": 52.3759, "lng": 9.7320,  # Hanover
        "soc": 8.0, "status": "maintenance",
        "nextStop": {"label": "Kassel Hub", "distanceKm": 165.0, "payloadT": 0.0, "temperatureC": 10.0},
    },
    {
        "id": "EA-09",
        "name": "eActros 600 #09",
        "lat": 51.3397, "lng": 12.3731,  # Leipzig
        "soc": 30.0, "status": "in_transit",
        "nextStop": {"label": "Berlin SCC", "distanceKm": 190.0, "payloadT": 20.0, "temperatureC": 2.0},
    },
    {
        "id": "EA-10",
        "name": "eActros 600 #10",
        "lat": 50.9375, "lng": 6.9603,  # Cologne
        "soc": 88.0, "status": "available",
        "nextStop": {"label": "Frankfurt DC", "distanceKm": 190.0, "payloadT": 6.0, "temperatureC": 11.0},
    },
]

#: Defaults for the two leg features the roster does not vary per-truck. A
#: motorway cruise speed and flat net gradient keep the model focused on the
#: dispatcher-relevant inputs (SOC, distance, payload, temperature).
_DEFAULT_SPEED_KPH: float = 75.0
_DEFAULT_GRADIENT_PCT: float = 0.0


def fleet_status(model_path=DEFAULT_MODEL_PATH) -> list[dict]:
    """Return the deterministic fleet roster, enriched with model verdicts.

    For every ``in_transit`` / ``available`` truck the next leg is run through
    :func:`nexdash.range.check_reachability`, adding:

    * ``reachable`` (bool) -- model says the leg is completable on current SOC,
    * ``marginKwh`` (float) -- usable-after-reserve energy minus predicted need,
    * ``remainingSocPct`` (float) -- estimated SOC on arrival,
    * ``atRisk`` (bool) -- ``not reachable`` OR margin below
      :data:`RISK_MARGIN_KWH` (inside the model's error band).

    Parked trucks (``charging`` / ``maintenance``) are not driving a leg, so
    their model fields are ``None`` and ``atRisk`` is ``False`` (not at risk).

    Fail-soft: if the model artifact is missing or prediction raises, the
    model-derived fields (including ``atRisk``) collapse to ``None`` for the
    affected trucks rather than propagating an exception.

    Args:
        model_path: Path to the trained energy model artifact.

    Returns:
        A list of JSON-serializable truck dicts (a deep, independent copy of
        the fixed roster so callers may mutate the result freely).
    """
    trucks: list[dict] = []
    for base in _FLEET:
        truck = {
            "id": base["id"],
            "name": base["name"],
            "lat": base["lat"],
            "lng": base["lng"],
            "soc": base["soc"],
            "status": base["status"],
            "nextStop": dict(base["nextStop"]),
        }

        if base["status"] in _DRIVING_STATUSES:
            verdict = _evaluate_leg(base, model_path)
            truck.update(verdict)
        else:
            # Parked: nothing to reach right now.
            truck.update(
                {
                    "reachable": None,
                    "marginKwh": None,
                    "remainingSocPct": None,
                    "atRisk": False,
                }
            )

        trucks.append(truck)

    return trucks


def _evaluate_leg(base: dict[str, Any], model_path) -> dict[str, Any]:
    """Run the next leg through the real model; fail soft to nulls on error."""
    stop = base["nextStop"]
    try:
        result = range_module.check_reachability(
            soc_pct=base["soc"],
            distance_km=stop["distanceKm"],
            payload_t=stop["payloadT"],
            speed_kph=_DEFAULT_SPEED_KPH,
            gradient_pct=_DEFAULT_GRADIENT_PCT,
            temperature_c=stop["temperatureC"],
            model_path=model_path,
        )
    except Exception:
        # Model missing / load failure / prediction error -> unknown verdict.
        return {
            "reachable": None,
            "marginKwh": None,
            "remainingSocPct": None,
            "atRisk": None,
        }

    reachable = bool(result["reaches"])
    margin = float(result["margin_kwh"])
    at_risk = (not reachable) or (margin < RISK_MARGIN_KWH)
    return {
        "reachable": reachable,
        "marginKwh": round(margin, 2),
        "remainingSocPct": round(float(result["remaining_soc_pct"]), 1),
        "atRisk": at_risk,
    }


def model_info(model_path=DEFAULT_MODEL_PATH) -> dict:
    """Return the trained energy model's headline metrics for the console.

    Prefers the metrics stored on the model artifact
    (:attr:`nexdash.model.EnergyModel.metrics` ``["hgb"]``); falls back to
    parsing ``reports/evaluation_report.md`` if the artifact is unavailable.
    Fully fail-soft: any missing piece becomes ``None``.

    The returned ``pct_range_error`` follows the report's definition -- MAE
    expressed against a nominal full-trip energy of the usable battery
    (``TRUCK.battery_kwh``), i.e. what fraction of a full charge the average
    miss represents.

    Returns:
        ``{mae_kwh, rmse_kwh, mape_pct, r2, pct_range_error}`` with float values
        where known and ``None`` where they could not be resolved.
    """
    info: dict[str, Optional[float]] = {
        "mae_kwh": None,
        "rmse_kwh": None,
        "mape_pct": None,
        "r2": None,
        "pct_range_error": None,
    }

    metrics = _metrics_from_artifact(model_path)
    if metrics is not None:
        info["mae_kwh"] = _as_float(metrics.get("mae_kwh"))
        info["rmse_kwh"] = _as_float(metrics.get("rmse_kwh"))
        info["mape_pct"] = _as_float(metrics.get("mape_pct"))
        info["r2"] = _as_float(metrics.get("r2"))
        if info["mae_kwh"] is not None and TRUCK.battery_kwh > 0:
            info["pct_range_error"] = round(
                info["mae_kwh"] / TRUCK.battery_kwh * 100.0, 3
            )
        return info

    # Artifact unavailable -> parse the human-readable report as a fallback.
    return _metrics_from_report(info)


def _metrics_from_artifact(model_path) -> Optional[dict[str, Any]]:
    """Load the primary (hgb) metrics dict from the model artifact, or None."""
    try:
        from .model import EnergyModel  # local import: keep module import light

        model = EnergyModel.load(model_path)
        hgb = model.metrics.get("hgb")
        if isinstance(hgb, dict) and hgb:
            return hgb
    except Exception:
        pass
    return None


def _metrics_from_report(info: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    """Parse headline metrics from reports/evaluation_report.md (fail-soft)."""
    report = REPORTS_DIR / "evaluation_report.md"
    try:
        text = Path(report).read_text(encoding="utf-8")
    except Exception:
        return info

    patterns = {
        "mae_kwh": r"\*\*MAE:\*\*\s*\*\*([\d.]+)\s*kWh\*\*",
        "rmse_kwh": r"\*\*RMSE:\*\*\s*([\d.]+)\s*kWh",
        "mape_pct": r"\*\*MAPE:\*\*\s*([\d.]+)\s*%",
        "r2": r"\*\*R\^2:\*\*\s*([\d.]+)",
        "pct_range_error": r"\*\*% range error:\*\*\s*\*\*([\d.]+)\s*%\*\*",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            info[key] = _as_float(match.group(1))
    return info


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float, returning None on failure (fail-soft helper)."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
