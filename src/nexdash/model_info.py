"""Headline metrics for the trained energy model, for the API / console.

Prefers the metrics stored on the model artifact
(:attr:`nexdash.model.EnergyModel.metrics` ``["hgb"]``) and falls back to parsing
``reports/evaluation_report.md`` if the artifact is unavailable. Fully fail-soft:
any missing piece becomes ``None``. All values are plain JSON-serializable types
for the FastAPI ``/api/model-info`` endpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_MODEL_PATH, REPORTS_DIR, TRUCK

__all__ = ["model_info"]


def model_info(model_path=DEFAULT_MODEL_PATH) -> dict:
    """Return the trained energy model's headline metrics.

    The returned ``pct_range_error`` follows the report's definition -- MAE
    expressed against a nominal full-trip energy of the usable battery
    (``TRUCK.battery_kwh``), i.e. what fraction of a full charge the average miss
    represents.

    Returns:
        ``{mae_kwh, rmse_kwh, mape_pct, r2, pct_range_error, model_version}`` with
        float values where known and ``None`` where they could not be resolved.
        ``model_version`` is the content-addressed lineage string (or ``None``).
    """
    info: dict[str, Optional[float]] = {
        "mae_kwh": None,
        "rmse_kwh": None,
        "mape_pct": None,
        "r2": None,
        "pct_range_error": None,
        "model_version": None,
    }

    # Content-addressed lineage (training-data SHA + code SHA), fail-soft: read
    # from the provenance sidecar written by run_pipeline / nexdash.registry.
    try:
        from . import registry

        sidecar = registry.read_sidecar(model_path)
        if sidecar:
            info["model_version"] = sidecar.get("model_version")
    except Exception:
        pass

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
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            info[key] = _as_float(match.group(1))
    # Derive pct_range_error from the parsed MAE exactly as the artifact path does,
    # rather than scrape the report's prose label (which has drifted before, e.g.
    # "% of a full charge" vs "% range error", silently yielding None). This keeps
    # the fallback consistent with model_info()'s own definition and wording-robust.
    if info["mae_kwh"] is not None and TRUCK.battery_kwh > 0:
        info["pct_range_error"] = round(info["mae_kwh"] / TRUCK.battery_kwh * 100.0, 3)
    return info


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float, returning None on failure (fail-soft helper)."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
