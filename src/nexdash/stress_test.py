"""Per-trip robustness stress test — the decision-side twin of calibration.

The calibration harness (:mod:`nexdash.calibration`) audits the *model's* honesty
offline. This module audits a *specific trip's* robustness to the operating
conditions a dispatcher most often mis-forecasts — temperature, payload, wind,
speed — at plan time. It converts the single fragile "+40 kWh margin" number into:

* a **breakpoint** per factor: the value at which the trip flips to NO-GO
  (e.g. "fails below -8 C", "fails above +5.5 t over plan"), and
* a **tornado ranking**: which lever erodes the safety margin fastest.

It is built entirely on :func:`nexdash.range.check_reachability` — reusing its
``margin_kwh`` / ``reaches`` / ``confidence`` contract verbatim, including the
directional physics cross-check that sets ``confidence="low"`` when the model is
optimistic on out-of-envelope inputs. Honesty discipline carries through: where a
swept point trips that low-confidence flag, the factor is tagged
``"low confidence beyond X"`` rather than quoting a precise breakpoint we cannot
trust. ~24 model calls, sub-second, deterministic, offline.

Honest limits (surfaced, not hidden): the sweeps are one-factor-at-a-time, so they
miss stacked adverse conditions (cold + headwind + traffic together); breakpoints
are trustworthy only where ``confidence="high"``; and the sweep bounds are
hand-set plausibility ranges clamped to the training envelope, not learned.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import TRUCK
from .range import check_reachability

__all__ = ["stress_test", "FACTORS"]

#: The four dispatcher-forecast factors swept, each with an adverse direction and
#: a plausibility range clamped to the model's training envelope. ``adverse`` is
#: the direction that *erodes* margin (so the tornado is intuitive).
FACTORS: dict[str, dict[str, Any]] = {
    "temperature_c": {"lo": -15.0, "hi": 40.0, "adverse": "low", "unit": "C"},
    "payload_t": {"lo": 0.0, "hi": TRUCK.max_payload_t, "adverse": "high", "unit": "t"},
    "wind_mps": {"lo": -12.0, "hi": 12.0, "adverse": "high", "unit": "m/s headwind"},
    "speed_kph": {"lo": 30.0, "hi": 85.0, "adverse": "high", "unit": "km/h"},
}

_N_POINTS = 6


def _grid(base: float, lo: float, hi: float, adverse: str, n: int = _N_POINTS) -> list[float]:
    """Adverse-direction sweep grid from the base value to the envelope edge.

    Sweeps from ``base`` toward whichever edge erodes margin (``adverse``), so we
    probe the dangerous direction; clamps to ``[lo, hi]`` so we never sweep into
    regions the model never saw.
    """
    base = max(lo, min(hi, base))
    edge = lo if adverse == "low" else hi
    if edge == base:
        return [base]
    return [base + (edge - base) * (k / (n - 1)) for k in range(n)]


def _interp_breakpoint(points: list[tuple[float, float]]) -> Optional[float]:
    """Linear-interpolated factor value where margin first crosses zero, or None.

    ``points`` is ordered ``[(factor_value, margin_kwh), ...]`` along the sweep.
    Returns the first crossing from >=0 to <0; ``None`` if margin never goes
    negative within the band.
    """
    for (x0, m0), (x1, m1) in zip(points, points[1:]):
        if m0 >= 0 > m1:
            if m0 == m1:
                return x1
            return x0 + (x1 - x0) * (m0 / (m0 - m1))
    return None


def stress_test(
    *,
    soc_pct: float,
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    wind_mps: float = 0.0,
    reserve_pct: float = 10.0,
    model_path=None,
) -> dict[str, Any]:
    """Sweep the four dispatcher-forecast factors and rank what threatens the trip.

    Returns a JSON-serialisable dict with ``baseline``, a ``factors`` list ranked
    worst-first (tornado order) carrying per-factor sensitivity / margin erosion /
    breakpoint / note / confidence flag, the ``dominant_threat`` factor, and honest
    ``assumptions``.

    ``breakpoint`` is the factor value where the trip flips to NO-GO (``None`` if
    it never fails in-band). When any swept point trips ``confidence="low"`` (the
    physics OOD cross-check), the numeric breakpoint is suppressed and
    ``breakpoint_note`` reads ``"low confidence beyond X"`` — we refuse to quote a
    crossing we can't trust.
    """
    base_kw = dict(
        soc_pct=soc_pct, distance_km=distance_km, payload_t=payload_t,
        speed_kph=speed_kph, gradient_pct=gradient_pct, temperature_c=temperature_c,
        wind_mps=wind_mps, reserve_pct=reserve_pct,
    )
    if model_path is not None:
        base_kw["model_path"] = model_path

    base = check_reachability(**base_kw)
    base_margin = float(base["margin_kwh"])

    base_vals = {
        "temperature_c": temperature_c, "payload_t": payload_t,
        "wind_mps": wind_mps, "speed_kph": speed_kph,
    }

    factor_rows: list[dict[str, Any]] = []
    for name, spec in FACTORS.items():
        grid = _grid(base_vals[name], spec["lo"], spec["hi"], spec["adverse"])
        pts: list[tuple[float, float]] = []
        any_low = False
        first_low_x: Optional[float] = None
        for v in grid:
            kw = dict(base_kw)
            kw[name] = v
            r = check_reachability(**kw)
            pts.append((v, float(r["margin_kwh"])))
            if r["confidence"] == "low" and not any_low:
                any_low = True
                first_low_x = v

        x0, m0 = pts[0]
        x1, m1 = pts[-1]
        sensitivity = (m1 - m0) / (x1 - x0) if x1 != x0 else 0.0
        margin_erosion = m0 - min(m for _, m in pts)  # how far margin can fall

        # Precedence: an OOD low-confidence flag is the most important honesty
        # signal, so it overrides the "already failing" note — we never imply a
        # trustworthy verdict in a region the physics cross-check flagged.
        if any_low:
            breakpoint = None
            note = f"low confidence beyond {first_low_x:.1f} {spec['unit']}"
        elif base_margin < 0:
            breakpoint, note = None, "already failing at base conditions"
        else:
            bp = _interp_breakpoint(pts)
            if bp is None:
                breakpoint, note = None, "no NO-GO crossing within the plausible band"
            else:
                breakpoint = round(float(bp), 2)
                note = f"NO-GO at {breakpoint} {spec['unit']}"

        factor_rows.append(
            {
                "factor": name,
                "unit": spec["unit"],
                "adverse": spec["adverse"],
                "sensitivity_kwh_per_unit": round(float(sensitivity), 3),
                "margin_erosion_kwh": round(float(margin_erosion), 2),
                "breakpoint": breakpoint,
                "breakpoint_note": note,
                "confidence_flips": bool(any_low),
            }
        )

    factor_rows.sort(key=lambda f: f["margin_erosion_kwh"], reverse=True)
    dominant = factor_rows[0]["factor"] if factor_rows and factor_rows[0]["margin_erosion_kwh"] > 0 else None

    assumptions = [
        "One-factor-at-a-time sweeps: real adverse days stack cold + headwind + "
        "traffic, which this does not combine.",
        "Breakpoints are trustworthy only where confidence is high; an OOD factor "
        "reports 'low confidence beyond X' instead of a precise crossing.",
        "Sweep bounds are plausibility ranges clamped to the training envelope, "
        "not learned.",
    ]
    if base["confidence"] == "low":
        assumptions.insert(
            0,
            "Baseline trip itself reads LOW CONFIDENCE (physics cross-check fired) "
            "— treat the whole stress test as indicative, not precise.",
        )

    return {
        "baseline": {
            "margin_kwh": base["margin_kwh"],
            "reaches": base["reaches"],
            "confidence": base["confidence"],
        },
        "factors": factor_rows,
        "dominant_threat": dominant,
        "assumptions": assumptions,
    }
