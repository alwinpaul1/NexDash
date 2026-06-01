"""Range / reachability reasoning for the eActros 600 fleet.

This module turns a single energy prediction into an operational answer:
*given the current state of charge, can the truck complete this segment?*

The heavy lifting (physics-informed energy estimation) lives in
:mod:`nexdash.model`. Here we only:

* convert state of charge (SOC) into usable kWh on board,
* subtract a configurable safety reserve,
* compare it against the model's predicted energy demand,
* and translate the leftover energy into a remaining SOC / range estimate.

Everything returned is plain Python scalars/bools so the result is directly
JSON-serializable (e.g. for the FastAPI dashboard endpoint and MCP tools).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import DEFAULT_MODEL_PATH, TRUCK
from .model import EnergyModel, predict_energy
from .physics import segment_energy_kwh

#: Coarse CONSERVATIVE fallback error band (kWh), used only when the trained
#: artifact's held-out MAE cannot be read (e.g. a degenerate/missing metric). The
#: live confidence note prefers the model's *actual* held-out MAE — see
#: :func:`_held_out_mae_kwh`. Pinned at/above the trained model's measured held-out
#: MAE (~6 kWh) so the fallback never UNDER-states error (the over-confident
#: direction); it is a deliberately rough default, not a measured value.
TYPICAL_MODEL_MAE_KWH: float = 6.0


@lru_cache(maxsize=8)
def _held_out_mae_cached(model_path: str, mtime_ns: int) -> float:
    """Cached read of the HGB held-out MAE, keyed by path AND file mtime so a
    retrain to the same path is never served stale (``mtime_ns`` busts the key)."""
    try:
        mae = float(EnergyModel.load(model_path).metrics.get("hgb", {}).get("mae_kwh"))
        if mae == mae and mae > 0:  # not NaN and positive
            return mae
    except Exception:
        pass
    return TYPICAL_MODEL_MAE_KWH


def _held_out_mae_kwh(model_path) -> float:
    """Return the HGB held-out MAE (kWh) from the model artifact, fail-soft.

    Falls back to :data:`TYPICAL_MODEL_MAE_KWH` if the artifact/metric is missing.
    """
    try:
        resolved = Path(model_path).resolve()
        return _held_out_mae_cached(str(resolved), resolved.stat().st_mtime_ns)
    except OSError:
        return TYPICAL_MODEL_MAE_KWH


def check_reachability(
    soc_pct: float,
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    *,
    wind_mps: float = 0.0,
    model_path=DEFAULT_MODEL_PATH,
    reserve_pct: float = 10.0,
) -> dict:
    """Decide whether the truck can reach its destination on the current SOC.

    The predicted energy demand for the segment comes from the trained energy
    model (:func:`nexdash.model.predict_energy`). Available energy is derived
    from the battery capacity and the current state of charge, less a safety
    reserve the operator never wants to dip below.

    Args:
        soc_pct: Current state of charge, percent (0-100).
        distance_km: Segment distance to travel (km).
        payload_t: Cargo payload (tonnes).
        speed_kph: Average travel speed (km/h).
        gradient_pct: Net road gradient (percent; negative = downhill).
        temperature_c: Ambient temperature (degrees Celsius).
        wind_mps: Head/tail wind component (m/s); positive = headwind.
        model_path: Path to the trained model artifact.
        reserve_pct: Battery percentage kept in reserve and not counted as
            usable for this trip (default 10%).

    Returns:
        A JSON-serializable dict with keys:

        * ``energy_needed_kwh`` -- model-predicted energy demand for the segment.
        * ``energy_available_kwh`` -- energy on board from current SOC.
        * ``usable_after_reserve_kwh`` -- on-board energy minus the safety reserve.
        * ``reaches`` -- ``True`` if usable energy covers predicted demand.
        * ``margin_kwh`` -- usable-after-reserve minus needed (negative = short).
        * ``remaining_soc_pct`` -- estimated SOC after the trip (0-100).
        * ``remaining_range_km`` -- estimated further range after the trip, using
          this segment's average kWh/km consumption.
        * ``confidence_note`` -- plain-language caveat referencing the model's
          approximate error band.
    """
    battery_kwh = TRUCK.battery_kwh

    # Clamp operational inputs to physically valid ranges. A SOC > 100 or a
    # negative reserve would otherwise INFLATE the usable energy and produce an
    # unsafely optimistic "reaches" verdict, so we bound them rather than trust a
    # bad sensor reading or an LLM-supplied out-of-range value.
    soc_pct = min(100.0, max(0.0, float(soc_pct)))
    reserve_pct = min(100.0, max(0.0, float(reserve_pct)))

    # A moving segment needs a positive speed. Reject it early with a clear message
    # so this path fails consistently with the physics layer (segment_energy_kwh
    # also rejects speed<=0) instead of crashing deep inside it — and so the model's
    # optimistic speed=0 extrapolation is never quietly used. A non-positive speed is
    # invalid input, NOT "unreachable" (a false NO-GO would be its own dishonest verdict).
    if speed_kph <= 0:
        raise ValueError(f"speed_kph must be > 0 for a moving segment; got {speed_kph}")

    # Energy demand predicted by the trained model from raw features.
    features = {
        "distance_km": distance_km,
        "payload_t": payload_t,
        "speed_kph": speed_kph,
        "gradient_pct": gradient_pct,
        "temperature_c": temperature_c,
        "wind_mps": wind_mps,
    }
    model_kwh = float(predict_energy(features, model_path=model_path))

    # Physics sanity cross-check. The data-driven model can only be trusted inside
    # the envelope it was trained on; handed a physically implausible segment (e.g.
    # a sustained steep grade over a long distance, which never occurs in real
    # data), it extrapolates and can *under*-predict badly — the dangerous
    # direction. We therefore compute a first-principles estimate and, when the
    # model is OPTIMISTIC relative to physics by more than ~3 error bands (or 15%),
    # refuse to quote the optimistic number: we use the more conservative (higher)
    # value and flag low confidence.
    #
    # The test is DIRECTIONAL on purpose. Only model under-prediction (model far
    # BELOW physics) is dangerous. When the model predicts MORE than physics — which
    # is normal on a regen-dominated descent, where the first-principles estimate can
    # even go negative — that is the safe/conservative direction, so we keep high
    # confidence and quote the model. A symmetric ``abs(...)`` test (and a
    # ``0.15 * physics`` band that collapses to zero when physics is negative) would
    # otherwise falsely flag every routine downhill leg — which is squarely inside the
    # -6..+6% training envelope — as "outside the envelope".
    physics_kwh = float(
        segment_energy_kwh(
            distance_km=distance_km,
            payload_t=payload_t,
            speed_kph=speed_kph,
            gradient_pct=gradient_pct,
            temperature_c=temperature_c,
            wind_mps=wind_mps,
            truck=TRUCK,
        )
    )
    mae_band = _held_out_mae_kwh(str(model_path))
    divergence_band = max(3.0 * mae_band, 0.15 * abs(physics_kwh))
    diverges = (physics_kwh - model_kwh) > divergence_band
    energy_needed_kwh = max(model_kwh, physics_kwh) if diverges else model_kwh
    confidence = "low" if diverges else "high"

    # Energy currently on board, and what is usable after holding back a reserve.
    energy_available_kwh = battery_kwh * (soc_pct / 100.0)
    reserve_kwh = battery_kwh * (reserve_pct / 100.0)
    usable_after_reserve_kwh = energy_available_kwh - reserve_kwh

    margin_kwh = usable_after_reserve_kwh - energy_needed_kwh
    reaches = margin_kwh >= 0.0

    # SOC remaining after the trip (clamped to a sane 0-100 range).
    remaining_soc_pct = (energy_available_kwh - energy_needed_kwh) / battery_kwh * 100.0
    remaining_soc_pct = max(0.0, min(100.0, remaining_soc_pct))

    # Estimate how much further the truck could go after this segment, assuming
    # the same average consumption (kWh/km) as the predicted segment. Energy
    # below the reserve is not counted toward usable remaining range.
    #
    # The segment rate is FLOORED at the truck's nominal flat consumption: a
    # downhill leg can have a near-zero (or net-regen negative) kWh/km, but a truck
    # cannot descend forever, so extrapolating that rate would quote a physically
    # impossible 1000+ km of remaining range. Flooring at the rated flat rate keeps
    # the estimate a sane "further range on average terrain", and leaves uphill
    # legs (rate above nominal) conservatively unchanged.
    remaining_range_km = 0.0
    if distance_km > 0 and energy_needed_kwh > 0:
        nominal_kwh_per_km = battery_kwh / TRUCK.nominal_range_km
        kwh_per_km = max(energy_needed_kwh / distance_km, nominal_kwh_per_km)
        usable_remaining_kwh = max(
            0.0, (energy_available_kwh - energy_needed_kwh) - reserve_kwh
        )
        remaining_range_km = usable_remaining_kwh / kwh_per_km

    if diverges:
        used_label = "physics" if energy_needed_kwh == physics_kwh else "the model"
        confidence_note = (
            "LOW CONFIDENCE: the data-driven model "
            f"({model_kwh:.0f} kWh) and a first-principles physics estimate "
            f"({physics_kwh:.0f} kWh) disagree sharply, which means this segment "
            "is outside the envelope the model was trained on. The more "
            f"conservative (higher) estimate of {energy_needed_kwh:.0f} kWh "
            f"({used_label}) is used for this decision; treat it as indicative "
            "only and keep a wide reserve."
        )
    else:
        confidence_note = (
            "Estimate from a HistGradientBoosting energy model whose held-out mean "
            f"absolute error is about +/-{mae_band:.0f} kWh (physics cross-check "
            f"agrees within {abs(model_kwh - physics_kwh):.0f} kWh). Treat margins "
            "smaller than this band as uncertain and keep the safety reserve."
        )

    return {
        "energy_needed_kwh": round(energy_needed_kwh, 3),
        "energy_available_kwh": round(energy_available_kwh, 3),
        "usable_after_reserve_kwh": round(usable_after_reserve_kwh, 3),
        "reaches": bool(reaches),
        "margin_kwh": round(margin_kwh, 3),
        "remaining_soc_pct": round(remaining_soc_pct, 2),
        "remaining_range_km": round(remaining_range_km, 1),
        "confidence": confidence,
        "model_kwh": round(model_kwh, 3),
        "physics_kwh": round(physics_kwh, 3),
        "confidence_note": confidence_note,
    }


__all__ = ["check_reachability", "TYPICAL_MODEL_MAE_KWH"]
