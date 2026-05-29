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

from .config import DEFAULT_MODEL_PATH, TRUCK
from .model import predict_energy

#: Approximate model error band (kWh) quoted in the confidence note. This is a
#: human-facing, deliberately rounded figure for the HistGradientBoosting model
#: trained on the synthetic eActros dataset; it is not read back from the model
#: artifact so that this function stays pure and dependency-light. Treat it as a
#: rule-of-thumb, not a guarantee.
TYPICAL_MODEL_MAE_KWH: float = 3.0


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

    # Energy demand predicted by the trained model from raw features.
    features = {
        "distance_km": distance_km,
        "payload_t": payload_t,
        "speed_kph": speed_kph,
        "gradient_pct": gradient_pct,
        "temperature_c": temperature_c,
        "wind_mps": wind_mps,
    }
    energy_needed_kwh = float(predict_energy(features, model_path=model_path))

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
    remaining_range_km = 0.0
    if distance_km > 0 and energy_needed_kwh > 0:
        kwh_per_km = energy_needed_kwh / distance_km
        usable_remaining_kwh = max(
            0.0, (energy_available_kwh - energy_needed_kwh) - reserve_kwh
        )
        remaining_range_km = usable_remaining_kwh / kwh_per_km

    confidence_note = (
        "Estimate from a HistGradientBoosting energy model with a typical error "
        f"band of about +/-{TYPICAL_MODEL_MAE_KWH:.0f} kWh (approximate). Treat "
        "margins smaller than this band as uncertain and keep the safety reserve."
    )

    return {
        "energy_needed_kwh": round(energy_needed_kwh, 3),
        "energy_available_kwh": round(energy_available_kwh, 3),
        "usable_after_reserve_kwh": round(usable_after_reserve_kwh, 3),
        "reaches": bool(reaches),
        "margin_kwh": round(margin_kwh, 3),
        "remaining_soc_pct": round(remaining_soc_pct, 2),
        "remaining_range_km": round(remaining_range_km, 1),
        "confidence_note": confidence_note,
    }


__all__ = ["check_reachability", "TYPICAL_MODEL_MAE_KWH"]
