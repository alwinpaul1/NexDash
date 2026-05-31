"""Deterministic physics "ground truth" for eActros 600 segment energy.

This module computes the traction energy required to drive a single road
segment from first principles of longitudinal vehicle dynamics. It is the
*ground truth* generator used to synthesise the training dataset (see
:mod:`nexdash.data_gen`); the machine-learning model in :mod:`nexdash.model`
then learns to approximate it from noisy samples.

Physical model
--------------
For a constant-speed segment of length ``d`` (m) at speed ``v`` (m/s) the
energy delivered at the wheels is the sum of four resistive contributions,
each integrated over the distance:

* **Rolling resistance** — ``F_roll = Crr(v, T) * m * g`` so the energy is
  ``E_roll = Crr(v, T) * m * g * d``. ``Crr`` is no longer constant: it rises
  modestly with speed (SAE J2452) and on the cold side of ~20 C (tyre pressure
  loss + rubber stiffening), both normalised to 1.0 at the 80 km/h / 20 C
  reference (see :func:`_crr_factor`).
* **Aerodynamic drag** — ``F_aero = 0.5 * rho(T) * Cd * A * v_air^2`` where the
  air speed ``v_air = v + wind`` is the headwind-relative speed (a positive
  ``wind_mps`` is an opposing headwind, a negative one a tailwind) and the air
  density ``rho(T)`` now follows the ideal gas law — denser (more drag) in the
  cold, lighter when hot, with ``rho(15 C) = 1.225`` as the pivot (see
  :func:`_air_density`). Energy: ``E_aero = 0.5 * rho(T) * Cd * A * v_air^2 * d``.
* **Gradient / potential energy** — ``F_grade = m * g * sin(theta)`` with
  ``theta = atan(grade_pct / 100)``. On climbs this adds energy; on descents it
  is negative and a fraction of that downhill potential energy is recovered
  through regenerative braking. That fraction is no longer fixed at
  ``regen_eff``: it tapers in the cold (BMS limits charge-acceptance) and on
  very steep descents (braking power exceeds the regen cap, so friction brakes
  dissipate the excess) — see :func:`_regen_fraction`. Mild descents in mild
  weather keep the full baseline recovery.
* **Auxiliary / HVAC** — a power draw ``P_aux`` (kW) sustained for the travel
  time ``t = d / v``. ``P_aux`` is U-shaped in ambient temperature: minimal in
  the ~18-22 C comfort band and rising at both cold extremes (battery
  conditioning + cabin heating) and hot extremes (air-conditioning). Energy:
  ``E_aux = P_aux * t``.

The traction terms (rolling, aero, gradient) are divided by the drivetrain
efficiency ``drivetrain_eff`` to convert wheel energy into battery draw, since
losses in the motor/inverter/gearbox mean more energy must leave the battery
than reaches the road. The auxiliary load is drawn directly from the battery
(it does not pass through the traction drivetrain) and so is *not* divided by
``drivetrain_eff``. Regenerated downhill energy is credited after the same
drivetrain-efficiency scaling, which approximates the round-trip loss on the
recovery path.

All energies are returned in **kWh** (Joules / 3.6e6). For a mid-load truck on
flat ground at motorway speed this yields roughly 1.0-1.6 kWh/km at mild
temperatures, rising toward ~1.5 kWh/km in the cold (-10 C) as denser air and
stiffer tyres bite — consistent with published eActros 600 real-world figures.
"""

from __future__ import annotations

import math

from nexdash.config import (
    G,
    P_SEA_LEVEL_PA,
    R_SPECIFIC_DRY_AIR,
    T_KELVIN_OFFSET,
    TRUCK,
    Truck,
)

__all__ = ["segment_energy_kwh", "energy_breakdown"]

#: Conversion factor from Joules to kilowatt-hours.
_J_PER_KWH: float = 3.6e6

#: Comfort temperature (C) at which auxiliary/HVAC load is minimal.
_COMFORT_TEMP_C: float = 20.0

#: Extra HVAC power per degree Celsius below the comfort band (kW/C).
#: Heating an EV cabin plus conditioning the battery in winter is power-hungry,
#: so the cold-side slope is steeper than the hot side. Calibrated to the
#: eActros winter test (~6-7 kW HVAC+aux at -10 C). See
#: docs/REAL_WORLD_CALIBRATION.md. [S7][S10]
_AUX_COLD_SLOPE_KW_PER_C: float = 0.18

#: Extra HVAC power per degree Celsius above the comfort band (kW/C).
#: Air-conditioning is comparatively efficient, hence a gentler slope
#: (~4-5 kW HVAC+aux near 38 C). [S10][S12]
_AUX_HOT_SLOPE_KW_PER_C: float = 0.13

#: Half-width (C) of the comfort band around :data:`_COMFORT_TEMP_C` within
#: which only the baseline auxiliary load applies (comfort band 20 +/- 3 C).
_COMFORT_HALF_WIDTH_C: float = 3.0

# --------------------------------------------------------------------------- #
# Rolling-resistance speed/temperature dependence (Crr is not actually constant)
# --------------------------------------------------------------------------- #
#: Crr rises modestly with speed for long-haul tyres (SAE J2452); normalised to
#: 1.0 at the reference speed so the calibration anchor (80 km/h) is unchanged.
_CRR_SPEED_SLOPE_PER_KPH: float = 0.0015
_CRR_SPEED_REF_KPH: float = 80.0
#: Floor on the speed factor so very low speeds don't drive Crr implausibly low.
_CRR_SPEED_FLOOR: float = 0.90
#: Cold-side stiffening: tyres lose pressure and rubber stiffens below ~20 C,
#: raising Crr ~0.4%/C (deliberately conservative; literature 0.6-0.9%/C from
#: surface temp conflates self-heating). Warm side is flat (clamped to 1.0).
_CRR_TEMP_SLOPE_PER_C: float = 0.004
_CRR_TEMP_REF_C: float = 20.0

# --------------------------------------------------------------------------- #
# Regen taper: cold batteries accept less charge; steep descents exceed the
# regen power cap so friction braking dissipates the excess.
# --------------------------------------------------------------------------- #
_REGEN_T_FULL_C: float = 10.0  # full regen at/above this temperature
_REGEN_T_COLD_C: float = -15.0  # regen floor reached at/below this temperature
_REGEN_TEMP_FLOOR: float = 0.45  # cold multiplier floor on regen_base
_REGEN_GRADE_KNEE_PCT: float = 5.0  # |descent| beyond which recovery tapers
_REGEN_GRADE_MAX_PCT: float = 10.0  # |descent| at which the grade floor is hit
# Steeper descents recover a smaller *fraction* (regen power cap → friction), but
# the floor is chosen so the recovered *total* energy stays monotonically
# non-decreasing in |grade| — a steeper descent never returns LESS charge than a
# gentler one. (With a lower floor the fraction taper out-ran the rising
# potential energy between -8% and -10%, which is physically wrong.)
_REGEN_GRADE_FLOOR: float = 0.70


def _air_density(temperature_c: float) -> float:
    """Air density (kg/m^3) from the ideal gas law ``P / (R * T_kelvin)``.

    Colder air is denser, so winter segments carry more aerodynamic drag. By
    construction ``_air_density(15) == 1.225`` (the ISA sea-level reference), so
    this is continuous with the previous constant. Sea-level pressure is assumed
    (no altitude term — see the module's residual-limitations note).
    """
    return P_SEA_LEVEL_PA / (R_SPECIFIC_DRY_AIR * (temperature_c + T_KELVIN_OFFSET))


def _crr_factor(speed_kph: float, temperature_c: float, truck: Truck) -> float:
    """Effective rolling-resistance coefficient.

    Base ``truck.crr`` scaled by a modest speed rise (SAE J2452) and a cold-side
    stiffening ramp. Both factors are 1.0 at the reference (80 km/h, >= 20 C), so
    the flat-motorway calibration anchor is unchanged.
    """
    f_speed = max(
        1.0 + _CRR_SPEED_SLOPE_PER_KPH * (speed_kph - _CRR_SPEED_REF_KPH),
        _CRR_SPEED_FLOOR,
    )
    f_temp = (
        1.0 + _CRR_TEMP_SLOPE_PER_C * (_CRR_TEMP_REF_C - temperature_c)
        if temperature_c < _CRR_TEMP_REF_C
        else 1.0
    )
    return truck.crr * f_speed * f_temp


def _regen_fraction(temperature_c: float, gradient_pct: float, truck: Truck) -> float:
    """Effective fraction of downhill potential energy recovered on a descent.

    ``truck.regen_eff`` tapered by two physical limits: cold battery
    charge-acceptance (BMS caps regen current in the cold) and very steep
    descents whose braking power exceeds the motor/regen cap (friction brakes
    dissipate the excess). Mild descents in mild weather keep the full baseline.
    """
    # Temperature ramp: full at/above T_full, linear down to the floor at T_cold.
    if temperature_c >= _REGEN_T_FULL_C:
        g_temp = 1.0
    elif temperature_c <= _REGEN_T_COLD_C:
        g_temp = _REGEN_TEMP_FLOOR
    else:
        frac = (temperature_c - _REGEN_T_COLD_C) / (_REGEN_T_FULL_C - _REGEN_T_COLD_C)
        g_temp = _REGEN_TEMP_FLOOR + (1.0 - _REGEN_TEMP_FLOOR) * frac

    # Grade ramp: full up to the knee, linear down to the floor at the max grade.
    steep = abs(gradient_pct)
    if steep <= _REGEN_GRADE_KNEE_PCT:
        g_grade = 1.0
    elif steep >= _REGEN_GRADE_MAX_PCT:
        g_grade = _REGEN_GRADE_FLOOR
    else:
        frac = (steep - _REGEN_GRADE_KNEE_PCT) / (_REGEN_GRADE_MAX_PCT - _REGEN_GRADE_KNEE_PCT)
        g_grade = 1.0 - (1.0 - _REGEN_GRADE_FLOOR) * frac

    return truck.regen_eff * g_temp * g_grade


def _auxiliary_power_kw(temperature_c: float, truck: Truck) -> float:
    """Return the auxiliary/HVAC power draw (kW) for an ambient temperature.

    The curve is U-shaped: a flat baseline (``truck.aux_base_kw``) inside the
    comfort band ``[20 +/- 2] C`` and a linear rise on each side, steeper for
    cold (battery + cabin heating) than for hot (air-conditioning).

    Args:
        temperature_c: Ambient air temperature in degrees Celsius.
        truck: Vehicle specification supplying ``aux_base_kw``.

    Returns:
        Auxiliary power draw in kilowatts (always >= ``aux_base_kw``).
    """
    lower = _COMFORT_TEMP_C - _COMFORT_HALF_WIDTH_C
    upper = _COMFORT_TEMP_C + _COMFORT_HALF_WIDTH_C

    if temperature_c < lower:
        extra = _AUX_COLD_SLOPE_KW_PER_C * (lower - temperature_c)
    elif temperature_c > upper:
        extra = _AUX_HOT_SLOPE_KW_PER_C * (temperature_c - upper)
    else:
        extra = 0.0

    return truck.aux_base_kw + extra


def energy_breakdown(
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    *,
    wind_mps: float = 0.0,
    truck: Truck = TRUCK,
) -> dict[str, float]:
    """Compute the per-component energy breakdown for a driving segment.

    See the module docstring for the full physical model. All component values
    are battery-side energies in kWh.

    Args:
        distance_km: Segment length (km). Must be > 0 for the result to be
            meaningful; non-positive distances yield all-zero components.
        payload_t: Cargo payload (tonnes); added to the kerb mass.
        speed_kph: Average travel speed (km/h). Must be > 0; a non-positive
            speed raises ``ValueError`` (a stationary truck over a non-zero
            distance is undefined).
        gradient_pct: Road grade in percent (rise/run * 100); negative is
            downhill.
        temperature_c: Ambient air temperature (C), drives the HVAC load.
        wind_mps: Headwind speed (m/s); positive opposes motion. Defaults to 0.
        truck: Vehicle specification. Defaults to the canonical
            :data:`nexdash.config.TRUCK`.

    Returns:
        A dict with keys ``rolling``, ``aero``, ``gradient``, ``aux``,
        ``regen`` and ``total`` (all kWh). ``gradient`` is the (possibly
        negative) net gradient term *including* regen on descents; ``regen``
        is the (>= 0) magnitude of energy recovered downhill, reported
        separately for diagnostics. ``total`` equals
        ``rolling + aero + gradient + aux`` and matches
        :func:`segment_energy_kwh`.
    """
    # --- Guard against degenerate inputs ---------------------------------- #
    if distance_km <= 0.0:
        return {
            "rolling": 0.0,
            "aero": 0.0,
            "gradient": 0.0,
            "aux": 0.0,
            "regen": 0.0,
            "total": 0.0,
        }
    # A moving segment must have a positive speed: aux energy is power x (d/speed),
    # so speed -> 0 diverges. Fail loud rather than returning an absurd number
    # (a previous version clamped to 1e-6 and produced ~2e8 kWh).
    if speed_kph <= 0.0:
        raise ValueError(f"speed_kph must be > 0 for a moving segment; got {speed_kph}")

    # --- Unit conversions ------------------------------------------------- #
    distance_m = distance_km * 1000.0
    speed_mps = speed_kph / 3.6
    air_speed_mps = speed_mps + wind_mps  # headwind-relative air speed
    mass_kg = truck.kerb_mass_kg + payload_t * 1000.0
    travel_time_h = distance_km / speed_kph  # hours, for kW * h -> kWh

    # --- Wheel-side traction forces and energies (Joules) ----------------- #
    f_rolling = _crr_factor(speed_kph, temperature_c, truck) * mass_kg * G
    e_rolling_j = f_rolling * distance_m

    rho_air = _air_density(temperature_c)  # denser in the cold -> more drag
    f_aero = 0.5 * rho_air * truck.cd * truck.frontal_area_m2 * air_speed_mps**2
    e_aero_j = f_aero * distance_m

    theta = math.atan(gradient_pct / 100.0)
    f_grade = mass_kg * G * math.sin(theta)
    e_grade_j = f_grade * distance_m  # >0 uphill, <0 downhill

    # --- Convert to battery-side kWh through the drivetrain --------------- #
    rolling_kwh = (e_rolling_j / _J_PER_KWH) / truck.drivetrain_eff
    aero_kwh = (e_aero_j / _J_PER_KWH) / truck.drivetrain_eff

    if e_grade_j >= 0.0:
        # Climbing: full potential energy charged against the battery.
        gradient_kwh = (e_grade_j / _J_PER_KWH) / truck.drivetrain_eff
        regen_kwh = 0.0
    else:
        # Descending: only a fraction is recovered (tapered by cold battery and
        # steep-grade power limits), then scaled by drivetrain efficiency to
        # approximate the round-trip recovery loss.
        recovered_j = -e_grade_j * _regen_fraction(temperature_c, gradient_pct, truck)
        regen_kwh = (recovered_j / _J_PER_KWH) * truck.drivetrain_eff
        gradient_kwh = -regen_kwh  # net battery credit (negative energy)

    # --- Auxiliary / HVAC load (drawn directly from the battery) ---------- #
    aux_kw = _auxiliary_power_kw(temperature_c, truck)
    aux_kwh = aux_kw * travel_time_h

    total_kwh = rolling_kwh + aero_kwh + gradient_kwh + aux_kwh

    return {
        "rolling": rolling_kwh,
        "aero": aero_kwh,
        "gradient": gradient_kwh,
        "aux": aux_kwh,
        "regen": regen_kwh,
        "total": total_kwh,
    }


def segment_energy_kwh(
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    *,
    wind_mps: float = 0.0,
    truck: Truck = TRUCK,
) -> float:
    """Return the battery energy (kWh) required to drive one segment.

    Thin wrapper over :func:`energy_breakdown` returning only the ``total``.
    The value is normally positive but may be slightly negative on a strong
    downhill where regenerated energy exceeds the rolling/aero/aux demand.

    Args:
        distance_km: Segment length (km).
        payload_t: Cargo payload (tonnes).
        speed_kph: Average travel speed (km/h).
        gradient_pct: Road grade in percent (negative is downhill).
        temperature_c: Ambient air temperature (C).
        wind_mps: Headwind speed (m/s); positive opposes motion. Default 0.
        truck: Vehicle specification. Defaults to :data:`nexdash.config.TRUCK`.

    Returns:
        Total battery-side energy for the segment in kWh.
    """
    return energy_breakdown(
        distance_km,
        payload_t,
        speed_kph,
        gradient_pct,
        temperature_c,
        wind_mps=wind_mps,
        truck=truck,
    )["total"]
