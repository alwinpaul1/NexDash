"""Model-driven SOC-drain simulation and trip planning for the NexOS planner.

This module is the backend brain behind the frontend route planner. Given a
total route distance + duration (the frontend supplies these from the TomTom
truck-routing API), a start/min state of charge, payload and ambient
temperature, it walks the route in fixed-distance chunks and uses the trained
:func:`nexdash.model.predict_energy` model to estimate per-chunk energy draw.
SOC is accumulated against the eActros 600's ~600 kWh battery; whenever the
projected SOC would fall below the operator's ``min_soc`` floor, an *adaptive*
DC fast-charge stop is inserted on a ~400 kW CCS charger -- topping up only as
high as the rest of the route needs (see ``_adaptive_target_soc``). EU 561
driving rules are layered on top (a 45 min break after 4.5 h driving and an 11 h
daily rest once a day hits the 9 h driving cap, splitting long routes across
calendar days). The result mirrors the ``PlanResult`` shape the frontend
expects (minus ``geometry``, which the frontend owns).

Geometry-enriched mode
----------------------
When the frontend also supplies the road ``geometry`` (the TomTom polyline),
:func:`plan_route` enriches it via :func:`nexdash.geodata.enrich_route` (real
per-segment gradient, temperature and wind from Open-Meteo) and simulates SOC
over it. The many short enriched segments are *aggregated* into ~``CHUNK_KM``
windows that carry the distance-weighted-average gradient/temperature/wind, and
the model is called once per window. This aggregation matters for honesty: the
energy model over-predicts on very short (<~5 km) legs, and a real polyline is
downsampled to ~2-10 km segments, so predicting per raw segment and summing
would inflate total route energy with polyline *density* rather than physics
(finer slicing -> higher kWh). Averaging is faithful for the dominant terms
(avg gradient * window distance equals the window's net climb). The response
then also carries ``elevationProfile`` + ``conditions`` and a
``summary.elevationGainM`` (total climb). When geometry is absent the planner
falls back to the flat-assumption mode described below.

Honest approximations
---------------------
* **Flat gradient (fallback only).** Without geometry, ``gradient_pct`` is
  fixed at 0 for every chunk, so net climb/descent over the route is assumed to
  cancel out. With geometry, the real per-segment gradient is used.
* **Wind handling.** The energy model takes a scalar ``wind_mps`` *signed*
  headwind component (positive = headwind, negative = tailwind). In geometry
  mode :func:`nexdash.geodata.enrich_route` projects Open-Meteo's wind
  (magnitude + direction) onto each segment's travel bearing --
  ``wind_mps = speed * cos(windFromDir - travelBearing)`` -- so a head-on wind
  adds drag and a following wind relieves it. The model is trained on the same
  signed convention (see ``data_gen``). Without geometry a mild constant 3 m/s
  headwind is assumed throughout.
* **Per-segment speed (geometry mode).** Speed is redistributed across segments
  by gradient -- slower on sustained climbs, capped on descents -- then anchored
  so the total drive time still equals the routing engine's measured duration
  (see ``_segment_speed_shape``). The absolute per-segment speeds are a gradient
  heuristic, not measured traffic/road-class speeds; the engine's true
  per-segment travel time is not supplied to the model. In the flat fallback a
  single ``distance_km / duration_h`` average is applied to every chunk.
* **Payload decay.** With >=2 waypoints carrying ``dropWeightKg`` the truck
  lightens at each stop: ``payload_t`` steps down at the stop's distance (see
  ``_payload_t_at``) so later legs cost less energy. The leg INTO a stop still
  carries the full pre-drop weight (conservative). Without per-stop drop weights
  (or on a single-leg route) payload is held constant for the whole trip --
  conservative (over-estimates energy on later legs).
* **Charging model.** Inserted stops are DC fast-charge at a rated ~400 kW CCS
  (the eActros 600's current real capability; MCS 1 MW is not yet shipping).
  Charge *time* follows a power-vs-SOC **taper** (full power to ~80% SOC,
  derating into the tail; see ``_charge_minutes``), so the slow ``80->100%``
  region is not under-counted. The recharge target is **adaptive** (see
  ``_adaptive_target_soc``): a stop tops up only as high as the rest of the route
  needs (arriving at the reserve floor), capped at 100% and reaching into the
  slow tail only when that avoids a second stop; ``charge_target_soc`` is the
  soft ceiling for intermediate stops, not a hard cap. Energy is priced here at a
  flat 0.45 EUR/kWh fallback; when a real per-station tariff feed is available the
  frontend re-costs each stop at the selected station's actual EUR/kWh, so charge
  cost is indicative (flat) unless a station tariff is present -- never contractual.
* **Driver hours (EU 561).** A 45 min break is inserted after 4.5 h continuous
  driving, and an 11 h daily rest once a calendar day reaches the 9 h daily
  driving cap -- so a long route is split across days (``driver.perDay`` carries
  the per-shift breakdown, ``driver.dailyH`` is the heaviest single day and
  ``driver.weeklyH`` the heaviest 7-day driving window against 56 h). The extended
  10 h driving day (max 2x/week) is opt-in via ``allow_extended_days`` and any duty
  already worked this week via ``hours_already_driven_this_week``; with both at
  their defaults the daily cap stays 9 h and the week is assumed fresh at
  departure. The reduced (9 h) daily rest and multi-manning are not modelled.
* **Linear SOC within a chunk.** The SOC profile interpolates linearly across
  each chunk; the model only predicts the chunk endpoints.

All returned values are plain JSON-serializable Python types.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

from . import geodata
from .config import DEFAULT_MODEL_PATH, FIELD_CALIBRATION_FACTOR, TRUCK
from .model import predict_energy
from .physics import segment_energy_kwh
from .range import _held_out_mae_kwh

# --------------------------------------------------------------------------- #
# Planner constants (approximations documented in the module docstring)
# --------------------------------------------------------------------------- #

#: Distance of each simulation chunk (km). Smaller -> finer SOC profile.
CHUNK_KM: float = 25.0

#: Recharge target SOC (%) when a charging stop is inserted. Defaults to 95% — the
#: long-haul "charge it up" target (matching how real corridor planners top up), so
#: each stop leaves a large arrival buffer and the route needs fewer stops. This is
#: ABOVE the ~80% power-vs-SOC taper knee, so the 80->95% tail charges slower (see
#: ``_charge_minutes``): a deliberate trade of charge time for fewer stops + buffer.
#: (The taper knee itself stays 80%; only the target SOC changed.)
CHARGE_TARGET_SOC: float = 95.0

#: Rated DC fast-charge power assumed for charge-time estimates (kW). The
#: Mercedes-Benz eActros 600 charges at up to ~400 kW on CCS today; MCS (1 MW)
#: hardware is not yet shipping (electrive, 2026), so 400 kW CCS is the honest
#: current-reality default rather than a fictitious "MCS" rate.
CHARGER_KW: float = 400.0

#: SOC (%) above which DC charging power begins to taper (CP->CV knee). Heavy-BEV
#: packs hold near-peak power to ~80% then derate steeply to protect the cells.
CHARGE_TAPER_KNEE_SOC: float = 80.0

#: Charging power at 100% SOC as a fraction of the rated power (end of the taper).
CHARGE_TAPER_FLOOR_FRAC: float = 0.2

#: Flat energy tariff used for charging-cost estimates (EUR/kWh).
PRICE_EUR_PER_KWH: float = 0.45

#: Assumed steady headwind component (m/s). Zero: no unphysical constant headwind
#: baked into the flat-fallback / per-segment energy (steady-state anchor is still-air).
WIND_MPS: float = 0.0

#: Assumed net road gradient (percent). Flat-route approximation.
GRADIENT_PCT: float = 0.0

# EU Regulation 561/2006 driving-time limits (the subset we model).
EU561_MAX_DRIVE_BEFORE_BREAK_MIN: float = 4.5 * 60.0  # 4h30 continuous driving
EU561_BREAK_MIN: float = 45.0                          # mandatory break length
EU561_DAILY_MAX_DRIVE_H: float = 9.0                   # standard daily driving
EU561_EXT_DAILY_MAX_DRIVE_H: float = 10.0              # extended daily driving (opt-in)
EU561_MAX_EXT_DAYS_PER_WEEK: int = 2                   # max 10 h days per week
EU561_WEEKLY_MAX_DRIVE_H: float = 56.0                 # max weekly driving
#: Daily rest inserted once the daily driving cap is hit. Regular EU 561 daily
#: rest is 11 h; the 9 h reduced-rest option (max 3x/week) is not modelled — 11 h
#: is the conservative, always-legal choice (it inserts rest at least as early).
#: The 10 h extended driving day (max 2x/week) is opt-in via
#: ``allow_extended_days``; with the default 0 the cap stays 9 h.
EU561_DAILY_REST_H: float = 11.0

# --- Per-segment speed shaping (geometry mode only) ------------------------- #
#: A loaded truck loses speed on sustained climbs and is speed-governed (barely
#: faster) on descents. These shape each segment's speed RELATIVE to the route
#: average; the shaped speeds are then re-anchored so the total drive time still
#: equals the routing engine's measured duration — speed is redistributed across
#: segments, never added or removed in total. Absolute values are a gradient
#: heuristic, not measured traffic/road-class speeds.
SPEED_GRAD_K_UP: float = 0.06        # fractional slowdown per +1% of climb
SPEED_GRAD_K_DOWN: float = 0.015     # fractional speed-up per 1% of descent
SPEED_DESC_CAP_FRAC: float = 1.05    # descents at most 5% faster (governed)
#: Energy-model speed clamp, bounded to the model's ACTUAL training envelope
#: (data_gen samples speed in [30, 85]). Feeding speeds outside it makes the GBM
#: extrapolate flat and under-predict energy — the optimistic / strand direction.
SEG_SPEED_MIN_KPH: float = 30.0
SEG_SPEED_MAX_KPH: float = 85.0

#: Safety headroom (SOC %) added to an adaptive charge target so the linear-SOC
#: interpolation + rounding never lands the truck a hair below the floor.
CHARGE_HEADROOM_PCT: float = 2.0


def _charge_power_kw(soc_pct: float, rated_kw: float) -> float:
    """Available DC charging power (kW) at a given SOC, with a CP->CV taper.

    Full ``rated_kw`` up to :data:`CHARGE_TAPER_KNEE_SOC` (~80%), then a linear
    derate to ``rated_kw * CHARGE_TAPER_FLOOR_FRAC`` at 100% SOC. A flat-power
    model (no taper) materially understates session time because the slow
    80->100% tail is the most tapered region.
    """
    if soc_pct <= CHARGE_TAPER_KNEE_SOC:
        return rated_kw
    if soc_pct >= 100.0:
        return rated_kw * CHARGE_TAPER_FLOOR_FRAC
    frac = (soc_pct - CHARGE_TAPER_KNEE_SOC) / (100.0 - CHARGE_TAPER_KNEE_SOC)
    return rated_kw * (1.0 - (1.0 - CHARGE_TAPER_FLOOR_FRAC) * frac)


def _charge_minutes(
    arrive_soc: float,
    target_soc: float,
    rated_kw: float,
    battery_kwh: float,
    *,
    step: float = 0.5,
) -> float:
    """Minutes to charge from ``arrive_soc`` to ``target_soc`` under the taper.

    Numerically integrates ``(kWh added per SOC step) / power(SOC)`` across the
    band using the midpoint power, so charging into the tapered tail correctly
    costs more time per percent than charging from near-empty. Returns 0 for a
    non-positive band or non-positive power.
    """
    if target_soc <= arrive_soc or rated_kw <= 0 or battery_kwh <= 0:
        return 0.0
    kwh_per_pct = battery_kwh / 100.0
    minutes = 0.0
    soc = float(arrive_soc)
    while soc < target_soc - 1e-9:
        s = min(step, target_soc - soc)
        power = _charge_power_kw(soc + s / 2.0, rated_kw)
        if power <= 0:
            break
        minutes += (kwh_per_pct * s) / power * 60.0
        soc += s
    return minutes


def _segment_speed_shape(gradient_pct: float) -> float:
    """Unitless speed factor vs the route average for a segment's gradient.

    ``<1`` on climbs (a loaded truck loses speed on sustained grades), slightly
    ``>1`` on descents but capped (trucks are speed-governed, not free-rolling),
    ``1`` on the flat. Monotone in gradient, so a steeper climb is always slower.
    """
    if gradient_pct > 0:
        return 1.0 / (1.0 + SPEED_GRAD_K_UP * gradient_pct)
    if gradient_pct < 0:
        return min(SPEED_DESC_CAP_FRAC, 1.0 + SPEED_GRAD_K_DOWN * (-gradient_pct))
    return 1.0


def _adaptive_target_soc(
    remaining_energy_kwh: float,
    arrive_soc: float,
    battery_kwh: float,
    *,
    charge_floor: float,
    soft_ceiling_soc: float,
    uncertainty_kwh: float = 0.0,
    headroom: float = CHARGE_HEADROOM_PCT,
) -> float:
    """Depart SOC for a charging stop: charge UP TO the target, capped at 100%.

    Each stop tops up to the ``soft_ceiling_soc`` TARGET (default 95%) — the
    "charge it up" long-haul policy — leaving a large arrival buffer and fewer
    stops. It reaches ABOVE the target, into the slow 80->100% tail, only when a
    single charge needs more than the target to finish the route in one stop
    (``need_depart``). When even 100% cannot finish in one stop, it charges to the
    target and the on-demand trigger inserts another stop later. It never charges
    below the arrival SOC.

    ``uncertainty_kwh`` is a forecast-error cushion (the caller passes
    ``mae_band * sqrt(n_remaining_chunks)`` — the sqrt-of-n scaling for roughly
    independent per-chunk errors) added to the energy still owed, so a one-stop
    target in the tail absorbs sub-divergence-band model optimism rather than
    sizing the charge on the model's own (possibly optimistic) number.
    Deterministic.
    """
    if battery_kwh <= 0:
        return min(100.0, max(soft_ceiling_soc, arrive_soc))
    drop_to_dest_pct = ((remaining_energy_kwh + max(0.0, uncertainty_kwh)) / battery_kwh) * 100.0
    need_depart = charge_floor + drop_to_dest_pct + headroom
    if need_depart <= 100.0:
        # One charge can finish: top to the TARGET, or higher into the tail when
        # finishing the route in a single stop needs more than the target.
        depart = max(soft_ceiling_soc, need_depart)
    else:
        # One charge cannot finish: charge to the target and stop again later.
        depart = soft_ceiling_soc
    return min(100.0, max(depart, arrive_soc + 0.5))


def _parse_departure(departure: Optional[str]) -> datetime:
    """Parse an ISO local datetime string; fall back to 'now' on any failure."""
    if departure:
        try:
            return datetime.fromisoformat(departure.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        except (ValueError, TypeError):
            pass
    return datetime.now().replace(second=0, microsecond=0)


def _hhmm(dt: datetime) -> str:
    """Format a datetime as a short ``HH:MM`` clock label."""
    return dt.strftime("%H:%M")


def _iso(dt: datetime) -> str:
    """Format a datetime as an ISO local string (no timezone)."""
    return dt.isoformat(timespec="minutes")


def _parse_iso_opt(value: Optional[str]) -> Optional[datetime]:
    """Parse an optional ISO datetime to naive-local, or ``None`` (no fallback).

    Unlike :func:`_parse_departure` this returns ``None`` on a missing/invalid
    value rather than defaulting to "now", so a missing ``deliverBy`` reads as
    "no deadline" (feasibility unknown) instead of a spurious past deadline.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _build_stops(
    waypoints: Optional[list[dict[str, Any]]],
    total_km: float,
    geometry: Optional[list[list[float]]] = None,
) -> list[dict[str, Any]]:
    """Resolve destination waypoints into per-stop legs along the route.

    Returns one entry per *destination* (the origin, index 0, is excluded), each
    with its cumulative distance from the origin plus any per-stop delivery data
    the caller attached: ``dropWeightKg`` (cargo shed at the stop), ``unloadMin``
    (dwell), and ``deliverBy`` (ISO deadline). Returns ``[]`` when there are fewer
    than two usable waypoints, so the planner falls back to a single continuous leg
    with constant payload (unchanged legacy behaviour).

    Distance placement (``cumKm``):

    * **With ``geometry``** (the routing-engine polyline): each waypoint is snapped
      to the nearest point on the polyline and its ALONG-polyline arc length is
      accumulated (see :func:`_snap_km_on_geometry`), then rescaled to ``total_km``
      (the polyline is great-circle-sampled, so its raw arc length underestimates
      road length — the same rescale used by :func:`_build_chunks`). This puts a
      payload drop where the truck actually reaches it on a winding road, not where
      the straight origin->stop line would place it.
    * **Without geometry** (fallback): the legacy estimate — great-circle leg
      lengths between consecutive waypoints, scaled so the final stop lands exactly
      at ``total_km``. Byte-identical to before this option existed.
    """
    if not waypoints or total_km <= 0:
        return []
    pts = [w for w in waypoints if w.get("lat") is not None and w.get("lng") is not None]
    if len(pts) < 2:
        return []
    coords = [(float(w["lat"]), float(w["lng"])) for w in pts]

    # Geometry mode: snap each waypoint onto the road polyline and accumulate its
    # along-polyline arc length, rescaled to the routing engine's total_km. Only
    # used when EVERY destination snaps successfully; any failure falls back to the
    # great-circle estimate so a partial/garbage polyline never silently mis-places
    # a single stop.
    snapped_cum: Optional[list[float]] = None
    snap_geo = _snap_km_on_geometry(coords[0], geometry)
    if snap_geo is not None:
        poly_total = snap_geo[1]
        scale_geo = total_km / poly_total if poly_total > 0 else 0.0
        if scale_geo > 0:
            acc: list[float] = []
            ok = True
            for i in range(1, len(pts)):
                snap = _snap_km_on_geometry(coords[i], geometry)
                if snap is None:
                    ok = False
                    break
                acc.append(snap[0] * scale_geo)
            if ok and acc:
                snapped_cum = acc

    stops: list[dict[str, Any]] = []
    if snapped_cum is not None:
        for i in range(1, len(pts)):
            w = pts[i]
            stops.append(
                {
                    "label": w.get("label") or f"Stop {i}",
                    "cumKm": min(total_km, round(snapped_cum[i - 1], 3)),
                    "dropWeightKg": max(0.0, float(w.get("dropWeightKg", 0) or 0)),
                    "unloadMin": max(0.0, float(w.get("unloadMin", 0) or 0)),
                    "deliverBy": w.get("deliverBy") or None,
                }
            )
    else:
        legs = [_haversine_km(coords[i - 1], coords[i]) for i in range(1, len(coords))]
        gc_total = sum(legs)
        if gc_total <= 0:
            return []
        scale = total_km / gc_total
        cum = 0.0
        for i in range(1, len(pts)):
            cum += legs[i - 1] * scale
            w = pts[i]
            stops.append(
                {
                    "label": w.get("label") or f"Stop {i}",
                    "cumKm": min(total_km, round(cum, 3)),
                    "dropWeightKg": max(0.0, float(w.get("dropWeightKg", 0) or 0)),
                    "unloadMin": max(0.0, float(w.get("unloadMin", 0) or 0)),
                    "deliverBy": w.get("deliverBy") or None,
                }
            )
    stops[-1]["cumKm"] = total_km  # final destination sits exactly at route end
    return stops


def plan_route(
    *,
    distance_km: float,
    duration_s: float,
    start_soc: float,
    min_soc: float,
    payload_kg: float,
    reserve_pct: float = 10.0,
    max_charge_kw: float = CHARGER_KW,
    charge_target_soc: float = CHARGE_TARGET_SOC,
    field_calibration: float = FIELD_CALIBRATION_FACTOR,
    departure: Optional[str] = None,
    temperature_c: float = 15.0,
    waypoints: Optional[list[dict[str, Any]]] = None,
    geometry: Optional[list[list[float]]] = None,
    leg_timings: Optional[list[dict[str, Any]]] = None,
    speed_limits: Optional[list[dict[str, Any]]] = None,
    allow_extended_days: int = 0,
    hours_already_driven_this_week: float = 0.0,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Simulate SOC drain + charging + driver hours over a route.

    Args:
        distance_km: Total route distance (km), from the routing engine.
        duration_s: Total driving time (seconds), from the routing engine.
        start_soc: Starting state of charge (%).
        min_soc: SOC floor the operator never wants to dip below (%).
        payload_kg: Cargo payload (kg); converted to tonnes for the model.
        departure: ISO local datetime of departure (optional).
        temperature_c: Ambient temperature (deg C); drives HVAC load in model.
            Used as the per-chunk temperature only in the flat fallback mode;
            when ``geometry`` is supplied, real per-segment temperatures from
            :func:`nexdash.geodata.enrich_route` are used instead.
        waypoints: Optional ordered ``[{lat,lng,label?}]`` list (origin first,
            then destinations). Destinations may carry per-stop delivery data:
            ``dropWeightKg`` (cargo shed at the stop -> the truck lightens and
            later legs cost less energy), ``unloadMin`` (dwell folded into the
            ETA for intermediate stops), and ``deliverBy`` (ISO deadline checked
            for feasibility). With >=2 waypoints the simulation is split per leg
            so the response carries per-stop arrival SOC + ETA; with fewer it
            falls back to a single continuous leg at constant payload. Waypoints
            also give inserted charging stops a sensible on-route coordinate.
        geometry: Optional ``[[lat, lng], ...]`` road polyline from the routing
            engine. When present the planner enriches it (elevation gradient +
            weather) and simulates SOC per enriched segment; when absent it
            falls back to the flat-route approximation.
        allow_extended_days: How many of this trip's calendar days may use the
            EU 561 extended 10 h driving cap instead of the standard 9 h (clamped
            to ``[0, EU561_MAX_EXT_DAYS_PER_WEEK]`` = ``[0, 2]``). A day "uses" an
            extended slot the moment its driving crosses 9 h; once the allowance
            is spent later days revert to the 9 h cap (inserting the 11 h rest
            earlier). With the default ``0`` every day caps at 9 h and the output
            is byte-identical to before this option existed.
        hours_already_driven_this_week: Driving hours already worked earlier in
            the current EU 561 fixed week, before this trip's departure (clamped
            ``>= 0``). Seeded as prior driving into the heaviest 7-day window so a
            mid-week driver sits closer to the 56 h weekly cap and ``eu561ok``
            reflects it. With the default ``0.0`` the week is assumed fresh at
            departure, exactly as before.
        model_path: Path to the trained energy model artifact.

    Returns:
        A dict with ``socProfile``, ``segments``, ``chargingStops``, ``stops``
        (per-destination arrival SOC / ETA / deliver-by feasibility) and
        ``summary`` keys, matching the frontend ``PlanResult`` contract (minus
        ``geometry``, which the frontend supplies). ``summary.assumptions`` is a
        machine-readable list of the honest modelling caveats. When ``geometry``
        is supplied it additionally carries ``elevationProfile`` + ``conditions``
        and ``summary.elevationGainM``.
    """
    battery_kwh = TRUCK.battery_kwh
    payload_t = max(0.0, payload_kg) / 1000.0
    # Clamp SOC inputs to physical bounds (mirrors range.check_reachability), so a
    # stray start_soc>100 or negative min_soc cannot render SOC>100%/<0% or silently
    # defeat the charge trigger (the dangerous optimistic direction).
    start_soc = min(100.0, max(0.0, float(start_soc)))
    min_soc = min(100.0, max(0.0, float(min_soc)))
    reserve_pct = max(0.0, min(100.0, float(reserve_pct)))
    # Field-calibration factor: clamp to (0, 1] so it can ONLY lower the displayed
    # energy headline (>1 cannot inflate; <=0 falls back to the conservative figure).
    # It is applied solely to summary.energyKwh / kwhPer100 below — never to the SOC
    # walk or charge trigger, which stay on the conservative estimate for safety.
    field_calibration = float(field_calibration)
    # <=0 means "disabled" (use the conservative figure); positive values are floored
    # at 0.5 so an absurd knob value can't make the headline physically implausible.
    field_calibration = 1.0 if field_calibration <= 0.0 else min(1.0, max(0.5, field_calibration))
    # EU 561 narrowings (opt-in; defaults reproduce today's output exactly).
    # allow_extended_days: how many days may use the 10 h cap (clamped [0, 2]).
    allow_extended_days = int(max(0, min(EU561_MAX_EXT_DAYS_PER_WEEK, int(allow_extended_days))))
    # hours_already_driven_this_week: prior duty seeded into the weekly window.
    hours_already_driven_this_week = max(0.0, float(hours_already_driven_this_week))

    distance_km = max(0.0, float(distance_km))
    duration_h = max(0.0, float(duration_s)) / 3600.0
    # Average speed across the whole route (flat approximation). Guard /0; a missing
    # routing duration falls back to 70 km/h but is disclosed in `assumptions`.
    _duration_missing = duration_h <= 0.0 and distance_km > 0.0
    avg_speed_kph = (distance_km / duration_h) if duration_h > 0 else 70.0
    avg_speed_kph = max(20.0, min(95.0, avg_speed_kph))

    depart_dt = _parse_departure(departure)
    clock = depart_dt

    # Running trip state.
    soc = float(start_soc)
    cum_km = 0.0
    soc_profile: list[dict[str, float]] = [{"distKm": 0.0, "soc": round(soc, 2)}]
    segments: list[dict[str, Any]] = []
    charging_stops: list[dict[str, Any]] = []

    total_energy_kwh = 0.0
    min_soc_seen = soc

    # Driver-hours accounting (minutes).
    drive_since_break_min = 0.0
    total_drive_min = 0.0
    total_break_min = 0.0
    total_charge_min = 0.0
    total_unload_min = 0.0
    n_breaks = 0

    # EU 561 multi-day accounting: a calendar "day" of driving caps at 9 h, after
    # which an 11 h daily rest is inserted and the day resets; weekly driving
    # accrues toward 56 h. per_day records each shift for the UI breakdown.
    day_drive_min = 0.0
    week_drive_min = 0.0
    day_index = 0
    day_breaks = 0
    total_daily_rest_min = 0.0
    per_day: list[dict[str, Any]] = []
    # Extended-day accounting: a day's applied driving cap is 10 h while extended
    # slots remain (or this day already became extended), else 9 h. A day "uses" a
    # slot the moment its driving crosses 9 h. `day_extended` is the current day's
    # flag, recorded on each per_day entry.
    ext_days_used = 0
    day_extended = False

    def _day_cap_h() -> float:
        """Applied daily driving cap (h) for the current day, EU 561 extended-aware."""
        if day_extended or ext_days_used < allow_extended_days:
            return EU561_EXT_DAILY_MAX_DRIVE_H
        return EU561_DAILY_MAX_DRIVE_H

    # Per-destination legs (for per-stop SOC/ETA, payload decay, deliver-by).
    starting_payload_t = payload_t
    stops_meta = _build_stops(waypoints, distance_km, geometry)
    stops_out: list[dict[str, Any]] = []
    next_stop_idx = 0

    # Open drive segment we accumulate chunks into until a break/charge/end.
    seg_open = distance_km > 0
    seg_km = 0.0
    seg_drive_min = 0.0
    seg_soc_start = soc
    seg_start_clock = clock

    def _close_drive_segment() -> None:
        """Flush the currently open drive segment into the timeline."""
        nonlocal seg_km, seg_drive_min, seg_soc_start, seg_start_clock
        if seg_km <= 0:
            return
        end_clock = seg_start_clock + timedelta(minutes=seg_drive_min)
        segments.append(
            {
                "type": "drive",
                "km": round(seg_km, 1),
                "durationMin": round(seg_drive_min),
                "socStart": round(seg_soc_start, 1),
                "socEnd": round(soc, 1),
                "startTime": _hhmm(seg_start_clock),
                "endTime": _hhmm(end_clock),
                "limitMin": round(EU561_MAX_DRIVE_BEFORE_BREAK_MIN),
            }
        )
        seg_km = 0.0
        seg_drive_min = 0.0

    # Build the list of chunks covering the route. Each chunk carries its own
    # physical conditions: in flat fallback mode these are the constant
    # approximations; in geometry mode they come from the enriched profile.
    enrichment = _enrich(geometry, departure, distance_km, leg_timings)
    chunks, chunk_measured_speeds = _build_chunks(distance_km, enrichment, temperature_c)

    # Payload carried during each chunk, accounting for per-stop drops: the truck
    # lightens as it sheds cargo at each destination, so later legs cost less
    # energy. A chunk uses the payload active at its START distance (drops happen
    # AT a stop); with no per-stop data this stays the constant starting payload,
    # exactly as before. Clamped at 0 if the drops sum to more than the start load.
    chunk_start_km: list[float] = []
    _acc = 0.0
    for chunk_km, *_rest in chunks:
        chunk_start_km.append(_acc)
        _acc += chunk_km

    def _payload_t_at(dist_km: float) -> float:
        dropped_kg = sum(s["dropWeightKg"] for s in stops_meta if s["cumKm"] <= dist_km + 1e-6)
        return max(0.0, starting_payload_t - dropped_kg / 1000.0)

    chunk_payloads = [_payload_t_at(c0) for c0 in chunk_start_km]

    # Per-chunk posted speed limit (km/h) from the routing engine's speedLimit
    # sections: the distance-weighted HARMONIC mean of the posted limits over the
    # chunk's span (harmonic = the speed reproducing the travel time across a mixed
    # autobahn/town/village span). None where no limit data covers the chunk.
    def _chunk_speed_limit_kph(start_km: float, end_km: float) -> Optional[float]:
        if not speed_limits or end_km <= start_km:
            return None
        covered = 0.0
        time_h = 0.0
        for sl in speed_limits:
            try:
                f = max(start_km, float(sl.get("fromKm", 0.0)))
                t = min(end_km, float(sl.get("toKm", 0.0)))
                kmh = float(sl.get("kmh", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
            if t > f and kmh > 0:
                covered += t - f
                time_h += (t - f) / kmh
        if covered <= 0 or time_h <= 0:
            return None
        return covered / time_h

    chunk_speed_limits = [
        _chunk_speed_limit_kph(chunk_start_km[i], chunk_start_km[i] + chunks[i][0])
        for i in range(len(chunks))
    ]

    # Per-segment speed (geometry mode): redistribute the route-average speed by
    # gradient — slower on climbs, capped on descents — then ANCHOR so the total
    # drive time still equals the routing engine's measured duration (speed moves
    # between segments, the total is never changed). Energy uses the clamped speed
    # (the model's training domain); drive time uses the UNCLAMPED anchored speed
    # so the ETA stays exact. Flat-fallback mode has no per-segment gradient, so
    # every chunk keeps the single average exactly as before.
    if (
        enrichment
        and chunks
        and speed_limits
        and avg_speed_kph > 0
        and all(s is not None and s > 0 for s in chunk_speed_limits)
    ):
        # Tier S — shape per-segment speed by the REAL posted limits (autobahn 80,
        # town 50, village 30), then ANCHOR so the total drive time still equals the
        # routing engine's measured duration: relative road speeds are preserved and
        # the overall ETA is unchanged. This replaces one flat average with a
        # road-aware speed profile. Energy uses the clamped speed (training domain).
        _shapes = [float(s) for s in chunk_speed_limits]
        _dists = [c[0] for c in chunks]
        _tot_d = sum(_dists)
        _weighted = sum(d / s for d, s in zip(_dists, _shapes) if s > 0)
        _base = avg_speed_kph * (_weighted / _tot_d) if _tot_d > 0 else avg_speed_kph
        drive_speeds = [max(1e-6, _base * s) for s in _shapes]  # unclamped -> exact ETA
        chunk_speeds = [min(SEG_SPEED_MAX_KPH, max(SEG_SPEED_MIN_KPH, v)) for v in drive_speeds]
        _speed_source = "speed-limit"
    elif enrichment and chunks and all(m is not None for m in chunk_measured_speeds):
        # Tier A — REAL per-leg speed measured from the routing engine's travel time
        # is available for every chunk: use it directly (traffic/road-class aware).
        # ETA from these speeds already equals the engine's duration (same source),
        # so no harmonic anchor is needed. Energy still clamps to the training domain.
        drive_speeds = [float(m) for m in chunk_measured_speeds]
        chunk_speeds = [min(SEG_SPEED_MAX_KPH, max(SEG_SPEED_MIN_KPH, v)) for v in drive_speeds]
        _speed_source = "measured"
    elif enrichment and len(chunks) > 1 and avg_speed_kph > 0:
        _shapes = [_segment_speed_shape(c[1]) for c in chunks]
        _dists = [c[0] for c in chunks]
        _tot_d = sum(_dists)
        _weighted = sum(d / s for d, s in zip(_dists, _shapes) if s > 0)
        _base_kph = avg_speed_kph * (_weighted / _tot_d) if _tot_d > 0 else avg_speed_kph
        drive_speeds = [max(1e-6, _base_kph * s) for s in _shapes]  # unclamped -> exact ETA
        chunk_speeds = [min(SEG_SPEED_MAX_KPH, max(SEG_SPEED_MIN_KPH, v)) for v in drive_speeds]
        _speed_source = "heuristic"
    else:
        # ETA uses the true route average; the ENERGY input is clamped to the
        # model's training envelope so a high average speed is not extrapolated.
        drive_speeds = [avg_speed_kph] * len(chunks)
        _energy_spd = min(SEG_SPEED_MAX_KPH, max(SEG_SPEED_MIN_KPH, avg_speed_kph))
        chunk_speeds = [_energy_spd] * len(chunks)
        _speed_source = "average"

    # Predict energy for every chunk up front (model-driven; the wind magnitude is
    # used directly as the headwind component, see module docstring). Doing this
    # before the walk lets the charging check look AHEAD at the energy still owed to
    # the destination, so we only charge when genuinely required.
    #
    # PHYSICS CROSS-CHECK (mirrors range.check_reachability's directional guard):
    # the data-driven model can *under*-predict on out-of-envelope terrain (e.g. a
    # sustained steep grade that never occurs in training) — the dangerous,
    # optimistic direction that would silently delay a charge and strand the truck.
    # So for each chunk we also compute a first-principles segment_energy_kwh; when
    # the model sits below physics by more than the divergence band, we use the
    # conservative (higher) physics value for the SOC walk + charge trigger and
    # flag the chunk out-of-envelope. The band matches range.py: max(3*MAE, 15%).
    mae_band = _held_out_mae_kwh(str(model_path))
    chunk_energies: list[float] = []
    n_low_confidence = 0
    sum_model_kwh = 0.0
    sum_physics_kwh = 0.0
    for i, (chunk_km, chunk_grad, chunk_temp, chunk_wind) in enumerate(chunks):
        feats = {
            "distance_km": chunk_km,
            "payload_t": chunk_payloads[i],
            "speed_kph": chunk_speeds[i],
            "gradient_pct": chunk_grad,
            "temperature_c": chunk_temp,
            "wind_mps": chunk_wind,
        }
        model_kwh = max(0.0, float(predict_energy(feats, model_path=model_path)))
        physics_kwh = float(
            segment_energy_kwh(
                distance_km=chunk_km,
                payload_t=chunk_payloads[i],
                speed_kph=drive_speeds[i],  # TRUE (unclamped) speed: cross-check at real-speed physics
                gradient_pct=chunk_grad,
                temperature_c=chunk_temp,
                wind_mps=chunk_wind,
                truck=TRUCK,
            )
        )
        sum_model_kwh += model_kwh
        sum_physics_kwh += physics_kwh
        diverges = (physics_kwh - model_kwh) > max(3.0 * mae_band, 0.15 * abs(physics_kwh))
        if diverges:
            n_low_confidence += 1
            chunk_energies.append(max(model_kwh, physics_kwh))
        else:
            chunk_energies.append(model_kwh)

    for i, (chunk_km, chunk_grad, chunk_temp, chunk_wind) in enumerate(chunks):
        chunk_energy = chunk_energies[i]
        chunk_drive_min = (chunk_km / drive_speeds[i]) * 60.0
        chunk_soc_drop = (chunk_energy / battery_kwh) * 100.0
        projected_soc = soc - chunk_soc_drop

        # --- Charging check: is a charge genuinely required to finish the route? ---
        # The reserve raises a SOFT trigger so that, WHEN a charge is needed, we take
        # it early and keep a cushion mid-route. But we only actually charge if
        # continuing without charging would drop below the HARD min_soc floor before
        # the destination -- i.e. the remaining route is truly unreachable on the
        # current charge. Without this look-ahead a short leg that merely dips into
        # the reserve band (yet finishes well above min_soc) would trigger a spurious
        # full recharge at the origin, contradicting check_reachability, which calls
        # the very same trip reachable.
        #
        # The en-route floor is the HIGHER of the two operator bounds, NOT their sum:
        # ``min_soc`` ("arrive with at least") is the destination minimum, and
        # ``reserve_pct`` ("safety reserve") is the cushion never to dip below en
        # route. Holding ``max(min_soc, reserve_pct)`` keeps SOC above both without
        # double-counting -- adding them (the old ``min_soc + reserve_pct``) reserved
        # 35% for a 15%-arrival/20%-reserve trip, forcing premature/extra charges and
        # disagreeing with check_reachability, which holds back reserve_pct ALONE.
        charge_floor = max(min_soc, max(0.0, reserve_pct))
        remaining_energy_kwh = sum(chunk_energies[i:])
        soc_at_end_without_charge = soc - (remaining_energy_kwh / battery_kwh) * 100.0
        if projected_soc < charge_floor and soc_at_end_without_charge < min_soc:
            # Close the running drive segment, then charge before continuing.
            if seg_open:
                _close_drive_segment()
            arrive_soc = soc
            # Adaptive target: charge only as high as the rest of the route needs
            # (arriving at the destination at the reserve floor), capped at 100% —
            # dipping into the slow 80->100% tail only when it secures the trip in
            # one stop, otherwise topping to the soft ceiling and charging again
            # later. `charge_target_soc` becomes that soft ceiling, not a hard cap.
            # Forecast-uncertainty cushion: the adaptive target follows the model's
            # own energy number, which can be optimistic within the divergence band.
            # Add mae_band * sqrt(n_remaining_chunks) (sqrt-of-n for roughly
            # independent per-chunk errors) so the depart SOC absorbs that drift
            # rather than arriving exactly at the floor on an optimistic estimate.
            n_remaining = max(1, len(chunk_energies) - i)
            depart_soc = _adaptive_target_soc(
                remaining_energy_kwh,
                arrive_soc=arrive_soc,
                battery_kwh=battery_kwh,
                charge_floor=charge_floor,
                soft_ceiling_soc=charge_target_soc,
                uncertainty_kwh=mae_band * math.sqrt(n_remaining),
            )
            kwh_added = max(0.0, (depart_soc - arrive_soc) / 100.0 * battery_kwh)
            charge_kw = max_charge_kw if max_charge_kw and max_charge_kw > 0 else CHARGER_KW
            # Taper-aware charge time: the 80->target tail charges slower than a
            # flat-rate model would (see _charge_minutes). Energy/cost are
            # unaffected by the taper (it changes time, not kWh delivered).
            charge_min = _charge_minutes(arrive_soc, depart_soc, charge_kw, battery_kwh)
            cost_eur = kwh_added * PRICE_EUR_PER_KWH

            ch_start = clock
            ch_end = ch_start + timedelta(minutes=charge_min)
            # Place the stop ON the actual road polyline at this distance.
            # Falls back to the straight waypoint line only if geometry is absent.
            lat, lng = _interp_on_geometry(geometry, cum_km, distance_km)
            if lat is None:
                lat, lng = _interp_point(waypoints, cum_km, distance_km)
            station = {
                "name": f"DC Fast-Charge Hub {len(charging_stops) + 1}",
                "lat": lat,
                "lng": lng,
            }
            segments.append(
                {
                    "type": "charge",
                    "station": station,
                    "startTime": _hhmm(ch_start),
                    "endTime": _hhmm(ch_end),
                    "durationMin": round(charge_min),
                    "socStart": round(arrive_soc, 1),
                    "socEnd": round(depart_soc, 1),
                    "kWh": round(kwh_added, 1),
                    "costEur": round(cost_eur, 2),
                }
            )
            charging_stops.append(
                {
                    "index": len(charging_stops),
                    "name": station["name"],
                    "distKm": round(cum_km, 1),  # route distance to the stop (elevation-chart marker)
                    "lat": lat,
                    "lng": lng,
                    "arriveSoc": round(arrive_soc, 1),
                    "departSoc": round(depart_soc, 1),
                    "kWh": round(kwh_added, 1),
                    "costEur": round(cost_eur, 2),
                    "durationMin": round(charge_min),
                }
            )
            # Charging dwell time (credited against the 45-min EU 561 break clock
            # only if the dwell is itself >= 45 min — guarded below).
            total_charge_min += charge_min
            clock = ch_end
            soc = depart_soc
            # Record the charge as an INSTANTANEOUS SOC jump at this distance, so the
            # SOC-coloured route shows the battery rising AT the charger — not
            # gradually over the chunk driven away from it. Without these two points
            # the profile interpolates linearly from the pre-charge low to the
            # post-charge high across the next chunk, which reads as the truck
            # gaining charge while it drives.
            soc_profile.append({"distKm": round(cum_km, 1), "soc": round(arrive_soc, 2)})
            soc_profile.append({"distKm": round(cum_km, 1), "soc": round(depart_soc, 2)})
            # A charge satisfies the EU 561 45-min break ONLY if its dwell is itself
            # >= 45 min; a short adaptive top-up must NOT reset the continuous-driving
            # clock, else a ~4-min splash would suppress a legally-owed break and
            # under-state the ETA.
            if charge_min >= EU561_BREAK_MIN:
                drive_since_break_min = 0.0
                # The dwell is itself a valid EU 561 break (>= 45 min off the wheel),
                # so count it alongside dedicated rest breaks: the driver rested here
                # even though the stop's purpose was charging. Without this a route
                # whose break need is met by a long charge under-reports the breaks
                # actually taken (showed 1 where the driver took 2 — a rest + a charge).
                n_breaks += 1
                day_breaks += 1
            # Reopen a fresh drive segment after the charge.
            seg_open = True
            seg_soc_start = soc
            seg_start_clock = clock
            projected_soc = soc - chunk_soc_drop

        # --- EU 561 daily driving cap: insert an 11 h overnight rest. ---
        # Checked before the 4.5 h break so the daily limit takes priority; an
        # 11 h rest also satisfies the 45 min break, splitting a long route across
        # calendar days so no single day exceeds the applied driving limit. The cap
        # is 10 h while extended slots remain (opt-in via allow_extended_days), else
        # the standard 9 h — so with the default the rest lands exactly as before.
        if day_drive_min + chunk_drive_min > _day_cap_h() * 60.0:
            if seg_open:
                _close_drive_segment()
            per_day.append(
                {
                    "day": day_index + 1,
                    "dateLabel": clock.strftime("%a %d %b"),
                    "drivingH": round(day_drive_min / 60.0, 2),
                    "breaks": day_breaks,
                    "extended": day_extended,
                }
            )
            rest_start = clock
            rest_end = rest_start + timedelta(hours=EU561_DAILY_REST_H)
            segments.append(
                {
                    "type": "daily_rest",
                    "startTime": _hhmm(rest_start),
                    "endTime": _hhmm(rest_end),
                    "durationMin": round(EU561_DAILY_REST_H * 60.0),
                    "label": "Daily Rest (11h)",
                    "dateLabel": rest_start.strftime("%a %d %b"),
                }
            )
            total_daily_rest_min += EU561_DAILY_REST_H * 60.0
            clock = rest_end
            day_drive_min = 0.0
            drive_since_break_min = 0.0  # the 11 h rest also clears the break clock
            day_breaks = 0
            day_index += 1
            day_extended = False  # a fresh day starts on the standard cap
            seg_open = True
            seg_soc_start = soc
            seg_start_clock = clock

        # --- EU 561 break check: 4.5h continuous driving cap. ---
        if drive_since_break_min + chunk_drive_min > EU561_MAX_DRIVE_BEFORE_BREAK_MIN:
            # Drive the slice of THIS chunk that still fits before the 4.5 h cap so the
            # break lands EXACTLY on the limit (4:30 / 4:30), not at the previous
            # ~25 km chunk boundary (which left a few minutes of allowance unused).
            # ETA-neutral: partial + remainder = the same chunk, plus the same 45 min
            # break -- only WHERE in the chunk the break sits moves. The chunk is then
            # shrunk to its remainder, driven normally by the block below.
            t_until = EU561_MAX_DRIVE_BEFORE_BREAK_MIN - drive_since_break_min
            if 0.0 < t_until < chunk_drive_min:
                frac = t_until / chunk_drive_min
                p_km = chunk_km * frac
                p_soc = chunk_soc_drop * frac
                p_energy = chunk_energy * frac
                soc -= p_soc
                min_soc_seen = min(min_soc_seen, soc)
                cum_km += p_km
                clock = clock + timedelta(minutes=t_until)
                seg_km += p_km
                seg_drive_min += t_until
                total_drive_min += t_until
                day_drive_min += t_until
                total_energy_kwh += p_energy
                # Remainder of the chunk -> driven after the break by the block below.
                chunk_km -= p_km
                chunk_soc_drop -= p_soc
                chunk_energy -= p_energy
                chunk_drive_min -= t_until
                projected_soc = soc - chunk_soc_drop
            if seg_open:
                _close_drive_segment()
            br_start = clock
            br_end = br_start + timedelta(minutes=EU561_BREAK_MIN)
            segments.append(
                {
                    "type": "rest",
                    "startTime": _hhmm(br_start),
                    "endTime": _hhmm(br_end),
                    "durationMin": round(EU561_BREAK_MIN),
                    "label": "Rest Break",
                }
            )
            total_break_min += EU561_BREAK_MIN
            n_breaks += 1
            day_breaks += 1
            clock = br_end
            drive_since_break_min = 0.0
            seg_open = True
            seg_soc_start = soc
            seg_start_clock = clock

        # --- Drive the chunk. ---
        soc = projected_soc
        min_soc_seen = min(min_soc_seen, soc)
        total_energy_kwh += chunk_energy
        cum_km += chunk_km
        clock = clock + timedelta(minutes=chunk_drive_min)

        seg_km += chunk_km
        seg_drive_min += chunk_drive_min
        drive_since_break_min += chunk_drive_min
        total_drive_min += chunk_drive_min
        day_drive_min += chunk_drive_min
        week_drive_min += chunk_drive_min

        # The moment the current day's driving crosses the standard 9 h cap it has
        # used an extended (10 h) slot: mark it so the rest of the day keeps the
        # 10 h cap, and consume one allowance so later days revert to 9 h once the
        # slots run out. Only fires when allow_extended_days > 0 (else the rollover
        # above already inserted the rest at 9 h and the day never reaches here >9 h).
        if (
            not day_extended
            and ext_days_used < allow_extended_days
            and day_drive_min > EU561_DAILY_MAX_DRIVE_H * 60.0 + 1e-9
        ):
            day_extended = True
            ext_days_used += 1

        soc_profile.append({"distKm": round(cum_km, 1), "soc": round(soc, 2)})

        # --- Destination arrival(s) reached within this chunk. Record per-stop
        # arrival SOC + ETA, check the deliver-by deadline, and add unload dwell
        # for intermediate stops. (Payload decay is already baked into the
        # precomputed chunk energies, so energy on later legs is already lower.)
        while (
            next_stop_idx < len(stops_meta)
            and cum_km + 1e-6 >= stops_meta[next_stop_idx]["cumKm"]
        ):
            stop = stops_meta[next_stop_idx]
            is_final = next_stop_idx == len(stops_meta) - 1
            deadline = _parse_iso_opt(stop["deliverBy"])
            on_time = None if deadline is None else bool(clock <= deadline)
            stops_out.append(
                {
                    "index": next_stop_idx,
                    "label": stop["label"],
                    "distKm": round(stop["cumKm"], 1),
                    "arriveSoc": round(soc, 1),
                    "etaLabel": _hhmm(clock),
                    "etaIso": _iso(clock),
                    "dropWeightKg": round(stop["dropWeightKg"], 1),
                    "payloadAfterT": round(_payload_t_at(cum_km), 2),
                    "unloadMin": round(stop["unloadMin"]),
                    "deliverBy": stop["deliverBy"],
                    "onTime": on_time,
                    "isFinal": is_final,
                }
            )
            # Unload dwell advances the clock for intermediate stops; the final
            # delivery's unload happens after arrival so it does not push the ETA.
            if stop["unloadMin"] > 0 and not is_final:
                if seg_open:
                    _close_drive_segment()
                u_start = clock
                u_end = u_start + timedelta(minutes=stop["unloadMin"])
                segments.append(
                    {
                        "type": "unload",
                        "label": f"Unload — {stop['label']}",
                        "startTime": _hhmm(u_start),
                        "endTime": _hhmm(u_end),
                        "durationMin": round(stop["unloadMin"]),
                    }
                )
                total_unload_min += stop["unloadMin"]
                clock = u_end
                seg_open = True
                seg_soc_start = soc
                seg_start_clock = clock
            next_stop_idx += 1

    # Flush the final open drive segment.
    if seg_open:
        _close_drive_segment()

    # Flush the final (partial) day into the per-day breakdown.
    per_day.append(
        {
            "day": day_index + 1,
            "dateLabel": clock.strftime("%a %d %b"),
            "drivingH": round(day_drive_min / 60.0, 2),
            "breaks": day_breaks,
            "extended": day_extended,
        }
    )

    # --- Summary aggregation. ---
    arrival_dt = clock
    driving_h = total_drive_min / 60.0
    charging_min_total = total_charge_min
    total_min = (
        total_drive_min + total_break_min + total_charge_min
        + total_unload_min + total_daily_rest_min
    )
    total_h = total_min / 60.0

    # DISPLAYED energy is field-calibrated (see config.FIELD_CALIBRATION_FACTOR):
    # the steady-state physics figure is mapped to real laden-route consumption.
    # `total_energy_kwh` itself stays conservative (it drove the SOC walk + charge
    # plan above); only the reported headline below is discounted, so charging and
    # reachability are unaffected.
    displayed_energy_kwh = total_energy_kwh * field_calibration
    kwh_per_100 = (displayed_energy_kwh / distance_km * 100.0) if distance_km > 0 else 0.0
    charging_cost = sum(s["costEur"] for s in charging_stops)

    # EU 561 compliance is now judged on the REAL day/week split: the machine
    # inserts an 11 h rest whenever a day would exceed 9 h driving, so the
    # heaviest single day is <= 9 h on a legal plan (a long multi-day trip is
    # Compliant, not the false "Violation" the single-shift model used to flag);
    # the weekly cap is checked against true accrued driving.
    max_day_drive_h = max((d["drivingH"] for d in per_day), default=driving_h)
    # Per-day legality is judged against each day's APPLIED cap: an extended day is
    # legal up to 10 h, a standard day up to 9 h. With allow_extended_days=0 no day
    # is extended, so this is exactly the old "every day <= 9 h" check.
    daily_ok = all(
        d["drivingH"]
        <= (EU561_EXT_DAILY_MAX_DRIVE_H if d.get("extended") else EU561_DAILY_MAX_DRIVE_H)
        + 1e-6
        for d in per_day
    )
    # weeklyH is the heaviest 7-CONSECUTIVE-DAY driving window, not the whole-trip
    # total: EU 561's 56 h is a weekly (rolling) cap, so a multi-week haul must not
    # be falsely flagged. `hours_already_driven_this_week` is prepended as prior
    # driving (one synthetic leading day) so a mid-week departure starts closer to
    # the 56 h cap; with the default 0.0 the week is fresh and the window is the
    # trip's own days exactly as before.
    _per_day_h = [d["drivingH"] for d in per_day]
    _window_h = (
        [hours_already_driven_this_week] + _per_day_h
        if hours_already_driven_this_week > 0
        else _per_day_h
    )
    week_drive_h = max(
        (sum(_window_h[i : i + 7]) for i in range(len(_window_h))), default=driving_h
    )
    eu561ok = (
        daily_ok
        and week_drive_h <= EU561_WEEKLY_MAX_DRIVE_H + 1e-6
    )

    elevation_profile = enrichment["elevationProfile"] if enrichment else []
    # The profile's distance axis is built from great-circle hops between the
    # downsampled polyline points, so it underestimates the real road length
    # (e.g. 559 vs the routing engine's 587 km). Rescale it to distance_km so the
    # chart axis matches the Route Info "Total Distance" the dispatcher sees.
    if elevation_profile and distance_km > 0:
        prof_total = float(elevation_profile[-1].get("distKm", 0.0) or 0.0)
        if prof_total > 0:
            scale = distance_km / prof_total
            elevation_profile = [
                {**p, "distKm": round(float(p["distKm"]) * scale, 3)}
                for p in elevation_profile
            ]
    conditions = enrichment["conditions"] if enrichment else {}
    elevation_gain_m = float(conditions.get("climbM", 0.0)) if conditions else 0.0

    # Machine-readable honest-limitations so the dispatcher/UI consuming this JSON
    # can see the model's caveats, not just the docstring (the brief's #1 axis).
    assumptions: list[str] = []
    if stops_meta and any(s["dropWeightKg"] > 0 for s in stops_meta):
        assumptions.append(
            "Payload decays per stop (truck lightens after each drop), so later legs cost less energy."
        )
    else:
        assumptions.append(
            "Payload held constant for the whole trip (no drop-offs set) — conservative, so later "
            "legs may be over-estimated. Set a stop's drop-off weight to model the truck lightening: "
            "after that stop the payload becomes the remaining load (previous payload minus the "
            "unloaded weight), and the legs after it cost less energy."
        )
    if not enrichment:
        assumptions.append("Flat-route fallback: gradient assumed 0 (no per-segment terrain).")
    assumptions.append(
        f"Charge stops top up to ~{round(charge_target_soc)}% at "
        f"~{round(max_charge_kw or CHARGER_KW)} kW CCS (reaching toward 100% only when one stop "
        f"must top up that high to finish the route), leaving a wide buffer above your {round(reserve_pct)}% reserve and "
        f"fewer stops. Charging past ~80% runs the slower power-vs-SOC taper, so a higher target trades "
        f"charge time for fewer stops; energy follows the model's forecast, bounded by the physics "
        f"cross-check and the min-SOC floor."
    )
    if field_calibration < 1.0:
        assumptions.append(
            f"Displayed energy is scaled by a documented field-calibration factor of "
            f"{field_calibration:.2f} so the headline matches observed laden eActros 600 "
            f"consumption (real-world laden 40 t tests cluster at ~0.96-1.03 kWh/km — Daimler tour "
            f"1.03, Vandijck 0.96, ADAC 0.88) rather than the higher constant-speed steady-state "
            f"physics (~1.22 kWh/km warm anchor at the calibrated CdA 5.0). Charging and "
            f"reachability decisions still use "
            f"the un-discounted conservative estimate, so this only affects the displayed total, "
            f"never whether or when the truck charges. See REAL_WORLD_CALIBRATION.md."
        )
    if _speed_source == "speed-limit":
        assumptions.append(
            "Per-segment speed follows the route's POSTED speed limits (e.g. ~30 km/h in a "
            "village, 50 in town, 80 on the autobahn — capped at the 80 km/h truck limit), then "
            "scaled uniformly so the total drive time equals the routing engine's measured "
            "(traffic-aware) duration — so road-by-road speeds vary while the overall ETA is exact."
        )
    elif _speed_source == "measured":
        assumptions.append(
            "Per-segment speed is MEASURED from the routing engine's per-leg travel time "
            "(traffic / road-class aware); within a leg, variation is still distributed by gradient."
        )
    elif enrichment:
        assumptions.append(
            "Per-segment speed varies with gradient (slower on climbs, capped on descents), "
            "re-anchored so the total ETA still matches the routing engine; the absolute "
            "per-segment speeds are a gradient heuristic, not measured traffic or road-class speeds."
        )
    else:
        assumptions.append("Single average speed applied to every segment (flat fallback).")
    # With BOTH defaults (allow_extended_days=0, hours_already_driven_this_week=0.0)
    # this caveat is emitted verbatim — byte-identical to before these options
    # existed. When an option is opted into, the corresponding "not modelled"
    # clause NARROWS (it is now modelled), per the honest-limitations contract.
    if allow_extended_days == 0 and hours_already_driven_this_week == 0.0:
        assumptions.append(
            "Driver hours follow EU 561: a 45-minute break after 4.5 hours of driving, and an "
            "11-hour daily rest once the 9-hour daily driving limit is reached — so a long route is "
            "split across calendar days and arrival times include the overnight rests. Weekly driving "
            "is capped at 56 hours assuming a fresh week at departure; reduced/compensated rest, "
            "the extended 10-hour day, multi-manning, and hours already worked this week are not modelled."
        )
    else:
        _not_modelled = ["reduced/compensated rest", "multi-manning"]
        if allow_extended_days > 0:
            _daily_cap_clause = (
                f"the daily driving limit is reached (up to {allow_extended_days} day(s) "
                f"of this trip may use the extended 10-hour cap, used {ext_days_used}; "
                "the rest cap at 9 hours)"
            )
        else:
            _daily_cap_clause = "the 9-hour daily driving limit is reached"
            _not_modelled.append("the extended 10-hour day")
        if hours_already_driven_this_week > 0:
            _week_clause = (
                f"is capped at 56 hours, seeded with {round(hours_already_driven_this_week, 1)} "
                "hour(s) already driven earlier this week"
            )
        else:
            _week_clause = "is capped at 56 hours assuming a fresh week at departure"
            _not_modelled.append("hours already worked this week")
        assumptions.append(
            "Driver hours follow EU 561: a 45-minute break after 4.5 hours of driving, and an "
            f"11-hour daily rest once {_daily_cap_clause} — so a long route is "
            "split across calendar days and arrival times include the overnight rests. Weekly driving "
            f"{_week_clause}; " + ", ".join(_not_modelled) + " are not modelled."
        )
    if n_low_confidence > 0:
        assumptions.append(
            f"LOW CONFIDENCE on {n_low_confidence} segment(s): the data-driven model "
            "and a first-principles physics estimate disagree sharply (terrain outside "
            "the model's training envelope). The conservative (higher) physics value "
            "was used for the SOC/charging decision on those segments — treat the plan "
            "as indicative there and keep a wide reserve."
        )
    # Route-level cumulative cross-check: a consistent-sign under-prediction can stay
    # under every per-chunk band yet sum to a dangerous optimism, so we also compare
    # the whole-route model vs physics totals (band proportional to the route).
    if (sum_physics_kwh - sum_model_kwh) > max(3.0 * mae_band, 0.10 * sum_physics_kwh):
        assumptions.append(
            "ROUTE-LEVEL LOW CONFIDENCE: across the whole route the data-driven model "
            "predicts materially less energy than the first-principles physics estimate "
            "— a consistent optimistic drift the per-segment check can miss. Treat the "
            "SOC and charging plan as indicative and keep a wide reserve."
        )
    if _duration_missing:
        assumptions.append(
            "Routing duration was missing or zero; ETA, breaks and driver-hours are "
            "derived from an assumed 70 km/h average, not a measured route time."
        )
    late = [s for s in stops_out if s.get("onTime") is False]
    if late:
        assumptions.append(
            f"{len(late)} stop(s) miss their deliver-by deadline on this plan."
        )
    total_drop_t = sum(s["dropWeightKg"] for s in stops_meta) / 1000.0
    if total_drop_t > starting_payload_t + 1e-9:
        assumptions.append(
            "Drop weights exceed the starting payload; payload clamped at 0 on the affected legs."
        )

    summary = {
        "distanceKm": round(distance_km, 1),
        "drivingTimeH": round(driving_h, 2),
        "chargingTimeMin": round(charging_min_total),
        "totalTimeH": round(total_h, 2),
        "etaLabel": _hhmm(arrival_dt),
        "etaIso": _iso(arrival_dt),
        "startSoc": round(float(start_soc), 1),
        "arrivalSoc": round(soc, 1),
        # `minSoc` is the LOWEST SOC actually reached on the trip (the true low
        # point); `minSocFloor` is the operator's "arrive with at least" SETTING
        # (the floor the plan must stay above). The gauge shows the floor (matching
        # the slider the user set); the achieved low stays available for detail views.
        "minSoc": round(min_soc_seen, 1),
        "minSocFloor": round(float(min_soc), 1),
        "energyKwh": round(displayed_energy_kwh, 1),
        "kwhPer100": round(kwh_per_100, 1),
        "chargingCostEur": round(charging_cost, 2),
        "chargingStops": len(charging_stops),
        "unloadTimeMin": round(total_unload_min),
        "elevationGainM": round(elevation_gain_m, 1),
        "driver": {
            "drivingH": round(driving_h, 2),
            "breaks": n_breaks,
            "totalH": round(total_h, 2),
            # dailyH is the HEAVIEST single calendar day's driving (the value the
            # 9 h daily-limit bar reflects), not the trip total; weeklyH is the
            # heaviest 7-day driving window. Long trips are split by inserted 11 h rests,
            # so perDay carries the per-shift breakdown the UI renders.
            "dailyH": round(max_day_drive_h, 2),
            "dailyMaxH": EU561_DAILY_MAX_DRIVE_H,
            "weeklyH": round(week_drive_h, 2),
            "weeklyMaxH": EU561_WEEKLY_MAX_DRIVE_H,
            "days": len(per_day),
            "perDay": per_day,
            "eu561ok": bool(eu561ok),
        },
        "assumptions": assumptions,
    }

    result: dict[str, Any] = {
        "socProfile": soc_profile,
        "segments": segments,
        "chargingStops": charging_stops,
        "stops": stops_out,
        "summary": summary,
    }
    # Surface the enriched physical context only when geometry was supplied.
    if enrichment:
        result["elevationProfile"] = elevation_profile
        result["conditions"] = conditions
    return result


def _enrich(
    geometry: Optional[list[list[float]]],
    departure: Optional[str],
    distance_km: float,
    leg_timings: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """Enrich ``geometry`` into per-segment conditions, or ``None`` if absent.

    Fails soft: if :func:`nexdash.geodata.enrich_route` yields no usable
    segments (empty/garbage geometry, network down) we return ``None`` so the
    planner uses its flat-route fallback. When ``leg_timings`` (the routing
    engine's per-leg travel time) is supplied, it is forwarded so enrich_route can
    stamp a measured per-segment speed; absent, the call shape is unchanged.
    """
    if not geometry or distance_km <= 0:
        return None
    try:
        if leg_timings:
            enriched = geodata.enrich_route(
                geometry, departure_iso=departure, leg_timings=leg_timings
            )
        else:
            enriched = geodata.enrich_route(geometry, departure_iso=departure)
    except Exception:  # pragma: no cover - geodata is contractually no-raise
        return None
    if not enriched or not enriched.get("segments"):
        return None
    return enriched


def _build_chunks(
    distance_km: float,
    enrichment: Optional[dict[str, Any]],
    temperature_c: float,
) -> tuple[list[tuple[float, float, float, float]], list[Optional[float]]]:
    """Build the ordered list of simulation chunks + their measured speeds.

    Returns ``(chunks, measured_speeds)``. Each chunk is
    ``(km, gradient_pct, temperature_c, wind_mps)``; ``measured_speeds[i]`` is the
    distance-weighted real per-leg speed (km/h) for chunk ``i`` when the routing
    engine supplied per-leg travel time (Tier A), else ``None`` (heuristic).

    * Flat fallback (no enrichment): even ``CHUNK_KM`` slices carrying the
      constant gradient/temperature/wind approximations.
    * Geometry mode: one or more sub-chunks per enriched segment, each
      sub-chunk <= ``CHUNK_KM`` so the SOC profile stays fine-grained while
      inheriting the segment's real gradient/temperature/wind. The enriched
      segment distances are rescaled to the routing engine's total
      ``distance_km`` (great-circle sampling underestimates road distance), so
      energy and SOC are accounted against the true route length.
    """
    chunks: list[tuple[float, float, float, float]] = []
    # Parallel to ``chunks``: the distance-weighted MEASURED speed (km/h) for each
    # chunk when the routing engine supplied real per-leg travel time (Tier A),
    # else ``None`` so the caller falls back to the gradient-speed heuristic.
    measured: list[Optional[float]] = []

    if enrichment is None:
        remaining = distance_km
        while remaining > 1e-6:
            step = min(CHUNK_KM, remaining)
            chunks.append((step, GRADIENT_PCT, temperature_c, WIND_MPS))
            remaining -= step
        return chunks, [None] * len(chunks)

    segs = enrichment["segments"]
    sampled_total = sum(max(0.0, float(s.get("distKm", 0.0))) for s in segs)
    scale = (distance_km / sampled_total) if sampled_total > 0 else 1.0

    # Aggregate consecutive enriched segments into ~CHUNK_KM windows, each carrying
    # the distance-weighted-average gradient/temperature/wind over the window, and
    # predict ONCE per window (back in plan_route). This is the crux of geometry-mode
    # accuracy: the model over-predicts on sub-~5 km legs (a positive small-distance
    # bias), and a real routing polyline is downsampled to ~2-10 km segments, so
    # predicting per raw segment and summing would inflate route energy with polyline
    # DENSITY (finer slicing -> higher kWh) instead of with physics. Averaging is
    # faithful: for the gradient/potential-energy term, avg_grade * window_dist equals
    # the window's net climb, and the temperature/wind drag terms are near-linear over
    # a 25 km span. The window distance still stays <= CHUNK_KM so the SOC profile and
    # charge-stop placement remain fine-grained.
    win_km = 0.0
    win_grad_km = 0.0  # distance-weighted accumulators (value * km)
    win_temp_km = 0.0
    win_wind_km = 0.0
    win_meas_km = 0.0   # measured-speed * km, over the window portion that HAD a speed
    win_meas_dist = 0.0

    def _flush_window() -> None:
        nonlocal win_km, win_grad_km, win_temp_km, win_wind_km, win_meas_km, win_meas_dist
        if win_km > 1e-6:
            chunks.append(
                (win_km, win_grad_km / win_km, win_temp_km / win_km, win_wind_km / win_km)
            )
            # Measured speed only if the routing engine timed this window's segments.
            measured.append(win_meas_km / win_meas_dist if win_meas_dist > 1e-6 else None)
        win_km = win_grad_km = win_temp_km = win_wind_km = win_meas_km = win_meas_dist = 0.0

    for s in segs:
        seg_km = max(0.0, float(s.get("distKm", 0.0))) * scale
        if seg_km <= 1e-6:
            continue
        grad = float(s.get("gradientPct", GRADIENT_PCT))
        temp = float(s.get("temperatureC", temperature_c))
        wind = float(s.get("windMps", WIND_MPS))
        seg_meas = s.get("measuredSpeedKph")  # real per-leg speed, when available
        remaining = seg_km
        while remaining > 1e-6:
            step = min(remaining, CHUNK_KM - win_km)
            win_km += step
            win_grad_km += grad * step
            win_temp_km += temp * step
            win_wind_km += wind * step
            if seg_meas is not None and float(seg_meas) > 0:
                win_meas_km += float(seg_meas) * step
                win_meas_dist += step
            remaining -= step
            if win_km >= CHUNK_KM - 1e-9:
                _flush_window()
    _flush_window()

    if not chunks:  # Degenerate enrichment -> fall back to flat.
        return _build_chunks(distance_km, None, temperature_c)
    return chunks, measured


def _interp_point(
    waypoints: Optional[list[dict[str, Any]]],
    cum_km: float,
    total_km: float,
) -> tuple[Optional[float], Optional[float]]:
    """Estimate a [lat,lng] for a charging stop along the route.

    Linearly interpolates between the first and last supplied waypoint by the
    fraction of total distance covered. This is a coarse placement (it ignores
    the true polyline shape) but gives the map a plausible marker location.
    Returns ``(None, None)`` when no usable waypoints are available.
    """
    if not waypoints or total_km <= 0:
        return (None, None)
    pts = [w for w in waypoints if w.get("lat") is not None and w.get("lng") is not None]
    if not pts:
        return (None, None)
    if len(pts) == 1:
        return (float(pts[0]["lat"]), float(pts[0]["lng"]))
    frac = max(0.0, min(1.0, cum_km / total_km))
    a, b = pts[0], pts[-1]
    lat = float(a["lat"]) + (float(b["lat"]) - float(a["lat"])) * frac
    lng = float(a["lng"]) + (float(b["lng"]) - float(a["lng"])) * frac
    return (round(lat, 5), round(lng, 5))


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance (km) between two ``(lat, lng)`` points."""
    r = 6371.0
    lat1, lng1 = math.radians(a[0]), math.radians(a[1])
    lat2, lng2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _interp_on_geometry(
    geometry: Optional[list[list[float]]],
    cum_km: float,
    total_km: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return the ``[lat,lng]`` point that lies ON the road polyline at the
    fraction ``cum_km / total_km`` of its arc length.

    Unlike :func:`_interp_point` (which interpolates the straight origin->dest
    line), this walks the actual ``geometry`` so a charging stop always sits on
    the drawn route. Returns ``(None, None)`` if geometry is unusable.
    """
    if not geometry or total_km <= 0:
        return (None, None)
    pts = [(float(p[0]), float(p[1])) for p in geometry if len(p) >= 2]
    if len(pts) < 2:
        return (None, None)

    seg = [_haversine_km(pts[i - 1], pts[i]) for i in range(1, len(pts))]
    total = sum(seg)
    if total <= 0:
        return (round(pts[0][0], 5), round(pts[0][1], 5))

    frac = max(0.0, min(1.0, cum_km / total_km))
    target = frac * total
    acc = 0.0
    for i, d in enumerate(seg):
        if acc + d >= target:
            r = 0.0 if d == 0 else (target - acc) / d
            lat = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r
            lng = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r
            return (round(lat, 5), round(lng, 5))
        acc += d
    return (round(pts[-1][0], 5), round(pts[-1][1], 5))


def _snap_km_on_geometry(
    point: tuple[float, float],
    geometry: Optional[list[list[float]]],
) -> Optional[tuple[float, float]]:
    """Snap ``point`` to the nearest spot on the road polyline; return its
    ALONG-polyline arc length and the polyline's total arc length (both km).

    Mirrors :func:`_interp_on_geometry` (same vertex walk, same
    :func:`_haversine_km` between vertices) but in reverse: instead of mapping an
    arc length to a coordinate, it maps a coordinate to its arc length. For each
    polyline segment the waypoint is projected onto the segment in a local
    equirectangular plane (lat/lng scaled by ``cos(lat)`` — faithful for the short
    hops of a downsampled polyline), the projection clamped to the segment, and the
    great-circle distance to that foot point measured. The closest foot point wins;
    its arc length is the summed segment lengths before it plus the projected
    fraction of its own segment. Returns ``None`` when the geometry is unusable so
    the caller falls back to the great-circle leg estimate.

    WHY: payload-drop placement (``_build_stops``/``_payload_t_at``) previously
    located a stop by scaling straight-line origin->...->stop leg lengths, which
    can sit a drop kilometres off where the truck actually passes it on a winding
    road — moving the payload step onto the wrong chunk. Snapping to true road
    distance puts the drop where the route really reaches it.
    """
    if not geometry:
        return None
    pts = [(float(p[0]), float(p[1])) for p in geometry if len(p) >= 2]
    if len(pts) < 2:
        return None
    seg = [_haversine_km(pts[i - 1], pts[i]) for i in range(1, len(pts))]
    total = sum(seg)
    if total <= 0:
        return None

    plat, plng = point
    # Equirectangular scaling so 1 deg lng ≈ 1 deg lat in distance near this lat.
    coslat = math.cos(math.radians(plat))
    best_dist = float("inf")
    best_arc = 0.0
    acc = 0.0
    for i, d in enumerate(seg):
        ax, ay = pts[i][0], pts[i][1] * coslat
        bx, by = pts[i + 1][0], pts[i + 1][1] * coslat
        px, py = plat, plng * coslat
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom <= 0 else ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        foot = (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
        gd = _haversine_km(point, foot)
        if gd < best_dist:
            best_dist = gd
            best_arc = acc + d * t
        acc += d
    return (best_arc, total)


__all__ = ["plan_route"]
