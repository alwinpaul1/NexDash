"""Model-driven SOC-drain simulation and trip planning for the NexOS planner.

This module is the backend brain behind the frontend route planner. Given a
total route distance + duration (the frontend supplies these from the TomTom
truck-routing API), a start/min state of charge, payload and ambient
temperature, it walks the route in fixed-distance chunks and uses the trained
:func:`nexdash.model.predict_energy` model to estimate per-chunk energy draw.
SOC is accumulated against the eActros 600's ~600 kWh battery; whenever the
projected SOC would fall below the operator's ``min_soc`` floor, a charging
stop is inserted (recharge to ~95% on a ~350 kW MCS charger). EU 561 driving
rules are layered on top (max 4.5 h driving before a 45 min break; 9 h daily
driving limit). The result mirrors the ``PlanResult`` shape the frontend
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
* **Constant average speed.** A single average speed is derived from
  ``distance_km / duration_h`` and applied to every chunk. Stop-and-go,
  motorway vs. urban mix, and traffic variation within the route are not
  modelled at the chunk level.
* **Constant payload.** ``payload_t`` is held constant for the whole trip even
  though multi-drop routes shed cargo at each stop. This is conservative
  (over-estimates energy on later legs).
* **Charging model.** Inserted stops are DC fast-charge to a configurable target
  (default ~80%), at a rated ~400 kW CCS (the eActros 600's current real capability;
  MCS 1 MW is not yet shipping). Charge *time* follows a power-vs-SOC **taper**
  (full power to ~80% SOC, derating into the tail; see ``_charge_minutes``), so the
  slow ``80->100%`` region is no longer under-counted. Energy is priced at a flat
  0.45 EUR/kWh; tariffs vary, so charge cost is indicative, not contractual.
* **Driver hours (EU 561).** Only the 4.5 h continuous-driving / 45 min break
  rule is simulated. The ``driver.dailyH`` and ``driver.weeklyH`` fields are the
  single trip's total driving time (no calendar day/week split and no 11 h daily
  rest are modelled), so for a trip longer than one shift they read the same
  value; ``eu561ok`` conservatively flags any total over the 9 h daily cap.
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
from .config import DEFAULT_MODEL_PATH, TRUCK
from .model import predict_energy

# --------------------------------------------------------------------------- #
# Planner constants (approximations documented in the module docstring)
# --------------------------------------------------------------------------- #

#: Distance of each simulation chunk (km). Smaller -> finer SOC profile.
CHUNK_KM: float = 25.0

#: Recharge target SOC (%) when a charging stop is inserted. Defaults to 80%:
#: on-route DC fast-charge stops top to ~80% because the power-vs-SOC taper makes
#: the 80->100% tail slow and driver-time-expensive (see ``_charge_minutes``).
CHARGE_TARGET_SOC: float = 80.0

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

#: Assumed steady headwind component (m/s).
WIND_MPS: float = 3.0

#: Assumed net road gradient (percent). Flat-route approximation.
GRADIENT_PCT: float = 0.0

# EU Regulation 561/2006 driving-time limits (the subset we model).
EU561_MAX_DRIVE_BEFORE_BREAK_MIN: float = 4.5 * 60.0  # 4h30 continuous driving
EU561_BREAK_MIN: float = 45.0                          # mandatory break length
EU561_DAILY_MAX_DRIVE_H: float = 9.0                   # standard daily driving
EU561_WEEKLY_MAX_DRIVE_H: float = 56.0                 # max weekly driving


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
    waypoints: Optional[list[dict[str, Any]]], total_km: float
) -> list[dict[str, Any]]:
    """Resolve destination waypoints into per-stop legs along the route.

    Returns one entry per *destination* (the origin, index 0, is excluded), each
    with its cumulative distance from the origin (great-circle leg lengths scaled
    so the final stop lands exactly at ``total_km``) plus any per-stop delivery
    data the caller attached: ``dropWeightKg`` (cargo shed at the stop),
    ``unloadMin`` (dwell), and ``deliverBy`` (ISO deadline). Returns ``[]`` when
    there are fewer than two usable waypoints, so the planner falls back to a
    single continuous leg with constant payload (unchanged legacy behaviour).
    """
    if not waypoints or total_km <= 0:
        return []
    pts = [w for w in waypoints if w.get("lat") is not None and w.get("lng") is not None]
    if len(pts) < 2:
        return []
    coords = [(float(w["lat"]), float(w["lng"])) for w in pts]
    legs = [_haversine_km(coords[i - 1], coords[i]) for i in range(1, len(coords))]
    gc_total = sum(legs)
    if gc_total <= 0:
        return []
    scale = total_km / gc_total

    stops: list[dict[str, Any]] = []
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
    departure: Optional[str] = None,
    temperature_c: float = 15.0,
    waypoints: Optional[list[dict[str, Any]]] = None,
    geometry: Optional[list[list[float]]] = None,
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

    distance_km = max(0.0, float(distance_km))
    duration_h = max(0.0, float(duration_s)) / 3600.0
    # Average speed across the whole route (flat approximation). Guard /0.
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

    # Per-destination legs (for per-stop SOC/ETA, payload decay, deliver-by).
    starting_payload_t = payload_t
    stops_meta = _build_stops(waypoints, distance_km)
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
    enrichment = _enrich(geometry, departure, distance_km)
    chunks = _build_chunks(distance_km, enrichment, temperature_c)

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

    # Predict energy for every chunk up front (model-driven; the wind magnitude is
    # used directly as the headwind component, see module docstring). Doing this
    # before the walk lets the charging check look AHEAD at the energy still owed to
    # the destination, so we only charge when genuinely required.
    chunk_energies = [
        max(
            0.0,
            float(
                predict_energy(
                    {
                        "distance_km": chunk_km,
                        "payload_t": chunk_payloads[i],
                        "speed_kph": avg_speed_kph,
                        "gradient_pct": chunk_grad,
                        "temperature_c": chunk_temp,
                        "wind_mps": chunk_wind,
                    },
                    model_path=model_path,
                )
            ),
        )
        for i, (chunk_km, chunk_grad, chunk_temp, chunk_wind) in enumerate(chunks)
    ]

    for i, (chunk_km, chunk_grad, chunk_temp, chunk_wind) in enumerate(chunks):
        chunk_energy = chunk_energies[i]
        chunk_drive_min = (chunk_km / avg_speed_kph) * 60.0
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
        charge_floor = min_soc + max(0.0, reserve_pct)
        remaining_energy_kwh = sum(chunk_energies[i:])
        soc_at_end_without_charge = soc - (remaining_energy_kwh / battery_kwh) * 100.0
        if projected_soc < charge_floor and soc_at_end_without_charge < min_soc:
            # Close the running drive segment, then charge before continuing.
            if seg_open:
                _close_drive_segment()
            arrive_soc = soc
            depart_soc = charge_target_soc
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
                    "lat": lat,
                    "lng": lng,
                    "arriveSoc": round(arrive_soc, 1),
                    "departSoc": round(depart_soc, 1),
                    "kWh": round(kwh_added, 1),
                    "costEur": round(cost_eur, 2),
                    "durationMin": round(charge_min),
                }
            )
            # A charge also counts as a rest, satisfying the 561 break clock.
            total_charge_min += charge_min
            clock = ch_end
            soc = depart_soc
            drive_since_break_min = 0.0
            # Reopen a fresh drive segment after the charge.
            seg_open = True
            seg_soc_start = soc
            seg_start_clock = clock
            projected_soc = soc - chunk_soc_drop

        # --- EU 561 break check: 4.5h continuous driving cap. ---
        if drive_since_break_min + chunk_drive_min > EU561_MAX_DRIVE_BEFORE_BREAK_MIN:
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

    # --- Summary aggregation. ---
    arrival_dt = clock
    driving_h = total_drive_min / 60.0
    charging_min_total = total_charge_min
    total_min = total_drive_min + total_break_min + total_charge_min + total_unload_min
    total_h = total_min / 60.0

    kwh_per_100 = (total_energy_kwh / distance_km * 100.0) if distance_km > 0 else 0.0
    charging_cost = sum(s["costEur"] for s in charging_stops)

    eu561ok = (
        driving_h <= EU561_DAILY_MAX_DRIVE_H
        and driving_h <= EU561_WEEKLY_MAX_DRIVE_H
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
            "Payload held constant for the whole trip — conservative (over-estimates later legs)."
        )
    if not enrichment:
        assumptions.append("Flat-route fallback: gradient assumed 0 (no per-segment terrain).")
    assumptions.append(
        f"Charge time uses a power-vs-SOC taper to a {round(charge_target_soc)}% target "
        f"at ~{round(max_charge_kw or CHARGER_KW)} kW CCS; the 80->100% tail is slower."
    )
    assumptions.append("Single average speed applied to every segment.")
    assumptions.append(
        "EU 561: only the 4.5 h / 45 min break is modelled — single-shift, no 11 h daily "
        "rest or day/week split. drivingH/dailyH/weeklyH are the trip total."
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
        "minSoc": round(min_soc_seen, 1),
        "energyKwh": round(total_energy_kwh, 1),
        "kwhPer100": round(kwh_per_100, 1),
        "chargingCostEur": round(charging_cost, 2),
        "chargingStops": len(charging_stops),
        "unloadTimeMin": round(total_unload_min),
        "elevationGainM": round(elevation_gain_m, 1),
        "driver": {
            "drivingH": round(driving_h, 2),
            "breaks": n_breaks,
            "totalH": round(total_h, 2),
            # Single-shift model: dailyH/weeklyH are the trip total (no day/week
            # split or 11 h daily rest is simulated — see `assumptions`).
            "dailyH": round(driving_h, 2),
            "dailyMaxH": EU561_DAILY_MAX_DRIVE_H,
            "weeklyH": round(driving_h, 2),
            "weeklyMaxH": EU561_WEEKLY_MAX_DRIVE_H,
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
) -> Optional[dict[str, Any]]:
    """Enrich ``geometry`` into per-segment conditions, or ``None`` if absent.

    Fails soft: if :func:`nexdash.geodata.enrich_route` yields no usable
    segments (empty/garbage geometry, network down) we return ``None`` so the
    planner uses its flat-route fallback.
    """
    if not geometry or distance_km <= 0:
        return None
    try:
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
) -> list[tuple[float, float, float, float]]:
    """Build the ordered list of simulation chunks.

    Each chunk is ``(km, gradient_pct, temperature_c, wind_mps)``.

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

    if enrichment is None:
        remaining = distance_km
        while remaining > 1e-6:
            step = min(CHUNK_KM, remaining)
            chunks.append((step, GRADIENT_PCT, temperature_c, WIND_MPS))
            remaining -= step
        return chunks

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

    def _flush_window() -> None:
        nonlocal win_km, win_grad_km, win_temp_km, win_wind_km
        if win_km > 1e-6:
            chunks.append(
                (win_km, win_grad_km / win_km, win_temp_km / win_km, win_wind_km / win_km)
            )
        win_km = win_grad_km = win_temp_km = win_wind_km = 0.0

    for s in segs:
        seg_km = max(0.0, float(s.get("distKm", 0.0))) * scale
        if seg_km <= 1e-6:
            continue
        grad = float(s.get("gradientPct", GRADIENT_PCT))
        temp = float(s.get("temperatureC", temperature_c))
        wind = float(s.get("windMps", WIND_MPS))
        remaining = seg_km
        while remaining > 1e-6:
            step = min(remaining, CHUNK_KM - win_km)
            win_km += step
            win_grad_km += grad * step
            win_temp_km += temp * step
            win_wind_km += wind * step
            remaining -= step
            if win_km >= CHUNK_KM - 1e-9:
                _flush_window()
    _flush_window()

    if not chunks:  # Degenerate enrichment -> fall back to flat.
        return _build_chunks(distance_km, None, temperature_c)
    return chunks


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


__all__ = ["plan_route"]
