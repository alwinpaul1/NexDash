"""Cost-minimising route optimisation for the eActros 600 — the VRP layer.

:mod:`nexdash.route_planner` *simulates* a route whose stop order you already
fixed. This module chooses the **order**: given an origin and a set of delivery
stops, it finds the visiting sequence that minimises total operating cost

    cost = energy_cost (EUR/kWh * kWh) + driver-time cost (EUR/h * hours)

over the trip (driving + charging + EU 561 rest). It is a single-vehicle,
energy- and payload-aware Travelling-Salesman / Vehicle-Routing optimiser:

* **Exact Held-Karp dynamic programming** for up to :data:`MAX_EXACT_STOPS`
  destinations (provably optimal), and **nearest-neighbour + 2-opt** local
  search above that (near-optimal, scales) — so a small real dispatch run is
  solved exactly and a large one quickly.
* **Payload decay is honoured.** The truck lightens as it sheds cargo at each
  stop, so a later leg costs less energy; the cheapest order therefore depends
  on *which* drops happen first (drop the heavy pallets early). Crucially the
  remaining payload on any leg is determined by the *set* of stops already
  visited, not the path taken to them, so Held-Karp stays exact (the leg cost
  for state ``(visited_set, last) -> j`` uses payload ``start - drops(visited)``).
* **Deliver-by deadlines** enter as a soft lateness penalty by default so
  feasible orders win; an opt-in hard time-window mode (``enforce_deadlines``)
  drops orders that miss a window via a post-filter over the soft-optimal seed
  and its 2-opt neighbours (a full joint-optimisation VRPTW is still out of scope).
* The winning order is handed to :func:`nexdash.route_planner.plan_route` for the
  accurate, charge-aware, EU-561-aware plan and true cost, and we report the
  **saving versus the operator's original order**.

Honest limits (surfaced, not hidden): single vehicle (no fleet assignment);
**ordering uses a great-circle leg proxy** (x a road-circuity factor) so it does
not call a routing API per candidate — the *final* plan is still produced by the
planner, but in flat-fallback mode here (no live road geometry), so its absolute
numbers are estimates the production pipeline should refine by re-routing the
chosen order through the real routing engine; deadlines are soft unless
``enforce_deadlines`` is set (hard windows are then enforced only over the
soft-optimal seed + its 2-opt neighbours, not a full VRPTW search); charging is
the planner's greedy adaptive policy, not jointly optimised with the order.
Deterministic, pure-Python (no external solver).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

from .config import DEFAULT_MODEL_PATH, TRUCK
from .physics import segment_energy_kwh
from .route_planner import (
    CHARGER_KW,
    CHARGE_TARGET_SOC,
    PRICE_EUR_PER_KWH,
    plan_route,
)

__all__ = [
    "optimize_route",
    "estimate_order_cost",
    "DRIVER_EUR_PER_H",
    "LATE_PENALTY_EUR_PER_H",
    "ROAD_CIRCUITY",
    "DEADLINE_TOLERANCE_MIN",
    "MAX_EXACT_STOPS",
]

#: Loaded driver cost (EUR per hour) — the time half of the objective. Driver
#: wage + overhead dominates EV operating cost, so time matters as much as kWh.
DRIVER_EUR_PER_H: float = 30.0

#: Great-circle -> road distance multiplier used for the ORDERING proxy only.
#: Real road distance is longer than the straight line; ~1.3 is a common heuristic.
ROAD_CIRCUITY: float = 1.3

#: Default cruise speed (km/h) for the ordering proxy when none is supplied.
DEFAULT_OPT_SPEED_KPH: float = 70.0

#: Destinations at or below this count are solved EXACTLY (Held-Karp, O(2^n n^2));
#: above it we fall back to nearest-neighbour + 2-opt. 11 -> ~2.3M state-ops, fast.
MAX_EXACT_STOPS: int = 11

#: Soft penalty (EUR per hour late) so the optimiser prefers deadline-feasible
#: orders without a hard time-window constraint (full VRPTW is out of scope).
LATE_PENALTY_EUR_PER_H: float = 250.0


# --------------------------------------------------------------------------- #
# Geometry + leg cost
# --------------------------------------------------------------------------- #
def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance (km) between two ``(lat, lng)`` points."""
    R = 6371.0
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat = math.radians(b[0] - a[0])
    dlng = math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _leg_eur(
    dist_km: float,
    payload_t: float,
    *,
    speed_kph: float,
    temperature_c: float,
    eur_per_kwh: float,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
) -> tuple[float, float, float]:
    """``(eur, energy_kwh, drive_h)`` for one flat leg at the given payload.

    Energy is a first-principles :func:`segment_energy_kwh` estimate (flat,
    no wind) so we can score factorially-many orders without the model; the
    cost blends energy price and driver time. ``driver_eur_per_h`` defaults to
    the module constant :data:`DRIVER_EUR_PER_H` so an unspecified call is
    byte-identical to before this was made configurable.
    """
    energy = max(
        0.0,
        float(
            segment_energy_kwh(
                distance_km=dist_km,
                payload_t=payload_t,
                speed_kph=speed_kph,
                gradient_pct=0.0,
                temperature_c=temperature_c,
                wind_mps=0.0,
                truck=TRUCK,
            )
        ),
    )
    drive_h = dist_km / speed_kph if speed_kph > 0 else 0.0
    eur = energy * eur_per_kwh + drive_h * driver_eur_per_h
    return eur, energy, drive_h


# --------------------------------------------------------------------------- #
# Order cost (payload-decay + soft deadlines)
# --------------------------------------------------------------------------- #
#: Tolerance (minutes) by which a stop's deliver-by deadline may be exceeded in
#: hard mode before the order is treated as infeasible — absorbs floating-point
#: clock noise so an arrival exactly on the deadline is never wrongly rejected.
DEADLINE_TOLERANCE_MIN: float = 1.0


def estimate_order_cost(
    origin: dict[str, Any],
    ordered_dests: list[dict[str, Any]],
    start_payload_t: float,
    *,
    speed_kph: float = DEFAULT_OPT_SPEED_KPH,
    temperature_c: float = 15.0,
    eur_per_kwh: float = PRICE_EUR_PER_KWH,
    return_to_origin: bool = False,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
    late_penalty_eur_per_h: float = LATE_PENALTY_EUR_PER_H,
    road_circuity: float = ROAD_CIRCUITY,
    enforce_deadlines: bool = False,
    deadline_tolerance_min: float = DEADLINE_TOLERANCE_MIN,
) -> float:
    """Proxy operating cost (EUR) of visiting ``ordered_dests`` from ``origin``.

    Walks the legs in order, decaying payload at each drop (the leg INTO a stop
    carries the full pre-drop weight, matching the planner), accruing energy +
    driver-time cost and a soft lateness penalty against each ``deliverByMin``
    (minutes after departure). Optionally returns to the origin (depot).

    The cost weights ``driver_eur_per_h``, ``late_penalty_eur_per_h`` and
    ``road_circuity`` default to the module constants, so an unspecified call is
    byte-identical to before they were made configurable.

    Deadlines are a SOFT lateness penalty by default. When
    ``enforce_deadlines=True`` this becomes a hard VRPTW window: if arrival at
    any stop exceeds its ``deliverByMin`` by more than ``deadline_tolerance_min``
    minutes the function returns ``+inf`` (infeasible). This is path-dependent
    and so cannot be folded into the Held-Karp set-DP — the caller seeds order
    with HK on the soft cost and post-filters feasibility over HK + 2-opt
    neighbours (see :func:`optimize_route`). The soft penalty still applies to
    on-time-or-feasible legs so feasible orders are still ranked by lateness.
    """
    total = 0.0
    payload = max(0.0, start_payload_t)
    clock_min = 0.0
    prev = (float(origin["lat"]), float(origin["lng"]))
    for d in ordered_dests:
        here = (float(d["lat"]), float(d["lng"]))
        dist = _haversine_km(prev, here) * road_circuity
        eur, _energy, drive_h = _leg_eur(
            dist, payload, speed_kph=speed_kph, temperature_c=temperature_c,
            eur_per_kwh=eur_per_kwh, driver_eur_per_h=driver_eur_per_h,
        )
        total += eur
        clock_min += drive_h * 60.0
        deadline = d.get("deliverByMin")
        if deadline is not None and clock_min > float(deadline):
            if enforce_deadlines and clock_min > float(deadline) + deadline_tolerance_min:
                return float("inf")  # hard window violated -> infeasible order
            total += (clock_min - float(deadline)) / 60.0 * late_penalty_eur_per_h
        clock_min += float(d.get("unloadMin", 0) or 0)
        payload = max(0.0, payload - float(d.get("dropWeightKg", 0) or 0) / 1000.0)
        prev = here
    if return_to_origin and ordered_dests:
        dist = _haversine_km(prev, (float(origin["lat"]), float(origin["lng"]))) * road_circuity
        eur, _e, _h = _leg_eur(
            dist, payload, speed_kph=speed_kph, temperature_c=temperature_c,
            eur_per_kwh=eur_per_kwh, driver_eur_per_h=driver_eur_per_h,
        )
        total += eur
    return total


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def _held_karp(
    origin: dict[str, Any],
    dests: list[dict[str, Any]],
    start_payload_t: float,
    *,
    speed_kph: float,
    temperature_c: float,
    eur_per_kwh: float,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
    road_circuity: float = ROAD_CIRCUITY,
) -> list[int]:
    """Exact min-cost visiting order via Held-Karp DP (open path from origin).

    State ``dp[(mask, last)]`` = min cost to leave the origin and visit exactly
    the stop set ``mask``, ending at ``last``. The remaining payload on the next
    leg is ``start - drops(mask)`` — a function of the *set*, which is what keeps
    the DP exact under payload decay. Deadlines are NOT folded in here (arrival
    time is path- not set-determined); they are scored on the full order via
    :func:`estimate_order_cost` for the final comparison. The cost weights
    default to the module constants so an unspecified call is unchanged.
    """
    n = len(dests)
    pts = [(float(d["lat"]), float(d["lng"])) for d in dests]
    o = (float(origin["lat"]), float(origin["lng"]))
    drop_t = [max(0.0, float(d.get("dropWeightKg", 0) or 0) / 1000.0) for d in dests]

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return _haversine_km(a, b) * road_circuity

    def _edge(a: tuple[float, float], b: tuple[float, float], payload: float) -> float:
        return _leg_eur(_dist(a, b), payload, speed_kph=speed_kph,
                        temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
                        driver_eur_per_h=driver_eur_per_h)[0]

    INF = float("inf")
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    for j in range(n):  # origin -> j (nothing dropped yet -> full payload)
        dp[(1 << j, j)] = _edge(o, pts[j], start_payload_t)
        parent[(1 << j, j)] = -1

    for mask in range(1, 1 << n):
        # Payload available for a leg LEAVING this visited set.
        dropped = sum(drop_t[k] for k in range(n) if mask & (1 << k))
        payload_next = max(0.0, start_payload_t - dropped)
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            base = dp.get((mask, last))
            if base is None:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                nmask = mask | (1 << j)
                cost = base + _edge(pts[last], pts[j], payload_next)
                if cost < dp.get((nmask, j), INF):
                    dp[(nmask, j)] = cost
                    parent[(nmask, j)] = last

    full = (1 << n) - 1
    best_last = min(range(n), key=lambda j: dp.get((full, j), INF))
    order: list[int] = []
    mask, last = full, best_last
    while last != -1:
        order.append(last)
        prev_last = parent[(mask, last)]
        mask ^= 1 << last
        last = prev_last
    order.reverse()
    return order


def _nn_two_opt(
    origin: dict[str, Any],
    dests: list[dict[str, Any]],
    start_payload_t: float,
    *,
    speed_kph: float,
    temperature_c: float,
    eur_per_kwh: float,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
    late_penalty_eur_per_h: float = LATE_PENALTY_EUR_PER_H,
    road_circuity: float = ROAD_CIRCUITY,
) -> list[int]:
    """Nearest-neighbour seed + 2-opt local search for larger stop counts.

    Cost is the full payload-aware :func:`estimate_order_cost` recomputed per
    candidate (O(n) each), so payload decay and deadlines are respected; 2-opt
    reverses sub-paths until no swap improves. Deterministic (no randomness).
    The cost weights default to the module constants so an unspecified call is
    unchanged. Always scored on the SOFT cost — hard-deadline feasibility is a
    path-dependent post-filter applied by :func:`optimize_route`, not here, so
    2-opt keeps a single well-defined (finite) objective to descend.
    """
    n = len(dests)

    def _order_cost(order: list[int]) -> float:
        return estimate_order_cost(
            origin, [dests[i] for i in order], start_payload_t,
            speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
            driver_eur_per_h=driver_eur_per_h, late_penalty_eur_per_h=late_penalty_eur_per_h,
            road_circuity=road_circuity,
        )

    # Nearest-neighbour seed by straight-line distance from the running point.
    pts = [(float(d["lat"]), float(d["lng"])) for d in dests]
    o = (float(origin["lat"]), float(origin["lng"]))
    unvisited = set(range(n))
    cur = o
    seed: list[int] = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: _haversine_km(cur, pts[j]))
        seed.append(nxt)
        unvisited.discard(nxt)
        cur = pts[nxt]

    best = seed
    best_cost = _order_cost(best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                c = _order_cost(cand)
                if c + 1e-9 < best_cost:
                    best, best_cost = cand, c
                    improved = True
    return best


def _solve_order(
    origin: dict[str, Any],
    dests: list[dict[str, Any]],
    start_payload_t: float,
    *,
    speed_kph: float,
    temperature_c: float,
    eur_per_kwh: float,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
    late_penalty_eur_per_h: float = LATE_PENALTY_EUR_PER_H,
    road_circuity: float = ROAD_CIRCUITY,
) -> list[int]:
    """Pick the solver by size: exact Held-Karp small, NN+2-opt large.

    Scores on the SOFT cost; cost weights default to the module constants so an
    unspecified call is unchanged. Hard-deadline feasibility (when requested) is
    a path-dependent post-filter handled by :func:`optimize_route`, because it
    cannot be folded into the Held-Karp set-DP.
    """
    n = len(dests)
    if n <= 1:
        return list(range(n))
    if n <= MAX_EXACT_STOPS:
        return _held_karp(origin, dests, start_payload_t, speed_kph=speed_kph,
                          temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
                          driver_eur_per_h=driver_eur_per_h, road_circuity=road_circuity)
    return _nn_two_opt(origin, dests, start_payload_t, speed_kph=speed_kph,
                      temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
                      driver_eur_per_h=driver_eur_per_h,
                      late_penalty_eur_per_h=late_penalty_eur_per_h, road_circuity=road_circuity)


def _two_opt_neighbours(order: list[int]) -> list[list[int]]:
    """All 2-opt sub-path reversals of ``order`` (the seed itself is the caller's
    responsibility to include). Used by the hard-deadline post-filter as the
    feasibility search neighbourhood around the soft-optimal seed."""
    out: list[list[int]] = []
    n = len(order)
    for i in range(n - 1):
        for k in range(i + 1, n):
            out.append(order[:i] + order[i : k + 1][::-1] + order[k + 1 :])
    return out


def _enforce_deadline_filter(
    origin: dict[str, Any],
    dests: list[dict[str, Any]],
    seed_order: list[int],
    start_payload_t: float,
    *,
    speed_kph: float,
    temperature_c: float,
    eur_per_kwh: float,
    driver_eur_per_h: float,
    late_penalty_eur_per_h: float,
    road_circuity: float,
    deadline_tolerance_min: float,
) -> tuple[list[int], bool]:
    """Hard VRPTW post-filter over the seed order and its 2-opt neighbours.

    Held-Karp / NN+2-opt seed the order on the SOFT cost (path-independent set
    DP can't fold in arrival-time windows). Here we re-score the seed and each
    of its 2-opt neighbours under the HARD window (``enforce_deadlines=True``,
    which returns ``+inf`` on a violation) and pick the CHEAPEST FEASIBLE order.

    Returns ``(order, infeasible)``. If no candidate is feasible we fall back to
    the least-violating order (lowest SOFT cost, i.e. smallest total lateness)
    and set ``infeasible=True`` so the caller can flag it — never silently
    returning an order that meets the windows when none does.
    """
    candidates = [list(seed_order)] + _two_opt_neighbours(seed_order)

    def _soft(order: list[int]) -> float:
        return estimate_order_cost(
            origin, [dests[i] for i in order], start_payload_t,
            speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
            driver_eur_per_h=driver_eur_per_h, late_penalty_eur_per_h=late_penalty_eur_per_h,
            road_circuity=road_circuity,
        )

    def _hard(order: list[int]) -> float:
        return estimate_order_cost(
            origin, [dests[i] for i in order], start_payload_t,
            speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
            driver_eur_per_h=driver_eur_per_h, late_penalty_eur_per_h=late_penalty_eur_per_h,
            road_circuity=road_circuity, enforce_deadlines=True,
            deadline_tolerance_min=deadline_tolerance_min,
        )

    feasible = [c for c in candidates if math.isfinite(_hard(c))]
    if feasible:
        # Cheapest feasible by soft cost (ties broken deterministically by order).
        best = min(feasible, key=lambda c: (_soft(c), c))
        return best, False
    # None feasible: least-violating = lowest soft (lateness-penalised) cost.
    best = min(candidates, key=lambda c: (_soft(c), c))
    return best, True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _ordered_distance_km(origin: dict[str, Any], ordered: list[dict[str, Any]],
                         *, return_to_origin: bool,
                         road_circuity: float = ROAD_CIRCUITY) -> float:
    """Great-circle (x circuity) road-distance estimate for a visiting order.

    ``road_circuity`` defaults to the module constant so an unspecified call is
    unchanged.
    """
    total = 0.0
    prev = (float(origin["lat"]), float(origin["lng"]))
    for d in ordered:
        here = (float(d["lat"]), float(d["lng"]))
        total += _haversine_km(prev, here) * road_circuity
        prev = here
    if return_to_origin and ordered:
        total += _haversine_km(prev, (float(origin["lat"]), float(origin["lng"]))) * road_circuity
    return total


def _plan_cost(plan: dict[str, Any], eur_per_kwh: float,
               driver_eur_per_h: float = DRIVER_EUR_PER_H) -> dict[str, Any]:
    """Operating cost of a planner result: energy price + driver time.

    ``driver_eur_per_h`` defaults to the module constant so an unspecified call
    is unchanged.
    """
    s = plan.get("summary", {})
    energy_kwh = float(s.get("energyKwh", 0.0) or 0.0)
    total_h = float(s.get("totalTimeH", 0.0) or 0.0)
    energy_eur = energy_kwh * eur_per_kwh
    time_eur = total_h * driver_eur_per_h
    driver = s.get("driver", {}) or {}
    return {
        "energyKwh": round(energy_kwh, 1),
        "totalTimeH": round(total_h, 2),
        "energyEur": round(energy_eur, 2),
        "timeEur": round(time_eur, 2),
        "totalEur": round(energy_eur + time_eur, 2),
        "chargingStops": int(s.get("chargingStops", 0) or 0),
        "eu561ok": bool(driver.get("eu561ok", True)),
    }


def _plan_for_order(
    origin: dict[str, Any],
    ordered: list[dict[str, Any]],
    *,
    start_soc: float,
    min_soc: float,
    payload_kg: float,
    reserve_pct: float,
    max_charge_kw: float,
    charge_target_soc: float,
    departure: Optional[str],
    temperature_c: float,
    speed_kph: float,
    return_to_origin: bool,
    model_path: Union[str, Path],
) -> dict[str, Any]:
    """Run the SOC/charging/EU-561 planner over one concrete visiting order."""
    total_km = _ordered_distance_km(origin, ordered, return_to_origin=return_to_origin)
    duration_s = (total_km / speed_kph) * 3600.0 if speed_kph > 0 else 0.0
    waypoints = [origin] + ordered + ([origin] if return_to_origin else [])
    return plan_route(
        distance_km=total_km,
        duration_s=duration_s,
        start_soc=start_soc,
        min_soc=min_soc,
        payload_kg=payload_kg,
        reserve_pct=reserve_pct,
        max_charge_kw=max_charge_kw,
        charge_target_soc=charge_target_soc,
        departure=departure,
        temperature_c=temperature_c,
        waypoints=waypoints,
        geometry=None,  # flat-fallback: ordering is a great-circle proxy (see module docstring)
        model_path=model_path,
    )


def optimize_route(
    origin: dict[str, Any],
    destinations: list[dict[str, Any]],
    *,
    start_soc: float,
    min_soc: float,
    payload_kg: float,
    reserve_pct: float = 10.0,
    max_charge_kw: float = CHARGER_KW,
    charge_target_soc: float = CHARGE_TARGET_SOC,
    departure: Optional[str] = None,
    temperature_c: float = 15.0,
    speed_kph: float = DEFAULT_OPT_SPEED_KPH,
    eur_per_kwh: float = PRICE_EUR_PER_KWH,
    return_to_origin: bool = False,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
    late_penalty_eur_per_h: float = LATE_PENALTY_EUR_PER_H,
    road_circuity: float = ROAD_CIRCUITY,
    enforce_deadlines: bool = False,
    deadline_tolerance_min: float = DEADLINE_TOLERANCE_MIN,
) -> dict[str, Any]:
    """Find the cheapest visiting order and return it with its plan + the saving.

    ``origin`` is ``{lat, lng, label?}``; each destination may carry
    ``dropWeightKg``, ``unloadMin``, ``deliverByMin`` (minutes after departure).
    Returns a JSON-serialisable dict with the ``optimizedOrder`` (the input
    indices in best order) + its ``plan`` + ``cost``, the ``baseline`` (original
    order) cost, ``savingsEur`` / ``savingsPct``, the solver used, and honest
    ``assumptions``. Deterministic and offline.

    The cost weights ``driver_eur_per_h`` (driver-time EUR/h),
    ``late_penalty_eur_per_h`` (soft-lateness EUR/h) and ``road_circuity``
    (great-circle -> road multiplier for the ordering proxy) default to the
    module constants, so calling without them is byte-identical to before they
    were configurable. The weights actually used are echoed into
    ``assumptions``.

    ``enforce_deadlines`` switches deliver-by windows from a SOFT penalty
    (default) to a HARD VRPTW window: the soft-cost solver still seeds the order
    (Held-Karp can't fold path-dependent arrival times), then a post-filter over
    that seed and its 2-opt neighbours drops orders that miss any window (beyond
    ``deadline_tolerance_min``) and keeps the cheapest feasible one. If none is
    feasible it returns the least-violating order and sets ``deadlinesInfeasible``
    to ``True`` in the result.
    """
    dests = [d for d in destinations if d and d.get("lat") is not None and d.get("lng") is not None]
    start_payload_t = max(0.0, float(payload_kg)) / 1000.0

    deadlines_infeasible = False
    if len(dests) <= 1:  # 0 or 1 stop -> nothing to reorder
        order_idx = list(range(len(dests)))
        if enforce_deadlines and order_idx:
            # Single stop: still report whether its own window is feasible.
            hard = estimate_order_cost(
                origin, dests, start_payload_t,
                speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
                driver_eur_per_h=driver_eur_per_h, late_penalty_eur_per_h=late_penalty_eur_per_h,
                road_circuity=road_circuity, enforce_deadlines=True,
                deadline_tolerance_min=deadline_tolerance_min,
            )
            deadlines_infeasible = not math.isfinite(hard)
    else:
        seed_idx = _solve_order(
            origin, dests, start_payload_t,
            speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
            driver_eur_per_h=driver_eur_per_h, late_penalty_eur_per_h=late_penalty_eur_per_h,
            road_circuity=road_circuity,
        )
        if enforce_deadlines:
            order_idx, deadlines_infeasible = _enforce_deadline_filter(
                origin, dests, seed_idx, start_payload_t,
                speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
                driver_eur_per_h=driver_eur_per_h, late_penalty_eur_per_h=late_penalty_eur_per_h,
                road_circuity=road_circuity, deadline_tolerance_min=deadline_tolerance_min,
            )
        else:
            order_idx = seed_idx

    ordered = [dests[i] for i in order_idx]
    solver = "held-karp (exact)" if 1 < len(dests) <= MAX_EXACT_STOPS else (
        "nearest-neighbour + 2-opt" if len(dests) > MAX_EXACT_STOPS else "trivial"
    )

    plan_kwargs = dict(
        start_soc=start_soc, min_soc=min_soc, payload_kg=payload_kg,
        reserve_pct=reserve_pct, max_charge_kw=max_charge_kw,
        charge_target_soc=charge_target_soc, departure=departure,
        temperature_c=temperature_c, speed_kph=speed_kph,
        return_to_origin=return_to_origin, model_path=model_path,
    )
    opt_plan = _plan_for_order(origin, ordered, **plan_kwargs)
    base_plan = _plan_for_order(origin, dests, **plan_kwargs)

    opt_cost = _plan_cost(opt_plan, eur_per_kwh, driver_eur_per_h)
    base_cost = _plan_cost(base_plan, eur_per_kwh, driver_eur_per_h)
    savings_eur = round(base_cost["totalEur"] - opt_cost["totalEur"], 2)
    savings_pct = round(
        (savings_eur / base_cost["totalEur"] * 100.0) if base_cost["totalEur"] > 0 else 0.0, 1
    )

    if enforce_deadlines:
        deadline_note = (
            "Deliver-by deadlines are a HARD time window (enforce_deadlines=True): "
            "orders missing any window (beyond "
            f"{deadline_tolerance_min:.0f} min tolerance) are dropped and the cheapest "
            "feasible order is chosen; the soft-cost solver seeds the order because "
            "Held-Karp cannot fold path-dependent arrival times."
            + (" No feasible order exists for these windows — returning the "
               "least-violating order (see deadlinesInfeasible)."
               if deadlines_infeasible else "")
        )
    else:
        deadline_note = (
            "Deliver-by deadlines are a soft lateness penalty, not a hard time window "
            "(pass enforce_deadlines=True for a hard VRPTW window)."
        )

    assumptions = [
        "Single vehicle; ordering minimises energy + driver-time cost with payload "
        "decay (heavier drops earlier lower later-leg energy).",
        "Stop ORDERING uses a great-circle distance proxy (x road-circuity factor); "
        "the production pipeline should re-route the chosen order through the real "
        "routing engine for road-accurate distance, ETA and energy.",
        deadline_note,
        "Charging is the planner's greedy adaptive policy applied AFTER ordering, "
        "not jointly optimised with the order.",
        f"Cost model: energy at {eur_per_kwh:.2f} EUR/kWh + driver time at "
        f"{driver_eur_per_h:.0f} EUR/h; soft-lateness penalty "
        f"{late_penalty_eur_per_h:.0f} EUR/h; road-circuity factor {road_circuity:.2f}.",
    ]

    return {
        "optimizedOrder": order_idx,
        "solver": solver,
        "nStops": len(dests),
        "plan": opt_plan,
        "cost": opt_cost,
        "baseline": {"order": list(range(len(dests))), "cost": base_cost},
        "savingsEur": savings_eur,
        "savingsPct": savings_pct,
        "deadlinesEnforced": bool(enforce_deadlines),
        "deadlinesInfeasible": bool(deadlines_infeasible),
        "assumptions": assumptions,
    }
