"""Cost-minimising route optimisation for the eActros 600 — the VRP layer.

:mod:`nexdash.route_planner` *simulates* a route whose stop order you already
fixed. This module chooses the **order**: given an origin and a set of delivery
stops, it finds the visiting sequence that minimises total operating cost

    cost = energy_cost (EUR/kWh * kWh) + driver-time cost (EUR/h * hours)

over the trip (driving + charging + EU 561 rest). It is a single-vehicle,
energy- and payload-aware Travelling-Salesman / Vehicle-Routing optimiser:

* **Exact Held-Karp dynamic programming** for up to :data:`MAX_EXACT_STOPS`
  destinations (provably optimal), and **nearest-neighbour + 2-opt** local
  search above that (near-optimal, scales).
* **Payload decay is honoured.** The truck lightens as it sheds cargo at each
  stop, so a later leg costs less energy; the cheapest order therefore depends
  on *which* drops happen first. The remaining payload on any leg is determined
  by the *set* of stops already visited, not the path, so Held-Karp stays exact.
* **Deliver-by deadlines** enter as a soft lateness penalty by default; an opt-in
  hard time-window mode (``enforce_deadlines``) drops orders that miss a window.

**All-factors leg costing (:class:`_LegCtx`).** The leg cost is no longer a flat
great-circle proxy. When the caller supplies them, the optimiser uses:

* a **real road distance/time matrix** (from the routing engine) instead of a
  great-circle x circuity estimate — trucks run on roads, not straight lines;
* **per-leg gradient** derived from **endpoint elevations** (Open-Meteo, fetched
  server-side once per node — O(N), not O(N^2)) plus a representative **wind**;
* leg energy from the **trained ML model** (``use_model``) rather than the
  physics ground truth, so the order is scored with the same model the rest of
  the system serves.

It degrades gracefully: with none of these supplied, the cost reproduces the
original great-circle + physics proxy exactly, so existing callers/tests are
unchanged. The winning order is handed to :func:`nexdash.route_planner.plan_route`
for the charge-/EU-561-aware plan and the saving vs the operator's order, and an
optional **reach-a-charger-from-the-final-stop** check (``final_charger_distance_km``)
flags an order that would strand the truck at the drop-off.

Honest limits (surfaced in ``assumptions``, not hidden): single vehicle (no fleet
assignment); charging is the planner's greedy adaptive policy applied AFTER
ordering, **not jointly co-optimised** with the order (full E-VRPTW is NP-hard);
per-leg terrain is a representative endpoint gradient, not the full polyline; the
final absolute plan numbers come from re-routing the chosen order through the real
engine. Deterministic, pure-Python (no external solver); the ML model and the
elevation/matrix inputs are the only non-physics ingredients.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

from .config import DEFAULT_MODEL_PATH, TRUCK
from .model import predict_energy
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

#: Great-circle -> road distance multiplier used for the ORDERING proxy ONLY when
#: no real road matrix is supplied. Real road distance is longer than the straight
#: line; ~1.3 is a common heuristic.
ROAD_CIRCUITY: float = 1.3

#: Default cruise speed (km/h) for the ordering proxy when none is supplied.
DEFAULT_OPT_SPEED_KPH: float = 70.0

#: Destinations at or below this count are solved EXACTLY (Held-Karp, O(2^n n^2));
#: above it we fall back to nearest-neighbour + 2-opt. 11 -> ~2.3M state-ops, fast.
MAX_EXACT_STOPS: int = 11

#: Soft penalty (EUR per hour late) so the optimiser prefers deadline-feasible
#: orders without a hard time-window constraint (full VRPTW is out of scope).
LATE_PENALTY_EUR_PER_H: float = 250.0

#: Tolerance (minutes) by which a stop's deliver-by deadline may be exceeded in
#: hard mode before the order is treated as infeasible — absorbs floating-point
#: clock noise so an arrival exactly on the deadline is never wrongly rejected.
DEADLINE_TOLERANCE_MIN: float = 1.0

#: Hard clamp on the per-leg gradient derived from endpoint elevations (%). Two
#: stops can sit at very different altitudes a short road apart; without a clamp a
#: near-zero road distance would imply an absurd grade and blow up the energy term.
_MAX_ENDPOINT_GRADIENT_PCT: float = 12.0


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance (km) between two ``(lat, lng)`` points."""
    R = 6371.0
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat = math.radians(b[0] - a[0])
    dlng = math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _is_square_matrix(m: Any, n: int) -> bool:
    """True iff ``m`` is an ``n x n`` matrix of finite numbers (else ignore it)."""
    if not isinstance(m, (list, tuple)) or len(m) != n:
        return False
    for row in m:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            return False
        for v in row:
            try:
                if not math.isfinite(float(v)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _node_xy(p: Any) -> tuple[float, float]:
    """Coerce an origin/destination (dict or pair) to a ``(lat, lng)`` tuple."""
    if isinstance(p, dict):
        return (float(p["lat"]), float(p["lng"]))
    return (float(p[0]), float(p[1]))


# --------------------------------------------------------------------------- #
# Leg cost context — the all-factors core
# --------------------------------------------------------------------------- #
class _LegCtx:
    """Per-optimisation leg cost over a fixed node set.

    Node index space: **0 = origin**, **1..N = the N destinations** (in the
    filtered ``dests`` order). For a leg ``a -> b`` at a given remaining payload:

    * distance/time come from the supplied **road matrix** when present, else a
      great-circle x circuity proxy (distance) and distance/speed (time);
    * **gradient** is ``(elev[b] - elev[a]) / road_distance_m * 100`` from the
      per-node elevations when present, else 0 (flat);
    * **energy** comes from the trained ML model when ``use_model``, else the
      physics ground truth — both fed the same (distance, payload, speed,
      gradient, temperature, wind) features.

    With no matrix / elevations / model the result is identical to the original
    great-circle + flat-terrain + physics proxy, so existing behaviour is kept.
    Leg energy is memoised per ``(a, b, payload-bucket)`` (0.5 t buckets) so the
    exact Held-Karp DP can score with the model without exploding model calls.
    """

    def __init__(
        self,
        nodes: list[Any],
        *,
        dist_matrix: Optional[list[list[float]]] = None,
        time_matrix: Optional[list[list[float]]] = None,
        elevations: Optional[list[float]] = None,
        wind_mps: float = 0.0,
        speed_kph: float = DEFAULT_OPT_SPEED_KPH,
        temperature_c: float = 15.0,
        eur_per_kwh: float = PRICE_EUR_PER_KWH,
        driver_eur_per_h: float = DRIVER_EUR_PER_H,
        road_circuity: float = ROAD_CIRCUITY,
        use_model: bool = False,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    ) -> None:
        self.nodes = [_node_xy(p) for p in nodes]
        n = len(self.nodes)
        # Validate matrices/elevations against the node count; a shape mismatch
        # falls back to the proxy rather than crashing or silently mis-indexing.
        self.dist_matrix = dist_matrix if _is_square_matrix(dist_matrix, n) else None
        self.time_matrix = time_matrix if _is_square_matrix(time_matrix, n) else None
        self.elevations = (
            [float(e) for e in elevations]
            if (isinstance(elevations, (list, tuple)) and len(elevations) == n)
            else None
        )
        self.wind = float(wind_mps or 0.0)
        self.speed = float(speed_kph) if speed_kph and speed_kph > 0 else DEFAULT_OPT_SPEED_KPH
        self.temp = float(temperature_c)
        self.price = float(eur_per_kwh)
        self.driver = float(driver_eur_per_h)
        self.circuity = float(road_circuity)
        self.use_model = bool(use_model)
        self.model_path = model_path
        self._energy_memo: dict[tuple[int, int, float], float] = {}

    # -- geometry ---------------------------------------------------------- #
    def dist_km(self, a: int, b: int) -> float:
        if self.dist_matrix is not None:
            return max(0.0, float(self.dist_matrix[a][b]))
        return _haversine_km(self.nodes[a], self.nodes[b]) * self.circuity

    def time_h(self, a: int, b: int) -> float:
        if self.time_matrix is not None:
            return max(0.0, float(self.time_matrix[a][b]))
        d = self.dist_km(a, b)
        return d / self.speed if self.speed > 0 else 0.0

    def _gradient_pct(self, a: int, b: int, dist_km: float) -> float:
        if self.elevations is not None and dist_km > 0:
            g = (self.elevations[b] - self.elevations[a]) / (dist_km * 1000.0) * 100.0
            return max(-_MAX_ENDPOINT_GRADIENT_PCT, min(_MAX_ENDPOINT_GRADIENT_PCT, g))
        return 0.0

    # -- cost -------------------------------------------------------------- #
    def energy_kwh(self, a: int, b: int, payload_t: float) -> float:
        dist_km = self.dist_km(a, b)
        grad = self._gradient_pct(a, b, dist_km)
        # 0.5 t payload bucket: the exact DP evaluates a leg at many payloads,
        # but distance/gradient/wind are fixed per (a, b), so bucketing payload
        # bounds distinct model calls to ~N^2 x buckets. Well within label noise.
        bucket = round(max(0.0, payload_t) * 2.0) / 2.0
        key = (a, b, bucket)
        hit = self._energy_memo.get(key)
        if hit is not None:
            return hit
        feats = {
            "distance_km": dist_km,
            "payload_t": bucket,
            "speed_kph": self.speed,
            "gradient_pct": grad,
            "temperature_c": self.temp,
            "wind_mps": self.wind,
        }
        if self.use_model:
            energy = float(predict_energy(feats, self.model_path))
        else:
            energy = float(
                segment_energy_kwh(
                    distance_km=dist_km,
                    payload_t=bucket,
                    speed_kph=self.speed,
                    gradient_pct=grad,
                    temperature_c=self.temp,
                    wind_mps=self.wind,
                    truck=TRUCK,
                )
            )
        energy = max(0.0, energy)
        self._energy_memo[key] = energy
        return energy

    def leg_eur(self, a: int, b: int, payload_t: float) -> float:
        return self.energy_kwh(a, b, payload_t) * self.price + self.time_h(a, b) * self.driver


def _node_index(d: dict[str, Any], fallback_pos: int) -> int:
    """Node index for a destination dict — its stamped ``_node`` or a positional
    fallback (used when :func:`estimate_order_cost` builds its own proxy ctx)."""
    try:
        return int(d["_node"])
    except (KeyError, TypeError, ValueError):
        return fallback_pos


# --------------------------------------------------------------------------- #
# Order cost (payload-decay + soft deadlines)
# --------------------------------------------------------------------------- #
def estimate_order_cost(
    origin: Optional[dict[str, Any]],
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
    ctx: Optional[_LegCtx] = None,
) -> float:
    """Proxy operating cost (EUR) of visiting ``ordered_dests`` from ``origin``.

    Walks the legs in order, decaying payload at each drop (the leg INTO a stop
    carries the full pre-drop weight), accruing energy + driver-time cost and a
    soft lateness penalty against each ``deliverByMin`` (minutes after departure).
    Optionally returns to the origin (depot).

    When ``ctx`` is supplied the cost uses that all-factors :class:`_LegCtx`
    (real road matrix / terrain / ML model) and each destination's stamped
    ``_node`` index; otherwise it builds a great-circle proxy ctx from the
    keyword weights and indexes positionally, so an unspecified call is
    equivalent to the original proxy (modulo the 0.5 t payload bucket).

    Deadlines are a SOFT lateness penalty by default. With ``enforce_deadlines``
    a stop arrival exceeding its ``deliverByMin`` by more than
    ``deadline_tolerance_min`` makes the order infeasible (returns ``+inf``).
    """
    if ctx is None:
        ctx = _LegCtx(
            [origin] + list(ordered_dests),
            speed_kph=speed_kph,
            temperature_c=temperature_c,
            eur_per_kwh=eur_per_kwh,
            driver_eur_per_h=driver_eur_per_h,
            road_circuity=road_circuity,
        )
        positional = True
    else:
        positional = False

    total = 0.0
    payload = max(0.0, start_payload_t)
    clock_min = 0.0
    prev = 0  # origin
    for k, d in enumerate(ordered_dests):
        nd = (k + 1) if positional else _node_index(d, k + 1)
        total += ctx.leg_eur(prev, nd, payload)
        clock_min += ctx.time_h(prev, nd) * 60.0
        deadline = d.get("deliverByMin")
        if deadline is not None and clock_min > float(deadline):
            if enforce_deadlines and clock_min > float(deadline) + deadline_tolerance_min:
                return float("inf")  # hard window violated -> infeasible order
            total += (clock_min - float(deadline)) / 60.0 * late_penalty_eur_per_h
        clock_min += float(d.get("unloadMin", 0) or 0)
        payload = max(0.0, payload - float(d.get("dropWeightKg", 0) or 0) / 1000.0)
        prev = nd
    if return_to_origin and ordered_dests:
        total += ctx.leg_eur(prev, 0, payload)
    return total


# --------------------------------------------------------------------------- #
# Solvers (operate on a shared _LegCtx; node index = dest local index + 1)
# --------------------------------------------------------------------------- #
def _drops_t(dests: list[dict[str, Any]]) -> list[float]:
    return [max(0.0, float(d.get("dropWeightKg", 0) or 0) / 1000.0) for d in dests]


def _held_karp(ctx: _LegCtx, dests: list[dict[str, Any]], start_payload_t: float) -> list[int]:
    """Exact min-cost visiting order via Held-Karp DP (open path from origin).

    State ``dp[(mask, last)]`` = min cost to leave the origin (node 0) and visit
    exactly the destination set ``mask``, ending at ``last`` (local dest index).
    Remaining payload on the next leg is ``start - drops(mask)`` — a function of
    the *set*, which keeps the DP exact under payload decay. Costs come from
    ``ctx`` (real road/terrain/ML when supplied). Deadlines are NOT folded in
    here (arrival time is path- not set-determined); they are scored on the full
    order via :func:`estimate_order_cost` for the final comparison.
    """
    n = len(dests)
    drop_t = _drops_t(dests)
    INF = float("inf")
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    for j in range(n):  # origin -> j (nothing dropped yet -> full payload)
        dp[(1 << j, j)] = ctx.leg_eur(0, j + 1, start_payload_t)
        parent[(1 << j, j)] = -1

    for mask in range(1, 1 << n):
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
                cost = base + ctx.leg_eur(last + 1, j + 1, payload_next)
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
    ctx: _LegCtx,
    dests: list[dict[str, Any]],
    start_payload_t: float,
    *,
    late_penalty_eur_per_h: float = LATE_PENALTY_EUR_PER_H,
) -> list[int]:
    """Nearest-neighbour seed + 2-opt local search for larger stop counts.

    Cost is the full payload-aware :func:`estimate_order_cost` (with ``ctx``)
    recomputed per candidate, so payload decay and soft deadlines are respected;
    2-opt reverses sub-paths until no swap improves. Deterministic.
    """
    n = len(dests)

    def _order_cost(order: list[int]) -> float:
        return estimate_order_cost(
            None,
            [dests[i] for i in order],
            start_payload_t,
            late_penalty_eur_per_h=late_penalty_eur_per_h,
            ctx=ctx,
        )

    # Nearest-neighbour seed by road (or proxy) distance from the running node.
    unvisited = set(range(n))
    cur = 0  # origin node
    seed: list[int] = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: ctx.dist_km(cur, j + 1))
        seed.append(nxt)
        unvisited.discard(nxt)
        cur = nxt + 1

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
    origin: Optional[dict[str, Any]],
    dests: list[dict[str, Any]],
    start_payload_t: float,
    *,
    speed_kph: float = DEFAULT_OPT_SPEED_KPH,
    temperature_c: float = 15.0,
    eur_per_kwh: float = PRICE_EUR_PER_KWH,
    driver_eur_per_h: float = DRIVER_EUR_PER_H,
    late_penalty_eur_per_h: float = LATE_PENALTY_EUR_PER_H,
    road_circuity: float = ROAD_CIRCUITY,
    ctx: Optional[_LegCtx] = None,
) -> list[int]:
    """Pick the solver by size: exact Held-Karp small, NN+2-opt large.

    Costs come from ``ctx`` (the all-factors :class:`_LegCtx`) when supplied;
    otherwise a great-circle proxy ctx is built from ``origin`` + the weight
    kwargs, so a direct call with the legacy signature still works. Scores on the
    SOFT cost; hard-deadline feasibility (when requested) is a path-dependent
    post-filter handled by :func:`optimize_route`.
    """
    n = len(dests)
    if n <= 1:
        return list(range(n))
    # Stamp node indices so ctx-aware costing (the NN+2-opt path) maps each dest
    # to the right matrix/elevation row regardless of how the order is permuted.
    for k, d in enumerate(dests):
        d["_node"] = k + 1
    if ctx is None:
        ctx = _LegCtx(
            [origin] + list(dests),
            speed_kph=speed_kph, temperature_c=temperature_c, eur_per_kwh=eur_per_kwh,
            driver_eur_per_h=driver_eur_per_h, road_circuity=road_circuity,
        )
    if n <= MAX_EXACT_STOPS:
        return _held_karp(ctx, dests, start_payload_t)
    return _nn_two_opt(ctx, dests, start_payload_t, late_penalty_eur_per_h=late_penalty_eur_per_h)


def _two_opt_neighbours(order: list[int]) -> list[list[int]]:
    """All 2-opt sub-path reversals of ``order`` (used by the hard-deadline
    feasibility search neighbourhood around the soft-optimal seed)."""
    out: list[list[int]] = []
    n = len(order)
    for i in range(n - 1):
        for k in range(i + 1, n):
            out.append(order[:i] + order[i : k + 1][::-1] + order[k + 1 :])
    return out


def _enforce_deadline_filter(
    ctx: _LegCtx,
    dests: list[dict[str, Any]],
    seed_order: list[int],
    start_payload_t: float,
    *,
    late_penalty_eur_per_h: float,
    deadline_tolerance_min: float,
) -> tuple[list[int], bool]:
    """Hard VRPTW post-filter over the seed order and its 2-opt neighbours.

    Re-scores the seed and each 2-opt neighbour under the HARD window
    (``enforce_deadlines=True`` -> ``+inf`` on a violation) and picks the
    CHEAPEST FEASIBLE order. Returns ``(order, infeasible)``; if none is feasible
    it returns the least-violating order (lowest SOFT cost) and ``infeasible=True``.
    """
    candidates = [list(seed_order)] + _two_opt_neighbours(seed_order)

    def _soft(order: list[int]) -> float:
        return estimate_order_cost(
            None, [dests[i] for i in order], start_payload_t,
            late_penalty_eur_per_h=late_penalty_eur_per_h, ctx=ctx,
        )

    def _hard(order: list[int]) -> float:
        return estimate_order_cost(
            None, [dests[i] for i in order], start_payload_t,
            late_penalty_eur_per_h=late_penalty_eur_per_h, ctx=ctx,
            enforce_deadlines=True, deadline_tolerance_min=deadline_tolerance_min,
        )

    feasible = [c for c in candidates if math.isfinite(_hard(c))]
    if feasible:
        best = min(feasible, key=lambda c: (_soft(c), c))
        return best, False
    best = min(candidates, key=lambda c: (_soft(c), c))
    return best, True


# --------------------------------------------------------------------------- #
# Plan cost over a concrete order
# --------------------------------------------------------------------------- #
def _ordered_metrics(ctx: _LegCtx, ordered: list[dict[str, Any]], *, return_to_origin: bool) -> tuple[float, float]:
    """Total (distance_km, time_h) for a visiting order using ``ctx`` (real road
    matrix when present, else the great-circle proxy)."""
    total_km = 0.0
    total_h = 0.0
    prev = 0
    for d in ordered:
        nd = _node_index(d, prev)
        total_km += ctx.dist_km(prev, nd)
        total_h += ctx.time_h(prev, nd)
        prev = nd
    if return_to_origin and ordered:
        total_km += ctx.dist_km(prev, 0)
        total_h += ctx.time_h(prev, 0)
    return total_km, total_h


def _ordered_distance_km(
    origin: dict[str, Any], ordered: list[dict[str, Any]], *, return_to_origin: bool,
    road_circuity: float = ROAD_CIRCUITY,
) -> float:
    """Great-circle (x circuity) road-distance estimate for a visiting order
    (used only when no real road matrix is available)."""
    total = 0.0
    prev = (float(origin["lat"]), float(origin["lng"]))
    for d in ordered:
        here = (float(d["lat"]), float(d["lng"]))
        total += _haversine_km(prev, here) * road_circuity
        prev = here
    if return_to_origin and ordered:
        total += _haversine_km(prev, (float(origin["lat"]), float(origin["lng"]))) * road_circuity
    return total


def _plan_cost(plan: dict[str, Any], eur_per_kwh: float, driver_eur_per_h: float = DRIVER_EUR_PER_H) -> dict[str, Any]:
    """Operating cost of a planner result: energy price + driver time."""
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
    ctx: Optional[_LegCtx] = None,
) -> dict[str, Any]:
    """Run the SOC/charging/EU-561 planner over one concrete visiting order.

    Total distance/time come from ``ctx`` (the real road matrix when supplied),
    else the great-circle proxy. ``geometry=None`` keeps this in flat-fallback
    mode — the absolute numbers are estimates the production pipeline refines by
    re-routing the chosen order through the real engine for the displayed plan.
    """
    if ctx is not None:
        total_km, total_h = _ordered_metrics(ctx, ordered, return_to_origin=return_to_origin)
        duration_s = total_h * 3600.0
    else:
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
        geometry=None,
        model_path=model_path,
    )


# --------------------------------------------------------------------------- #
# Reach-a-charger-from-the-final-stop check
# --------------------------------------------------------------------------- #
def _destination_charger_check(
    plan: dict[str, Any],
    *,
    final_charger_distance_km: float,
    final_payload_t: float,
    min_soc: float,
    speed_kph: float,
    temperature_c: float,
    wind_mps: float,
    model_path: Union[str, Path],
) -> dict[str, Any]:
    """Can the truck still reach the nearest charger AFTER the final drop-off?

    A truck that arrives at the final destination on fumes is effectively
    stranded if the destination has no charger. We estimate the energy of the
    ``destination -> nearest charger`` limp leg (the ML model, at the lightened
    final payload) and require the arrival SOC to cover it AND still land at the
    charger at or above ``min_soc``. Returns a verdict dict; ``reachable=False``
    is the conservative red flag.
    """
    arrival_soc = float(plan.get("summary", {}).get("arrivalSoc", 0.0) or 0.0)
    batt = float(TRUCK.battery_kwh)
    if final_charger_distance_km > 0:
        limp_kwh = float(
            predict_energy(
                {
                    "distance_km": float(final_charger_distance_km),
                    "payload_t": max(0.0, final_payload_t),
                    "speed_kph": speed_kph,
                    "gradient_pct": 0.0,
                    "temperature_c": temperature_c,
                    "wind_mps": wind_mps,
                },
                model_path,
            )
        )
    else:
        limp_kwh = 0.0
    limp_kwh = max(0.0, limp_kwh)
    soc_drop_pct = (limp_kwh / batt * 100.0) if batt > 0 else 0.0
    soc_at_charger = arrival_soc - soc_drop_pct
    reachable = soc_at_charger >= float(min_soc) - 1e-6
    if reachable:
        note = (
            f"Arrives at the final stop at {arrival_soc:.0f}% SOC and can reach the nearest charger "
            f"(~{final_charger_distance_km:.0f} km, ~{limp_kwh:.0f} kWh), landing at ~{soc_at_charger:.0f}% "
            f"(>= the {min_soc:.0f}% floor)."
        )
    else:
        note = (
            f"WARNING: arrives at the final stop at {arrival_soc:.0f}% SOC but the nearest charger is "
            f"~{final_charger_distance_km:.0f} km (~{limp_kwh:.0f} kWh) away — it would land at "
            f"~{soc_at_charger:.0f}%, below the {min_soc:.0f}% floor. The truck risks being stranded at "
            f"the drop-off; charge more en route or pick a closer charger."
        )
    return {
        "distanceKm": round(float(final_charger_distance_km), 1),
        "limpKwh": round(limp_kwh, 1),
        "arrivalSoc": round(arrival_soc, 1),
        "socAtCharger": round(soc_at_charger, 1),
        "minSoc": round(float(min_soc), 1),
        "reachable": bool(reachable),
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
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
    # --- all-factors inputs (optional; absent => great-circle + physics) --- #
    dist_matrix_km: Optional[list[list[float]]] = None,
    time_matrix_h: Optional[list[list[float]]] = None,
    node_elevations_m: Optional[list[float]] = None,
    wind_mps: float = 0.0,
    use_model: bool = False,
    final_charger_distance_km: Optional[float] = None,
    charger_km_by_dest: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Find the cheapest visiting order and return it with its plan + the saving.

    ``origin`` is ``{lat, lng, label?}``; each destination may carry
    ``dropWeightKg``, ``unloadMin``, ``deliverByMin`` (minutes after departure).

    **All-factors costing.** When supplied, the order is scored on a real road
    ``dist_matrix_km`` / ``time_matrix_h`` (index 0 = origin, 1..N = destinations
    in the given order), per-leg gradient from ``node_elevations_m`` (same
    indexing), a representative ``wind_mps``, and — with ``use_model`` — the
    trained ML energy model. Absent these, it reproduces the original
    great-circle + flat + physics proxy, so existing callers are unchanged.

    If ``final_charger_distance_km`` is given (km from the final destination to
    the nearest charger), the result carries a ``destinationCharger`` verdict:
    whether the truck can still reach that charger on its arrival SOC (P2-D5).

    Returns a JSON-serialisable dict: ``optimizedOrder`` (input indices, best
    order) + ``plan`` + ``cost``, the ``baseline`` (input order) cost,
    ``savingsEur`` / ``savingsPct``, ``solver``, ``dataSources`` (what each cost
    ingredient actually used), an optional ``destinationCharger``, and honest
    ``assumptions``. Deterministic.
    """
    dests = [d for d in destinations if d and d.get("lat") is not None and d.get("lng") is not None]
    start_payload_t = max(0.0, float(payload_kg)) / 1000.0

    # Stamp each kept destination with its node index (0 = origin, 1..N = dests).
    for k, d in enumerate(dests):
        d["_node"] = k + 1

    ctx = _LegCtx(
        [origin] + dests,
        dist_matrix=dist_matrix_km,
        time_matrix=time_matrix_h,
        elevations=node_elevations_m,
        wind_mps=wind_mps,
        speed_kph=speed_kph,
        temperature_c=temperature_c,
        eur_per_kwh=eur_per_kwh,
        driver_eur_per_h=driver_eur_per_h,
        road_circuity=road_circuity,
        use_model=use_model,
        model_path=model_path,
    )

    deadlines_infeasible = False
    if len(dests) <= 1:  # 0 or 1 stop -> nothing to reorder
        order_idx = list(range(len(dests)))
        if enforce_deadlines and order_idx:
            hard = estimate_order_cost(
                None, dests, start_payload_t,
                late_penalty_eur_per_h=late_penalty_eur_per_h, ctx=ctx,
                enforce_deadlines=True, deadline_tolerance_min=deadline_tolerance_min,
            )
            deadlines_infeasible = not math.isfinite(hard)
    else:
        seed_idx = _solve_order(
            origin, dests, start_payload_t,
            late_penalty_eur_per_h=late_penalty_eur_per_h, ctx=ctx,
        )
        if enforce_deadlines:
            order_idx, deadlines_infeasible = _enforce_deadline_filter(
                ctx, dests, seed_idx, start_payload_t,
                late_penalty_eur_per_h=late_penalty_eur_per_h,
                deadline_tolerance_min=deadline_tolerance_min,
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
        return_to_origin=return_to_origin, model_path=model_path, ctx=ctx,
    )
    opt_plan = _plan_for_order(origin, ordered, **plan_kwargs)
    base_plan = _plan_for_order(origin, dests, **plan_kwargs)

    opt_cost = _plan_cost(opt_plan, eur_per_kwh, driver_eur_per_h)
    base_cost = _plan_cost(base_plan, eur_per_kwh, driver_eur_per_h)
    savings_eur = round(base_cost["totalEur"] - opt_cost["totalEur"], 2)
    savings_pct = round(
        (savings_eur / base_cost["totalEur"] * 100.0) if base_cost["totalEur"] > 0 else 0.0, 1
    )
    # Energy-based saving (the case study is about energy, not money) — this is what
    # the dashboard surfaces; the EUR figures stay internal/auditable only.
    savings_kwh = round(base_cost["energyKwh"] - opt_cost["energyKwh"], 1)
    savings_pct_kwh = round(
        (savings_kwh / base_cost["energyKwh"] * 100.0) if base_cost["energyKwh"] > 0 else 0.0, 1
    )

    # Reach-a-charger-from-the-final-stop verdict (P2-D5). The relevant stop is the
    # LAST one in the CHOSEN order, so prefer a per-destination charger-distance
    # array (indexed like the input dests) and pick the optimised final stop; fall
    # back to an explicit scalar (used by direct/test callers).
    dest_charger = None
    final_dist = final_charger_distance_km
    if final_dist is None and charger_km_by_dest and order_idx:
        last_local = order_idx[-1]
        if 0 <= last_local < len(charger_km_by_dest) and charger_km_by_dest[last_local] is not None:
            final_dist = float(charger_km_by_dest[last_local])
    if final_dist is not None and float(final_dist) >= 0 and dests:
        final_payload_t = max(0.0, start_payload_t - sum(_drops_t(dests)))
        dest_charger = _destination_charger_check(
            opt_plan,
            final_charger_distance_km=float(final_dist),
            final_payload_t=final_payload_t,
            min_soc=min_soc,
            speed_kph=speed_kph,
            temperature_c=temperature_c,
            wind_mps=wind_mps,
            model_path=model_path,
        )

    data_sources = {
        "distance": "tomtom-road-matrix" if ctx.dist_matrix is not None else "great-circle-proxy",
        "time": "tomtom-road-matrix" if ctx.time_matrix is not None else f"distance/{round(speed_kph)}kph",
        "terrain": "open-meteo-endpoint-gradient" if ctx.elevations is not None else "flat",
        "energy": "ml-model" if use_model else "physics-ground-truth",
        "wind": "supplied" if (ctx.elevations is not None and wind_mps) else "none",
    }

    # Honest assumptions, reflecting what was ACTUALLY used for the ordering.
    if ctx.dist_matrix is not None:
        order_geo = (
            "Stop ORDERING uses REAL road distances/times (routing-engine matrix); "
            "the final displayed plan re-routes the chosen order through the engine for road-accurate numbers."
        )
    else:
        order_geo = (
            "Stop ORDERING uses a great-circle distance proxy (x road-circuity factor) — no road matrix was "
            "supplied; the production pipeline should re-route the chosen order for road-accurate distance/ETA."
        )
    energy_src = (
        "Leg energy scored with the trained ML model."
        if use_model else
        "Leg energy scored with the first-principles physics ground truth."
    )
    terrain_src = (
        "Per-leg gradient derived from endpoint elevations (Open-Meteo); a representative wind is applied."
        if ctx.elevations is not None else
        "Ordering assumes flat terrain and no wind (no elevation supplied); terrain enters only the final plan."
    )
    assumptions = [
        "Single vehicle; ordering minimises energy + driver-time cost with payload decay "
        "(heavier drops earlier lower later-leg energy).",
        order_geo,
        energy_src,
        terrain_src,
        "Charging is the planner's greedy adaptive policy applied AFTER ordering, NOT jointly "
        "co-optimised with the order (full E-VRPTW is out of scope).",
        f"Cost model: energy at {eur_per_kwh:.2f} EUR/kWh + driver time at {driver_eur_per_h:.0f} EUR/h; "
        f"soft-lateness penalty {late_penalty_eur_per_h:.0f} EUR/h; road-circuity factor {road_circuity:.2f}.",
    ]
    if enforce_deadlines:
        assumptions.append(
            "Deliver-by deadlines are a HARD time window: orders missing any window are dropped and the "
            "cheapest feasible order chosen"
            + (" — NO feasible order meets these windows; returning the least-violating order "
               "(see deadlinesInfeasible)." if deadlines_infeasible else ".")
        )
    else:
        assumptions.append(
            "Deliver-by deadlines are a soft lateness penalty, not a hard time window "
            "(pass enforce_deadlines=True for a hard VRPTW window)."
        )
    if dest_charger is not None and not dest_charger["reachable"]:
        assumptions.insert(0, dest_charger["note"])

    return {
        "optimizedOrder": order_idx,
        "solver": solver,
        "nStops": len(dests),
        "plan": opt_plan,
        "cost": opt_cost,
        "baseline": {"order": list(range(len(dests))), "cost": base_cost},
        "savingsEur": savings_eur,
        "savingsPct": savings_pct,
        "savingsKwh": savings_kwh,
        "savingsPctKwh": savings_pct_kwh,
        "deadlinesEnforced": bool(enforce_deadlines),
        "deadlinesInfeasible": bool(deadlines_infeasible),
        "dataSources": data_sources,
        "destinationCharger": dest_charger,
        "assumptions": assumptions,
    }
