"""Tests for :mod:`nexdash.optimizer` — the cost-minimising VRP layer.

These encode WHY the optimiser matters, not just that it runs:

* The exact Held-Karp solver must return the TRUE minimum-cost order — pinned by
  comparing it to brute-force over all permutations (incl. payload decay), so a
  bug in the DP recurrence or the payload-by-visited-set accounting is caught.
* Payload decay must make the order genuinely order-dependent (dropping a heavy
  load early lowers later-leg energy) — a flat model would make every order equal.
* optimize_route must never return an order MORE expensive than the operator's
  original (the whole point), and must expose the saving + an auditable plan.
* The result must be deterministic and JSON-serialisable.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from nexdash import optimizer as opt
from nexdash.config import DEFAULT_MODEL_PATH

MODEL = str(DEFAULT_MODEL_PATH)
_ORIGIN = {"lat": 52.52, "lng": 13.40, "label": "Berlin"}

# A small spread of destinations around the origin (lat/lng), some carrying drops.
_DESTS = [
    {"lat": 52.52, "lng": 14.20, "label": "E1", "dropWeightKg": 6000},
    {"lat": 51.34, "lng": 12.37, "label": "Leipzig", "dropWeightKg": 3000},
    {"lat": 53.55, "lng": 10.00, "label": "Hamburg", "dropWeightKg": 5000},
    {"lat": 52.38, "lng": 9.73, "label": "Hannover", "dropWeightKg": 4000},
    {"lat": 51.05, "lng": 13.74, "label": "Dresden", "dropWeightKg": 2000},
]

_KW = dict(speed_kph=70.0, temperature_c=15.0, eur_per_kwh=0.45)


def test_held_karp_matches_brute_force_with_payload_decay():
    """Exactness guard: Held-Karp must equal the brute-force optimum, INCLUDING the
    payload-decay coupling (leg cost depends on which drops happened before it).
    A wrong DP recurrence or set-payload accounting would diverge from brute force."""
    dests = _DESTS  # 5 dests carrying drops
    start_t = 18.0
    order = opt._solve_order(_ORIGIN, dests, start_t, **_KW)
    hk_cost = opt.estimate_order_cost(_ORIGIN, [dests[i] for i in order], start_t, **_KW)

    best = min(
        permutations(range(len(dests))),
        key=lambda p: opt.estimate_order_cost(_ORIGIN, [dests[i] for i in p], start_t, **_KW),
    )
    best_cost = opt.estimate_order_cost(_ORIGIN, [dests[i] for i in best], start_t, **_KW)
    assert hk_cost == pytest.approx(best_cost, rel=1e-9), (
        f"Held-Karp order cost {hk_cost} != brute-force optimum {best_cost}"
    )


def test_payload_decay_makes_order_matter():
    """WHY: with payload decay, dropping a heavy load before a long leg is cheaper,
    so the order is not arbitrary. Two orders of the SAME stops must be able to
    differ in cost — otherwise the energy/payload coupling isn't being applied."""
    a = opt.estimate_order_cost(_ORIGIN, _DESTS, 18.0, **_KW)
    b = opt.estimate_order_cost(_ORIGIN, list(reversed(_DESTS)), 18.0, **_KW)
    assert a != pytest.approx(b), "payload-aware leg costs must make order matter"


def test_optimize_route_never_worse_than_original_and_is_auditable():
    """optimize_route must (a) expose the full auditable contract, and (b) the
    chosen order's modelled cost must be <= the operator's original-order cost
    (the optimiser can tie but must never make it worse)."""
    out = opt.optimize_route(
        _ORIGIN, _DESTS, start_soc=95, min_soc=15, payload_kg=18000, model_path=MODEL
    )
    for key in ("optimizedOrder", "solver", "plan", "cost", "baseline", "savingsEur", "savingsPct", "assumptions"):
        assert key in out
    assert sorted(out["optimizedOrder"]) == list(range(len(_DESTS)))  # a permutation
    assert out["cost"]["totalEur"] <= out["baseline"]["cost"]["totalEur"] + 1e-6
    assert out["savingsEur"] >= -1e-6
    assert "held-karp" in out["solver"]  # 5 stops -> exact


def test_optimize_route_is_deterministic_and_json_safe():
    """Same inputs -> identical chosen order + savings (no randomness), JSON-safe."""
    import json

    kw = dict(start_soc=80, min_soc=10, payload_kg=12000, model_path=MODEL)
    a = opt.optimize_route(_ORIGIN, _DESTS, **kw)
    b = opt.optimize_route(_ORIGIN, _DESTS, **kw)
    assert a["optimizedOrder"] == b["optimizedOrder"]
    assert a["savingsEur"] == b["savingsEur"]
    json.dumps(a)  # must not raise


def test_single_stop_is_trivial():
    """0 or 1 stop has nothing to reorder; the optimiser must not choke on it."""
    out = opt.optimize_route(
        _ORIGIN, _DESTS[:1], start_soc=90, min_soc=15, payload_kg=8000, model_path=MODEL
    )
    assert out["optimizedOrder"] == [0]
    assert out["solver"] == "trivial"


# --------------------------------------------------------------------------- #
# (C5) Configurable cost weights — default-preserving + auditable
# --------------------------------------------------------------------------- #
# A heavy stop due NORTH and a deadline-bearing light stop due SOUTH, equidistant
# from the depot. Dropping the heavy load first lightens the long return leg, so
# the ENERGY-cheapest order visits the heavy stop first and reaches the southern
# deadline LATE. Whether that lateness is worth it depends entirely on the late
# penalty — which is exactly the knob C5 makes configurable.
_HEAVY_NORTH = {"lat": 52.90, "lng": 13.40, "label": "H_heavy", "dropWeightKg": 18000}
_LATE_SOUTH = {"lat": 52.14, "lng": 13.40, "label": "L_deadline",
               "dropWeightKg": 500, "deliverByMin": 50}
_TWO = [_HEAVY_NORTH, _LATE_SOUTH]  # index 0 = heavy-first(late), 1 = deadline-first(on-time)


def _cheapest_of_two(penalty: float) -> list[int]:
    return min(
        ([0, 1], [1, 0]),
        key=lambda p: opt.estimate_order_cost(
            _ORIGIN, [_TWO[i] for i in p], 18.0,
            speed_kph=70.0, temperature_c=15.0, eur_per_kwh=0.45,
            late_penalty_eur_per_h=penalty,
        ),
    )


def test_configurable_late_penalty_shifts_chosen_order():
    """WHY C5 exists: the cost weights are policy, not constants. With a tiny
    late penalty the cheapest order tolerates lateness to save energy (heavy drop
    first); with a large penalty the on-time order wins. The SAME stops must pick
    a DIFFERENT order purely because the configurable weight changed — proving the
    kwarg is actually threaded into the scoring, not ignored."""
    assert _cheapest_of_two(0.5) == [0, 1]      # heavy-first, slightly late
    assert _cheapest_of_two(250.0) == [1, 0]    # deadline-first, on time
    assert _cheapest_of_two(0.5) != _cheapest_of_two(250.0)


def test_configurable_road_circuity_scales_cost():
    """The road-circuity multiplier must actually scale the distance proxy: a
    larger factor means longer modelled roads and so strictly higher cost. If the
    kwarg were dropped (still reading the global) the two costs would be equal."""
    base = dict(speed_kph=70.0, temperature_c=15.0, eur_per_kwh=0.45)
    lo = opt.estimate_order_cost(_ORIGIN, _DESTS, 18.0, road_circuity=1.0, **base)
    hi = opt.estimate_order_cost(_ORIGIN, _DESTS, 18.0, road_circuity=2.0, **base)
    assert hi > lo


def test_weights_default_to_today_and_are_byte_identical():
    """Default-preservation contract: omitting the new weight kwargs must give the
    EXACT same number as passing today's module constants explicitly. A drifted
    default (or a hard-coded literal) would break this and silently change every
    existing caller's result."""
    base = dict(speed_kph=70.0, temperature_c=15.0, eur_per_kwh=0.45)
    implicit = opt.estimate_order_cost(_ORIGIN, _DESTS, 18.0, **base)
    explicit = opt.estimate_order_cost(
        _ORIGIN, _DESTS, 18.0,
        driver_eur_per_h=opt.DRIVER_EUR_PER_H,
        late_penalty_eur_per_h=opt.LATE_PENALTY_EUR_PER_H,
        road_circuity=opt.ROAD_CIRCUITY,
        **base,
    )
    assert implicit == explicit


def test_default_optimize_route_order_and_cost_unchanged_by_C5():
    """End-to-end default-preservation: optimize_route called WITHOUT the new
    kwargs must produce the identical chosen order, cost and savings as calling it
    WITH the weights pinned to today's module constants. C5 may only ADD the echo,
    never move the answer."""
    kw = dict(start_soc=95, min_soc=15, payload_kg=18000, model_path=MODEL)
    a = opt.optimize_route(_ORIGIN, _DESTS, **kw)
    b = opt.optimize_route(
        _ORIGIN, _DESTS,
        driver_eur_per_h=opt.DRIVER_EUR_PER_H,
        late_penalty_eur_per_h=opt.LATE_PENALTY_EUR_PER_H,
        road_circuity=opt.ROAD_CIRCUITY,
        **kw,
    )
    assert a["optimizedOrder"] == b["optimizedOrder"]
    assert a["cost"] == b["cost"]
    assert a["savingsEur"] == b["savingsEur"]


def test_actual_weights_are_echoed_into_assumptions():
    """C5 requires the weights actually used to be auditable in the output. With
    non-default weights the assumptions must report THOSE numbers, not the module
    constants — so a reviewer can see the policy the run was costed under."""
    out = opt.optimize_route(
        _ORIGIN, _DESTS, start_soc=95, min_soc=15, payload_kg=18000, model_path=MODEL,
        driver_eur_per_h=42.0, late_penalty_eur_per_h=99.0, road_circuity=1.55,
    )
    cost_line = next(a for a in out["assumptions"] if a.startswith("Cost model:"))
    assert "42 EUR/h" in cost_line
    assert "99 EUR/h" in cost_line
    assert "1.55" in cost_line


# --------------------------------------------------------------------------- #
# (C3) Hard VRPTW window option — opt-in, soft stays the default
# --------------------------------------------------------------------------- #
def test_estimate_order_cost_returns_inf_on_hard_window_violation():
    """The hard-window primitive: when enforce_deadlines=True an order that misses
    a deliver-by window (beyond tolerance) must be +inf (infeasible), whereas the
    same order under the soft default is merely penalised (finite). This is the
    feasibility signal the post-filter relies on."""
    base = dict(speed_kph=70.0, temperature_c=15.0, eur_per_kwh=0.45,
                late_penalty_eur_per_h=0.5)
    late_order = [_TWO[i] for i in (0, 1)]  # heavy-first -> south deadline missed
    soft = opt.estimate_order_cost(_ORIGIN, late_order, 18.0, **base)
    hard = opt.estimate_order_cost(_ORIGIN, late_order, 18.0, enforce_deadlines=True, **base)
    assert soft < float("inf")          # soft only penalises
    assert hard == float("inf")         # hard rejects


def test_hard_mode_picks_ontime_while_soft_keeps_cheap_but_late():
    """The whole point of C3: with a low late penalty the SOFT optimiser keeps the
    cheaper heavy-first order even though it arrives late at the deadline stop;
    HARD mode must instead pick the on-time order (visiting the deadline stop
    first), trading a little money for feasibility. Same inputs, opposite orders,
    selected only by enforce_deadlines."""
    common = dict(start_soc=90, min_soc=15, payload_kg=18500, model_path=MODEL,
                  late_penalty_eur_per_h=0.5)
    soft = opt.optimize_route(_ORIGIN, _TWO, **common)
    hard = opt.optimize_route(_ORIGIN, _TWO, enforce_deadlines=True, **common)

    assert soft["optimizedOrder"] == [0, 1]       # cheap-but-late kept
    assert soft["deadlinesEnforced"] is False
    assert hard["optimizedOrder"] == [1, 0]       # on-time chosen
    assert hard["deadlinesEnforced"] is True
    assert hard["deadlinesInfeasible"] is False   # a feasible order existed


def test_hard_mode_flags_infeasible_but_still_returns_least_violating():
    """If NO order can meet the windows, hard mode must NOT silently pretend it
    found a feasible plan: it flags deadlinesInfeasible=True and still returns a
    usable (least-violating) order so the caller has something to dispatch. Fail
    loud, don't fabricate feasibility."""
    impossible = [
        _HEAVY_NORTH,
        {"lat": 52.14, "lng": 13.40, "label": "L_tight",
         "dropWeightKg": 500, "deliverByMin": 1},  # unreachable in time by any order
    ]
    out = opt.optimize_route(
        _ORIGIN, impossible, enforce_deadlines=True,
        start_soc=90, min_soc=15, payload_kg=18500, model_path=MODEL,
        late_penalty_eur_per_h=0.5,
    )
    assert out["deadlinesInfeasible"] is True
    assert sorted(out["optimizedOrder"]) == [0, 1]   # still a full permutation
    assert any("least-violating" in a for a in out["assumptions"])


def test_default_mode_is_soft_and_does_not_set_infeasible():
    """Regression guard: the default (no enforce_deadlines) must stay SOFT — never
    drops an order for lateness, never flags infeasible — so existing callers are
    unaffected by C3 being present."""
    out = opt.optimize_route(
        _ORIGIN, _TWO, start_soc=90, min_soc=15, payload_kg=18500, model_path=MODEL,
        late_penalty_eur_per_h=0.5,
    )
    assert out["deadlinesEnforced"] is False
    assert out["deadlinesInfeasible"] is False
    assert out["optimizedOrder"] == [0, 1]  # the cheap-but-late soft optimum
