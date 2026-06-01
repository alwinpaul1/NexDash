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
