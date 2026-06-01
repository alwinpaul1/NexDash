"""Tests for :mod:`nexdash.failure_miner` — auto-discovered failure modes.

These encode WHY the miner exists, not just that it returns rows:

* **Planted-pocket recovery** (the load-bearing test): inject inflated error on a
  KNOWN feature region, and assert the miner's top-ranked pocket is exactly that
  region. If the miner couldn't recover a planted failure, its discoveries on
  real data would be meaningless.
* **Support floor**: a too-small error pocket must NOT be reported — the guard
  against flukes that the report's own "n<30 indicative" caveat demands.
* **No-signal case**: uniform error yields no pockets (nothing over the lift bar),
  so the miner doesn't manufacture failures that aren't there.
* **Determinism**: a fixed seed yields identical pockets (reproducible report).
"""

from __future__ import annotations

import numpy as np

from nexdash import failure_miner

FEATURES = ["distance_km", "payload_t", "speed_kph", "gradient_pct", "temperature_c", "wind_mps"]


def _synthetic(n=1200, seed=0):
    """A feature matrix over the realistic envelope, with baseline small error."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(1, 350, n),     # distance_km
        rng.uniform(0, 22, n),      # payload_t
        rng.uniform(30, 85, n),     # speed_kph
        rng.uniform(-6, 6, n),      # gradient_pct
        rng.uniform(-15, 40, n),    # temperature_c
        rng.uniform(-12, 12, n),    # wind_mps
    ])
    abs_err = np.abs(rng.normal(0, 1.0, n))  # baseline ~1 kWh error everywhere
    return X, abs_err


def test_recovers_a_planted_failure_pocket():
    """Inflate error where gradient>4 AND payload>15; the top pocket must be it.

    This is the test that proves the miner works: if it can't surface a failure
    we deliberately planted, it can't be trusted to find real ones.
    """
    X, abs_err = _synthetic(n=1500, seed=0)
    grad, payload = X[:, 3], X[:, 1]
    planted = (grad > 4.0) & (payload > 15.0)
    abs_err = abs_err.copy()
    abs_err[planted] += 30.0  # large, unambiguous error inflation

    pockets = failure_miner.mine_failure_modes(
        X, abs_err, feature_names=FEATURES, min_support=20, seed=0
    )
    assert pockets, "miner found no pocket despite a planted one"
    top = pockets[0]
    # The worst pocket must reference BOTH planted dimensions (the tree may add an
    # upper gradient bound too, e.g. '4.0<gradient_pct<=4.9 AND payload_t>15.0',
    # so check membership of the feature names, not a specific '>' form).
    assert "gradient_pct" in top["conditions"] and "payload_t" in top["conditions"], top["conditions"]
    assert top["lift"] > 2.0
    assert top["lift_ci_low"] > 1.0  # CI clears 1x -> statistically real


def test_small_pocket_is_suppressed():
    """An inflated-error region smaller than min_support must NOT be reported.

    WHY: a 5-row fluke must never ship as a 'discovered failure mode' — the
    support floor is the honesty guard.
    """
    X, abs_err = _synthetic(n=1000, seed=1)
    abs_err = abs_err.copy()
    abs_err[:5] += 50.0  # only 5 rows
    pockets = failure_miner.mine_failure_modes(
        X, abs_err, feature_names=FEATURES, min_support=30, seed=1
    )
    for p in pockets:
        assert p["n"] >= 30


def test_noise_false_pocket_rate_is_low_at_default_threshold():
    """On pure noise, the default threshold must rarely invent a 'failure'.

    WHY (the real guard, not a single lucky seed): a depth-3 tree *selects* its
    worst leaf, so on genuinely structureless error it can still carve a low-lift
    pocket whose naive within-leaf CI clears 1.0 — a selection-bias artefact. A
    single-seed "no pockets" assertion hides this. Instead we measure the
    false-positive RATE across many independent noise draws and require the
    shipped default (min_lift=1.8) to keep it small (< 5%). This encodes the
    actual claim the report relies on: the threshold sits ABOVE the noise band.
    """
    n_seeds = 60
    flagged = 0
    for s in range(n_seeds):
        X, abs_err = _synthetic(n=1000, seed=1000 + s)
        pockets = failure_miner.mine_failure_modes(
            X, abs_err, feature_names=FEATURES, min_support=30, seed=1000 + s
        )
        if pockets:
            flagged += 1
    fp_rate = flagged / n_seeds
    assert fp_rate < 0.05, f"noise false-pocket rate {fp_rate:.2%} too high at default min_lift"


def test_deterministic_under_fixed_seed():
    """Same inputs + seed -> identical pockets (the report must reproduce)."""
    X, abs_err = _synthetic(n=1500, seed=0)
    abs_err = abs_err.copy()
    abs_err[(X[:, 3] > 4.0) & (X[:, 1] > 15.0)] += 30.0
    a = failure_miner.mine_failure_modes(X, abs_err, feature_names=FEATURES, min_support=20, seed=7)
    b = failure_miner.mine_failure_modes(X, abs_err, feature_names=FEATURES, min_support=20, seed=7)
    assert a == b
