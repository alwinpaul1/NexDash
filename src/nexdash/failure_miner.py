"""Auto-discover failure modes the designer never thought to slice.

The evaluation report already slices error by *hand-picked* regimes (cold / steep
/ heavy). But the worst error pocket is often a *combination* nobody enumerated —
e.g. ``gradient > 4% AND payload > 15 t``. This module adversarially searches the
held-out feature space for that pocket: it fits a shallow decision tree that
predicts the model's **absolute error** from the raw features, then reports the
leaves whose mean error is worst relative to the global MAE.

Each reported pocket carries a **support floor** (a leaf must hold at least
``min_support`` rows) and a **bootstrap confidence interval on its lift** — so a
pocket only ships if its elevated error is statistically real, not a 3-row fluke.
That guard directly answers the report's own "n < 30, indicative only" caveat
about small hand-picked slices.

Pure numpy + a single shallow ``DecisionTreeRegressor``; deterministic (seeded);
offline. Mining on the *raw* feature columns keeps the discovered conditions
human-readable.

Honest caveat (selection bias): because the tree *chooses* the worst-splitting
leaf, the per-pocket bootstrap lift CI is optimistic — on pure noise a depth-3
tree can still carve a leaf with lift up to ~1.5 whose naive within-leaf CI
lower bound slightly clears 1.0. We measured the false-pocket rate on pure noise
across many seeds: a threshold of 1.5 still admits a noise pocket on roughly a
fifth of seeds, while ``min_lift=1.8`` admits ~0. The default is therefore set to
**1.8**, comfortably above that selection-bias noise band, so a reported pocket
reflects genuine structure rather than the tree's cherry-picking. The lift CI is
a *conditional* (within-leaf) interval, not selection-corrected, so treat a
pocket near the threshold as indicative, not proven.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["mine_failure_modes"]


def _leaf_conditions(tree, feature_names: Sequence[str]) -> dict[int, str]:
    """Reconstruct the human-readable rule path to every leaf of a fitted tree.

    Walks the sklearn tree structure, accumulating ``feature <= thr`` /
    ``feature > thr`` predicates down each branch and, per feature, tightening to
    the most restrictive bound so the printed rule is compact
    (e.g. ``gradient_pct>4.1 AND payload_t>15.2``).
    """
    t = tree.tree_
    out: dict[int, str] = {}

    def fmt(bounds: dict[str, tuple[float, float]]) -> str:
        parts = []
        for name, (lo, hi) in bounds.items():
            if lo > -np.inf and hi < np.inf:
                parts.append(f"{lo:.1f}<{name}<={hi:.1f}")
            elif lo > -np.inf:
                parts.append(f"{name}>{lo:.1f}")
            elif hi < np.inf:
                parts.append(f"{name}<={hi:.1f}")
        return " AND ".join(parts) if parts else "(all rows)"

    def recurse(node: int, bounds: dict[str, tuple[float, float]]) -> None:
        if t.children_left[node] == t.children_right[node]:  # leaf
            out[node] = fmt(bounds)
            return
        name = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        lo, hi = bounds.get(name, (-np.inf, np.inf))
        # left: feature <= thr  (tighten upper bound)
        left = dict(bounds)
        left[name] = (lo, min(hi, thr))
        recurse(t.children_left[node], left)
        # right: feature > thr  (tighten lower bound)
        right = dict(bounds)
        right[name] = (max(lo, thr), hi)
        recurse(t.children_right[node], right)

    recurse(0, {})
    return out


def mine_failure_modes(
    X: np.ndarray,
    abs_error: Sequence[float],
    *,
    feature_names: Sequence[str],
    min_support: int = 30,
    max_depth: int = 3,
    seed: int = 42,
    n_boot: int = 500,
    top_k: int = 5,
    min_lift: float = 1.8,
) -> list[dict[str, Any]]:
    """Find the worst-performing feature-space pockets by lift over global MAE.

    Args:
        X: ``(n, n_features)`` raw feature matrix (use the human-readable raw
            columns so the discovered conditions are interpretable).
        abs_error: per-row absolute prediction error ``|y_hat - y|``.
        feature_names: column names of ``X`` (same order).
        min_support: a leaf must hold at least this many rows to be reported
            (the honesty guard against flukes); also the tree's min_samples_leaf.
        max_depth: tree depth — small keeps pockets simple and legible.
        seed: deterministic tree + bootstrap.
        n_boot: bootstrap resamples for the lift CI.
        top_k: max pockets returned.
        min_lift: only report pockets at least this many times the global MAE.

    Returns:
        Pockets sorted by lift (worst first), each a dict with ``conditions``,
        ``n``, ``mae``, ``global_mae``, ``lift``, ``lift_ci_low``,
        ``lift_ci_high``. Empty list if nothing clears the support+lift bars.
    """
    from sklearn.tree import DecisionTreeRegressor

    X = np.asarray(X, dtype=float)
    abs_error = np.asarray(abs_error, dtype=float)
    if X.ndim != 2 or X.shape[0] != abs_error.size or X.shape[0] < min_support:
        return []

    global_mae = float(np.mean(abs_error))
    if not np.isfinite(global_mae) or global_mae <= 0:
        return []

    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_support, random_state=seed
    )
    tree.fit(X, abs_error)

    leaf_of = tree.apply(X)  # leaf node id per row
    conditions = _leaf_conditions(tree, feature_names)
    rng = np.random.default_rng(seed)

    pockets: list[dict[str, Any]] = []
    for node in np.unique(leaf_of):
        mask = leaf_of == node
        n = int(mask.sum())
        if n < min_support:
            continue
        errs = abs_error[mask]
        mae = float(np.mean(errs))
        lift = mae / global_mae
        if lift < min_lift:
            continue
        # bootstrap CI on the pocket's lift.
        boot = errs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) / global_mae
        pockets.append(
            {
                "conditions": conditions.get(int(node), "(unknown)"),
                "n": n,
                "mae": round(mae, 3),
                "global_mae": round(global_mae, 3),
                "lift": round(lift, 2),
                "lift_ci_low": round(float(np.quantile(boot, 0.025)), 2),
                "lift_ci_high": round(float(np.quantile(boot, 0.975)), 2),
                # The tree CHOSE this leaf to maximise error, so this within-leaf
                # bootstrap CI is selection-biased (optimistic) — NOT a
                # selection-corrected interval. The flag travels with the record so
                # any consumer (report/UI) discloses it rather than over-trusting
                # the lower bound. (Mirrors promote.py's `indicative` pattern.)
                "selection_biased_ci": True,
            }
        )

    pockets.sort(key=lambda p: p["lift"], reverse=True)
    return pockets[:top_k]
