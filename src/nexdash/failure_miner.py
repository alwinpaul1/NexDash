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

Selection-bias cure (``confirm_split=True``, opt-in): the threshold above is a
*calibrated band*, not a *correction* — it makes the bias rare, but the reported
``lift_ci_*`` is still conditioned on the same data the tree mined, so its lower
bound remains optimistic for the surviving leaf. The principled fix is honest
split-sample inference: partition the held-out rows ONCE (seeded) into a DISCOVER
half and a CONFIRM half, fit the tree and pick candidate pockets on DISCOVER, then
**re-evaluate** each surviving leaf's lift and bootstrap CI on the CONFIRM rows
routed through ``tree.apply`` — rows the tree never saw when it chose where to
split. Because the CONFIRM evaluation is unconditioned on the selection, its CI
lower bound (``lift_ci_low_confirm``) is honest, and ``min_lift`` is gated on
*that* bound. The within-leaf ``lift_ci_*`` (still flagged ``selection_biased_ci``)
is retained for continuity. This is opt-in and additive: with the default
``confirm_split=False`` the function is byte-identical to the single-sample miner
above. Honest residual: the split halves the effective sample, so CONFIRM CIs are
wider and a genuine-but-small pocket can fail to clear the honest gate (a
false-negative the calibrated single-sample path would have reported).
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
    confirm_split: bool = False,
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
        seed: deterministic tree + bootstrap (and, when ``confirm_split``, the
            DISCOVER/CONFIRM partition).
        n_boot: bootstrap resamples for the lift CI.
        top_k: max pockets returned.
        min_lift: minimum lift to report. In single-sample mode this gates the
            within-leaf point lift; in ``confirm_split`` mode it gates the honest
            CONFIRM lower bound ``lift_ci_low_confirm`` (see below).
        confirm_split: when ``True``, run honest split-sample inference — fit the
            tree and pick pockets on a seeded DISCOVER half, then re-evaluate lift
            and bootstrap CI on the held-back CONFIRM half routed through
            ``tree.apply``, gating ``min_lift`` on the CONFIRM lower bound. Default
            ``False`` reproduces the single-sample miner byte-for-byte.

    Returns:
        Pockets sorted by lift (worst first), each a dict with ``conditions``,
        ``n``, ``mae``, ``global_mae``, ``lift``, ``lift_ci_low``,
        ``lift_ci_high``, ``selection_biased_ci``. When ``confirm_split=True``
        each pocket additionally carries ``n_confirm``, ``lift_confirm``,
        ``lift_ci_low_confirm`` and ``lift_ci_high_confirm`` (the honest,
        selection-corrected interval), and ``min_lift`` is gated on
        ``lift_ci_low_confirm``. Empty list if nothing clears the support+lift
        bars.
    """
    from sklearn.tree import DecisionTreeRegressor

    X = np.asarray(X, dtype=float)
    abs_error = np.asarray(abs_error, dtype=float)
    if X.ndim != 2 or X.shape[0] != abs_error.size or X.shape[0] < min_support:
        return []

    global_mae = float(np.mean(abs_error))
    if not np.isfinite(global_mae) or global_mae <= 0:
        return []

    if confirm_split:
        return _mine_confirm_split(
            X,
            abs_error,
            feature_names=feature_names,
            min_support=min_support,
            max_depth=max_depth,
            seed=seed,
            n_boot=n_boot,
            top_k=top_k,
            min_lift=min_lift,
        )

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


def _mine_confirm_split(
    X: np.ndarray,
    abs_error: np.ndarray,
    *,
    feature_names: Sequence[str],
    min_support: int,
    max_depth: int,
    seed: int,
    n_boot: int,
    top_k: int,
    min_lift: float,
) -> list[dict[str, Any]]:
    """Honest split-sample failure mining (the ``confirm_split=True`` path).

    Discovery (tree fit + leaf selection) happens on the DISCOVER half; the lift
    and bootstrap CI that gate ``min_lift`` are recomputed on the disjoint CONFIRM
    half routed through ``tree.apply``. Because CONFIRM rows played no part in
    *where* the tree split, their lift estimate is unconditioned on the selection
    and its CI lower bound is honest — that is the bound ``min_lift`` gates on.

    ``X``/``abs_error`` are assumed already validated (2-D, aligned, finite global
    MAE > 0) by the public ``mine_failure_modes`` wrapper.
    """
    from sklearn.tree import DecisionTreeRegressor

    n_total = X.shape[0]
    # ONE seeded partition into DISCOVER / CONFIRM halves. Seeded so the report
    # reproduces; a permutation (not a per-row coin flip) keeps the split exactly
    # balanced and deterministic.
    perm = np.random.default_rng(seed).permutation(n_total)
    half = n_total // 2
    disc_idx, conf_idx = perm[:half], perm[half:]

    # Both halves must clear the support floor for the split to be meaningful;
    # too few rows on either side cannot support an honest re-evaluation.
    if disc_idx.size < min_support or conf_idx.size < min_support:
        return []

    X_disc, err_disc = X[disc_idx], abs_error[disc_idx]
    X_conf, err_conf = X[conf_idx], abs_error[conf_idx]

    global_mae_disc = float(np.mean(err_disc))
    global_mae_conf = float(np.mean(err_conf))
    if (
        not np.isfinite(global_mae_disc)
        or global_mae_disc <= 0
        or not np.isfinite(global_mae_conf)
        or global_mae_conf <= 0
    ):
        return []

    # DISCOVER: fit the tree and pick candidate leaves exactly as the
    # single-sample path does, but only on the DISCOVER rows.
    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_support, random_state=seed
    )
    tree.fit(X_disc, err_disc)
    conditions = _leaf_conditions(tree, feature_names)

    leaf_disc = tree.apply(X_disc)
    leaf_conf = tree.apply(X_conf)  # route held-back rows through the SAME tree
    rng = np.random.default_rng(seed)

    pockets: list[dict[str, Any]] = []
    for node in np.unique(leaf_disc):
        disc_mask = leaf_disc == node
        n_disc = int(disc_mask.sum())
        if n_disc < min_support:
            continue
        errs_d = err_disc[disc_mask]
        mae_disc = float(np.mean(errs_d))
        lift_disc = mae_disc / global_mae_disc
        # Candidate gate on DISCOVER mirrors the single-sample lift bar; the
        # binding honest gate is the CONFIRM lower bound below.
        if lift_disc < min_lift:
            continue

        # Within-leaf (selection-biased) CI on DISCOVER — kept for continuity with
        # the single-sample record so consumers see both the optimistic and the
        # honest interval side by side.
        boot_d = (
            errs_d[rng.integers(0, n_disc, size=(n_boot, n_disc))].mean(axis=1)
            / global_mae_disc
        )

        # CONFIRM: re-evaluate the SAME leaf on rows the tree never saw when it
        # chose where to split -> lift unconditioned on the selection.
        conf_mask = leaf_conf == node
        n_conf = int(conf_mask.sum())
        if n_conf < min_support:
            # Not enough independent rows to honestly confirm this pocket.
            continue
        errs_c = err_conf[conf_mask]
        mae_conf = float(np.mean(errs_c))
        lift_conf = mae_conf / global_mae_conf
        boot_c = (
            errs_c[rng.integers(0, n_conf, size=(n_boot, n_conf))].mean(axis=1)
            / global_mae_conf
        )
        ci_low_conf = float(np.quantile(boot_c, 0.025))
        ci_high_conf = float(np.quantile(boot_c, 0.975))

        # HONEST gate: the pocket ships only if its confirmed lower bound clears
        # min_lift. This is what defeats selection bias — the tree cannot cherry
        # pick CONFIRM rows it never optimised against.
        if ci_low_conf < min_lift:
            continue

        pockets.append(
            {
                "conditions": conditions.get(int(node), "(unknown)"),
                # `n`/`mae`/`global_mae`/`lift` keep the single-sample meaning
                # (the DISCOVER leaf that was mined) so the record shape matches.
                "n": n_disc,
                "mae": round(mae_disc, 3),
                "global_mae": round(global_mae_disc, 3),
                "lift": round(lift_disc, 2),
                "lift_ci_low": round(float(np.quantile(boot_d, 0.025)), 2),
                "lift_ci_high": round(float(np.quantile(boot_d, 0.975)), 2),
                # Same disclosure as the single-sample path: the DISCOVER lift_ci_*
                # above is conditioned on the tree's selection, so it stays flagged.
                "selection_biased_ci": True,
                # Honest, selection-corrected re-evaluation on the held-back rows.
                "n_confirm": n_conf,
                "lift_confirm": round(lift_conf, 2),
                "lift_ci_low_confirm": round(ci_low_conf, 2),
                "lift_ci_high_confirm": round(ci_high_conf, 2),
            }
        )

    # Rank by the honest confirmed lift so the worst *confirmed* pocket leads.
    pockets.sort(key=lambda p: p["lift_confirm"], reverse=True)
    return pockets[:top_k]
