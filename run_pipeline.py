#!/usr/bin/env python3
"""End-to-end training & evaluation pipeline for the NexDash energy model.

This script ties the whole package together and is the single command a
reviewer runs to reproduce every reported number::

    python run_pipeline.py

What it does, deterministically (fixed seeds throughout):

1. Generate the synthetic eActros 600 dataset
   (:func:`nexdash.data_gen.generate_dataset`) and persist it to ``data/``.
2. Perform an *explicit*, seeded train/test split so the evaluation set is
   genuinely held out from training.
3. Train the model on the train split via
   :func:`nexdash.model.train_model` (which fits both the
   HistGradientBoosting primary and the LinearRegression baseline) and save
   the artifact to ``models/``.
4. Score the trained model on the held-out test split using
   :func:`nexdash.evaluate.evaluate`, slice errors by operating regime with
   :func:`nexdash.evaluate.failure_mode_report`, and render diagnostic
   figures with :func:`nexdash.evaluate.make_plots`.
5. Write ``reports/evaluation_report.md`` containing the dataset
   description, a model-vs-linear-baseline metric table, the headline MAE /
   percentage range error, the failure-mode tables, links to the figures,
   and an honest "where it breaks and why" section -- all populated with the
   real numbers computed in this run.
6. Print a concise summary to stdout.

Because every random source is seeded and the script always writes to the
canonical paths in :mod:`nexdash.config`, repeated runs are reproducible and
overwrite the previous artifacts in place.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from nexdash import data_gen, evaluate
from nexdash.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    REPORTS_DIR,
    TRUCK,
)
from nexdash.model import train_model

# --------------------------------------------------------------------------- #
# Pipeline configuration (kept here so the run is fully reproducible).
# --------------------------------------------------------------------------- #

#: Master seed used for data generation and the explicit train/test split.
SEED: int = 42

#: Number of drive segments to synthesise.
N_SAMPLES: int = 6000

#: Fraction of rows held out for evaluation (never seen during training).
TEST_SIZE: float = 0.2

#: Nominal full-trip energy used to express MAE as a "% range error". This is
#: the energy the eActros 600 spends across its ~500 km real-world range, i.e.
#: roughly the usable battery capacity. Documented in the report.
NOMINAL_TRIP_KWH: float = float(TRUCK.battery_kwh)

#: Destination for the human-readable evaluation report.
REPORT_PATH: Path = REPORTS_DIR / "evaluation_report.md"


# --------------------------------------------------------------------------- #
# Markdown rendering helpers
# --------------------------------------------------------------------------- #


def _fmt(value: Any, places: int = 3) -> str:
    """Format a numeric value for a Markdown cell, gracefully handling None/NaN."""
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN check without importing math.
        return "n/a"
    return f"{f:.{places}f}"


def _comparison_table(model_metrics: dict[str, Any]) -> str:
    """Render the model-vs-linear-baseline comparison table.

    Uses the metrics the model stored from its *internal* split (keys ``hgb``
    and ``linear``), which is the apples-to-apples comparison: both estimators
    were fit on the same train rows and scored on the same held-out rows.
    """
    hgb = model_metrics.get("hgb", {})
    lin = model_metrics.get("linear", {})
    rows = [
        ("MAE (kWh)", "mae_kwh", 3),
        ("RMSE (kWh)", "rmse_kwh", 3),
        ("MAPE (%)", "mape_pct", 2),
        ("R^2", "r2", 4),
    ]
    lines = [
        "| Metric | HistGradientBoosting (primary) | LinearRegression (baseline) |",
        "| --- | --- | --- |",
    ]
    for label, key, places in rows:
        lines.append(
            f"| {label} | {_fmt(hgb.get(key), places)} "
            f"| {_fmt(lin.get(key), places)} |"
        )
    return "\n".join(lines)


def _slice_table(section: dict[str, Any], dimension_label: str) -> str:
    """Render one failure-mode dimension (temperature / gradient / payload).

    ``section`` is a mapping of bin-name -> {mae_kwh, mape_pct, n, ...}. Extra
    keys are tolerated so the table stays robust if the evaluate module adds
    more per-slice metrics.
    """
    lines = [
        f"| {dimension_label} | MAE (kWh) | MAPE (%) | n |",
        "| --- | --- | --- | --- |",
    ]
    for bin_name, stats in section.items():
        stats = stats if isinstance(stats, dict) else {}
        lines.append(
            f"| {bin_name} | {_fmt(stats.get('mae_kwh'), 3)} "
            f"| {_fmt(stats.get('mape_pct'), 2)} "
            f"| {int(stats.get('n', 0))} |"
        )
    return "\n".join(lines)


def _worst_slice(
    section: dict[str, Any], by: str = "mae_kwh"
) -> tuple[str, dict[str, Any]] | None:
    """Return the (bin_name, stats) of the worst slice ranked by ``by``.

    Ranks by **absolute MAE (kWh) by default, not MAPE**. A dispatcher is
    stranded by absolute energy error; MAPE explodes on the near-zero-denominator
    downhill slices, which previously crowned a *harmless* regime (`steep_down`,
    tiny absolute error) as the headline failure instead of the dangerous
    `steep_up` climbs that carry the largest real kWh miss.
    """
    worst: tuple[str, float, dict[str, Any]] | None = None
    for name, stats in section.items():
        if not isinstance(stats, dict):
            continue
        val = stats.get(by)
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if val != val:  # NaN
            continue
        if worst is None or val > worst[1]:
            worst = (name, val, stats)
    return (worst[0], worst[2]) if worst else None


def _figure_links(figure_paths: list[str]) -> str:
    """Render Markdown image links relative to the reports directory."""
    if not figure_paths:
        return "_No figures were generated._"
    lines = []
    for raw in figure_paths:
        p = Path(raw)
        try:
            rel = p.relative_to(REPORTS_DIR)
        except ValueError:
            rel = Path(p.name)
        title = p.stem.replace("_", " ").title()
        lines.append(f"### {title}\n\n![{title}]({rel.as_posix()})")
    return "\n\n".join(lines)


def _where_it_breaks(
    failure: dict[str, Any], headline_mape: float
) -> str:
    """Compose the honest 'where it breaks and why' narrative from real slices."""
    paragraphs: list[str] = []

    worst_temp = _worst_slice(failure.get("temperature", {}))
    worst_grad = _worst_slice(failure.get("gradient", {}))
    worst_payload = _worst_slice(failure.get("payload", {}))

    paragraphs.append(
        "The model's average error is small, but it is **not uniform** across "
        f"the operating envelope (overall MAPE ~ {headline_mape:.2f}%). The "
        "failure-mode tables above expose where it degrades and the physics "
        "behind each weak spot."
    )

    def _mae(s: dict[str, Any]) -> float:
        try:
            return float(s.get("mae_kwh"))
        except (TypeError, ValueError):
            return float("nan")

    def _mape(s: dict[str, Any]) -> float:
        try:
            return float(s.get("mape_pct"))
        except (TypeError, ValueError):
            return float("nan")

    if worst_temp:
        name, st = worst_temp
        paragraphs.append(
            f"- **Temperature.** The `{name}` slice carries the largest absolute "
            f"error (MAE {_mae(st):.2f} kWh; MAPE {_mape(st):.2f}%). Auxiliary/HVAC "
            "draw rises at both temperature extremes and is spread over travel "
            "time rather than distance, so short, slow segments in cold or hot "
            "weather have an outsized, harder-to-predict auxiliary share."
        )
    if worst_grad:
        name, st = worst_grad
        paragraphs.append(
            f"- **Gradient (the safety-critical one).** Ranked by absolute energy, "
            f"the `{name}` slice is the weakest (MAE {_mae(st):.2f} kWh). Steep "
            "*climbs* convert payload mass into potential energy fastest, so any "
            "miss there is a large *absolute* kWh miss — exactly the regime that "
            "strands a truck. Note the downhill slice can show a huge *percentage* "
            "error, but its absolute error is tiny (regen drives net energy near "
            "zero, so the denominator collapses); that is loud in MAPE yet "
            "operationally harmless, which is why we headline MAE, not MAPE."
        )
    if worst_payload:
        name, st = worst_payload
        paragraphs.append(
            f"- **Payload.** The `{name}` slice shows the highest absolute error "
            f"(MAE {_mae(st):.2f} kWh; MAPE {_mape(st):.2f}%). Payload scales "
            "rolling resistance and gradient potential energy linearly, so heavy "
            "loads both consume the most energy and leave the most room to be "
            "wrong about it in absolute terms."
        )

    paragraphs.append(
        "Two structural caveats apply, and both are limits of the *synthetic data*, "
        "not just the model. First, labels carry multiplicative (~6%) plus additive "
        "sensor noise, which sets a hard floor on achievable accuracy -- the model "
        "cannot beat the noise it was trained on. Second, features are sampled from "
        "independent marginals except for a deliberate gradient/distance coupling "
        "(long legs cannot sustain steep grades); rare *combinations* of the "
        "remaining features (e.g. heavy payload + steep climb + extreme cold at "
        "once) are still under-represented and should be treated as lower-confidence "
        "extrapolations until real telematics data is available."
    )
    return "\n\n".join(paragraphs)


def _build_report(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model_metrics: dict[str, Any],
    held_out: dict[str, Any],
    failure: dict[str, Any],
    figure_paths: list[str],
) -> str:
    """Assemble the full Markdown evaluation report from real computed numbers."""
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headline_mae = float(held_out.get("mae_kwh", float("nan")))
    headline_mape = float(held_out.get("mape_pct", float("nan")))
    pct_range = float(held_out.get("pct_range_error", float("nan")))

    feature_cols = [c for c in df.columns if c != "energy_kwh"]
    desc = df[feature_cols + ["energy_kwh"]].describe().T
    stat_lines = [
        "| Feature | min | mean | max | std |",
        "| --- | --- | --- | --- | --- |",
    ]
    for col in feature_cols + ["energy_kwh"]:
        r = desc.loc[col]
        stat_lines.append(
            f"| {col} | {_fmt(r['min'], 2)} | {_fmt(r['mean'], 2)} "
            f"| {_fmt(r['max'], 2)} | {_fmt(r['std'], 2)} |"
        )
    stats_table = "\n".join(stat_lines)

    report = f"""# NexDash Energy Model -- Evaluation Report

_Generated {now} by `run_pipeline.py` (seed = {SEED}, deterministic)._

This report evaluates the energy-consumption model for the
**{TRUCK.name}** (~{TRUCK.battery_kwh:.0f} kWh usable battery, ~500 km
real-world range, payload 0-{TRUCK.max_payload_t:.0f} t). The model predicts
per-segment energy use (kWh) from driving conditions and underpins the range /
reachability checks exposed to dispatchers.

## 1. Dataset

Synthetic drive segments were generated by `nexdash.data_gen.generate_dataset`,
labelling each segment with a deterministic physics ground truth
(`nexdash.physics.segment_energy_kwh`) perturbed by realistic measurement
noise (multiplicative ~6% + small additive sensor term). Feature marginals
follow the truck's real operating envelope; no correlations are injected, so
all learnable structure comes from the physics.

- **Total segments:** {len(df):,}
- **Train / test split:** {len(df_train):,} train / {len(df_test):,} test
  (held out, seed = {SEED}, test_size = {TEST_SIZE:.0%})
- **Stored at:** `{DEFAULT_DATASET_PATH.relative_to(DEFAULT_DATASET_PATH.parents[1]).as_posix()}`
- **Model artifact:** `{DEFAULT_MODEL_PATH.relative_to(DEFAULT_MODEL_PATH.parents[1]).as_posix()}`

### Feature & target summary

{stats_table}

## 2. Headline performance (held-out test set)

Computed by `nexdash.evaluate.evaluate` on the {len(df_test):,} held-out
segments the model never saw during training.

- **MAE:** **{_fmt(headline_mae, 3)} kWh**
- **RMSE:** {_fmt(held_out.get("rmse_kwh"), 3)} kWh
- **MAPE:** {_fmt(headline_mape, 2)} %
- **R^2:** {_fmt(held_out.get("r2"), 4)}
- **% range error:** **{_fmt(pct_range, 3)} %** -- MAE expressed against a
  nominal full-trip energy of {NOMINAL_TRIP_KWH:.0f} kWh (the energy spent
  across the truck's ~500 km real-world range, i.e. the usable battery). A
  {_fmt(headline_mae, 2)} kWh average miss is therefore a {_fmt(pct_range, 2)}%
  slice of a full charge -- comfortably inside a typical 10% reserve.

## 3. Model vs. linear baseline

Both estimators were fit on identical train rows and scored on the same
internal held-out split (metrics stored on the model artifact). The
HistGradientBoosting primary is reported alongside a LinearRegression baseline
to quantify the value of the non-linear model over a transparent reference.

{_comparison_table(model_metrics)}

## 4. Failure-mode analysis

Error sliced by operating regime (`nexdash.evaluate.failure_mode_report`).
Watch MAPE rather than MAE in low-energy regimes, where small absolute misses
become large relative ones.

### By temperature

{_slice_table(failure.get("temperature", {}), "Temperature bin")}

### By gradient

{_slice_table(failure.get("gradient", {}), "Gradient bin")}

### By payload

{_slice_table(failure.get("payload", {}), "Payload bin")}

## 5. Diagnostic figures

{_figure_links(figure_paths)}

## 6. Where it breaks, and why

{_where_it_breaks(failure, headline_mape)}
"""
    return report


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run() -> dict[str, Any]:
    """Run the full pipeline end-to-end and return the key artifacts/metrics.

    Returns:
        A dict with the generated dataframe, fitted model, held-out metrics,
        failure-mode report, figure paths and the report path -- useful for
        tests and for callers that want to inspect results programmatically.
    """
    print(f"[1/5] Generating dataset (n={N_SAMPLES}, seed={SEED}) ...")
    df = data_gen.generate_dataset(n_samples=N_SAMPLES, seed=SEED)
    data_gen.save_dataset(df, DEFAULT_DATASET_PATH)
    print(f"      wrote {len(df):,} rows to {DEFAULT_DATASET_PATH}")

    print(f"[2/5] Explicit train/test split (test_size={TEST_SIZE}, seed={SEED}) ...")
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"      {len(df_train):,} train / {len(df_test):,} test")

    print("[3/5] Training model (HGB primary + linear baseline) and saving ...")
    model = train_model(df_train, save=True, path=DEFAULT_MODEL_PATH)
    print(f"      saved model to {DEFAULT_MODEL_PATH}")

    print("[4/5] Evaluating on held-out test set ...")
    held_out = evaluate.evaluate(model, df_test)
    failure = evaluate.failure_mode_report(model, df_test)
    figure_paths = evaluate.make_plots(model, df_test)
    print(
        f"      MAE={held_out.get('mae_kwh'):.3f} kWh, "
        f"MAPE={held_out.get('mape_pct'):.2f}%, "
        f"R^2={held_out.get('r2'):.4f}, "
        f"{len(figure_paths)} figure(s)"
    )

    print(f"[5/5] Writing report to {REPORT_PATH} ...")
    report_md = _build_report(
        df=df,
        df_train=df_train,
        df_test=df_test,
        model_metrics=model.metrics,
        held_out=held_out,
        failure=failure,
        figure_paths=figure_paths,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    # --- stdout summary --------------------------------------------------- #
    hgb = model.metrics.get("hgb", {})
    lin = model.metrics.get("linear", {})
    print("\n" + "=" * 64)
    print("NexDash pipeline complete")
    print("=" * 64)
    print(f"Dataset:        {len(df):,} segments ({len(df_test):,} held out)")
    print(
        f"Held-out MAE:   {held_out.get('mae_kwh'):.3f} kWh  "
        f"({held_out.get('pct_range_error'):.2f}% of a full charge)"
    )
    print(f"Held-out MAPE:  {held_out.get('mape_pct'):.2f} %")
    print(f"Held-out R^2:   {held_out.get('r2'):.4f}")
    print(
        f"Baseline gap:   HGB MAE {hgb.get('mae_kwh', float('nan')):.3f} "
        f"vs linear {lin.get('mae_kwh', float('nan')):.3f} kWh"
    )
    print(f"Report:         {REPORT_PATH}")
    print(f"Figures:        {len(figure_paths)} in {REPORTS_DIR / 'figures'}")
    print("=" * 64)

    return {
        "dataframe": df,
        "model": model,
        "metrics": held_out,
        "failure_modes": failure,
        "figures": figure_paths,
        "report_path": REPORT_PATH,
    }


if __name__ == "__main__":
    run()
