"""Which input matters most to energy consumption? — a sensitivity sweep.

Run live to SHOW (not assert) the ranking, straight from the deployed predictor
(physics + ML residual). Method: hold a realistic baseline segment, vary ONE
input at a time across its full operating range, and measure the resulting swing
in energy (kWh / 100 km). The input with the biggest swing matters most.

Why a sweep and not the model's feature_importances_? Because the ML half predicts
the *residual* (the correction to physics), so its importances describe the small
correction, not total consumption. Sweeping the full physics+ML stack is the
honest way to answer "what moves consumption."

Usage:
    python scripts/feature_sensitivity.py            # table only
    python scripts/feature_sensitivity.py --chart    # also save a tornado chart
"""

from __future__ import annotations

import argparse

from nexdash.model import predict_energy

# A realistic mid-load motorway baseline. Energy is reported per 100 km, so the
# segment distance is 100 km and the raw kWh IS the kWh/100 km figure.
DIST_KM = 100.0
BASELINE = dict(
    distance_km=DIST_KM,
    payload_t=11.0,
    speed_kph=70.0,
    gradient_pct=0.0,
    temperature_c=15.0,
    wind_mps=0.0,
)

# Each input's full operating range (the same envelope the model was trained on).
SWEEPS = [
    ("Elevation (gradient)", "gradient_pct", [-6, -3, 0, 3, 6], "%"),
    ("Speed", "speed_kph", [20, 40, 60, 80, 90], "km/h"),
    ("Payload", "payload_t", [0, 5, 11, 16, 22], "t"),
    ("Temperature", "temperature_c", [-15, 0, 15, 30, 40], "°C"),
]


def _energy(**override: float) -> float:
    """kWh per 100 km for the baseline segment with one input overridden."""
    return predict_energy({**BASELINE, **override})


def run() -> list[tuple[str, float, float, float, str, list[tuple[float, float]]]]:
    base = _energy()
    rows = []
    for name, key, vals, unit in SWEEPS:
        pts = [(v, _energy(**{key: v})) for v in vals]
        energies = [e for _, e in pts]
        lo, hi = min(energies), max(energies)
        rows.append((name, hi - lo, lo, hi, unit, pts))
    rows.sort(key=lambda r: -r[1])  # biggest swing first
    return base, rows


def print_table(base: float, rows) -> None:
    print(f"\nBaseline: {DIST_KM:.0f} km, 11 t, 70 km/h, flat, 15 °C  ->  "
          f"{base:.1f} kWh/100 km\n")
    print("Which input moves consumption most? (swing across its full range)\n")
    print(f"  {'rank':<5}{'input':<22}{'swing kWh/100km':>16}   range")
    for i, (name, swing, lo, hi, unit, _) in enumerate(rows, 1):
        print(f"  {i:<5}{name:<22}{swing:>16.1f}   {lo:6.1f} -> {hi:6.1f}")
    print("\nPer-point detail:")
    for name, swing, lo, hi, unit, pts in rows:
        detail = "  ".join(f"{v:>5}{unit}:{e:6.1f}" for v, e in pts)
        print(f"  {name:<22} {detail}")
    print(
        "\nNote: sustained +/-6%% over 100 km is geographically impossible (a long "
        "leg\naverages near-flat). Elevation is still the biggest PHYSICAL driver; "
        "on a flat\nhighway leg, SPEED is the biggest lever a dispatcher actually "
        "controls.\n"
    )


def save_chart(rows, path: str = "docs/feature_sensitivity.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r[0] for r in rows]
    swings = [r[1] for r in rows]
    green = "#1f9d57"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # Left: tornado bar chart — the answer at a glance.
    y = range(len(names))
    ax1.barh(list(y), swings, color=green)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Energy swing (kWh / 100 km)")
    ax1.set_title("Which input matters most", fontweight="bold", loc="left")
    for i, s in enumerate(swings):
        ax1.text(s, i, f" {s:.0f}", va="center", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: the sweep curves — shows elevation's steepness + sign change.
    for name, _, _, _, unit, pts in rows:
        xs = [v for v, _ in pts]
        ys = [e for _, e in pts]
        ax2.plot(xs, ys, marker="o", label=f"{name} ({unit})")
    ax2.axhline(0, color="#bbb", lw=0.8)
    ax2.set_xlabel("Input value (each over its own range)")
    ax2.set_ylabel("Energy (kWh / 100 km)")
    ax2.set_title("Sweep curves (others held at baseline)", fontweight="bold", loc="left")
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "eActros 600 energy sensitivity (physics + ML model)",
        fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    print(f"Chart saved -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", action="store_true", help="also save a tornado chart")
    args = ap.parse_args()
    base, rows = run()
    print_table(base, rows)
    if args.chart:
        save_chart(rows)


if __name__ == "__main__":
    main()
