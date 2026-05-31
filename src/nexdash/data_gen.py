"""Synthetic dataset generation for the NexDash energy model.

This module produces a labelled training/evaluation dataset for the
Mercedes-Benz eActros 600 energy-consumption model. Each row describes a
single drive *segment* (its operating conditions) together with the energy
consumed over that segment.

The ground-truth label is computed by :func:`nexdash.physics.segment_energy_kwh`
and then perturbed with realistic measurement noise:

* a *multiplicative* gaussian factor ``(1 + N(0, noise_frac))`` modelling the
  proportional uncertainty of real-world driving (driver behaviour, traffic,
  road surface, tyre pressure, etc.), and
* a small *additive* gaussian term ``N(0, 0.3)`` kWh modelling sensor/telemetry
  measurement error.

Feature values are drawn from realistic marginal distributions. The only
deliberate coupling is **physical, not statistical**: the average gradient a
segment can sustain is attenuated as its distance grows, because a long leg
cannot hold a steep grade without implying an impossible net elevation change
(see the ``gradient_pct`` sampling below). No *other* correlations are injected;
any further structure the model learns comes from the physics relationship and
the genuine variability of the operating envelope, not from synthetic shortcuts.

.. note::
   This generator is a deliberate *stand-in* for future real telematics data.
   When live fleet telemetry from eActros 600 units becomes available it should
   replace this module wholesale; the column schema (see below) is the contract
   that keeps the rest of the pipeline unchanged.

The produced :class:`pandas.DataFrame` has exactly these columns, in order:

    ``distance_km, payload_t, speed_kph, gradient_pct, temperature_c,
    wind_mps, energy_kwh``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nexdash.config import DEFAULT_DATASET_PATH, TRUCK
from nexdash.physics import segment_energy_kwh

#: Column order of the generated dataset (features first, target last).
COLUMNS: list[str] = [
    "distance_km",
    "payload_t",
    "speed_kph",
    "gradient_pct",
    "temperature_c",
    "wind_mps",
    "energy_kwh",
]

#: Lower bound on the energy label (kWh). A real BEV energy meter CAN log a
#: *net-negative* segment on a long steep descent (regen returns more charge than
#: the segment spends), so we deliberately do NOT clamp at zero: clamping would
#: erase the regenerative-braking signal, bias the steep-down failure slice by
#: ~+10 kWh, and inject gradient-correlated structure into the label noise. We
#: only reject numerically absurd values far below any plausible regen return.
_ENERGY_FLOOR_KWH: float = -50.0

#: Maximum implied net elevation change (m) for a single segment. The gradient is
#: capped per row so ``distance * sin(atan(gradient/100))`` never exceeds this,
#: keeping labels geographically plausible (German Autobahn tops out ~1000 m) and
#: preventing the impossible multi-kilometre climbs a naive sampler produces on
#: long legs.
_MAX_NET_CLIMB_M: float = 1000.0


def generate_dataset(
    n_samples: int = 6000,
    seed: int = 42,
    noise_frac: float = 0.06,
) -> pd.DataFrame:
    """Generate a synthetic eActros 600 energy-consumption dataset.

    Args:
        n_samples: Number of drive segments (rows) to generate.
        seed: Seed for :func:`numpy.random.default_rng` (reproducibility).
        noise_frac: Standard deviation of the multiplicative gaussian noise
            applied to the physics ground truth (e.g. ``0.06`` == 6%).

    Returns:
        A :class:`pandas.DataFrame` with the columns listed in
        :data:`COLUMNS`. Feature distributions follow the realistic operating
        envelope of the truck; ``energy_kwh`` is the noisy observed label.
    """
    rng = np.random.default_rng(seed)

    # --- Feature sampling (independent, realistic marginals) --------------- #
    # See docs/REAL_WORLD_CALIBRATION.md for the German-ops calibration.
    # Distances skew toward regional legs with a long right tail for inter-hub
    # runs (250-350 km between chargers; EU 561/2006 break rules), 1-350 km.
    distance_km = np.clip(rng.gamma(shape=2.0, scale=30.0, size=n_samples), 1.0, 350.0)

    # Payload is uniform across the legal range, including frequent empty runs.
    payload_t = rng.uniform(0.0, TRUCK.max_payload_t, size=n_samples)

    # Average segment speed: German Lkw Autobahn limit 80 km/h, limiter 90 (not
    # legally driveable loaded). Cluster 70-80 on Autobahn, bounded 30-85 kph.
    speed_kph = np.clip(rng.normal(loc=72.0, scale=12.0, size=n_samples), 30.0, 85.0)

    # Road gradient is symmetric about flat; most segments are gentle.
    #
    # CRITICAL realism constraint: the *average* gradient a real leg can sustain
    # shrinks as the leg lengthens. A 3 km ramp can average +6 %, but a 110 km
    # leg averaging +4.5 % would imply a ~5 km net climb — higher than any
    # German road reaches. We therefore sample a symmetric spread and then cap
    # |gradient| per row so the implied net elevation change
    # (``distance * sin(atan(gradient/100))``) never exceeds ``_MAX_NET_CLIMB_M``.
    # Short legs keep the full +/-6 % spread (which populates the steep-grade
    # failure-mode slices); long legs are forced near-flat, exactly as real
    # inter-hub Autobahn runs are. This is the exact physical version of the
    # earlier distance-attenuation heuristic and, unlike it, the net-climb bound
    # holds for every seed and sample size — no phantom-mountain climbs. (Note: a
    # rare long + heavy + cold + headwind leg can still legitimately need MORE than
    # one charge of energy; that is a real "must charge mid-route" segment, not a
    # bug, so labels are not clamped to the battery capacity.)
    raw_gradient = np.clip(rng.normal(loc=0.0, scale=2.8, size=n_samples), -6.0, 6.0)
    distance_m = distance_km * 1000.0
    # Largest |grade| (%) whose net climb over this leg stays within the ceiling.
    climb_ratio = np.minimum(1.0, _MAX_NET_CLIMB_M / distance_m)
    grad_cap_pct = np.minimum(100.0 * np.tan(np.arcsin(climb_ratio)), 6.0)
    gradient_pct = np.clip(raw_gradient, -grad_cap_pct, grad_cap_pct)

    # German ambient temperature across the year (-15..40 C).
    temperature_c = np.clip(rng.normal(loc=12.0, scale=11.0, size=n_samples), -15.0, 40.0)

    # Wind enters the model as a *signed headwind component* (positive = headwind
    # opposing travel, negative = tailwind), because at inference we project real
    # Open-Meteo wind direction onto the truck's heading. We mirror that here: draw
    # a light wind magnitude (gamma) and a uniformly-random bearing relative to
    # travel, then take the along-track component speed*cos(angle). This teaches the
    # model that tailwinds *reduce* energy — without it the model would never see
    # negative wind and would extrapolate on every following wind.
    wind_speed = np.clip(rng.gamma(shape=2.0, scale=2.0, size=n_samples), 0.0, 14.0)
    wind_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    wind_mps = np.clip(wind_speed * np.cos(wind_angle), -12.0, 12.0)

    # --- Ground-truth physics label --------------------------------------- #
    # segment_energy_kwh is scalar; vectorise over the sampled rows.
    truth_kwh = np.fromiter(
        (
            segment_energy_kwh(
                distance_km=float(d),
                payload_t=float(p),
                speed_kph=float(s),
                gradient_pct=float(g),
                temperature_c=float(t),
                wind_mps=float(w),
                truck=TRUCK,
            )
            for d, p, s, g, t, w in zip(
                distance_km, payload_t, speed_kph, gradient_pct, temperature_c, wind_mps
            )
        ),
        dtype=float,
        count=n_samples,
    )

    # --- Observation noise ------------------------------------------------- #
    multiplicative = 1.0 + rng.normal(loc=0.0, scale=noise_frac, size=n_samples)
    additive = rng.normal(loc=0.0, scale=0.3, size=n_samples)
    energy_kwh = np.clip(truth_kwh * multiplicative + additive, _ENERGY_FLOOR_KWH, None)

    df = pd.DataFrame(
        {
            "distance_km": distance_km,
            "payload_t": payload_t,
            "speed_kph": speed_kph,
            "gradient_pct": gradient_pct,
            "temperature_c": temperature_c,
            "wind_mps": wind_mps,
            "energy_kwh": energy_kwh,
        }
    )
    return df[COLUMNS]


def save_dataset(df: pd.DataFrame, path: str | Path = DEFAULT_DATASET_PATH) -> None:
    """Persist a generated dataset to CSV.

    Args:
        df: The dataset to write (as returned by :func:`generate_dataset`).
        path: Destination CSV path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m nexdash.data_gen --n 6000 --out <path>``."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic eActros 600 energy-consumption dataset."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6000,
        help="Number of segments (rows) to generate (default: 6000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Output CSV path (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--noise-frac",
        type=float,
        default=0.06,
        help="Multiplicative noise std-dev as a fraction (default: 0.06).",
    )
    args = parser.parse_args(argv)

    df = generate_dataset(n_samples=args.n, seed=args.seed, noise_frac=args.noise_frac)
    save_dataset(df, args.out)
    print(
        f"Wrote {len(df):,} rows to {args.out} "
        f"(energy_kwh mean={df['energy_kwh'].mean():.2f}, "
        f"min={df['energy_kwh'].min():.2f}, max={df['energy_kwh'].max():.2f})."
    )


if __name__ == "__main__":
    main()
