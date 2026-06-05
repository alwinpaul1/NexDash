"""Central configuration for NexDash.

This module is imported by virtually every other module in the package, so it
deliberately has no heavy dependencies. It defines:

* the :data:`TRUCK` specification (a frozen dataclass modelling the
  Mercedes-Benz eActros 600),
* shared physical constants (air density, gravity),
* canonical filesystem paths for data, models and reports.

The data/model/report directories are created on import so that downstream
code can write to them without first checking for their existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Truck specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Truck:
    """Physical / drivetrain specification of a battery-electric truck.

    Defaults model the Mercedes-Benz eActros 600: ~600 kWh usable battery,
    ~500 km real-world range, payload up to 22 t, GVW up to ~40 t.

    Attributes:
        name: Human-readable model name.
        battery_kwh: Usable battery capacity (kWh).
        max_payload_t: Maximum payload (tonnes).
        kerb_mass_kg: Empty/kerb mass of the tractor unit (kg).
        frontal_area_m2: Effective frontal area for aerodynamic drag (m^2).
        cd: Aerodynamic drag coefficient (dimensionless).
        crr: Coefficient of rolling resistance (dimensionless).
        drivetrain_eff: Battery-to-wheel drivetrain efficiency (0-1).
        regen_eff: Fraction of available braking/downhill energy recovered (0-1).
        aux_base_kw: Baseline auxiliary/HVAC power draw at mild temperature (kW).
        nominal_range_km: Rated real-world range on a usable charge (km). Used
            only as a sane flat-consumption floor when extrapolating a
            *remaining* range from a single segment (a regen-dominated descent's
            near-zero rate cannot be sustained over further distance).
    """

    # Values below are calibrated to real Mercedes-Benz eActros 600 figures and
    # defensible literature values. See docs/REAL_WORLD_CALIBRATION.md for the
    # full [S1]-[S13] source list and per-parameter justification.
    name: str = "Mercedes-Benz eActros 600"
    battery_kwh: float = 600.0  # usable; 621 kWh installed (3x207 LFP). [S1][S2][S3]
    max_payload_t: float = 22.0  # ~22 t with std EU semitrailer; GCW up to 44 t. [S1][S2]
    kerb_mass_kg: float = 18000.0  # loaded-rig baseline (tractor ~11.7 t + empty trailer); +22 t = 40 t GCW. [S3]
    frontal_area_m2: float = 10.0  # literature value for EU tractor-semitrailer (not published). [S3][S8]
    cd: float = 0.50  # ProCabin: generic 0.55 tractor-trailer x 0.91 (-9% cW). CdA 5.0. [S1][S6]
    crr: float = 0.0055  # long-haul LRR truck tyres ~0.005-0.007; tuned to the measured band. [S2][S8]
    drivetrain_eff: float = 0.85  # battery-to-wheel for 800 V e-axle at cruise. [S1]
    regen_eff: float = 0.60  # ~50-70% braking capture, ~25% favourable-stage recovery. [S4][S5][S11]
    aux_base_kw: float = 2.0  # steady electronics/aux floor on top of U-shaped HVAC. [S10][S12]
    nominal_range_km: float = 500.0  # ~500 km real-world on a usable charge. [S1][S2]


#: Canonical truck specification used throughout the package.
TRUCK = Truck()

# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

#: ISA sea-level air density at 15 C (kg/m^3). Retained as the reference / pivot:
#: the temperature-dependent :func:`nexdash.physics._air_density` reproduces this
#: exactly at 15 C, so the upgraded model stays continuous with the old constant.
AIR_DENSITY: float = 1.225

#: Standard sea-level pressure (Pa) and specific gas constant of dry air
#: (J/kg/K), used to compute temperature-dependent air density from the ideal
#: gas law ``rho = P / (R * T_kelvin)``. [ISO 2533 International Standard
#: Atmosphere; R_specific = R_universal / M_dry_air = 287.05.]
P_SEA_LEVEL_PA: float = 101325.0
R_SPECIFIC_DRY_AIR: float = 287.05
T_KELVIN_OFFSET: float = 273.15

#: Standard gravitational acceleration (m/s^2).
G: float = 9.81

#: Minimum |actual energy| (kWh) for a row to participate in MAPE. Shared by the
#: model's comparison metrics and the evaluation report so every MAPE figure in
#: the project uses ONE definition (avoids a 9% headline vs 16% table mismatch).
#: Set above 1 kWh to exclude near-zero net-regen downhill rows whose tiny
#: denominators would make MAPE explode meaninglessly.
MAPE_FLOOR_KWH: float = 1.0

#: CONSTANT field-calibration factor mapping the displayed STEADY-STATE energy DOWN
#: to field-observed laden eActros 600 consumption. The displayed headline is
#: ``max(model, physics)`` summed per chunk (physics-dominated) x this factor. At the
#: 40 t / 80 km/h / 20 C / flat anchor that raw figure is ~121.6 kWh/100km, and
#: 0.78 x 121.6 = ~95 — NexDash's NexOS flat field-real centre. The displayed total
#: still varies per route because the raw physics+ML does (it integrates gradient,
#: wind, temperature, payload and speed per chunk), so a constant still yields a
#: per-route-varying headline.
#:
#: HISTORY: 0.887 (Daimler tour anchor) -> 0.83 (2026-06-05, NexOS anchor, but
#: mistakenly computed against the model's 113.88 not the displayed physics 121.6, so
#: it actually showed ~101) -> a ROUTE-AWARE multiplier (PRs #108/#109) -> REVERTED
#: to a constant 0.78 (2026-06-05) after adversarial testing against real field data.
#: Why reverted: the needed field/raw ratio FALLS on hard routes (hilly Vandijck
#: needs ~0.48, flat ~0.71), because the steady-state physics — recovering only ~60%
#: regen — OVER-responds to terrain/payload/cold vs the compressed real field band
#: (~0.85-1.40 kWh/km). A route-aware factor that ROSE on hard routes was therefore
#: backwards (MAE 27 vs 13 for a constant). KNOWN LIMITATION: even a constant cannot
#: reconcile the physics over-spread; a hilly/heavy/cold route still reads somewhat
#: high (the conservative direction). The real fix is the REMOVAL CONDITION below.
#:
#: Applied ONLY to the DISPLAYED energy headline (summary.energyKwh / kwhPer100); the
#: SOC walk and EVERY charging/reachability decision use the un-discounted
#: conservative max(model, physics) estimate, so the factor can never delay a charge
#: or strand the truck. Clamped to (0, 1] at the call site (>1 cannot inflate energy);
#: 1.0 disables it. REMOVAL CONDITION: retune or remove once the ML model is retrained
#: against real FIELD (not steady-state) labels — that would fix the terrain/payload
#: over-spread directly and make the multiplier unnecessary.
#: [S3][S4][S5] (see docs/REAL_WORLD_CALIBRATION.md)
FIELD_CALIBRATION_FACTOR: float = 0.78

# --------------------------------------------------------------------------- #
# Filesystem paths
# --------------------------------------------------------------------------- #

#: Repository root (two levels up from this file: src/nexdash/config.py).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

#: Default location of the trained energy model artifact.
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "energy_model.joblib"

#: Default location of the generated dataset.
DEFAULT_DATASET_PATH: Path = DATA_DIR / "dataset.csv"

# Ensure the working directories exist so writers never have to check.
for _directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "figures"):
    _directory.mkdir(parents=True, exist_ok=True)

__all__ = [
    "Truck",
    "TRUCK",
    "AIR_DENSITY",
    "G",
    "FIELD_CALIBRATION_FACTOR",
    "ROOT_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_DATASET_PATH",
]
