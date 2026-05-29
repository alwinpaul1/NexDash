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
    """

    # Values below are calibrated to real Mercedes-Benz eActros 600 figures and
    # defensible literature values. See docs/REAL_WORLD_CALIBRATION.md for the
    # full [S1]-[S13] source list and per-parameter justification.
    name: str = "Mercedes-Benz eActros 600"
    battery_kwh: float = 600.0  # usable; 621 kWh installed (3x207 LFP). [S1][S2][S3]
    max_payload_t: float = 22.0  # ~22 t with std EU semitrailer; GCW up to 44 t. [S1][S2]
    kerb_mass_kg: float = 18000.0  # loaded-rig baseline (tractor ~11.7 t + empty trailer); +22 t = 40 t GCW. [S3]
    frontal_area_m2: float = 10.0  # literature value for EU tractor-semitrailer (not published). [S3][S8]
    cd: float = 0.55  # std aero tractor-trailer drag; consistent with ProCabin -9% cW claim. [S1][S6]
    crr: float = 0.0055  # long-haul LRR truck tyres ~0.005-0.007; tuned to the measured band. [S2][S8]
    drivetrain_eff: float = 0.85  # battery-to-wheel for 800 V e-axle at cruise. [S1]
    regen_eff: float = 0.60  # ~50-70% braking capture, ~25% favourable-stage recovery. [S4][S5][S11]
    aux_base_kw: float = 2.0  # steady electronics/aux floor on top of U-shaped HVAC. [S10][S12]


#: Canonical truck specification used throughout the package.
TRUCK = Truck()

# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

#: Air density at sea level / mild conditions (kg/m^3).
AIR_DENSITY: float = 1.225

#: Standard gravitational acceleration (m/s^2).
G: float = 9.81

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
    "ROOT_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_DATASET_PATH",
]
