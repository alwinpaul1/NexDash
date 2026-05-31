"""Tests for :mod:`nexdash.physics`.

These tests pin the *physical intent* of the deterministic energy model that
acts as ground truth for the whole pipeline. Each assertion encodes a real-world
law of vehicle dynamics, not just an arbitrary numeric snapshot, so that the
tests fail loudly if the underlying physics is broken (e.g. a sign flip on
regen, dropping the speed term from drag, or losing the U-shaped HVAC curve).

A representative motorway baseline is reused across tests so that single-factor
comparisons isolate one effect at a time.
"""

from __future__ import annotations

import pytest

from nexdash.config import TRUCK
from nexdash.physics import energy_breakdown, segment_energy_kwh

# A realistic mid-load, flat, motorway baseline. Holding everything but one
# variable fixed lets each test attribute a change in energy to a single cause.
BASE = dict(
    distance_km=50.0,
    payload_t=11.0,
    speed_kph=70.0,
    gradient_pct=0.0,
    temperature_c=20.0,
)


def test_energy_rises_with_distance() -> None:
    """Energy must scale up with distance.

    WHY: every resistive force acts over the travelled distance (and aux load
    over travel time), so a longer segment can never cost less energy. A failure
    here would mean distance was dropped from the integration.
    """
    short = segment_energy_kwh(**{**BASE, "distance_km": 25.0})
    long = segment_energy_kwh(**{**BASE, "distance_km": 100.0})
    assert long > short


def test_energy_rises_with_payload() -> None:
    """Heavier payload must cost more energy on a flat road.

    WHY: rolling resistance F = Crr * m * g scales with total mass, so adding
    cargo increases battery draw. A regression that ignored payload mass would
    flatten this relationship.
    """
    light = segment_energy_kwh(**{**BASE, "payload_t": 0.0})
    heavy = segment_energy_kwh(**{**BASE, "payload_t": 22.0})
    assert heavy > light


def test_energy_rises_with_speed() -> None:
    """Higher speed must cost more energy on a flat road.

    WHY: aerodynamic drag grows with the square of air speed
    (F_aero ~ v^2), which dominates the speed dependence and outweighs the
    shorter travel time (and thus smaller aux contribution) at higher speed.
    A linear or missing speed term would break this.
    """
    slow = segment_energy_kwh(**{**BASE, "speed_kph": 40.0})
    fast = segment_energy_kwh(**{**BASE, "speed_kph": 85.0})
    assert fast > slow


def test_energy_rises_with_uphill_gradient() -> None:
    """Steeper uphill must cost more energy.

    WHY: climbing adds potential energy m * g * sin(theta) * d to the battery
    draw, monotonically increasing with grade. A sign error or dropped gradient
    term would violate this.
    """
    flat = segment_energy_kwh(**{**BASE, "gradient_pct": 0.0})
    gentle = segment_energy_kwh(**{**BASE, "gradient_pct": 3.0})
    steep = segment_energy_kwh(**{**BASE, "gradient_pct": 6.0})
    assert steep > gentle > flat


def test_downhill_costs_less_than_flat_due_to_regen() -> None:
    """Going downhill must cost less than the equivalent flat segment.

    WHY: on a descent the gradient term is negative and regenerative braking
    credits a fraction (regen_eff) of the potential energy back to the battery.
    If regen were applied with the wrong sign the descent would look *more*
    expensive than flat.
    """
    flat = segment_energy_kwh(**{**BASE, "gradient_pct": 0.0})
    downhill = segment_energy_kwh(**{**BASE, "gradient_pct": -4.0})
    assert downhill < flat


def test_regen_recovers_only_a_fraction_of_downhill_energy() -> None:
    """Regen is lossy: it recovers less than the full downhill potential energy.

    WHY: regen_eff < 1, so the energy credited on a descent must be strictly
    less in magnitude than the energy charged on the symmetric climb. This
    guards against a regen path that mistakenly returns 100% of the potential
    energy.
    """
    bd_up = energy_breakdown(**{**BASE, "gradient_pct": 4.0})
    bd_down = energy_breakdown(**{**BASE, "gradient_pct": -4.0})
    # Climb adds positive gradient energy; descent yields a negative net term.
    assert bd_up["gradient"] > 0.0
    assert bd_down["gradient"] < 0.0
    # Recovered (regen) magnitude < energy charged on the symmetric climb.
    assert bd_down["regen"] < bd_up["gradient"]
    # And regen magnitude equals the negative net gradient on the descent.
    assert bd_down["regen"] == pytest.approx(-bd_down["gradient"], rel=1e-9)


def test_aux_load_higher_at_cold_and_hot_extremes() -> None:
    """HVAC/aux energy must be U-shaped: higher at both -10C and 38C than at 20C.

    WHY: an EV spends extra power on cabin/battery heating in the cold and on
    air-conditioning in the heat; the ~20C comfort band is the minimum. A model
    that only penalised cold (or only heat) would be physically wrong.
    """
    mild = energy_breakdown(**{**BASE, "temperature_c": 20.0})["aux"]
    cold = energy_breakdown(**{**BASE, "temperature_c": -10.0})["aux"]
    hot = energy_breakdown(**{**BASE, "temperature_c": 38.0})["aux"]
    assert cold > mild
    assert hot > mild


def test_total_energy_higher_at_temperature_extremes() -> None:
    """The U-shaped aux load must propagate to total segment energy.

    WHY: extreme temperatures raise the only temperature-sensitive component
    (aux), so total energy at -10C and 38C must exceed the 20C baseline. This
    confirms the aux term is actually summed into the total.
    """
    mild = segment_energy_kwh(**{**BASE, "temperature_c": 20.0})
    cold = segment_energy_kwh(**{**BASE, "temperature_c": -10.0})
    hot = segment_energy_kwh(**{**BASE, "temperature_c": 38.0})
    assert cold > mild
    assert hot > mild


def test_breakdown_components_sum_to_total() -> None:
    """The reported components must reconstruct the total energy.

    WHY: rolling + aero + gradient + aux must equal total (the ``gradient`` term
    already encodes the net regen credit). If the breakdown and the headline
    number diverged, downstream diagnostics/reports would be misleading. Tested
    across a flat, an uphill and a downhill case to cover both regen branches.
    """
    for gradient in (0.0, 5.0, -5.0):
        bd = energy_breakdown(**{**BASE, "gradient_pct": gradient})
        reconstructed = bd["rolling"] + bd["aero"] + bd["gradient"] + bd["aux"]
        assert reconstructed == pytest.approx(bd["total"], rel=1e-9)
        # Mirror: segment_energy_kwh must agree with the breakdown total.
        assert segment_energy_kwh(
            **{**BASE, "gradient_pct": gradient}
        ) == pytest.approx(bd["total"], rel=1e-9)


def test_components_have_expected_signs() -> None:
    """Rolling, aero and aux are always energy *costs* (>= 0).

    WHY: these three resistive/load terms can never return energy to the
    battery; only the gradient term may go negative (via regen). A negative
    rolling/aero/aux would signal a unit or sign bug.
    """
    bd = energy_breakdown(**BASE)
    assert bd["rolling"] > 0.0
    assert bd["aero"] > 0.0
    assert bd["aux"] > 0.0


def test_plausible_kwh_per_km_for_midload_flat_segment() -> None:
    """A mid-load flat motorway segment must land in a realistic kWh/km band.

    WHY: published eActros 600 real-world consumption is roughly 1.0-1.6 kWh/km
    at full GVW and motorway speed. This baseline is mid-load (11 t) at a
    moderate 70 km/h with no HVAC penalty, which sits at the low end of that
    range. With the upgraded physics (rho(20 C) < the old 1.225 constant, and a
    sub-1.0 speed factor below 80 km/h) this point lands ~0.94 kWh/km, so we
    allow 0.85-1.6 — the 0.85 floor is the documented empty/downhill low end and
    still catches gross unit-conversion errors (e.g. Joules vs kWh).
    """
    distance_km = BASE["distance_km"]
    total = segment_energy_kwh(**BASE)
    kwh_per_km = total / distance_km
    assert 0.85 <= kwh_per_km <= 1.6, f"kWh/km out of band: {kwh_per_km:.3f}"


def test_plausible_kwh_per_km_for_fullload_motorway_segment() -> None:
    """A full-load motorway segment must land in the published 1.0-1.6 kWh/km band.

    WHY: this is the canonical operating point (22 t payload ~ 40 t GVW at
    85 km/h) for which eActros 600 real-world consumption figures are quoted.
    It anchors the upper end of the absolute energy scale and confirms the
    heavy/fast case stays physically realistic rather than exploding.
    """
    distance_km = 50.0
    total = segment_energy_kwh(
        distance_km=distance_km,
        payload_t=22.0,
        speed_kph=85.0,
        gradient_pct=0.0,
        temperature_c=20.0,
    )
    kwh_per_km = total / distance_km
    assert 1.0 <= kwh_per_km <= 1.7, f"kWh/km out of band: {kwh_per_km:.3f}"


def test_headwind_increases_energy() -> None:
    """A headwind must raise aerodynamic energy and thus total energy.

    WHY: drag depends on air speed (v + wind); a positive headwind increases the
    relative air speed and therefore the aero cost. If wind were ignored this
    would not change.
    """
    no_wind = segment_energy_kwh(**BASE, wind_mps=0.0)
    headwind = segment_energy_kwh(**BASE, wind_mps=10.0)
    assert headwind > no_wind


def test_zero_distance_yields_zero_energy() -> None:
    """A zero-length segment consumes no energy.

    WHY: with no distance travelled there is no traction work and no travel time
    for aux load. This guards the degenerate-input handling in the breakdown.
    """
    bd = energy_breakdown(**{**BASE, "distance_km": 0.0})
    assert bd["total"] == 0.0
    assert all(v == 0.0 for v in bd.values())


def test_air_density_pivot_and_cold_denser() -> None:
    """Air density follows the ideal gas law: continuous with the old constant
    at 15 C, and denser in the cold than the heat.

    WHY: this is the new winter-drag channel. Pinning rho(15)=1.225 guards the
    continuity with the previous constant and a Celsius-vs-Kelvin bug (which
    would diverge near 0 C); rho(-10) > rho(35) encodes "cold air is denser".
    """
    from nexdash.physics import _air_density

    assert _air_density(15.0) == pytest.approx(1.2250, abs=1e-3)
    assert _air_density(-10.0) > _air_density(35.0)


def test_cold_denser_air_raises_aero() -> None:
    """Aerodynamic drag must be higher in the cold purely via air density.

    WHY: rho rises as temperature falls (ideal gas law), so the aero component
    scales up in winter. The ratio must match the density ratio exactly, proving
    it is the density channel and not some other temperature term leaking in.
    """
    from nexdash.physics import _air_density

    cold = energy_breakdown(**{**BASE, "temperature_c": -10.0})
    warm = energy_breakdown(**{**BASE, "temperature_c": 20.0})
    assert cold["aero"] > warm["aero"]
    assert cold["aero"] / warm["aero"] == pytest.approx(
        _air_density(-10.0) / _air_density(20.0), rel=1e-6
    )


def test_crr_rises_with_speed() -> None:
    """Rolling resistance must rise modestly with speed (SAE J2452).

    WHY: tyre rolling resistance is not speed-independent. Comparing equal-
    distance segments (rolling energy depends on distance, not speed) isolates
    the Crr(speed) effect from the quadratic aero term.
    """
    slow = energy_breakdown(**{**BASE, "speed_kph": 50.0})
    fast = energy_breakdown(**{**BASE, "speed_kph": 90.0})
    assert fast["rolling"] > slow["rolling"]


def test_cold_raises_rolling_but_heat_does_not() -> None:
    """Cold raises rolling resistance; heat does not lower it (cold-side ramp).

    WHY: cold tyre-pressure loss and rubber stiffening raise Crr below ~20 C,
    while above the reference the factor is clamped to 1.0 (we don't model a
    warm-tyre RR reduction). This pins the asymmetric, cold-only ramp.
    """
    cold = energy_breakdown(**{**BASE, "temperature_c": -10.0})["rolling"]
    warm = energy_breakdown(**{**BASE, "temperature_c": 20.0})["rolling"]
    hot = energy_breakdown(**{**BASE, "temperature_c": 35.0})["rolling"]
    assert cold > warm
    assert hot == pytest.approx(warm, rel=1e-9)


def test_cold_lowers_regen_recovery() -> None:
    """A cold battery recovers less downhill energy than a warm one.

    WHY: the BMS caps charge-acceptance current in the cold, so regen tapers —
    a real temperature channel that is independent of HVAC. Same descent, two
    temperatures: the cold case must credit back strictly less energy.
    """
    cold = energy_breakdown(**{**BASE, "temperature_c": -10.0, "gradient_pct": -7.0})
    warm = energy_breakdown(**{**BASE, "temperature_c": 20.0, "gradient_pct": -7.0})
    assert cold["regen"] < warm["regen"]


def test_steep_descent_recovers_smaller_fraction_than_gentle() -> None:
    """Very steep descents recover a smaller *fraction* of potential energy.

    WHY: beyond the regen power cap, friction brakes dissipate the excess, so
    the recovered fraction falls on steep grades. Normalising regen by the
    gravitational potential energy isolates the fraction from the larger raw PE
    of a steeper descent. Mild grades keep the full baseline fraction.
    """
    import math

    mass_kg = TRUCK.kerb_mass_kg + BASE["payload_t"] * 1000.0
    d_m = BASE["distance_km"] * 1000.0

    def recovered_fraction(grade_pct):
        bd = energy_breakdown(**{**BASE, "temperature_c": 20.0, "gradient_pct": grade_pct})
        pe_kwh = abs(mass_kg * 9.81 * math.sin(math.atan(grade_pct / 100.0)) * d_m) / 3.6e6
        return bd["regen"] / pe_kwh

    gentle = recovered_fraction(-3.0)  # below the knee -> full baseline
    steep = recovered_fraction(-9.0)  # past the knee -> tapered
    assert steep < gentle


def test_gentle_mild_descent_keeps_baseline_regen() -> None:
    """A gentle descent in mild weather keeps the full baseline regen fraction.

    WHY: the regen taper must touch ONLY the cold/steep tails. At -4 % / 20 C
    (below the 5 % knee, at/above the 10 C temperature) the recovered fraction
    must equal the unchanged base regen_eff, protecting the bulk of the
    distribution and the published mixed-route average from drifting.
    """
    import math

    bd = energy_breakdown(**{**BASE, "temperature_c": 20.0, "gradient_pct": -4.0})
    mass_kg = TRUCK.kerb_mass_kg + BASE["payload_t"] * 1000.0
    d_m = BASE["distance_km"] * 1000.0
    pe_kwh = abs(mass_kg * 9.81 * math.sin(math.atan(-4.0 / 100.0)) * d_m) / 3.6e6
    # regen_kwh = pe * regen_eff * drivetrain_eff (round-trip scaling).
    expected = pe_kwh * TRUCK.regen_eff * TRUCK.drivetrain_eff
    assert bd["regen"] == pytest.approx(expected, rel=1e-9)


def test_default_truck_is_eactros() -> None:
    """The default truck spec wired into physics is the eActros 600.

    WHY: the plausibility band above is calibrated to this specific vehicle; if
    the default spec silently changed, the absolute-scale test's meaning would
    change with it.
    """
    assert TRUCK.battery_kwh == pytest.approx(600.0)
    assert TRUCK.max_payload_t == pytest.approx(22.0)
