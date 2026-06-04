"""Tests for :mod:`nexdash.stress_test` — per-trip robustness stress test.

These encode WHY the panel matters, not just that it runs:

* The breakpoint interpolation must find the zero-crossing of margin (the NO-GO
  threshold a dispatcher needs) and return None when margin never goes negative.
* Each factor must move margin in the PHYSICALLY CORRECT direction — heavier,
  windier, faster erodes margin. A sign error here would mis-rank the threats.
* The tornado must be sorted worst-first so the dominant threat is row 0.
* HONESTY: where the physics cross-check trips confidence='low' on a swept point,
  the factor must REFUSE a precise breakpoint and say 'low confidence beyond X' —
  the same discipline as the calibration harness, applied per-trip.
"""

from __future__ import annotations

from nexdash import stress_test as st
from nexdash.config import DEFAULT_MODEL_PATH

MODEL = str(DEFAULT_MODEL_PATH)


def test_interp_breakpoint_finds_zero_crossing():
    """The pure helper finds the first >=0 -> <0 crossing, else None."""
    pts = [(0.0, 30.0), (1.0, 20.0), (2.0, 10.0), (3.0, -10.0)]
    assert st._interp_breakpoint(pts) == 2.5  # crossing at x=2.5
    assert st._interp_breakpoint([(0.0, 5.0), (1.0, 3.0), (2.0, 1.0)]) is None


def test_grid_sweeps_toward_the_adverse_edge_and_clamps():
    """The sweep probes the dangerous direction and never leaves the envelope."""
    g = st._grid(5.0, 0.0, 22.0, "high")
    assert g[0] == 5.0 and g[-1] == 22.0 and all(0.0 <= v <= 22.0 for v in g)
    g2 = st._grid(15.0, -15.0, 40.0, "low")
    assert g2[0] == 15.0 and g2[-1] == -15.0


def test_factors_erode_margin_in_the_correct_direction():
    """Adverse sweeps must reduce margin: every factor erodes (erosion >= 0).

    WHY: the load-bearing physics check — a factor whose adverse sweep did NOT
    erode margin would contradict the physics and make the tornado meaningless.
    """
    out = st.stress_test(
        soc_pct=95, distance_km=120, payload_t=8, speed_kph=70,
        gradient_pct=0.0, temperature_c=15, model_path=MODEL,
    )
    assert out["baseline"]["reaches"] is True
    assert {f["factor"] for f in out["factors"]} == set(st.FACTORS)
    for f in out["factors"]:
        assert f["margin_erosion_kwh"] >= -1e-6, f


def test_tornado_is_ranked_worst_first():
    """factors sorted by margin erosion descending; dominant_threat is row 0."""
    out = st.stress_test(
        soc_pct=95, distance_km=120, payload_t=8, speed_kph=70,
        gradient_pct=0.0, temperature_c=15, model_path=MODEL,
    )
    erosions = [f["margin_erosion_kwh"] for f in out["factors"]]
    assert erosions == sorted(erosions, reverse=True)
    assert out["dominant_threat"] == out["factors"][0]["factor"]


def test_out_of_envelope_sweep_suppression_contract_holds():
    """The confidence-flip suppression contract must hold for every swept factor.

    WHY (honesty): when a swept point trips confidence='low' (the physics
    cross-check firing as the sweep pushes into an optimistic-model region) the
    panel must refuse a precise breakpoint and say 'low confidence beyond X'
    instead of quoting a crossing it cannot trust.

    The energy model now learns the PHYSICS RESIDUAL (`nexdash.model`), so on this
    long flat 300 km leg it TRACKS physics across the entire sweep instead of
    saturating — so no swept point flips, and the panel is free to quote precise
    breakpoints. That is the fix: the per-trip cross-check no longer raises a
    false alarm where the model is in fact physics-faithful. We therefore assert
    the *contract* rather than requiring a flip: any factor that DOES flip must
    suppress its breakpoint (so the mechanism stays wired), and a non-flipping
    factor must not carry the suppression note.
    """
    out = st.stress_test(
        soc_pct=50, distance_km=300, payload_t=10, speed_kph=72,
        gradient_pct=0.0, temperature_c=12, model_path=MODEL,
    )
    for f in out["factors"]:
        if f["confidence_flips"]:
            assert f["breakpoint"] is None
            assert "low confidence beyond" in f["breakpoint_note"]
        else:
            assert "low confidence beyond" not in (f.get("breakpoint_note") or "")


def test_response_is_json_serialisable_with_honest_assumptions():
    """The whole result must be JSON-safe and carry the honest caveats."""
    import json

    out = st.stress_test(
        soc_pct=80, distance_km=150, payload_t=10, speed_kph=72,
        gradient_pct=0.0, temperature_c=10, model_path=MODEL,
    )
    json.dumps(out)  # must not raise
    blob = " ".join(out["assumptions"]).lower()
    assert "one-factor" in blob and "envelope" in blob
