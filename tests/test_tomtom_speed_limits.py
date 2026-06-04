"""tomtom.truck_route must extract posted speed-limit spans (sectionType=speedLimit)
exactly like the browser planner, so a server-routed / MCP trip shapes per-segment
speed by the real road and matches the web dashboard's energy."""
from __future__ import annotations

import pytest

from nexdash import tomtom


def _fake_route_response():
    """Minimal TomTom calculateRoute payload: one leg of 4 points + two speedLimit
    sections (one above the truck cap, to prove the cap is applied)."""
    return {
        "routes": [
            {
                "legs": [
                    {
                        "points": [
                            {"latitude": 52.5200, "longitude": 13.4050},  # Berlin
                            {"latitude": 52.0000, "longitude": 13.0000},
                            {"latitude": 51.5000, "longitude": 12.5000},
                            {"latitude": 51.0000, "longitude": 12.0000},
                        ],
                        "summary": {
                            "lengthInMeters": 200000,
                            "travelTimeInSeconds": 9000,
                        },
                    }
                ],
                "sections": [
                    # 0->1: autobahn posted 120 -> must be capped at the truck's 80.
                    {
                        "sectionType": "SPEED_LIMIT",
                        "maxSpeedLimitInKmh": 120,
                        "startPointIndex": 0,
                        "endPointIndex": 1,
                    },
                    # 2->3: town 50 -> kept as-is.
                    {
                        "sectionType": "SPEED_LIMIT",
                        "maxSpeedLimitInKmh": 50,
                        "startPointIndex": 2,
                        "endPointIndex": 3,
                    },
                ],
                "summary": {
                    "lengthInMeters": 200000,
                    "travelTimeInSeconds": 9000,
                },
            }
        ]
    }


def test_truck_route_extracts_and_caps_speed_limits(monkeypatch):
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")
    monkeypatch.setattr(tomtom, "_get_json", lambda url: _fake_route_response())

    out = tomtom.truck_route(
        [{"lat": 52.52, "lng": 13.405}, {"lat": 51.0, "lng": 12.0}]
    )

    sl = out["speed_limits"]
    assert len(sl) == 2
    # Same {fromKm, toKm, kmh} shape route_planner.plan_route consumes.
    assert set(sl[0]) == {"fromKm", "toKm", "kmh"}
    # 120 km/h autobahn limit is capped to the truck's legal max (80).
    assert sl[0]["kmh"] == pytest.approx(min(120, tomtom.TRUCK_SPEC["maxSpeedKph"]))
    # Town 50 stays 50.
    assert sl[1]["kmh"] == pytest.approx(50)
    # Spans are real cumulative-km distances, ascending and within the route length.
    assert sl[0]["fromKm"] == pytest.approx(0.0, abs=1e-6)
    assert sl[0]["toKm"] > sl[0]["fromKm"]
    assert sl[1]["toKm"] > sl[1]["fromKm"]
    assert sl[1]["toKm"] <= out["distance_km"] + 1.0


def test_truck_route_speed_limits_empty_when_no_sections(monkeypatch):
    resp = _fake_route_response()
    resp["routes"][0]["sections"] = []
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")
    monkeypatch.setattr(tomtom, "_get_json", lambda url: resp)

    out = tomtom.truck_route(
        [{"lat": 52.52, "lng": 13.405}, {"lat": 51.0, "lng": 12.0}]
    )
    # No section data -> empty list (route_planner then falls back to leg-average
    # speed), never a crash.
    assert out["speed_limits"] == []
