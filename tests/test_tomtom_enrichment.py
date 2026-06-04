"""Server-side ports of the browser planner's station enrichment + traffic
incidents must surface the same real-world data (operator, power, availability,
ETA-relevant incidents) the website shows — and never raise on a bad lookup."""
from __future__ import annotations

from nexdash import tomtom


# --------------------------------------------------------------------------- #
# enrich_charging_stations
# --------------------------------------------------------------------------- #
def _category_response():
    return {
        "results": [
            {
                "poi": {
                    "name": "AVIA VOLT",
                    "openingHours": {
                        "timeRanges": [
                            {
                                "startTime": {"hour": 0, "minute": 0},
                                "endTime": {"hour": 0, "minute": 0},
                            }
                        ]
                    },
                },
                "address": {"municipality": "Lauf an der Pegnitz"},
                "position": {"lat": 49.51, "lon": 11.28},
                "dist": 3200,
                "dataSources": {"chargingAvailability": {"id": "AVAIL-1"}},
                "chargingPark": {
                    "connectors": [
                        {"connectorType": "IEC62196Type2CCS", "ratedPowerKW": 400},
                        {"connectorType": "IEC62196Type2CCS", "ratedPowerKW": 150},
                    ]
                },
            }
        ]
    }


def _availability_response():
    return {
        "connectors": [
            {
                "type": "IEC62196Type2CCS",
                "total": 6,
                "availability": {"current": {"available": 5, "occupied": 1}},
            }
        ]
    }


def test_enrich_charging_stations_resolves_real_station(monkeypatch):
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")

    def fake_get_json(url):
        if "chargingAvailability.json" in url:
            return _availability_response()
        if "categorySearch" in url:
            return _category_response()
        return {}

    monkeypatch.setattr(tomtom, "_get_json", fake_get_json)

    stops = [{"name": "DC Fast-Charge Hub 1", "lat": 49.5, "lng": 11.3, "kWh": 300}]
    out = tomtom.enrich_charging_stations(stops, max_charge_kw=400)

    assert len(out) == 1
    st = out[0]["station"]
    assert st is not None
    assert st["name"] == "AVIA VOLT"
    assert st["address"] == "Lauf an der Pegnitz"
    assert st["max_power_kw"] == 400
    assert st["effective_power_kw"] == 400  # capped at the truck's max
    assert st["off_route_km"] == 3.2
    # Connectors deduped by label, highest power kept.
    assert st["connectors"] == [{"label": "CCS", "power_kw": 400}]
    # Live CCS availability surfaced.
    assert st["availability"] == {"available": 5, "total": 6}
    assert st["opening_hours"] == "Open 24/7"
    # Real operator name promoted onto the stop itself.
    assert out[0]["name"] == "AVIA VOLT"


def test_enrich_charging_stations_no_results_keeps_stop(monkeypatch):
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")
    monkeypatch.setattr(tomtom, "_get_json", lambda url: {"results": []})

    stops = [{"name": "DC Fast-Charge Hub 1", "lat": 49.5, "lng": 11.3, "kWh": 300}]
    out = tomtom.enrich_charging_stations(stops)
    assert out[0]["name"] == "DC Fast-Charge Hub 1"
    assert out[0]["station"] is None


def test_enrich_charging_stations_never_raises_on_error(monkeypatch):
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")

    def boom(url):
        raise tomtom.TomTomError("nope")

    monkeypatch.setattr(tomtom, "_get_json", boom)
    out = tomtom.enrich_charging_stations(
        [{"name": "x", "lat": 49.5, "lng": 11.3, "kWh": 300}]
    )
    assert out[0]["station"] is None


# --------------------------------------------------------------------------- #
# rank_chargers_by_time
# --------------------------------------------------------------------------- #
def test_rank_prefers_faster_charger_slightly_off_route():
    near_slow = {"dist": 500, "chargingPark": {"connectors": [{"ratedPowerKW": 150}]}}
    far_fast = {"dist": 4000, "chargingPark": {"connectors": [{"ratedPowerKW": 400}]}}
    ranked = tomtom.rank_chargers_by_time(
        [near_slow, far_fast], energy_kwh=300, max_charge_kw=400
    )
    # The 400 kW site finishes the stop sooner despite the longer detour.
    assert ranked[0]["c"] is far_fast


# --------------------------------------------------------------------------- #
# fetch_traffic_incidents
# --------------------------------------------------------------------------- #
def test_fetch_traffic_incidents_keeps_eta_relevant_on_route(monkeypatch):
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")

    incident = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [13.40, 52.50]},  # lng,lat ON route
        "properties": {
            "id": "inc1",
            "iconCategory": 6,  # traffic jam
            "magnitudeOfDelay": 3,
            "events": [{"description": "Queuing traffic"}],
            "from": "Junction A",
            "to": "Junction B",
            "delay": 120,
            "roadNumbers": ["A9"],
        },
    }
    monkeypatch.setattr(tomtom, "_get_json", lambda url: {"incidents": [incident]})

    geometry = [[52.50, 13.40], [52.00, 13.00], [51.50, 12.50]]
    out = tomtom.fetch_traffic_incidents(geometry)

    assert len(out) == 1  # deduped across the sampled bboxes
    inc = out[0]
    assert inc["category"] == "Traffic jam"
    assert inc["delay_s"] == 120
    assert inc["road"] == "A9"
    assert inc["description"] == "Queuing traffic"


def test_fetch_traffic_incidents_drops_off_corridor(monkeypatch):
    monkeypatch.setattr(tomtom, "get_api_key", lambda: "TESTKEY")
    far = {
        "geometry": {"type": "Point", "coordinates": [9.0, 48.0]},  # far from route
        "properties": {"id": "x", "iconCategory": 6, "delay": 300, "roadNumbers": []},
    }
    monkeypatch.setattr(tomtom, "_get_json", lambda url: {"incidents": [far]})
    out = tomtom.fetch_traffic_incidents([[52.50, 13.40], [52.00, 13.00]])
    assert out == []


def test_fetch_traffic_incidents_empty_geometry():
    assert tomtom.fetch_traffic_incidents([]) == []
