// Route engine — frontend lib (light dashboard Routes page).
//
// optimizeRoute(planner) -> PlanResult
//   1. Calls the TomTom Routing API (travelMode=truck) through origin +
//      destinations to get the road polyline + total distance/time.
//   2. POSTs the totals AND the full road geometry to the backend
//      /api/route-plan so the backend can enrich each segment with REAL
//      elevation / gradient / weather and run the model-driven SOC
//      simulation (charging stops, EU 561 driver hours).
//   3. Merges TomTom geometry with the backend simulation (incl. the
//      enriched elevationProfile + conditions) into a PlanResult.
//
// On ANY failure (missing key, network, backend down, no model) it returns a
// best-effort local fallback so the UI still renders something sensible.

const TOMTOM_KEY = import.meta.env.VITE_TOMTOM_API_KEY;

// Backend base URL. In dev the Vite proxy / same-origin usually works; allow an
// explicit override via VITE_API_BASE if the API runs elsewhere.
const API_BASE = import.meta.env.VITE_API_BASE || "";

const BATTERY_KWH = 600; // eActros 600 usable battery (matches backend TRUCK).
const COUNTRY_SET = "DE,AT,CH,NL,BE,FR,PL,CZ,DK";

// --------------------------------------------------------------------------- //
// Geocoding (TomTom Search API)
// --------------------------------------------------------------------------- //

export async function geocode(query) {
  const q = (query || "").trim();
  if (!q || !TOMTOM_KEY) return [];
  try {
    const url =
      `https://api.tomtom.com/search/2/geocode/${encodeURIComponent(q)}.json` +
      `?key=${TOMTOM_KEY}&limit=6&countrySet=${COUNTRY_SET}`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.results || []).map((r) => ({
      label: r.address?.freeformAddress || q,
      lat: r.position?.lat,
      lng: r.position?.lon,
    }));
  } catch {
    return [];
  }
}

// --------------------------------------------------------------------------- //
// Routing
// --------------------------------------------------------------------------- //

function collectWaypoints(planner) {
  const pts = [];
  if (planner?.origin?.lat != null && planner?.origin?.lng != null) {
    pts.push({
      label: planner.origin.label || "Origin",
      lat: planner.origin.lat,
      lng: planner.origin.lng,
    });
  }
  for (const d of planner?.destinations || []) {
    if (d?.lat != null && d?.lng != null) {
      pts.push({ label: d.label || "Destination", lat: d.lat, lng: d.lng });
    }
  }
  return pts;
}

// Call TomTom Calculate Route (truck profile) through all waypoints.
// Returns { geometry: [[lat,lng]...], distanceKm, durationS }.
async function tomtomRoute(waypoints) {
  if (!TOMTOM_KEY) throw new Error("missing TomTom key");
  if (waypoints.length < 2) throw new Error("need >= 2 waypoints");

  const locs = waypoints.map((w) => `${w.lat},${w.lng}`).join(":");
  const url =
    `https://api.tomtom.com/routing/1/calculateRoute/${locs}/json` +
    `?key=${TOMTOM_KEY}` +
    `&travelMode=truck` +
    `&routeType=fastest` +
    `&traffic=true` +
    `&vehicleMaxSpeed=89` +
    `&vehicleWeight=40000` +
    `&vehicleAxleWeight=11500` +
    `&vehicleLength=16.5` +
    `&vehicleWidth=2.55` +
    `&vehicleHeight=4` +
    `&vehicleCommercial=true`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`TomTom routing ${res.status}`);
  const data = await res.json();
  const route = data.routes?.[0];
  if (!route) throw new Error("no route returned");

  const geometry = [];
  for (const leg of route.legs || []) {
    for (const p of leg.points || []) {
      geometry.push([p.latitude, p.longitude]);
    }
  }
  const summary = route.summary || {};
  return {
    geometry,
    distanceKm: (summary.lengthInMeters || 0) / 1000,
    durationS: summary.travelTimeInSeconds || 0,
  };
}

// --------------------------------------------------------------------------- //
// Backend SOC simulation
// --------------------------------------------------------------------------- //

async function backendPlan({ waypoints, geometry, distanceKm, durationS, planner }) {
  const body = {
    waypoints: waypoints.map((w) => ({ lat: w.lat, lng: w.lng, label: w.label })),
    // Full road polyline so the backend can enrich every segment with real
    // elevation / gradient / temperature / wind from Open-Meteo.
    geometry: Array.isArray(geometry) ? geometry : [],
    distanceKm,
    durationS,
    startSoc: planner?.startSoc ?? 100,
    minSoc: planner?.minSoc ?? 15,
    payloadKg: planner?.payloadKg ?? 0,
    departure: planner?.departure || null,
    temperatureC: planner?.temperatureC ?? 15,
  };
  const res = await fetch(`${API_BASE}/api/route-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`route-plan ${res.status}`);
  return res.json();
}

// --------------------------------------------------------------------------- //
// Public entry point
// --------------------------------------------------------------------------- //

export async function optimizeRoute(planner) {
  const waypoints = collectWaypoints(planner);

  // 1. Geometry + totals from TomTom (truck). Fall back to straight line.
  let geometry;
  let distanceKm;
  let durationS;
  try {
    const r = await tomtomRoute(waypoints);
    geometry = r.geometry;
    distanceKm = r.distanceKm;
    durationS = r.durationS;
  } catch {
    const fb = straightLineRoute(waypoints);
    geometry = fb.geometry;
    distanceKm = fb.distanceKm;
    durationS = fb.durationS;
  }

  // 2. SOC simulation from the backend. Fall back to a local linear drain.
  try {
    const sim = await backendPlan({ waypoints, geometry, distanceKm, durationS, planner });
    return {
      geometry,
      socProfile: sim.socProfile || [],
      segments: sim.segments || [],
      chargingStops: sim.chargingStops || [],
      // Enriched real-world layers from the backend (Open-Meteo).
      elevationProfile: sim.elevationProfile || [],
      conditions: sim.conditions || {},
      summary: sim.summary || {},
    };
  } catch {
    return localFallback({ planner, geometry, distanceKm, durationS });
  }
}

// --------------------------------------------------------------------------- //
// Fallbacks
// --------------------------------------------------------------------------- //

function haversineKm(a, b) {
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b[0] - a[0]);
  const dLng = toRad(b[1] - a[1]);
  const lat1 = toRad(a[0]);
  const lat2 = toRad(b[0]);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// Straight-ish geometry between waypoints; great-circle distance; 70 km/h.
function straightLineRoute(waypoints) {
  if (waypoints.length < 2) {
    const p = waypoints[0] || { lat: 52.52, lng: 13.405 };
    return { geometry: [[p.lat, p.lng]], distanceKm: 0, durationS: 0 };
  }
  const geometry = waypoints.map((w) => [w.lat, w.lng]);
  let distanceKm = 0;
  for (let i = 1; i < geometry.length; i++) {
    distanceKm += haversineKm(geometry[i - 1], geometry[i]);
  }
  // Road factor ~1.25 to approximate detours.
  distanceKm *= 1.25;
  const durationS = (distanceKm / 70) * 3600;
  return { geometry, distanceKm, durationS };
}

// Local linear SOC drain when the backend is unreachable. Mirrors the backend
// contract closely enough for the UI to render a coherent result.
function localFallback({ planner, geometry, distanceKm, durationS }) {
  const startSoc = planner?.startSoc ?? 100;
  const minSoc = planner?.minSoc ?? 15;
  const payloadT = (planner?.payloadKg ?? 0) / 1000;

  // Rough consumption: ~110 kWh/100km base + 1.5 kWh/100km per tonne payload.
  const kwhPer100 = 110 + payloadT * 1.5;
  const energyKwh = (distanceKm / 100) * kwhPer100;
  const socDrop = (energyKwh / BATTERY_KWH) * 100;
  const arrivalSoc = Math.max(0, startSoc - socDrop);

  const depart = parseDeparture(planner?.departure);
  const drivingMin = (durationS || (distanceKm / 70) * 3600) / 60;
  const arrival = new Date(depart.getTime() + drivingMin * 60000);

  const socProfile = [
    { distKm: 0, soc: round1(startSoc) },
    { distKm: round1(distanceKm), soc: round1(arrivalSoc) },
  ];
  const segments = [
    {
      type: "drive",
      km: round1(distanceKm),
      durationMin: Math.round(drivingMin),
      socStart: round1(startSoc),
      socEnd: round1(arrivalSoc),
      startTime: hhmm(depart),
      endTime: hhmm(arrival),
      limitMin: 270,
    },
  ];
  const drivingH = drivingMin / 60;

  return {
    geometry,
    socProfile,
    segments,
    chargingStops: [],
    elevationProfile: [],
    conditions: {},
    summary: {
      distanceKm: round1(distanceKm),
      drivingTimeH: round2(drivingH),
      chargingTimeMin: 0,
      totalTimeH: round2(drivingH),
      etaLabel: hhmm(arrival),
      etaIso: arrival.toISOString().slice(0, 16),
      startSoc: round1(startSoc),
      arrivalSoc: round1(arrivalSoc),
      minSoc: round1(Math.min(minSoc, arrivalSoc)),
      energyKwh: round1(energyKwh),
      kwhPer100: round1(kwhPer100),
      chargingCostEur: 0,
      chargingStops: 0,
      elevationGainM: 0,
      driver: {
        drivingH: round2(drivingH),
        breaks: 0,
        totalH: round2(drivingH),
        dailyH: round2(drivingH),
        dailyMaxH: 9,
        weeklyH: round2(drivingH),
        weeklyMaxH: 56,
        eu561ok: drivingH <= 9,
      },
    },
  };
}

// --------------------------------------------------------------------------- //
// Small helpers
// --------------------------------------------------------------------------- //

function parseDeparture(iso) {
  if (iso) {
    const d = new Date(iso);
    if (!isNaN(d.getTime())) return d;
  }
  return new Date();
}

function hhmm(d) {
  return `${String(d.getHours()).padStart(2, "0")}:${String(
    d.getMinutes()
  ).padStart(2, "0")}`;
}

const round1 = (n) => Math.round(n * 10) / 10;
const round2 = (n) => Math.round(n * 100) / 100;
