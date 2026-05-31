// Route engine — frontend lib for the Routes planner console.
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

// Single source of truth for the routed vehicle, so what TomTom routes matches
// the eActros 600 the backend simulates (kerb ~18 t + 22 t payload = 40 t GCW,
// 5-axle artic). Keeping these here, rather than as scattered magic numbers,
// avoids the routed vehicle silently diverging from the simulated one.
const TRUCK_SPEC = {
  weightKg: 40000,
  axleWeightKg: 11500,
  numberOfAxles: 5,
  lengthM: 16.5,
  widthM: 2.55,
  heightM: 4.0,
  maxSpeedKph: 89,
};

// Resilient fetch for TomTom calls: a per-request timeout (AbortController) plus
// a bounded retry with exponential backoff on 429 / 5xx — TomTom's Search and
// Routing QPS cap is 5, and a multi-stop replan can burst past it, after which a
// bare fetch would silently drop stations/incidents. Permanent 4xx are NOT
// retried. Returns the Response; callers keep their own `if (!res.ok)` handling.
async function tomtomFetch(url, { timeoutMs = 8000, retries = 2 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(url, { signal: ctrl.signal });
      clearTimeout(timer);
      if ((res.status === 429 || res.status >= 500) && attempt < retries) {
        await new Promise((r) => setTimeout(r, 400 * 2 ** attempt));
        continue;
      }
      return res;
    } catch (err) {
      clearTimeout(timer);
      lastErr = err;
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 400 * 2 ** attempt));
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

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
    const res = await tomtomFetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    // Build clean "name + region" results (e.g. "Berlin" / "Berlin, Germany")
    // and de-duplicate so we don't show six near-identical suburb rows.
    const out = [];
    const seen = new Set();
    for (const r of data.results || []) {
      const a = r.address || {};
      const name =
        r.poi?.name || a.municipality || a.localName || (a.freeformAddress || q).split(",")[0].trim();
      const region = [a.countrySubdivision, a.country].filter(Boolean).join(", ");
      const lat = r.position?.lat;
      const lng = r.position?.lon;
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;
      const key = `${name}|${region}`.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ name, region, label: name, lat, lng });
    }
    return out;
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
      pts.push({
        label: d.label || "Destination",
        lat: d.lat,
        lng: d.lng,
        // Per-stop delivery data wired through to the backend per-leg simulation.
        dropWeightKg: d.dropWeightKg || 0,
        unloadMin: d.unloadMin || 0,
        deliverBy: d.deliverBy || "",
      });
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
    `&vehicleMaxSpeed=${TRUCK_SPEC.maxSpeedKph}` +
    `&vehicleWeight=${TRUCK_SPEC.weightKg}` +
    `&vehicleAxleWeight=${TRUCK_SPEC.axleWeightKg}` +
    `&vehicleNumberOfAxles=${TRUCK_SPEC.numberOfAxles}` +
    `&vehicleLength=${TRUCK_SPEC.lengthM}` +
    `&vehicleWidth=${TRUCK_SPEC.widthM}` +
    `&vehicleHeight=${TRUCK_SPEC.heightM}` +
    `&vehicleCommercial=true`;

  const res = await tomtomFetch(url);
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
    // Live-traffic delay baked into travelTime (routeType=fastest + traffic=true
    // means TomTom already routes around closures / heavy congestion).
    trafficDelayS: summary.trafficDelayInSeconds || 0,
  };
}

// --------------------------------------------------------------------------- //
// Live traffic incidents (TomTom Traffic Incident Details v5)
// --------------------------------------------------------------------------- //

// iconCategory → short human label (TomTom v5 codes).
const INCIDENT_LABEL = {
  0: "Traffic incident",
  1: "Accident",
  2: "Fog",
  3: "Dangerous conditions",
  4: "Rain",
  5: "Ice",
  6: "Traffic jam",
  7: "Lane closed",
  8: "Road closed",
  9: "Road works",
  10: "Wind",
  11: "Flooding",
  14: "Broken-down vehicle",
};

// Query incidents in small bboxes sampled along the route (the v5 bbox has an
// area limit, so we can't send one giant box for a 600 km corridor), then keep
// only those within ~6 km of the road and dedupe by incident id.
async function fetchIncidents(geometry) {
  if (!TOMTOM_KEY || !Array.isArray(geometry) || geometry.length < 2) return [];

  const N = Math.min(10, Math.max(2, Math.round(geometry.length / 60)));
  const step = Math.max(1, Math.floor(geometry.length / N));
  const samples = [];
  for (let i = 0; i < geometry.length; i += step) samples.push(geometry[i]);

  const fields = encodeURIComponent(
    "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},from,to,delay,roadNumbers}}}"
  );
  const pad = 0.18; // ~25 x 40 km box (~1000 km2), well under TomTom's 10000 km2 v5 limit

  const batches = await Promise.all(
    samples.map(async ([lat, lng]) => {
      const bbox = `${(lng - pad).toFixed(5)},${(lat - pad).toFixed(5)},${(lng + pad).toFixed(5)},${(lat + pad).toFixed(5)}`;
      // categoryFilter: only flow-affecting incidents (1 accident, 6 jam,
      // 7 lane-closed, 8 road-closed, 9 roadworks, 14 broken-down) — drop pure
      // weather (fog/rain/ice/wind/flooding) that doesn't change truck ETA.
      const url =
        `https://api.tomtom.com/traffic/services/5/incidentDetails` +
        `?key=${TOMTOM_KEY}&bbox=${bbox}&fields=${fields}&language=en-GB` +
        `&timeValidityFilter=present&categoryFilter=1,6,7,8,9,14`;
      try {
        const res = await tomtomFetch(url);
        if (!res.ok) return [];
        const data = await res.json();
        return data.incidents || [];
      } catch {
        return [];
      }
    })
  );

  const seen = new Set();
  const out = [];
  for (const inc of batches.flat()) {
    const p = inc.properties || {};
    if (p.id) {
      if (seen.has(p.id)) continue;
      seen.add(p.id);
    }
    const g = inc.geometry || {};
    let lat0;
    let lng0;
    if (g.type === "Point") {
      [lng0, lat0] = g.coordinates || [];
    } else if (g.type === "LineString" && Array.isArray(g.coordinates) && g.coordinates.length) {
      [lng0, lat0] = g.coordinates[Math.floor(g.coordinates.length / 2)];
    }
    if (!Number.isFinite(lat0) || !Number.isFinite(lng0)) continue;

    // Corridor filter: keep only incidents essentially ON the route (<= 2.5 km),
    // checked against a DENSE sample of the polyline. Also remember the nearest
    // route point so we can SNAP the marker onto the travelled line (otherwise it
    // floats on a parallel road, which looks wrong).
    const CORRIDOR_KM = 2.5;
    const corridorStride = Math.max(1, Math.floor(geometry.length / 400));
    let nearest = Infinity;
    let snapPt = [lat0, lng0];
    for (let i = 0; i < geometry.length; i += corridorStride) {
      const dd = haversineKm(geometry[i], [lat0, lng0]);
      if (dd < nearest) {
        nearest = dd;
        snapPt = geometry[i];
      }
    }
    if (nearest > CORRIDOR_KM) continue;

    out.push({
      lat: snapPt[0],
      lng: snapPt[1],
      category: p.iconCategory ?? 0,
      magnitude: p.magnitudeOfDelay ?? 0,
      description: p.events?.[0]?.description || INCIDENT_LABEL[p.iconCategory] || "Traffic incident",
      from: p.from || "",
      to: p.to || "",
      delayS: p.delay || 0,
      road: Array.isArray(p.roadNumbers) ? p.roadNumbers.join(", ") : "",
    });
  }

  // Keep only incidents that actually affect the truck's ETA: a measurable
  // delay, a major-severity event, a full road closure (forces a reroute), or a
  // traffic jam. This drops minor lane closures / roadworks with no delay that
  // were cluttering the map.
  const etaRelevant = (x) =>
    x.delayS >= 30 || x.magnitude >= 3 || x.category === 8 || x.category === 6;
  const relevant = out.filter(etaRelevant);
  // ETA impact first: biggest delay, then severity. Cap to keep the map clean.
  relevant.sort((a, b) => b.delayS - a.delayS || b.magnitude - a.magnitude);
  return relevant.slice(0, 8);
}

// --------------------------------------------------------------------------- //
// Backend SOC simulation
// --------------------------------------------------------------------------- //

async function backendPlan({ waypoints, geometry, distanceKm, durationS, planner }) {
  const body = {
    // Carry per-stop delivery data so the backend can run the per-leg simulation
    // (payload decay after each drop, unload dwell in the ETA, deliver-by check).
    waypoints: waypoints.map((w) => ({
      lat: w.lat,
      lng: w.lng,
      label: w.label,
      dropWeightKg: w.dropWeightKg || 0,
      unloadMin: w.unloadMin || 0,
      deliverBy: w.deliverBy || "",
    })),
    // Full road polyline so the backend can enrich every segment with real
    // elevation / gradient / temperature / wind from Open-Meteo.
    geometry: Array.isArray(geometry) ? geometry : [],
    distanceKm,
    durationS,
    startSoc: planner?.startSoc ?? 100,
    minSoc: planner?.minSoc ?? 15,
    payloadKg: planner?.payloadKg ?? 0,
    reservePct: planner?.reservePct ?? 10,
    maxChargeKw: planner?.maxChargeKw ?? 400,
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
// Real charging-station lookup (TomTom EV charging POIs, category 7309)
// --------------------------------------------------------------------------- //

// Map TomTom connector enums → short dispatcher-friendly labels. Falls back to
// the raw enum so we never hide an unknown connector type.
const CONNECTOR_LABEL = {
  IEC62196Type2CCS: "CCS",
  IEC62196Type2CableAttached: "Type 2",
  IEC62196Type2Outlet: "Type 2",
  Combo: "CCS",
  Chademo: "CHAdeMO",
  GBT20234Part2: "GB/T",
  GBT20234Part3: "GB/T",
  IEC62196Type1: "Type 1",
  IEC62196Type1CCS: "CCS",
  Tesla: "Tesla",
};
function connectorLabel(type) {
  if (!type) return "Unknown";
  if (CONNECTOR_LABEL[type]) return CONNECTOR_LABEL[type];
  if (type.startsWith("GBT")) return "GB/T";
  if (type.includes("CCS")) return "CCS";
  if (type.includes("Type2")) return "Type 2";
  return type;
}

// Live availability for one station (EV Charging Stations Availability API).
// Best-effort: returns { available, total } or null. Never throws.
async function fetchAvailability(availabilityId) {
  if (!availabilityId) return null;
  try {
    const url =
      `https://api.tomtom.com/search/2/chargingAvailability.json` +
      `?key=${TOMTOM_KEY}&chargingAvailability=${encodeURIComponent(availabilityId)}`;
    const res = await tomtomFetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    let available = 0;
    let total = 0;
    for (const c of data.connectors || []) {
      total += c.total || 0;
      available += c.availability?.current?.available || 0;
    }
    if (total === 0) return null;
    return { available, total };
  } catch {
    return null;
  }
}

// Replace each simulated charging stop with the nearest REAL EV charging
// station from TomTom, so stops show actual operator names (e.g. "Aral pulse",
// "IONITY") at real coordinates on/near the route instead of "MCS Charging Hub".
//
// Returns a richer station object per stop, preserving the caller's existing
// fields (name, lat, lng, address, + kWh attached later). Adds:
//   connectors:   [{ label, powerKw }]  — distinct connector types at the site
//   maxPowerKw:   number | null         — max ratedPowerKW across connectors
//   availability: { available, total } | null  — live "X of Y free" (best-effort)
//   openingHours: string | null         — short human string
async function enrichStations(stops, radiusKm = 30) {
  if (!TOMTOM_KEY || !Array.isArray(stops) || stops.length === 0) return stops || [];
  const radius = Math.max(1000, Math.round((radiusKm || 30) * 1000));
  // A 40 t eActros charges on high-power DC (CCS Combo / MCS), NOT the slow AC
  // "normal" points cars use. TomTom has no dedicated truck-charging category,
  // so we proxy "truck charging" with CCS connector + high power. Preference:
  //   1) truck-capable HPC: CCS, >=150 kW
  //   2) any DC fast charger (>=150 kW)
  //   3) nearest-any (last-resort fallback so a stop is never empty)
  const TRUCK_MIN_KW = 150;
  const TRUCK_CONNECTORS = "IEC62196Type2CCS"; // CCS Combo — the truck/HPC DC standard
  return Promise.all(
    stops.map(async (s) => {
      if (!Number.isFinite(s?.lat) || !Number.isFinite(s?.lng)) return s;
      try {
        const base =
          `https://api.tomtom.com/search/2/categorySearch/EV%20charging.json` +
          `?key=${TOMTOM_KEY}&lat=${s.lat}&lon=${s.lng}&radius=${radius}&categorySet=7309&limit=5` +
          `&openingHours=nextSevenDays&relatedPois=child`;
        const nearest = async ({ minPowerKW = 0, connectorSet = "" } = {}) => {
          let u = base;
          if (minPowerKW) u += `&minPowerKW=${minPowerKW}`;
          if (connectorSet) u += `&connectorSet=${connectorSet}`;
          const res = await tomtomFetch(u);
          if (!res.ok) return null;
          const data = await res.json();
          return (data.results || [])[0] || null;
        };
        const r =
          (await nearest({ minPowerKW: TRUCK_MIN_KW, connectorSet: TRUCK_CONNECTORS })) ||
          (await nearest({ minPowerKW: TRUCK_MIN_KW })) ||
          (await nearest());
        if (!r) return s;

        // Connectors: dedupe by label, keep the highest power seen per label.
        const rawConnectors = r.chargingPark?.connectors || [];
        let maxPowerKw = 0;
        const byLabel = new Map();
        for (const c of rawConnectors) {
          const powerKw = Number(c.ratedPowerKW) || 0;
          if (powerKw > maxPowerKw) maxPowerKw = powerKw;
          const label = connectorLabel(c.connectorType);
          const prev = byLabel.get(label);
          if (!prev || powerKw > prev.powerKw) byLabel.set(label, { label, powerKw });
        }
        const connectors = [...byLabel.values()].sort((a, b) => b.powerKw - a.powerKw);

        // Live availability (best-effort, never breaks the route).
        const availability = await fetchAvailability(r.dataSources?.chargingAvailability?.id);

        // Opening hours → short human string (e.g. "Mon 06:00-22:00 …").
        let openingHours = null;
        const oh = r.poi?.openingHours;
        if (oh?.timeRanges?.length) {
          const t = oh.timeRanges[0];
          const fmt = (x) =>
            x ? `${String(x.hour).padStart(2, "0")}:${String(x.minute).padStart(2, "0")}` : "";
          const open = fmt(t.startTime);
          const close = fmt(t.endTime);
          if (open && close) openingHours = open === "00:00" && close === "00:00" ? "Open 24/7" : `${open}–${close}`;
        }

        // Realistic charge time at THIS station: energy ÷ real connector power.
        // Divide by ~0.9 to roughly account for the charge-curve taper at high
        // SOC (a flat estimate would read optimistically fast). Null when we
        // lack either the planned kWh or the station's power.
        const powerKw = maxPowerKw > 0 ? maxPowerKw : null;
        const chargeMinutes =
          powerKw && Number.isFinite(s.kWh) && s.kWh > 0
            ? Math.round((s.kWh / (powerKw * 0.9)) * 60)
            : null;

        return {
          ...s,
          name: r.poi?.name || s.name,
          lat: r.position?.lat ?? s.lat,
          lng: r.position?.lon ?? s.lng,
          address: r.address?.municipality || r.address?.freeformAddress || s.address,
          connectors,
          maxPowerKw: powerKw ? Math.round(powerKw) : null,
          chargeMinutes,
          availability,
          openingHours,
        };
      } catch {
        return s;
      }
    })
  );
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
  let trafficDelayS = 0;
  try {
    const r = await tomtomRoute(waypoints);
    geometry = r.geometry;
    distanceKm = r.distanceKm;
    durationS = r.durationS;
    trafficDelayS = r.trafficDelayS || 0;
  } catch {
    const fb = straightLineRoute(waypoints);
    geometry = fb.geometry;
    distanceKm = fb.distanceKm;
    durationS = fb.durationS;
  }

  // 2. SOC simulation from the backend. Fall back to a local linear drain.
  try {
    // SOC sim + real charging stations + live traffic incidents — independent
    // calls, so run them together.
    const [sim, incidents] = await Promise.all([
      backendPlan({ waypoints, geometry, distanceKm, durationS, planner }),
      fetchIncidents(geometry),
    ]);
    // Swap simulated hubs for real nearby TomTom charging stations.
    const chargingStops = await enrichStations(sim.chargingStops || [], planner?.maxDetourKm);
    // Re-label the charge timeline segments to the enriched stations, in order.
    let ci = 0;
    const segments = (sim.segments || []).map((seg) => {
      if (seg.type === "charge" && chargingStops[ci]) {
        const e = chargingStops[ci++];
        return { ...seg, station: { name: e.name, lat: e.lat, lng: e.lng, address: e.address } };
      }
      return seg;
    });
    return {
      geometry,
      socProfile: sim.socProfile || [],
      segments,
      // Per-destination arrivals (SOC/ETA/deliver-by) from the per-leg sim — must
      // be forwarded or the "Delivery Stops" panel never renders.
      stops: sim.stops || [],
      chargingStops,
      // Enriched real-world layers from the backend (Open-Meteo).
      elevationProfile: sim.elevationProfile || [],
      conditions: sim.conditions || {},
      summary: sim.summary || {},
      traffic: { delayS: trafficDelayS, incidents },
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
