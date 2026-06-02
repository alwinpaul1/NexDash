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

const TOMTOM_KEY = import.meta.env?.VITE_TOMTOM_API_KEY;

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
  // German Lkw autobahn legal limit is 80 km/h for trucks > 7.5 t. The eActros 600
  // is electronically limited to ~89 km/h, but routing at that exceeds the legal
  // (and realistic) truck speed and makes ETAs too optimistic, so we cap routing at
  // the 80 km/h limit. Real averages then land ~75-78 km/h once ramps / Landstrasse
  // / traffic are mixed in — consistent with field data and the reference planner.
  maxSpeedKph: 80,
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
const API_BASE = import.meta.env?.VITE_API_BASE || "";

// Flat fallback charging tariff (mirrors backend PRICE_EUR_PER_KWH). A real
// per-station tariff overrides this when the station data carries one — see
// extractPricePerKwh. The categorySearch endpoint we use today returns no price,
// so this flat rate applies until a tariff feed (TomTom EV Search API
// `include=tariffs`, or another structured source) is wired.
const FLAT_PRICE_EUR_PER_KWH = 0.45;

// Pull a per-kWh ENERGY tariff (EUR) from a TomTom station result IF present.
// categorySearch carries none -> null (flat fallback). TomTom's EV Search API
// (`include=tariffs`) returns references.tariffs[] with elements[].priceComponents[]
// of type ENERGY|TIME|FLAT; we take the ENERGY price. Defensive: any shape miss -> null.
function extractPricePerKwh(r) {
  try {
    const tariffs = r?.references?.tariffs || r?.chargingPark?.tariffs;
    if (!Array.isArray(tariffs) || tariffs.length === 0) return null;
    for (const t of tariffs) {
      const comps = Array.isArray(t?.elements)
        ? t.elements.flatMap((e) => e?.priceComponents || [])
        : t?.priceComponents || [];
      const energy = comps.find((c) => c?.type === "ENERGY" && Number.isFinite(c?.price));
      if (energy) return energy.price;
    }
    return null;
  } catch {
    return null;
  }
}

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
    `&vehicleCommercial=true` +
    // Posted speed-limit sections along the route, so per-segment speed can follow
    // the real road (autobahn vs town vs 30-zone) instead of one flat average.
    `&sectionType=speedLimit`;

  const res = await tomtomFetch(url);
  if (!res.ok) throw new Error(`TomTom routing ${res.status}`);
  const data = await res.json();
  const route = data.routes?.[0];
  if (!route) throw new Error("no route returned");

  const geometry = [];
  const cumKm = []; // cumulative distance (km) at each geometry point
  const legTimings = [];
  let _cum = 0;
  for (const leg of route.legs || []) {
    for (const p of leg.points || []) {
      const pt = [p.latitude, p.longitude];
      if (geometry.length) _cum += haversineKm(geometry[geometry.length - 1], pt);
      geometry.push(pt);
      cumKm.push(_cum);
    }
    // Per-leg measured timing -> the backend derives a REAL per-segment speed
    // (traffic/road-class aware) from it instead of the gradient heuristic.
    const ls = leg.summary || {};
    legTimings.push({
      lengthM: ls.lengthInMeters || 0,
      travelTimeS: ls.travelTimeInSeconds || 0,
    });
  }
  // Posted speed limits -> distance spans (km), capped at the truck's legal max.
  // The backend uses these as the per-segment speed SHAPE (30 in a village, 80 on
  // the autobahn), anchored to the measured leg time so the total ETA is unchanged.
  const speedLimits = [];
  for (const sec of route.sections || []) {
    if (sec.sectionType && sec.sectionType !== "SPEED_LIMIT") continue;
    const kmh = Number(sec.maxSpeedLimitInKmh);
    const a = cumKm[sec.startPointIndex];
    const b = cumKm[sec.endPointIndex];
    if (!Number.isFinite(kmh) || kmh <= 0 || !Number.isFinite(a) || !Number.isFinite(b) || b <= a) continue;
    speedLimits.push({ fromKm: a, toKm: b, kmh: Math.min(kmh, TRUCK_SPEC.maxSpeedKph) });
  }
  const summary = route.summary || {};
  return {
    geometry,
    legTimings,
    speedLimits,
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

async function backendPlan({ waypoints, geometry, legTimings, speedLimits, distanceKm, durationS, planner }) {
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
    // Per-leg measured travel time -> backend derives REAL per-segment speed.
    legTimings: Array.isArray(legTimings) ? legTimings : [],
    // Posted speed-limit distance spans (capped at the truck limit) -> backend
    // shapes per-segment speed by road, anchored to the measured time.
    speedLimits: Array.isArray(speedLimits) ? speedLimits : [],
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
    // The eActros charges on CCS DC, so count ONLY CCS connectors — a stack of slow
    // AC outlets must not masquerade as "free" truck plugs. Critically, TomTom marks
    // many sites as status "unknown" (NOT a real free/occupied count); treat unknown
    // as UNKNOWN, never as "0 free". If no CCS connector has a definite live status
    // (available/occupied/reserved/outOfService), return null so the UI shows nothing
    // rather than a misleading "0 of N free". (Confirmed against the live API.)
    let available = 0;
    let total = 0;
    let definite = 0;
    for (const c of data.connectors || []) {
      const type = String(c.type || "");
      if (!(type.includes("CCS") || type.includes("Combo"))) continue;
      total += c.total || 0;
      const cur = c.availability?.current || {};
      const a = Number(cur.available) || 0;
      available += a;
      definite +=
        a + (Number(cur.occupied) || 0) + (Number(cur.reserved) || 0) + (Number(cur.outOfService) || 0);
    }
    if (total === 0 || definite === 0) return null;
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
// Rank charging candidates by TOTAL added time = charge time (at the station's
// real power, capped at the truck's max) + a round-trip detour penalty. The
// energy to add is identical for every candidate at a given stop, so this ranks a
// faster charger a little further off-route ABOVE a slow one right on the line —
// which is what actually minimises trip time. Pure + exported for tests. Returns
// [{ c, score }] sorted fastest-first (lowest added time).
export function rankChargersByTime(candidates, { energyKwh = 400, maxChargeKw = 400, detourKph = 60 } = {}) {
  const cap = Number.isFinite(maxChargeKw) && maxChargeKw > 0 ? maxChargeKw : 400;
  const e = Number.isFinite(energyKwh) && energyKwh > 0 ? energyKwh : 400;
  const powerOf = (c) =>
    (c?.chargingPark?.connectors || []).reduce((mx, x) => Math.max(mx, Number(x.ratedPowerKW) || 0), 0);
  const scoreOf = (c) => {
    const eff = Math.min(cap, powerOf(c)); // truck can't pull more than its own cap
    const chargeMin = eff > 0 ? (e / (eff * 0.9)) * 60 : Infinity; // unknown power -> last
    const detourMin = ((Number(c?.dist) || 0) / 1000 / detourKph) * 60 * 2; // out-and-back
    return chargeMin + detourMin;
  };
  return (Array.isArray(candidates) ? candidates : [])
    .map((c) => ({ c, score: scoreOf(c) }))
    .sort((a, b) => a.score - b.score);
}

async function enrichStations(stops, radiusKm = 30, minChargerKw = 150, maxChargeKw = 400) {
  if (!TOMTOM_KEY || !Array.isArray(stops) || stops.length === 0) return stops || [];
  const radius = Math.max(1000, Math.round((radiusKm || 30) * 1000));
  // A 40 t eActros charges on high-power DC (CCS Combo / MCS), NOT the slow AC
  // "normal" points cars use. TomTom has no dedicated truck-charging category,
  // so we proxy "truck charging" with CCS connector + high power. Preference:
  //   1) truck-capable HPC: CCS, >= the operator's Min Charger Speed
  //   2) any DC fast charger (>= Min Charger Speed)
  //   3) nearest-any (last-resort fallback so a stop is never empty)
  // The minimum power is the planner's "Min Charger Speed" slider (default 150 kW).
  const TRUCK_MIN_KW = Number.isFinite(minChargerKw) && minChargerKw > 0 ? minChargerKw : 150;
  const TRUCK_CONNECTORS = "IEC62196Type2CCS"; // CCS Combo — the truck/HPC DC standard
  return Promise.all(
    stops.map(async (s) => {
      if (!Number.isFinite(s?.lat) || !Number.isFinite(s?.lng)) return s;
      try {
        const base =
          `https://api.tomtom.com/search/2/categorySearch/EV%20charging.json` +
          `?key=${TOMTOM_KEY}&lat=${s.lat}&lon=${s.lng}&radius=${radius}&categorySet=7309&limit=12` +
          `&openingHours=nextSevenDays&relatedPois=child`;
        const nearestList = async ({ minPowerKW = 0, connectorSet = "" } = {}) => {
          let u = base;
          if (minPowerKW) u += `&minPowerKW=${minPowerKW}`;
          if (connectorSet) u += `&connectorSet=${connectorSet}`;
          const res = await tomtomFetch(u);
          if (!res.ok) return [];
          const data = await res.json();
          return data.results || [];
        };
        let candidates = await nearestList({ minPowerKW: TRUCK_MIN_KW, connectorSet: TRUCK_CONNECTORS });
        if (!candidates.length) candidates = await nearestList({ minPowerKW: TRUCK_MIN_KW });
        if (!candidates.length) candidates = await nearestList();
        if (!candidates.length) return s;

        // Choose the TIME-OPTIMAL station (fastest stop overall), not merely the
        // nearest: a higher-power charger a little further off-route finishes the
        // stop sooner, which is what actually minimises trip time. (Picking nearest
        // is why we used to land a slow 150 kW site when a 300 kW one was close by.)
        const cap = Number.isFinite(maxChargeKw) && maxChargeKw > 0 ? maxChargeKw : 400;
        const ranked = rankChargersByTime(candidates, { energyKwh: s.kWh, maxChargeKw: cap });
        // "Fastest FREE charger on the route." `ranked` is already ordered by total
        // added time (charge time at effective power + detour), so we check the top
        // few for a free slot — in parallel — and take the FASTEST one that has one.
        // Availability is a planning-time snapshot (the truck arrives hours later), so
        // if none of the top candidates report a free slot we fall back to the fastest
        // overall rather than strand the stop on a stale "busy".
        const topK = ranked.slice(0, 6);
        const avails = await Promise.all(
          topK.map(({ c }) => fetchAvailability(c.dataSources?.chargingAvailability?.id))
        );
        let r = ranked[0].c;
        let availability = avails[0] || null;
        for (let i = 0; i < topK.length; i += 1) {
          if (avails[i] && avails[i].available > 0) {
            r = topK[i].c;
            availability = avails[i];
            break;
          }
        }

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

        // (availability was resolved above, preferring a currently-free station)

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
        // Charge time uses the EFFECTIVE power — the station's rating capped at the
        // truck's max charge power (a 600 kW post can't push the eActros past ~400 kW).
        const effPowerKw = powerKw ? Math.min(cap, powerKw) : null;
        const chargeMinutes =
          effPowerKw && Number.isFinite(s.kWh) && s.kWh > 0
            ? Math.round((s.kWh / (effPowerKw * 0.9)) * 60)
            : null;

        return {
          ...s,
          name: r.poi?.name || s.name,
          pricePerKwh: extractPricePerKwh(r),
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

// Build a REAL road distance/time matrix over [origin, ...dests] via TomTom truck
// routing — one calculateRoute per node pair, symmetric so we only route the upper
// triangle (one-ways/traffic asymmetry is ignored for the ORDERING proxy only; the
// final displayed plan is the real directed route). Indexed 0=origin, 1..N=dests in
// the given order — exactly what the backend optimiser expects. Bounded to <=9 nodes
// to cap the call count; returns null on any failure (backend then uses great-circle).
async function buildRoadMatrix(nodes) {
  const n = nodes.length;
  if (!TOMTOM_KEY || n < 2 || n > 9) return null;
  const dist = Array.from({ length: n }, () => new Array(n).fill(0));
  const time = Array.from({ length: n }, () => new Array(n).fill(0));
  try {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const r = await tomtomRoute([nodes[i], nodes[j]]);
        dist[i][j] = dist[j][i] = r.distanceKm;
        time[i][j] = time[j][i] = (r.durationS || 0) / 3600;
      }
    }
  } catch {
    return null;
  }
  return { distMatrixKm: dist, timeMatrixH: time };
}

// Straight-line km to the nearest high-power (>=150 kW) charger near a point, via the
// same TomTom EV category search the stop enrichment uses. Feeds the "can the truck
// still reach a charger from the final stop?" check. null on failure.
async function nearestChargerKm(point, minChargerKw = 150) {
  if (!TOMTOM_KEY || !Number.isFinite(point?.lat) || !Number.isFinite(point?.lng)) return null;
  const minKw = Number.isFinite(minChargerKw) && minChargerKw > 0 ? Math.round(minChargerKw) : 150;
  try {
    const u =
      `https://api.tomtom.com/search/2/categorySearch/EV%20charging.json` +
      `?key=${TOMTOM_KEY}&lat=${point.lat}&lon=${point.lng}&radius=60000` +
      `&categorySet=7309&minPowerKW=${minKw}&limit=1`;
    const res = await tomtomFetch(u);
    if (!res.ok) return null;
    const data = await res.json();
    const r = (data.results || [])[0];
    if (!Number.isFinite(r?.position?.lat) || !Number.isFinite(r?.position?.lon)) return null;
    return haversineKm([point.lat, point.lng], [r.position.lat, r.position.lon]);
  } catch {
    return null;
  }
}

// Energy-minimising stop ORDER from the backend optimiser. Supplies a REAL road
// distance/time matrix + per-destination nearest-charger distances so the order is
// chosen on roads (not a straight-line proxy), scored with the ML energy model, and
// the reach-a-charger-from-the-final-stop check is anchored to the chosen last stop.
// Returns {optimizedOrder, savingsKwh, savingsPctKwh, solver, destinationCharger,
// dataSources} or null (keep the typed order) on failure or <2 valid destinations.
async function optimizeOrder(planner) {
  const dests = (planner?.destinations || []).filter((d) => d?.lat != null && d?.lng != null);
  if (dests.length < 2 || planner?.origin?.lat == null) return null;
  try {
    const origin = { lat: planner.origin.lat, lng: planner.origin.lng, label: planner.origin.label || "" };
    // Real road matrix + nearest-charger distances (best-effort; null -> backend proxy).
    const matrix = await buildRoadMatrix([origin, ...dests]);
    const chargerKmByDest = await Promise.all(dests.map((d) => nearestChargerKm(d, planner?.minChargerKw)));
    const res = await fetch(`${API_BASE}/api/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        origin,
        destinations: dests.map((d) => ({
          lat: d.lat, lng: d.lng, label: d.label || "", dropWeightKg: d.dropWeightKg || 0,
        })),
        startSoc: planner.startSoc ?? 100,
        minSoc: planner.minSoc ?? 15,
        payloadKg: planner.payloadKg ?? 0,
        temperatureC: planner.temperatureC ?? 15,
        reservePct: planner.reservePct ?? 10,
        maxChargeKw: planner.maxChargeKw ?? 400,
        windMps: planner.windMps ?? 0,
        distMatrixKm: matrix?.distMatrixKm ?? null,
        timeMatrixH: matrix?.timeMatrixH ?? null,
        chargerKmByDest,
      }),
    });
    if (!res.ok) return null;
    const d = await res.json();
    if (!Array.isArray(d?.optimizedOrder)) return null;
    return {
      optimizedOrder: d.optimizedOrder,
      savingsKwh: d.savingsKwh,
      savingsPctKwh: d.savingsPctKwh,
      solver: d.solver,
      destinationCharger: d.destinationCharger || null,
      dataSources: d.dataSources || null,
    };
  } catch {
    return null;
  }
}

// --------------------------------------------------------------------------- //
// Public entry point
// --------------------------------------------------------------------------- //

export async function optimizeRoute(planner) {
  // 0. Optimise the visiting ORDER first via the backend VRP (offline; no routing
  // key). On failure or <2 stops this is null and the typed order is kept. The
  // reorder applies to the FILTERED valid destinations (matching collectWaypoints,
  // which skips null-coord stops), so optimizedOrder indices line up.
  // Stop order ALWAYS follows the user's typed sequence (origin -> destinations as
  // listed): the route, map pins and delivery list all reflect exactly what the
  // dispatcher entered, and "Optimize Route" optimises the energy / charging /
  // SOC plan FOR that order. The VRP stop-order optimiser (`optimizeOrder` +
  // backend /api/optimize) is intentionally left available but NOT auto-invoked —
  // wire it to an explicit "suggest a better order" action if that's ever wanted,
  // rather than silently reordering what the user asked for.
  const opt = null;
  let activePlanner = planner;
  if (opt && Array.isArray(opt.optimizedOrder)) {
    const validDests = (planner.destinations || []).filter((d) => d?.lat != null && d?.lng != null);
    const reordered = opt.optimizedOrder.map((i) => validDests[i]).filter(Boolean);
    if (reordered.length === validDests.length) {
      activePlanner = { ...planner, destinations: reordered };
    }
  }
  const waypoints = collectWaypoints(activePlanner);

  // 1. Geometry + totals from TomTom (truck) for the (optimised) order. Fall back
  // to straight line.
  let geometry;
  let distanceKm;
  let durationS;
  let trafficDelayS = 0;
  let legTimings = [];
  let speedLimits = [];
  try {
    const r = await tomtomRoute(waypoints);
    geometry = r.geometry;
    distanceKm = r.distanceKm;
    durationS = r.durationS;
    trafficDelayS = r.trafficDelayS || 0;
    legTimings = r.legTimings || [];
    speedLimits = r.speedLimits || [];
  } catch {
    const fb = straightLineRoute(waypoints);
    geometry = fb.geometry;
    distanceKm = fb.distanceKm;
    durationS = fb.durationS;
  }

  // 2. SOC simulation from the backend. Fall back to a local linear drain.
  try {
    const [sim, incidents] = await Promise.all([
      backendPlan({ waypoints, geometry, legTimings, speedLimits, distanceKm, durationS, planner: activePlanner }),
      fetchIncidents(geometry),
    ]);
    // Swap simulated hubs for real nearby TomTom charging stations.
    const chargingStops = await enrichStations(sim.chargingStops || [], planner?.maxDetourKm, planner?.minChargerKw, planner?.maxChargeKw);

    // Re-route the DISPLAYED polyline THROUGH the real charging stations so the
    // drawn route visibly detours off the road into each station, instead of the
    // pin floating beside the highway. This is DISPLAY-ONLY: the SOC simulation,
    // charging plan, ETA, EU 561 and energy were already computed by the backend
    // on the direct route and are NOT recomputed — we only swap the geometry the
    // map draws. One extra TomTom call routes through all stops at once; any
    // failure silently keeps the original (direct) geometry, and summary.distanceKm
    // stays the backend value so the SOC-colour scale (totalKm/geomTotal) and the
    // calibrated energy headline are unaffected.
    if (chargingStops.length) {
      try {
        const rerouteNodes = [
          { lat: waypoints[0]?.lat, lng: waypoints[0]?.lng, distKm: 0 },
          ...chargingStops.map((s) => ({ lat: s.lat, lng: s.lng, distKm: s.distKm })),
          ...waypoints.slice(1).map((w, i) => ({
            lat: w?.lat,
            lng: w?.lng,
            distKm: Number.isFinite(sim.stops?.[i]?.distKm) ? sim.stops[i].distKm : Infinity,
          })),
        ]
          .filter((n) => Number.isFinite(n.lat) && Number.isFinite(n.lng))
          .sort((a, b) => a.distKm - b.distKm);
        // Worth a re-route only if a station sits between origin and a stop.
        if (rerouteNodes.length >= 3) {
          const rr = await tomtomRoute(rerouteNodes);
          if (rr && Array.isArray(rr.geometry) && rr.geometry.length >= 2) {
            geometry = rr.geometry;
          }
        }
      } catch {
        /* keep the original (direct) geometry on any failure */
      }
    }

    // Re-label the charge timeline segments to the enriched stations, in order.
    let ci = 0;
    const segments = (sim.segments || []).map((seg) => {
      if (seg.type === "charge" && chargingStops[ci]) {
        const e = chargingStops[ci++];
        return { ...seg, station: { name: e.name, lat: e.lat, lng: e.lng, address: e.address } };
      }
      return { ...seg };
    });
    const summary = { ...(sim.summary || {}) };
    // Honest caveats surfaced in the existing assumptions block (RouteResults reads it).
    const assumptions = Array.isArray(summary.assumptions) ? [...summary.assumptions] : [];
    if (opt) {
      const savedKwh = Number.isFinite(opt.savingsKwh) ? opt.savingsKwh : 0;
      const distSrc = opt.dataSources?.distance === "tomtom-road-matrix" ? "real road distances" : "a great-circle proxy";
      const energySrc = opt.dataSources?.energy === "ml-model" ? "the ML energy model" : "the physics model";
      assumptions.unshift(
        `Stop order optimised (${opt.solver}) on ${distSrc} + ${energySrc}: saves ~${savedKwh.toFixed(0)} kWh ` +
          `(${(opt.savingsPctKwh ?? 0).toFixed(0)}%) of energy vs the typed order.`
      );
    }
    // Reach-a-charger-from-the-final-stop verdict (P2-D5): surface it prominently.
    if (opt?.destinationCharger?.note) {
      assumptions.unshift(opt.destinationCharger.note);
    }
    // Be honest that the "X of Y free" badge is a planning-time snapshot.
    if (chargingStops.some((e) => e && e.availability)) {
      assumptions.push(
        "Charging-station \"X of Y free\" is a LIVE snapshot at planning time; the truck arrives later, so " +
          "availability may differ. We prefer a currently-free station where one exists nearby — not a guarantee."
      );
    }
    summary.assumptions = assumptions;
    summary.destinationCharger = opt?.destinationCharger || null;

    // Per-destination arrivals (SOC/ETA/deliver-by) from the per-leg sim — must
    // be forwarded or the "Delivery Stops" panel never renders. Copied so the
    // charge-time reconciliation below can re-stamp their ETAs without mutating sim.
    const stops = (sim.stops || []).map((s) => ({ ...s }));

    // Reconcile each planned charge time (the backend computes it at the truck's
    // max charge power) with the REAL matched station's rated power: a station
    // that can't deliver full power charges slower, so the charge — and the ETA,
    // total time and every later segment / stop — move later. Display-only; the
    // SOC walk, kWh and energy headline are unchanged (same electrons, more
    // minutes at the plug).
    reconcileChargeDurations({
      segments,
      stops,
      summary,
      chargingStops,
      maxChargeKw: activePlanner?.maxChargeKw,
      departure: activePlanner?.departure,
    });

    return {
      geometry,
      socProfile: sim.socProfile || [],
      segments,
      stops,
      chargingStops,
      // Enriched real-world layers from the backend (Open-Meteo).
      elevationProfile: sim.elevationProfile || [],
      conditions: sim.conditions || {},
      summary,
      optimization: opt || null,
      traffic: { delayS: trafficDelayS, incidents },
    };
  } catch {
    return localFallback({ planner: activePlanner, geometry, distanceKm, durationS });
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

  // Per-destination arrival estimates so the Delivery Stops panel still renders
  // when the backend SOC simulator is unreachable. Approximate: leg fractions
  // from great-circle waypoint spacing, constant consumption, linear SOC drop.
  const wpts = [];
  if (planner?.origin?.lat != null) wpts.push(planner.origin);
  for (const d of planner?.destinations || []) if (d?.lat != null) wpts.push(d);
  const legKm = [];
  for (let i = 1; i < wpts.length; i++) {
    legKm.push(haversineKm([wpts[i - 1].lat, wpts[i - 1].lng], [wpts[i].lat, wpts[i].lng]));
  }
  const gcTotal = legKm.reduce((a, b) => a + b, 0) || 1;
  const dests = wpts.slice(1);
  let cumGc = 0;
  const stops = dests.map((d, i) => {
    cumGc += legKm[i] || 0;
    const frac = Math.min(1, cumGc / gcTotal);
    return {
      index: i,
      label: d.label || `Stop ${i + 1}`,
      distKm: round1(distanceKm * frac),
      arriveSoc: round1(Math.max(0, startSoc - socDrop * frac)),
      etaLabel: "—",
      etaIso: null,
      dropWeightKg: d.dropWeightKg || 0,
      payloadAfterT: null,
      unloadMin: d.unloadMin || 0,
      deliverBy: d.deliverBy || null,
      onTime: null, // the offline fallback can't reliably verify deadlines
      isFinal: i === dests.length - 1,
    };
  });

  return {
    geometry,
    socProfile,
    segments,
    stops,
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
        days: 1,
        perDay: [{ day: 1, dateLabel: null, drivingH: round2(drivingH), breaks: 0 }],
        // The offline fallback can't split days or insert 11 h rests, so it does
        // NOT emit a confident EU 561 verdict (null -> the badge is hidden); only
        // the backend's day-split machine can judge compliance for a long trip.
        eu561ok: null,
      },
      assumptions: [
        "Client-side fallback estimate: the backend SOC simulator was unreachable, so this uses a constant-consumption linear model with no per-segment terrain, no charging insertion, and approximate per-stop arrivals. Start the backend (python dashboard/server.py) for the full physics-model plan.",
      ],
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

// --------------------------------------------------------------------------- //
// Charge-time reconciliation (real station power)
// --------------------------------------------------------------------------- //

// The backend plans each charge at the truck's max charge power (`maxChargeKw`,
// ~400 kW). The frontend then matches a REAL station, which may only deliver
// e.g. 300 kW — so the charge actually takes longer, and so do the ETA, total
// time, and every segment / delivery stop AFTER it. This re-stamps all of those.
//
// Scaling is one-directional (a charge is never made SHORTER): we multiply the
// planned minutes by `cap / stationKw` only when the station sits below the cap,
// which preserves the backend's taper model while correcting for the power
// deficit and can't introduce new optimism. Display-only — SOC, kWh and the
// energy headline are untouched. Mutates the passed plan objects in place.
export function reconcileChargeDurations({ segments, stops, summary, chargingStops, maxChargeKw, departure }) {
  const cap = Number.isFinite(maxChargeKw) && maxChargeKw > 0 ? maxChargeKw : 400;
  const chargeDeltas = []; // { distKm, deltaMin } per charge, in route order
  let ci = 0;
  let anyDelta = false;
  for (const seg of segments || []) {
    if (seg.type !== "charge") continue;
    const station = chargingStops?.[ci];
    const planned = Number(seg.durationMin) || 0;
    const kw = station ? Number(station.maxPowerKw) : NaN;
    let reconciled = planned;
    if (Number.isFinite(kw) && kw > 0 && kw < cap) {
      reconciled = Math.round(planned * (cap / kw));
    }
    const delta = Math.max(0, reconciled - planned);
    if (delta > 0) {
      anyDelta = true;
      seg.durationMin = reconciled;
    }
    // Always mirror the Charging-Stops card to the timeline's charge duration —
    // even when the station meets the truck's cap (delta 0) — so the card estimate
    // and the segment never disagree (they previously differed by the taper margin).
    if (station) {
      station.durationMin = reconciled;
      station.chargeMinutes = reconciled;
    }
    chargeDeltas.push({
      distKm: station && Number.isFinite(station.distKm) ? station.distKm : Infinity,
      deltaMin: delta,
    });
    ci += 1;
  }
  if (!anyDelta) return;

  // Re-stamp every segment's clock: reconstruct the backend's absolute times from
  // their HH:MM (anchored to the departure date, tracking midnight rollover) and
  // add the cumulative charge delta accrued up to each point. Walking `prevEnd`
  // on the ORIGINAL timeline keeps rollover reconstruction exact.
  const depart = parseDeparture(departure);
  depart.setSeconds(0, 0);
  let prevEnd = new Date(depart);
  let cumShift = 0; // minutes added by reconciled charges seen so far
  let cj = 0;
  let etaAbs = new Date(depart);
  for (const seg of segments) {
    const startAbs = absFromHhmm(prevEnd, seg.startTime);
    const endAbs = absFromHhmm(startAbs, seg.endTime);
    seg.startTime = hhmm(new Date(startAbs.getTime() + cumShift * 60000));
    if (seg.type === "charge") {
      cumShift += chargeDeltas[cj] ? chargeDeltas[cj].deltaMin : 0;
      cj += 1;
    }
    const newEnd = new Date(endAbs.getTime() + cumShift * 60000);
    seg.endTime = hhmm(newEnd);
    prevEnd = endAbs;
    etaAbs = newEnd;
  }

  // Summary: ETA + total time move by the full delta; charging time grows.
  const totalDelta = chargeDeltas.reduce((a, c) => a + c.deltaMin, 0);
  summary.etaLabel = hhmm(etaAbs);
  summary.etaIso = localIso(etaAbs);
  summary.totalTimeH = round2((etaAbs.getTime() - depart.getTime()) / 3600000);
  summary.chargingTimeMin = Math.round((Number(summary.chargingTimeMin) || 0) + totalDelta);
  if (Array.isArray(summary.assumptions)) {
    summary.assumptions.push(
      "Charge time reflects each matched station's real rated power; where that is below the truck's max " +
        "charge power the charge — and the ETA — run longer than a full-power estimate."
    );
  }

  // Delivery-stop ETAs after a reconciled charge shift by the deltas before them;
  // recompute the deliver-by verdict against the later arrival.
  for (const stop of stops || []) {
    const sh = chargeDeltas
      .filter((c) => c.distKm <= (Number(stop.distKm) || 0) + 1e-6)
      .reduce((a, c) => a + c.deltaMin, 0);
    if (sh <= 0 || !stop.etaIso) continue;
    const d = new Date(stop.etaIso);
    if (Number.isNaN(d.getTime())) continue;
    const nd = new Date(d.getTime() + sh * 60000);
    stop.etaIso = localIso(nd);
    stop.etaLabel = hhmm(nd);
    if (stop.deliverBy) {
      const dl = new Date(stop.deliverBy);
      if (!Number.isNaN(dl.getTime())) stop.onTime = nd.getTime() <= dl.getTime();
    }
  }
}

// First datetime >= `notBefore` whose wall-clock is `hh:mm` (handles the segment
// crossing midnight from the previous one). Anchored to notBefore's calendar day.
function absFromHhmm(notBefore, hhmmStr) {
  const [h, m] = String(hhmmStr || "0:0").split(":").map((x) => Number(x));
  const d = new Date(notBefore);
  d.setHours(h || 0, m || 0, 0, 0);
  if (d.getTime() < notBefore.getTime()) d.setDate(d.getDate() + 1);
  return d;
}

// Local (not UTC) "YYYY-MM-DDTHH:MM" — matches the naive ISO the rest of the app
// uses, so a later `new Date(iso)` re-parses in the same local frame.
function localIso(d) {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}
