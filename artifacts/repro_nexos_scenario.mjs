// Replicate the NexDash frontend pipeline for the NexOS screenshot scenario:
//   Munich -> Berlin, 100% start, 15% min, 20% reserve, 50km detour,
//   400kW min charger, 400kW max charge, 18000 kg, depart 2026-06-01 21:05.
// geocode (TomTom) -> truck route (TomTom) -> POST /api/route-plan (local backend).
// Prints the summary + segment timeline so we can diff against NexOS.

import { readFileSync } from 'fs';

// --- TomTom key from frontend/.env (VITE_TOMTOM_API_KEY) ---
const env = readFileSync(new URL('../frontend/.env', import.meta.url), 'utf8');
const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
if (!KEY) throw new Error('no TomTom key');

const COUNTRY_SET = 'DE,AT,CH,NL,BE,FR,PL,CZ,DK';
const TRUCK = { weightKg: 40000, axleWeightKg: 11500, numberOfAxles: 5, lengthM: 16.5, widthM: 2.55, heightM: 4.0, maxSpeedKph: 89 };
const API = 'http://localhost:8000';

async function geocode(q) {
  const url = `https://api.tomtom.com/search/2/geocode/${encodeURIComponent(q)}.json?key=${KEY}&limit=1&countrySet=${COUNTRY_SET}`;
  const r = await fetch(url); const d = await r.json();
  const p = d.results?.[0]?.position;
  return { label: q, lat: p.lat, lng: p.lon };
}

async function tomtomRoute(wps) {
  const locs = wps.map(w => `${w.lat},${w.lng}`).join(':');
  const url = `https://api.tomtom.com/routing/1/calculateRoute/${locs}/json?key=${KEY}`
    + `&travelMode=truck&routeType=fastest&traffic=true`
    + `&vehicleMaxSpeed=${TRUCK.maxSpeedKph}&vehicleWeight=${TRUCK.weightKg}`
    + `&vehicleAxleWeight=${TRUCK.axleWeightKg}&vehicleNumberOfAxles=${TRUCK.numberOfAxles}`
    + `&vehicleLength=${TRUCK.lengthM}&vehicleWidth=${TRUCK.widthM}&vehicleHeight=${TRUCK.heightM}&vehicleCommercial=true`;
  const r = await fetch(url); const d = await r.json();
  const route = d.routes[0];
  const geometry = [], legTimings = [];
  for (const leg of route.legs || []) {
    for (const p of leg.points || []) geometry.push([p.latitude, p.longitude]);
    const ls = leg.summary || {};
    legTimings.push({ lengthM: ls.lengthInMeters || 0, travelTimeS: ls.travelTimeInSeconds || 0 });
  }
  const s = route.summary || {};
  return { geometry, legTimings, distanceKm: (s.lengthInMeters || 0) / 1000, durationS: s.travelTimeInSeconds || 0, trafficDelayS: s.trafficDelayInSeconds || 0 };
}

const munich = await geocode('Munich');
const berlin = await geocode('Berlin');
console.log('Munich:', munich, '\nBerlin:', berlin);

const waypoints = [
  { ...munich },
  { ...berlin, dropWeightKg: 0, unloadMin: 0, deliverBy: '2026-06-02T21:10' },
];

const route = await tomtomRoute(waypoints);
console.log(`\nTomTom truck route: ${route.distanceKm.toFixed(1)} km, ${(route.durationS/3600).toFixed(2)} h drive, traffic delay ${(route.trafficDelayS/60).toFixed(0)} min, ${route.geometry.length} polyline pts, ${route.legTimings.length} legs`);

// Body EXACTLY as frontend backendPlan() builds it (note: NO chargeTargetSoc/strategy/minChargerKw/maxDetourKm sent)
const body = {
  waypoints: waypoints.map(w => ({ lat: w.lat, lng: w.lng, label: w.label, dropWeightKg: w.dropWeightKg || 0, unloadMin: w.unloadMin || 0, deliverBy: w.deliverBy || '' })),
  geometry: route.geometry,
  legTimings: route.legTimings,
  distanceKm: route.distanceKm,
  durationS: route.durationS,
  startSoc: 100,
  minSoc: 15,
  payloadKg: 18000,
  reservePct: 20,
  maxChargeKw: 400,
  departure: '2026-06-01T21:05',
  temperatureC: 15,
};

const res = await fetch(`${API}/api/route-plan`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
if (!res.ok) { console.error('route-plan failed', res.status, await res.text()); process.exit(1); }
const sim = await res.json();

const sm = sim.summary || {};
console.log('\n========== LOCAL NEXDASH SUMMARY ==========');
console.log(JSON.stringify({
  distanceKm: sm.distanceKm, drivingTimeH: sm.drivingTimeH, chargingTimeMin: sm.chargingTimeMin,
  totalTimeH: sm.totalTimeH, etaLabel: sm.etaLabel, etaIso: sm.etaIso,
  startSoc: sm.startSoc, arrivalSoc: sm.arrivalSoc, minSoc: sm.minSoc,
  energyKwh: sm.energyKwh, kwhPer100: sm.kwhPer100, chargingCostEur: sm.chargingCostEur,
  chargingStops: sm.chargingStops, elevationGainM: sm.elevationGainM,
}, null, 2));
console.log('\nDRIVER:', JSON.stringify(sm.driver, null, 2));
console.log('\n========== SEGMENT TIMELINE ==========');
for (const s of sim.segments || []) {
  const km = s.km != null ? `${s.km}km` : '';
  const dur = s.durationMin != null ? `${s.durationMin}min` : '';
  const soc = (s.socStart != null) ? `${s.socStart}->${s.socEnd}%` : '';
  const t = (s.startTime || s.endTime) ? `${s.startTime||''}-${s.endTime||''}` : '';
  const extra = [s.kWh!=null?`${s.kWh}kWh`:'', s.station?.name||'', s.label||''].filter(Boolean).join(' ');
  console.log(`  ${s.type.padEnd(11)} ${km.padEnd(8)} ${dur.padEnd(7)} ${soc.padEnd(13)} ${t.padEnd(14)} ${extra}`);
}
console.log('\n========== CHARGING STOPS (raw, pre-enrichment) ==========');
for (const c of sim.chargingStops || []) console.log('  ', JSON.stringify(c));
console.log('\n========== STOPS (deliveries) ==========');
for (const s of sim.stops || []) console.log('  ', JSON.stringify(s));
console.log('\n========== ASSUMPTIONS ==========');
for (const a of sm.assumptions || []) console.log('  -', a);
