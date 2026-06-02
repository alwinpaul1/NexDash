// Does our ETA actually respect posted limits + live traffic, or are we
// optimistically fast? Compares our plan's driving time to TomTom's measured
// (traffic-aware) truck duration, shows the traffic delay baked in, the posted
// speed-limit mix, and our resulting average speed vs NexOS (~75 km/h).
import { readFileSync } from "fs";
const env = readFileSync(new URL("../frontend/.env", import.meta.url), "utf8");
const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
const R = 6371, rad = (d) => (d * Math.PI) / 180;
const hav = (a, b) => { const dLat = rad(b[0]-a[0]), dLng = rad(b[1]-a[1]); const h = Math.sin(dLat/2)**2 + Math.cos(rad(a[0]))*Math.cos(rad(b[0]))*Math.sin(dLng/2)**2; return 2*R*Math.asin(Math.sqrt(h)); };
const T = { w: 40000, a: 11500, n: 5, l: 16.5, wd: 2.55, h: 4.0, s: 80 };

const url = `https://api.tomtom.com/routing/1/calculateRoute/52.52,13.405:48.137,11.575/json?key=${KEY}&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=${T.s}&vehicleWeight=${T.w}&vehicleAxleWeight=${T.a}&vehicleNumberOfAxles=${T.n}&vehicleLength=${T.l}&vehicleWidth=${T.wd}&vehicleHeight=${T.h}&vehicleCommercial=true&sectionType=speedLimit`;
const rt = (await (await fetch(url)).json()).routes[0];
const sm = rt.summary;
const distKm = sm.lengthInMeters / 1000;
const travelH = sm.travelTimeInSeconds / 3600;
const freeH = (sm.noTrafficTravelTimeInSeconds || sm.travelTimeInSeconds) / 3600;
const delayMin = (sm.trafficDelayInSeconds || 0) / 60;

console.log("=== TomTom (truck profile, traffic=true, vehicleMaxSpeed=80) ===");
console.log(`  distance:            ${distKm.toFixed(1)} km`);
console.log(`  travel time W/TRAFFIC ${travelH.toFixed(2)} h  -> avg ${(distKm/travelH).toFixed(1)} km/h   <- what we use`);
console.log(`  free-flow (no traffic)${freeH.toFixed(2)} h  -> avg ${(distKm/freeH).toFixed(1)} km/h`);
console.log(`  traffic delay baked in: +${delayMin.toFixed(0)} min`);

// geometry + posted speed-limit distribution
const geom = [], cum = []; let c = 0;
for (const lg of rt.legs || []) for (const p of lg.points || []) { const pt = [p.latitude, p.longitude]; if (geom.length) c += hav(geom[geom.length-1], pt); geom.push(pt); cum.push(c); }
const spans = {};
for (const sec of rt.sections || []) {
  if (sec.sectionType && sec.sectionType !== "SPEED_LIMIT") continue;
  const kmh = Math.min(Number(sec.maxSpeedLimitInKmh), 80);
  const a = cum[sec.startPointIndex], z = cum[sec.endPointIndex];
  if (Number.isFinite(kmh) && kmh > 0 && z > a) spans[kmh] = (spans[kmh] || 0) + (z - a);
}
console.log("  posted speed-limit mix (km of route at each cap, ALL <= 80):");
for (const k of Object.keys(spans).sort((a,b)=>a-b)) console.log(`     ${k} km/h: ${spans[k].toFixed(0)} km`);

// our plan
const body = { waypoints: [{lat:52.52,lng:13.405,label:"Berlin"},{lat:48.137,lng:11.575,label:"Munich"}], geometry: geom, legTimings: [], speedLimits: [], distanceKm: distKm, durationS: sm.travelTimeInSeconds, startSoc: 100, minSoc: 15, payloadKg: 18000, reservePct: 10, maxChargeKw: 400, departure: "2026-06-02T09:00", temperatureC: 15 };
const plan = await (await fetch("http://localhost:8000/api/route-plan", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body) })).json();
const s = plan.summary;
const driveH = s.drivingTimeH;
console.log("\n=== OUR PLAN ===");
console.log(`  driving time:        ${driveH} h  -> avg ${(s.distanceKm/driveH).toFixed(1)} km/h`);
console.log(`  == TomTom traffic-aware time? ${Math.abs(driveH - travelH) < 0.12 ? "YES (we use TomTom's measured, traffic+limit-aware duration)" : "NO — DIVERGES by " + ((driveH-travelH)*60).toFixed(0) + " min"}`);
const segSpeeds = (plan.segments||[]).filter(x=>x.type==="drive").map(x=>(x.km/(x.durationMin/60)));
console.log(`  drive-segment speeds: ${segSpeeds.map(v=>v.toFixed(0)).join(", ")} km/h (all <= 80 legal cap?)  ${segSpeeds.every(v=>v<=80.5)?"YES":"NO"}`);
console.log(`  NexOS for reference drove ~75 km/h (601 km / 8h02m).`);
