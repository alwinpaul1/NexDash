// Verifies the time-optimal charger selection: rankChargersByTime() must rank a
// faster charger (a little further off-route) ABOVE a slow one on the line, cap
// power at the truck's max, and push unknown-power sites last. Part 2 runs a LIVE
// TomTom search at the real Berlin->Munich charge point to show old (nearest) vs
// new (fastest) — proving we'd swap the slow 150 kW Allego for a faster site.
import { rankChargersByTime } from "../frontend/src/lib/routePlanner.js";
import { readFileSync } from "fs";

let fail = 0;
const ok = (label, cond) => { if (!cond) fail++; console.log(`  ${cond ? "✓" : "✗"} ${label}`); };
const mk = (name, kw, distM) => ({ poi: { name }, dist: distM, chargingPark: { connectors: [{ ratedPowerKW: kw }] } });

console.log("Unit — rankChargersByTime:");
{
  const r = rankChargersByTime([mk("Allego", 150, 1000), mk("Aral", 300, 8000)], { energyKwh: 480, maxChargeKw: 400 });
  ok("faster-but-farther (300kW@8km) beats slow-but-near (150kW@1km)", r[0].c.poi.name === "Aral");
}
{
  const r = rankChargersByTime([mk("Far", 300, 20000), mk("Near", 300, 2000)], { energyKwh: 480 });
  ok("equal power -> nearer wins (less detour)", r[0].c.poi.name === "Near");
}
{
  const r = rankChargersByTime([mk("NoPower", 0, 500), mk("Has", 200, 6000)], { energyKwh: 480 });
  ok("unknown-power site ranks last", r[r.length - 1].c.poi.name === "NoPower");
}
{
  const s6 = rankChargersByTime([mk("S600", 600, 0)], { energyKwh: 480, maxChargeKw: 400 })[0].score;
  const s4 = rankChargersByTime([mk("S400", 400, 0)], { energyKwh: 480, maxChargeKw: 400 })[0].score;
  ok("power capped at truck max (600 ≈ 400)", Math.abs(s6 - s4) < 0.01);
}
{
  // A modest detour for big power still wins; an extreme detour for tiny gain loses.
  const r = rankChargersByTime([mk("Near150", 150, 1000), mk("Mid300", 300, 5000)], { energyKwh: 480, maxChargeKw: 400 });
  ok("300kW@5km decisively beats 150kW@1km", r[0].c.poi.name === "Mid300");
}

console.log("\nLive — real candidates at the planned Berlin->Munich charge point (18t):");
try {
  const env = readFileSync(new URL("../frontend/.env", import.meta.url), "utf8");
  const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
  // Mirror the real frontend flow: route through TomTom (truck) FIRST so the
  // backend places the charge on the actual road, not a straight-line proxy.
  const T = { weightKg: 40000, axleWeightKg: 11500, numberOfAxles: 5, lengthM: 16.5, widthM: 2.55, heightM: 4.0, maxSpeedKph: 80 };
  const m = { lat: 52.52, lng: 13.405 }, b = { lat: 48.137, lng: 11.575 };
  const ru = `https://api.tomtom.com/routing/1/calculateRoute/${m.lat},${m.lng}:${b.lat},${b.lng}/json?key=${KEY}&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=${T.maxSpeedKph}&vehicleWeight=${T.weightKg}&vehicleAxleWeight=${T.axleWeightKg}&vehicleNumberOfAxles=${T.numberOfAxles}&vehicleLength=${T.lengthM}&vehicleWidth=${T.widthM}&vehicleHeight=${T.heightM}&vehicleCommercial=true`;
  const rt = (await (await fetch(ru)).json()).routes[0];
  const geometry = [];
  for (const leg of rt.legs || []) for (const p of leg.points || []) geometry.push([p.latitude, p.longitude]);
  const sm = rt.summary || {};
  const body = { waypoints: [{ lat: m.lat, lng: m.lng, label: "Berlin" }, { lat: b.lat, lng: b.lng, label: "Munich" }], geometry, legTimings: [], speedLimits: [], distanceKm: (sm.lengthInMeters || 0) / 1000, durationS: sm.travelTimeInSeconds || 0, startSoc: 100, minSoc: 15, payloadKg: 18000, reservePct: 10, maxChargeKw: 400, departure: "2026-06-02T09:00", temperatureC: 15 };
  const plan = await (await fetch("http://localhost:8000/api/route-plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  const cs = plan.chargingStops?.[0];
  const u = `https://api.tomtom.com/search/2/categorySearch/EV%20charging.json?key=${KEY}&lat=${cs.lat}&lon=${cs.lng}&radius=50000&categorySet=7309&limit=12&minPowerKW=150&connectorSet=IEC62196Type2CCS&relatedPois=child`;
  const cands = (await (await fetch(u)).json()).results || [];
  const powerOf = (c) => (c.chargingPark?.connectors || []).reduce((m, x) => Math.max(m, Number(x.ratedPowerKW) || 0), 0);
  const chMin = (c) => Math.round((cs.kWh / (Math.min(400, powerOf(c)) * 0.9)) * 60);
  const fmt = (c) => `${(c.poi?.name || "?").slice(0, 26)} — ${powerOf(c)}kW, ${((c.dist || 0) / 1000).toFixed(1)}km off → ~${chMin(c)}min`;
  // Mirror the app's live availability check (CCS connectors with a definite status).
  const freeSlots = async (id) => {
    if (!id) return null;
    try {
      const d = await (await fetch(`https://api.tomtom.com/search/2/chargingAvailability.json?key=${KEY}&chargingAvailability=${encodeURIComponent(id)}`)).json();
      let avail = 0, total = 0, definite = 0;
      for (const c of d.connectors || []) {
        const t = String(c.type || ""); if (!(t.includes("CCS") || t.includes("Combo"))) continue;
        total += c.total || 0; const cur = c.availability?.current || {};
        const a = Number(cur.available) || 0; avail += a;
        definite += a + (Number(cur.occupied) || 0) + (Number(cur.reserved) || 0) + (Number(cur.outOfService) || 0);
      }
      return total === 0 || definite === 0 ? null : { available: avail, total };
    } catch { return null; }
  };
  console.log(`  charge point ~${cs.distKm} km in, needs ${Math.round(cs.kWh)} kWh; ${cands.length} candidates`);
  const nearest = [...cands].sort((a, b) => (a.dist || 0) - (b.dist || 0))[0];
  const ranked = rankChargersByTime(cands, { energyKwh: cs.kWh, maxChargeKw: 400 });
  const topK = ranked.slice(0, 6);
  const avs = await Promise.all(topK.map(({ c }) => freeSlots(c.dataSources?.chargingAvailability?.id)));
  const fi = avs.findIndex((a) => a && a.available > 0);
  const fastestFree = fi >= 0 ? topK[fi].c : ranked[0].c;
  const freeNote = fi >= 0 ? `(${avs[fi].available} of ${avs[fi].total} free)` : "(no live free data → fastest overall)";
  console.log(`  OLD (nearest):      ${fmt(nearest)}`);
  console.log(`  NEW (fastest free): ${fmt(fastestFree)} ${freeNote}`);
  console.log(`  -> saves ~${chMin(nearest) - chMin(fastestFree)} min of charging`);
} catch (e) { console.log("  (live probe skipped:", e.message, ")"); }

console.log(fail === 0 ? "\nUNIT: ALL PASS" : `\nUNIT: ${fail} FAIL`);
process.exit(fail === 0 ? 0 : 1);
