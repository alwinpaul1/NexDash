// Probe: does TomTom Calculate Route return per-section posted SPEED LIMITS we
// could use to vary per-segment speed by road (autobahn 80 / town 50 / 30-zone)?
import { readFileSync } from 'fs';
const env = readFileSync(new URL('../frontend/.env', import.meta.url), 'utf8');
const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
const T = { weightKg:40000, axleWeightKg:11500, numberOfAxles:5, lengthM:16.5, widthM:2.55, heightM:4.0, maxSpeedKph:80 };
async function geo(q){const u=`https://api.tomtom.com/search/2/geocode/${encodeURIComponent(q)}.json?key=${KEY}&limit=1&countrySet=DE`;const d=await(await fetch(u)).json();const p=d.results[0].position;return`${p.lat},${p.lon}`;}
const m=await geo('Munich'), b=await geo('Berlin');
for (const st of ['speedLimit','motorway,tunnel,urban']) {
  const u=`https://api.tomtom.com/routing/1/calculateRoute/${m}:${b}/json?key=${KEY}&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=${T.maxSpeedKph}&vehicleWeight=${T.weightKg}&vehicleAxleWeight=${T.axleWeightKg}&vehicleNumberOfAxles=${T.numberOfAxles}&vehicleLength=${T.lengthM}&vehicleWidth=${T.widthM}&vehicleHeight=${T.heightM}&vehicleCommercial=true&sectionType=${encodeURIComponent(st)}`;
  const res=await fetch(u); const d=await res.json();
  console.log(`\n=== sectionType=${st} -> HTTP ${res.status} ===`);
  if (d.error || d.detailedError) { console.log('  error:', JSON.stringify(d.error||d.detailedError).slice(0,200)); continue; }
  const rt=d.routes?.[0]; const secs=rt?.sections||[];
  console.log(`  ${secs.length} sections; route ${(rt?.summary?.lengthInMeters/1000||0).toFixed(0)}km`);
  // show distinct keys + a sample
  const keys=new Set(); secs.forEach(s=>Object.keys(s).forEach(k=>keys.add(k)));
  console.log('  section keys:', [...keys].join(', '));
  console.log('  sample sections:', JSON.stringify(secs.slice(0,6), null, 0).slice(0,600));
  // if speed limits present, summarize the distribution
  const sls = secs.map(s=>s.maxSpeedInKmph ?? s.speedLimitInKmph ?? s.maxSpeedLimitInKmph).filter(v=>v!=null);
  if (sls.length) { const u2=[...new Set(sls)].sort((a,b)=>a-b); console.log('  distinct speed limits (kmph):', u2.join(', ')); }
}
