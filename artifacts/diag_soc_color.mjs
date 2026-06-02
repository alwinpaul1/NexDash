// Diagnose the post-charge SOC-colour misalignment caused by the display-only
// re-route through the charging station. Replicates RouteMap.buildSocSegments
// against the re-routed geometry + the (unchanged) backend socProfile.
import { readFileSync } from 'fs';
const env = readFileSync(new URL('../frontend/.env', import.meta.url), 'utf8');
const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
const CS = 'DE,AT,CH,NL,BE,FR,PL,CZ,DK';
const T = { weightKg:40000, axleWeightKg:11500, numberOfAxles:5, lengthM:16.5, widthM:2.55, heightM:4.0, maxSpeedKph:89 };
const R=6371, rad=d=>d*Math.PI/180;
const hav=(a,b)=>{const dLat=rad(b[0]-a[0]),dLng=rad(b[1]-a[1]);const h=Math.sin(dLat/2)**2+Math.cos(rad(a[0]))*Math.cos(rad(b[0]))*Math.sin(dLng/2)**2;return 2*R*Math.asin(Math.sqrt(h));};
async function geo(q){const u=`https://api.tomtom.com/search/2/geocode/${encodeURIComponent(q)}.json?key=${KEY}&limit=1&countrySet=${CS}`;const d=await(await fetch(u)).json();const p=d.results[0].position;return{lat:p.lat,lng:p.lon};}
async function route(wps){const locs=wps.map(w=>`${w.lat},${w.lng}`).join(':');const u=`https://api.tomtom.com/routing/1/calculateRoute/${locs}/json?key=${KEY}&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=${T.maxSpeedKph}&vehicleWeight=${T.weightKg}&vehicleAxleWeight=${T.axleWeightKg}&vehicleNumberOfAxles=${T.numberOfAxles}&vehicleLength=${T.lengthM}&vehicleWidth=${T.widthM}&vehicleHeight=${T.heightM}&vehicleCommercial=true`;const d=await(await fetch(u)).json();const rt=d.routes[0];const g=[],lt=[];for(const leg of rt.legs||[]){for(const p of leg.points||[])g.push([p.latitude,p.longitude]);const s=leg.summary||{};lt.push({lengthM:s.lengthInMeters||0,travelTimeS:s.travelTimeInSeconds||0});}const s=rt.summary||{};return{geometry:g,legTimings:lt,distanceKm:(s.lengthInMeters||0)/1000,durationS:s.travelTimeInSeconds||0};}

const munich=await geo('Munich'), berlin=await geo('Berlin');
const direct=await route([munich,berlin]);
const body={waypoints:[{...munich,label:'Munich'},{...berlin,label:'Berlin'}],geometry:direct.geometry,legTimings:direct.legTimings,distanceKm:direct.distanceKm,durationS:direct.durationS,startSoc:100,minSoc:15,payloadKg:18000,reservePct:20,maxChargeKw:400,departure:'2026-06-02T09:00',temperatureC:15};
const sim=await(await fetch('http://localhost:8000/api/route-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
const stop=sim.chargingStops[0];
// enrich -> real station coords
const sr=await(await fetch(`https://api.tomtom.com/search/2/categorySearch/EV%20charging.json?key=${KEY}&lat=${stop.lat}&lon=${stop.lng}&radius=30000&categorySet=7309&minPowerKW=150&limit=1`)).json();
const st=sr.results?.[0]?.position?{lat:sr.results[0].position.lat,lng:sr.results[0].position.lon}:{lat:stop.lat,lng:stop.lng};
const rr=await route([munich,st,berlin]);
console.log('charge stop distKm(socProfile):', stop.distKm, ' arriveSoc:', stop.arriveSoc, 'departSoc:', stop.departSoc);
console.log('direct dist:', direct.distanceKm.toFixed(1), ' re-routed dist:', rr.distanceKm.toFixed(1), ' (detour +', (rr.distanceKm-direct.distanceKm).toFixed(2),'km)');
console.log('summary.distanceKm (totalKm passed to buildSocSegments):', sim.summary.distanceKm);

// Replicate buildSocSegments mapping on the RE-ROUTED geometry
const geom=rr.geometry, sp=sim.socProfile, totalKm=sim.summary.distanceKm;
const cum=[0]; for(let i=1;i<geom.length;i++) cum.push(cum[i-1]+hav(geom[i-1],geom[i]));
const geomTotal=cum[cum.length-1]; const scale=totalKm/geomTotal;
const socAt=km=>{if(km<=sp[0].distKm)return sp[0].soc;const last=sp[sp.length-1];if(km>=last.distKm)return last.soc;for(let i=1;i<sp.length;i++){if(sp[i].distKm>=km){const a=sp[i-1],b=sp[i],f=(km-a.distKm)/((b.distKm-a.distKm)||1);return a.soc+f*(b.soc-a.soc);}}return last.soc;};
// station vertex on re-routed geom
let si=0,sd=Infinity; for(let i=0;i<geom.length;i++){const d=hav(geom[i],[st.lat,st.lng]);if(d<sd){sd=d;si=i;}}
console.log(`\nstation nearest vertex idx=${si}/${geom.length}, cum=${cum[si].toFixed(2)}km, ${(sd*1000).toFixed(0)}m off`);
// NEW anchored piecewise map (mirrors the fixed buildSocSegments)
const mono=[{g:0,s:0},{g:cum[si],s:stop.distKm},{g:geomTotal,s:totalKm}].sort((a,b)=>a.g-b.g);
const mapToSocDist=g=>{for(let i=1;i<mono.length;i++){if(g<=mono[i].g){const a=mono[i-1],b=mono[i],f=(g-a.g)/((b.g-a.g)||1);return a.s+f*(b.s-a.s);}}return mono[mono.length-1].s;};
console.log('\n=== AFTER FIX (anchored mapping) — soc walking PAST the station ===');
for(let i=Math.max(0,si-3);i<=Math.min(geom.length-1,si+12);i++){
  const oldSoc=socAt(cum[i]*scale), newSoc=socAt(mapToSocDist(cum[i])); const mark=i===si?'  <-- STATION':'';
  console.log(`  idx ${i}: cum=${cum[i].toFixed(2)}  OLD soc=${oldSoc.toFixed(1)}%  NEW soc=${newSoc.toFixed(1)}%${mark}`);
}
console.log('\n>>> NEW soc should jump to ~departSoc (green) right at/after the station vertex.');
