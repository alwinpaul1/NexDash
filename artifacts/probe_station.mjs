// Probe: where does our charging marker actually land vs the route line + the
// real station, for Munich->Berlin. Replicates the frontend enrichStations lookup.
import { readFileSync } from 'fs';
const env = readFileSync(new URL('../frontend/.env', import.meta.url), 'utf8');
const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
const COUNTRY_SET = 'DE,AT,CH,NL,BE,FR,PL,CZ,DK';
const TRUCK = { weightKg: 40000, axleWeightKg: 11500, numberOfAxles: 5, lengthM: 16.5, widthM: 2.55, heightM: 4.0, maxSpeedKph: 89 };

function hav(a, b) {
  const R = 6371, r = d => d * Math.PI / 180;
  const dLat = r(b[0] - a[0]), dLng = r(b[1] - a[1]);
  const h = Math.sin(dLat/2)**2 + Math.cos(r(a[0]))*Math.cos(r(b[0]))*Math.sin(dLng/2)**2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
async function geocode(q){const u=`https://api.tomtom.com/search/2/geocode/${encodeURIComponent(q)}.json?key=${KEY}&limit=1&countrySet=${COUNTRY_SET}`;const d=await (await fetch(u)).json();const p=d.results[0].position;return{lat:p.lat,lng:p.lon};}
async function route(wps){const locs=wps.map(w=>`${w.lat},${w.lng}`).join(':');const u=`https://api.tomtom.com/routing/1/calculateRoute/${locs}/json?key=${KEY}&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=${TRUCK.maxSpeedKph}&vehicleWeight=${TRUCK.weightKg}&vehicleAxleWeight=${TRUCK.axleWeightKg}&vehicleNumberOfAxles=${TRUCK.numberOfAxles}&vehicleLength=${TRUCK.lengthM}&vehicleWidth=${TRUCK.widthM}&vehicleHeight=${TRUCK.heightM}&vehicleCommercial=true`;const d=await (await fetch(u)).json();const rt=d.routes[0];const g=[],lt=[];for(const leg of rt.legs||[]){for(const p of leg.points||[])g.push([p.latitude,p.longitude]);const s=leg.summary||{};lt.push({lengthM:s.lengthInMeters||0,travelTimeS:s.travelTimeInSeconds||0});}const s=rt.summary||{};return{geometry:g,legTimings:lt,distanceKm:(s.lengthInMeters||0)/1000,durationS:s.travelTimeInSeconds||0};}

const munich=await geocode('Munich'), berlin=await geocode('Berlin');
const r=await route([munich,berlin]);
const body={waypoints:[{...munich,label:'Munich'},{...berlin,label:'Berlin'}],geometry:r.geometry,legTimings:r.legTimings,distanceKm:r.distanceKm,durationS:r.durationS,startSoc:100,minSoc:15,payloadKg:18000,reservePct:20,maxChargeKw:400,departure:'2026-06-01T21:05',temperatureC:15};
const sim=await (await fetch('http://localhost:8000/api/route-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
const stop=(sim.chargingStops||[])[0];
console.log('Backend on-road charge point:', stop.lat, stop.lng, '@', stop.distKm, 'km,', stop.name);
const minToRoute = pt => Math.min(...r.geometry.map(g=>hav(g,[pt.lat,pt.lng])));
console.log('  on-road point distance to nearest route vertex:', (minToRoute(stop)*1000).toFixed(0),'m (should be ~0, it IS on the line)');

// Replicate enrichStations: nearest CCS >=150kW within 30km
const radius=30000, minKw=150;
const search=async(extra='')=>{const u=`https://api.tomtom.com/search/2/categorySearch/EV%20charging.json?key=${KEY}&lat=${stop.lat}&lon=${stop.lng}&radius=${radius}&categorySet=7309&limit=5&openingHours=nextSevenDays&relatedPois=child${extra}`;const d=await (await fetch(u)).json();return d.results||[];};
let cands=await search(`&minPowerKW=${minKw}&connectorSet=IEC62196Type2CCS`);
if(!cands.length)cands=await search(`&minPowerKW=${minKw}`);
if(!cands.length)cands=await search();
const real=cands[0];
if(real){
  const rp={lat:real.position.lat,lng:real.position.lon};
  console.log('\nReal station picked:', real.poi?.name, '|', real.address?.freeformAddress||real.address?.municipality);
  console.log('  coords:', rp.lat, rp.lng);
  console.log('  distance from on-road charge point:', (hav([stop.lat,stop.lng],[rp.lat,rp.lng])*1000).toFixed(0),'m');
  console.log('  distance from nearest ROUTE vertex:', (minToRoute(rp)*1000).toFixed(0),'m  <-- if large, marker is OFF the route line and the route does NOT detour to it');
  // Verify the FIX: route origin -> realStation -> dest and re-measure.
  const rr = await route([munich, rp, berlin]);
  const minToReroute = Math.min(...rr.geometry.map(g=>hav(g,[rp.lat,rp.lng])));
  console.log('\n--- AFTER FIX (route through the station) ---');
  console.log('  re-routed distance:', rr.distanceKm.toFixed(1),'km (was', r.distanceKm.toFixed(1),'km; detour +'+(rr.distanceKm-r.distanceKm).toFixed(1)+'km)');
  console.log('  station distance from nearest re-routed vertex:', (minToReroute*1000).toFixed(0),'m  <-- should be ~0: the line now passes through the station');
}else{
  console.log('\nNo real station found -> marker stays at the on-road point.');
}
