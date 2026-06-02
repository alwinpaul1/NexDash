// Verify Tier S: posted speed limits shape per-segment speed, ETA stays exact.
import { readFileSync } from 'fs';
const env = readFileSync(new URL('../frontend/.env', import.meta.url), 'utf8');
const KEY = (env.match(/VITE_TOMTOM_API_KEY=(.+)/) || [])[1]?.trim();
const T = { weightKg:40000, axleWeightKg:11500, numberOfAxles:5, lengthM:16.5, widthM:2.55, heightM:4.0, maxSpeedKph:80 };
const R=6371, rad=d=>d*Math.PI/180;
const hav=(a,b)=>{const dLat=rad(b[0]-a[0]),dLng=rad(b[1]-a[1]);const h=Math.sin(dLat/2)**2+Math.cos(rad(a[0]))*Math.cos(rad(b[0]))*Math.sin(dLng/2)**2;return 2*R*Math.asin(Math.sqrt(h));};
async function geo(q){const u=`https://api.tomtom.com/search/2/geocode/${encodeURIComponent(q)}.json?key=${KEY}&limit=1&countrySet=DE`;const d=await(await fetch(u)).json();const p=d.results[0].position;return{lat:p.lat,lng:p.lon};}
const m=await geo('Munich'), b=await geo('Berlin');
const locs=`${m.lat},${m.lng}:${b.lat},${b.lng}`;
const u=`https://api.tomtom.com/routing/1/calculateRoute/${locs}/json?key=${KEY}&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=${T.maxSpeedKph}&vehicleWeight=${T.weightKg}&vehicleAxleWeight=${T.axleWeightKg}&vehicleNumberOfAxles=${T.numberOfAxles}&vehicleLength=${T.lengthM}&vehicleWidth=${T.widthM}&vehicleHeight=${T.heightM}&vehicleCommercial=true&sectionType=speedLimit`;
const rt=(await(await fetch(u)).json()).routes[0];
const geom=[],cum=[],lt=[]; let c=0;
for(const leg of rt.legs||[]){for(const p of leg.points||[]){const pt=[p.latitude,p.longitude]; if(geom.length)c+=hav(geom[geom.length-1],pt); geom.push(pt); cum.push(c);} const s=leg.summary||{}; lt.push({lengthM:s.lengthInMeters||0,travelTimeS:s.travelTimeInSeconds||0});}
const speedLimits=[];
for(const sec of rt.sections||[]){const kmh=Number(sec.maxSpeedLimitInKmh);const a=cum[sec.startPointIndex],z=cum[sec.endPointIndex];if(Number.isFinite(kmh)&&kmh>0&&Number.isFinite(a)&&Number.isFinite(z)&&z>a)speedLimits.push({fromKm:a,toKm:z,kmh:Math.min(kmh,T.maxSpeedKph)});}
console.log(`speedLimits: ${speedLimits.length} spans; distinct kmh:`, [...new Set(speedLimits.map(s=>s.kmh))].sort((a,b)=>a-b).join(','));
const s=rt.summary||{};
const body={waypoints:[{...m,label:'Munich'},{...b,label:'Berlin'}],geometry:geom,legTimings:lt,speedLimits,distanceKm:(s.lengthInMeters||0)/1000,durationS:s.travelTimeInSeconds||0,startSoc:100,minSoc:15,payloadKg:18000,reservePct:20,maxChargeKw:400,departure:'2026-06-02T09:00',temperatureC:15};
const sim=await(await fetch('http://localhost:8000/api/route-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
console.log('TomTom measured duration:', (body.durationS/3600).toFixed(3),'h');
console.log('Plan drivingTimeH (must match ~exactly = ETA preserved):', sim.summary.drivingTimeH);
console.log('Speed assumption present:', (sim.summary.assumptions||[]).some(a=>a.includes('POSTED speed limits')));
console.log('\nDrive-segment speeds (km / min -> km/h) — should VARY by road:');
for(const seg of (sim.segments||[]).filter(x=>x.type==='drive')) console.log(`  ${seg.km}km / ${seg.durationMin}min = ${(seg.km/(seg.durationMin/60)).toFixed(1)} km/h`);
