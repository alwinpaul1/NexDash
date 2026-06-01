export const meta = {
  name: 'nexos-vs-nexdash-parity',
  description: 'Drive the Vercel agent-browser CLI across the NexOS demo planner and local NexDash, capture API calls + results for the same Berlin->Munich route + filters, plus a deterministic backend ground-truth, then synthesize a parity verdict.',
  whenToUse: 'Comparing the NexOS demo planner vs local NexDash route results and explaining any divergence.',
  phases: [
    { title: 'NexDash API truth', detail: 'Deterministic curl: TomTom truck geometry -> /api/route-plan', model: 'sonnet' },
    { title: 'NexOS recon', detail: 'Vercel agent-browser: drive nexos-demo-planner.netlify.app, capture API calls + rendered values', model: 'sonnet' },
    { title: 'NexDash live', detail: 'Vercel agent-browser: drive localhost:4173, capture /api/route-plan call + rendered values', model: 'sonnet' },
    { title: 'Synthesis', detail: 'Compare, classify divergences as intentional vs defect, write verdict' },
  ],
}

// ---- Shared facts every agent needs (agents are context-free) -------------
const ROUTE = 'Berlin (52.5200, 13.4050) -> Munich/Muenchen (48.1372, 11.5756)'
const FILTERS = 'Truck = Mercedes eActros 600. startSoc=90%, minSoc=15%, reservePct=20% (=> ~22% effective floor), maxChargeKw=400, chargeTargetSoc=95%, payload=18000 kg, temperature=15C.'

// ---- Schemas (flat + nullable to avoid StructuredOutput non-compliance) ---
const SITE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['site', 'captureStatus', 'notes'],
  properties: {
    site: { type: 'string' },
    origin: { type: ['string', 'null'] },
    destination: { type: ['string', 'null'] },
    distanceKm: { type: ['number', 'null'] },
    energyKwh: { type: ['number', 'null'] },
    kwhPer100km: { type: ['number', 'null'] },
    chargingStops: { type: ['integer', 'null'] },
    chargeWindowPct: { type: ['string', 'null'], description: 'e.g. "22% -> 95%"' },
    arrivalSocPct: { type: ['number', 'null'] },
    totalTime: { type: ['string', 'null'] },
    strategy: { type: ['string', 'null'], description: 'e.g. Minimize Cost / Minimize Time' },
    exposedFilters: { type: ['string', 'null'], description: 'Which input filters the UI exposes (SoC, charger power, payload, etc.)' },
    apiCalls: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['url', 'method'],
        properties: {
          url: { type: 'string' },
          method: { type: 'string' },
          purpose: { type: ['string', 'null'], description: 'geocode / routing / data / energy / charging POIs' },
          hasEnergyField: { type: ['boolean', 'null'], description: 'true if the response carries an energy/kWh/SoC field' },
        },
      },
    },
    energyComputedClientSide: { type: ['boolean', 'null'] },
    captureStatus: { type: 'string', enum: ['ok', 'partial', 'failed'] },
    notes: { type: 'string' },
  },
}

const API_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['status', 'notes'],
  properties: {
    distanceKm: { type: ['number', 'null'] },
    drivingTimeH: { type: ['number', 'null'] },
    energyKwh: { type: ['number', 'null'] },
    kwhPer100km: { type: ['number', 'null'] },
    chargingStops: { type: ['integer', 'null'] },
    arrivalSocPct: { type: ['number', 'null'] },
    minSocPct: { type: ['number', 'null'] },
    totalTimeH: { type: ['number', 'null'] },
    elevationGainM: { type: ['number', 'null'] },
    usedGeometry: { type: ['boolean', 'null'], description: 'true if real TomTom polyline was sent (terrain-enriched)' },
    requestSummary: { type: ['string', 'null'] },
    status: { type: 'string', enum: ['ok', 'failed'] },
    notes: { type: 'string' },
  },
}

// Vercel agent-browser (https://github.com/vercel-labs/agent-browser) is installed on PATH (v0.27).
const AGENT_BROWSER_BOOT =
  'BROWSER TOOL: Use the VERCEL agent-browser CLI via Bash (it is on PATH as `agent-browser`, v0.27, Chrome-for-Testing already installed). Do NOT use any Playwright MCP — use agent-browser commands.\n' +
  'Run `export AGENT_BROWSER_DEFAULT_TIMEOUT=25000` first. Use an ISOLATED session by passing --session SESSION to EVERY command so you never collide with another browser.\n' +
  'HOOK NOTE: if a hook blocks your first Bash command asking you to "present facts", state (1) the task and (2) what the command does, then immediately retry the SAME command.\n' +
  'CORE WORKFLOW:\n' +
  '  agent-browser --session S open <url>                  # launch + navigate\n' +
  '  agent-browser --session S wait --load networkidle      # let it settle\n' +
  '  agent-browser --session S snapshot -i                  # interactive a11y tree with @eN refs\n' +
  '  agent-browser --session S find placeholder "<ph>" fill "<text>"   # fill by placeholder (or: fill @eN "text")\n' +
  '  agent-browser --session S click @eN                    # click a ref from the latest snapshot\n' +
  '  agent-browser --session S wait --text "kWh"            # wait for result text (or wait <ms>)\n' +
  '  agent-browser --session S get text @eN                 # read a value\n' +
  '  agent-browser --session S network requests --json      # list captured API/XHR calls\n' +
  '  agent-browser --session S network requests --filter <substr> --json   # filtered\n' +
  '  agent-browser --session S network request <id> --json  # full request+response detail (inspect for energy fields)\n' +
  '  agent-browser --session S screenshot <path>            # save a screenshot\n' +
  '  agent-browser --session S close                        # close when done\n' +
  'AUTOCOMPLETE INPUTS: after fill, run `wait 1800` then `snapshot -i` and CLICK the first suggestion option (a row/option whose text matches the city) before moving on. Re-snapshot after the page changes; refs are only valid for the latest snapshot.'

// =========================================================================
phase('NexDash API truth')
const apiTruth = await agent(
  'You produce the DETERMINISTIC ground-truth route plan from the local NexDash backend, reproducing exactly what the browser app sends.\n\n' +
  'Work in /Users/alwinpaul/Desktop/Project/NexDash. Use Bash (python3 + curl available).\n' +
  'HOOK NOTE: if a hook blocks your first Bash command asking you to "present facts", state (1) the task: reproduce NexDash backend route plan, (2) what the command does, then immediately retry the SAME command.\n\n' +
  'STEPS:\n' +
  '1. Read the TomTom key from frontend/.env (line VITE_TOMTOM_API_KEY=...).\n' +
  '2. Call TomTom Calculate Route (truck profile) with EXACTLY these params (this mirrors the app):\n' +
  '   https://api.tomtom.com/routing/1/calculateRoute/52.5200,13.4050:48.1372,11.5756/json?key=KEY&travelMode=truck&routeType=fastest&traffic=true&vehicleMaxSpeed=89&vehicleWeight=40000&vehicleAxleWeight=11500&vehicleNumberOfAxles=5&vehicleLength=16.5&vehicleWidth=2.55&vehicleHeight=4.0&vehicleCommercial=true\n' +
  '3. From routes[0]: geometry = [[lat,lng], ...] from every legs[].points[] ({latitude,longitude}); legTimings = [{lengthM: leg.summary.lengthInMeters, travelTimeS: leg.summary.travelTimeInSeconds}, ...]; distanceKm = routes[0].summary.lengthInMeters/1000; durationS = routes[0].summary.travelTimeInSeconds.\n' +
  '4. POST that to http://localhost:8000/api/route-plan as JSON: {waypoints:[{lat:52.52,lng:13.405,label:"Berlin"},{lat:48.1372,lng:11.5756,label:"Munich"}], geometry, legTimings, distanceKm, durationS, startSoc:90, minSoc:15, reservePct:20, payloadKg:18000, maxChargeKw:400, chargeTargetSoc:95, temperatureC:15}. Write the body to a temp file and use curl -d @file.\n' +
  '5. Parse response["summary"]: distanceKm, drivingTimeH, energyKwh, kwhPer100 (->kwhPer100km), chargingStops, arrivalSoc (->arrivalSocPct), minSoc (->minSocPct), totalTimeH, elevationGainM. Set usedGeometry=true.\n' +
  'FALLBACK: if TomTom fails (non-200/rate limit), POST the same body WITHOUT geometry/legTimings (flat fallback), set usedGeometry=false, say so in notes. ALWAYS return a result.\n\n' +
  'ROUTE: ' + ROUTE + '\nFILTERS: ' + FILTERS,
  { schema: API_SCHEMA, label: 'nexdash-api-truth', phase: 'NexDash API truth', model: 'sonnet' }
)
log('API truth: ' + (apiTruth && apiTruth.status === 'ok'
  ? (apiTruth.energyKwh + ' kWh, ' + apiTruth.kwhPer100km + ' kWh/100km, ' + apiTruth.chargingStops + ' stop(s), arr ' + apiTruth.arrivalSocPct + '%, elev +' + apiTruth.elevationGainM + 'm, geometry=' + apiTruth.usedGeometry)
  : 'FAILED'))

// =========================================================================
phase('NexOS recon')
const nexos = await agent(
  'You investigate the PUBLIC NexOS demo EV route planner with the Vercel agent-browser and report how it works for a Berlin->Munich truck route. Use session name "nexos".\n\n' +
  AGENT_BROWSER_BOOT + '\n\n' +
  'STEPS (session "nexos"):\n' +
  '1. open https://nexos-demo-planner.netlify.app/plan ; wait --load networkidle ; snapshot -i to read the UI.\n' +
  '2. ORIGIN: fill the first location/search input with "Berlin"; wait 1800; snapshot -i; click the first Germany suggestion.\n' +
  '3. DESTINATION: fill with "Muenchen" (if no results, try "München" then "Munich"); wait 1800; snapshot -i; click the first suggestion.\n' +
  '4. From the snapshots, record which FILTERS/inputs the UI exposes (start SoC? min SoC? charger power? payload? a strategy selector like "Minimize Cost"?) into exposedFilters and strategy.\n' +
  '5. Trigger the plan: click the Plan / Calculate / Go / Optimize button. Then `wait --text "kWh"` (allow up to ~30s; if "kWh" never shows, wait for a stops/summary panel via snapshot).\n' +
  '6. `network requests --json` to list meaningful API/XHR calls (geocode e.g. Nominatim, routing e.g. TomTom, data e.g. Supabase, charging POIs). Use `network request <id> --json` on the data/route calls to inspect their RESPONSE bodies. For each call set purpose + hasEnergyField. CRITICAL: if NO response carries an energy/kWh/SoC field, the planner computes energy CLIENT-SIDE -> set energyComputedClientSide=true.\n' +
  '7. Read rendered RESULT values (snapshot + get text): total distance km, energy kWh and kWh/100km if shown, number of charging stops, charge window (e.g. "22% -> 95%"), arrival SoC %, total time.\n' +
  '8. screenshot /tmp/nexos-result.png ; then close.\n' +
  '9. Return SITE_SCHEMA with site="NexOS demo". If a value is not visible, use null — DO NOT fabricate. captureStatus ok/partial/failed. Notable details (units, strategy, errors) in notes.\n\n' +
  'ROUTE: ' + ROUTE,
  { schema: SITE_SCHEMA, label: 'nexos-browser', phase: 'NexOS recon', model: 'sonnet' }
)
log('NexOS: ' + (nexos ? nexos.captureStatus + ' — ' + (nexos.energyKwh ?? '?') + ' kWh, ' + (nexos.kwhPer100km ?? '?') + ' kWh/100km, ' + (nexos.chargingStops ?? '?') + ' stop(s), clientSideEnergy=' + nexos.energyComputedClientSide : 'FAILED'))

// =========================================================================
phase('NexDash live')
const local = await agent(
  'You drive the Vercel agent-browser against our LOCAL NexDash app and capture both the rendered plan and the backend API call. Use session name "nexdash". The NexOS recon already finished.\n\n' +
  AGENT_BROWSER_BOOT + '\n\n' +
  'KNOWN UI (verified): origin input placeholder = "Where does the trip start?"; destination input placeholder = "Add a destination…"; the submit button text contains "Optimize Route" (it is DISABLED until both origin and a destination are set). Starting battery defaults to 100% — leave sliders at defaults; the deterministic API agent already covers exact-filter numbers.\n\n' +
  'STEPS (session "nexdash"):\n' +
  '1. open http://localhost:4173/ ; wait --load networkidle ; snapshot -i.\n' +
  '2. ORIGIN: `find placeholder "Where does the trip start?" fill "Berlin"` ; wait 1800 ; snapshot -i ; click the first Berlin suggestion option.\n' +
  '3. DESTINATION: `find placeholder "Add a destination…" fill "Munich"` ; wait 1800 ; snapshot -i ; click the first Munich suggestion option.\n' +
  '4. snapshot -i to confirm the "Optimize Route" button is now ENABLED, then click it (find text "Optimize Route" click, or its @ref).\n' +
  '5. `wait --text "kWh"` (allow up to ~40s) for the results panel with energy + charging stops.\n' +
  '6. `network requests --filter route-plan --json` to find the POST /api/route-plan; `network request <id> --json` to capture its REQUEST body (startSoc/minSoc/reservePct/payloadKg/maxChargeKw + geometry length) and RESPONSE (energyKwh, chargingStops). Confirm HTTP 200. Also note the TomTom routing call. Put a one-line summary in notes.\n' +
  '7. Read rendered RESULT values (snapshot + get text): total distance km, energy kWh + kWh/100km, charging stops, arrival SoC %, total time, elevation gain if shown.\n' +
  '8. screenshot /tmp/nexdash-result.png ; then close.\n' +
  '9. Return SITE_SCHEMA with site="NexDash local". This app computes energy SERVER-SIDE via /api/route-plan, so energyComputedClientSide=false; list /api/route-plan (purpose="energy", hasEnergyField=true) plus the TomTom routing call in apiCalls. Null for anything not visible. captureStatus ok/partial/failed; details in notes.\n\n' +
  'ROUTE: ' + ROUTE + '\nFILTERS to aim for (best-effort; defaults fine): ' + FILTERS,
  { schema: SITE_SCHEMA, label: 'nexdash-browser', phase: 'NexDash live', model: 'sonnet' }
)
log('NexDash live: ' + (local ? local.captureStatus + ' — ' + (local.energyKwh ?? '?') + ' kWh, ' + (local.kwhPer100km ?? '?') + ' kWh/100km, ' + (local.chargingStops ?? '?') + ' stop(s)' : 'FAILED'))

// =========================================================================
phase('Synthesis')
const report = await agent(
  'You are the synthesis analyst. Write a clear, honest comparison report (GitHub-flavored markdown) of the NexOS demo planner vs local NexDash for the SAME route + filters. FAIL LOUD: if any capture below failed or returned nulls, say so explicitly — never paper over a gap.\n\n' +
  'ROUTE: ' + ROUTE + '\nFILTERS: ' + FILTERS + '\n\n' +
  'DATA — NexOS demo (live Vercel agent-browser capture):\n' + JSON.stringify(nexos, null, 2) + '\n\n' +
  'DATA — NexDash local (live Vercel agent-browser capture):\n' + JSON.stringify(local, null, 2) + '\n\n' +
  'DATA — NexDash backend deterministic ground-truth (curl, real TomTom geometry):\n' + JSON.stringify(apiTruth, null, 2) + '\n\n' +
  'WRITE THESE SECTIONS:\n' +
  '1. "How NexOS makes its API calls / how it works" — from its captured apiCalls: geocoding, routing, data backend, and the KEY finding of whether energy is computed client-side (no energy field in any response) vs server-side. State its strategy and which filters it exposes.\n' +
  '2. "How NexDash works (same location + filters)" — TomTom truck routing in the browser for geometry, then POST /api/route-plan where the backend runs the energy model + Open-Meteo elevation/gradient enrichment (server-side). Contrast with NexOS.\n' +
  '3. A METRICS TABLE: rows = Distance km, Energy kWh, kWh/100km, Charging stops, Charge window, Arrival SoC %, Total time. Columns = NexOS | NexDash (live UI) | NexDash (API truth) | Delta (NexDash vs NexOS) | Classification. Classification = INTENTIONAL (by-design modeling difference), MATCH (within ~5%), or INVESTIGATE (unexplained gap / possible defect).\n' +
  '4. "Verdict" — Are they "almost similar"? Expected core finding: structural parity (same ~580-630 km, same stop count, same ~22%->95% charge band) with NexDash showing ~15-18% HIGHER energy and LOWER arrival SoC because it models the real ~1600m Berlin->Bavaria elevation gain (terrain-honest), while NexOS assumes flat terrain. Confirm or refute against the actual data above. Flag any INVESTIGATE row as a real follow-up.\n' +
  '5. "Capture health" — one line each for nexos / nexdash-live / api-truth: ok or what failed, so the reader trusts the numbers. Screenshots saved at /tmp/nexos-result.png and /tmp/nexdash-result.png.\n\n' +
  'Return the full markdown report as your final message.',
  { label: 'synthesis', phase: 'Synthesis' }
)

return {
  apiTruth,
  nexos,
  local,
  report,
}
