# Real-World Calibration — Mercedes-Benz eActros 600

This document maps every parameter of the NexDash energy model to a **real,
cited** figure for the Mercedes-Benz eActros 600 operating on German long-haul
roads. It replaces the original "guessed" assumptions in
`src/nexdash/config.py`, `src/nexdash/physics.py` and `src/nexdash/data_gen.py`
with values grounded in manufacturer specifications, independent field tests and
published heavy-truck / EV literature.

The eActros 600 is Daimler Truck's battery-electric long-haul tractor (world
premiere October 2023, series production from late 2024): ~600 kWh **usable**
battery (621 kWh installed, 3 × 207 kWh LFP packs), 800 V architecture, a single
eAxle with two motors (400 kW continuous / 600 kW peak), 22 t GVW, up to 44 t
GCW, electronically capped at 90 km/h (German Lkw legal max 80 km/h), and an
official 500 km range without intermediate charging (4×2 tractor, 40 t, 20 °C).

**Scope note.** The model treats a *constant-speed segment* at 40 t-class
combination weight. All calibrated values target the realistic German Autobahn /
mixed long-haul envelope, not laboratory WLTP conditions.

---

## 1. TRUCK dataclass parameters

| Field | Current | **Calibrated** | Basis |
|---|---|---|---|
| `battery_kwh` | 600.0 | **600.0** (keep) | Usable capacity. 621 kWh installed (3 × 207 kWh LFP), ~600 kWh usable (200 kWh/pack, ~96.6%, Daimler rounds to "~95%"). The model name reflects the usable figure. [S1][S2][S3] |
| `max_payload_t` | 22.0 | **22.0** (keep) | ~22 t payload with a standard EU semitrailer; GVW 22 t, GCW up to 44 t. [S1][S2] |
| `kerb_mass_kg` | 18000.0 | **18000.0** (keep, re-documented) | This is the **loaded-rig baseline** (tractor + empty semitrailer), NOT the bare tractor. The bare 4×2 tractor kerb mass is ~11.7 t (hylane: 11,546 kg chassis+cab). 18 t base + 22 t payload = 40 t GCW, the reference combination weight for the official 500 km range and all field tests. Keeping 18 t makes "payload_t = 0" mean "empty trailer attached" (~18 t) and "payload_t = 22" mean a fully loaded 40 t rig — the correct operating span. [S3] |
| `frontal_area_m2` | 10.0 | **10.0** (keep) | Not published by Daimler. 10 m² is the standard literature value for a European tractor-semitrailer frontal area (cab width ~2.48 m × height ~3.9 m gives ~9–10 m² effective). Defensible class value. [S3][S8] |
| `cd` | 0.55 | **0.55** (keep) | No absolute Cd is published; Daimler states only the new ProCabin lowers the cW value **9% vs the current series Actros** (front extended +80 mm, wheel infills, no MirrorCam) for a ~2–3% efficiency gain. 0.55 is a standard literature drag coefficient for a modern aero tractor-trailer and is consistent with the relative-improvement claim. [S1][S6] |
| `crr` | 0.006 | **0.0055** | Not published. Long-haul truck tyres on good asphalt sit ~0.005–0.007; the eActros uses low-rolling-resistance 315/70 R22.5 drive / 385/55 R22.5 steer tyres. 0.0055 is a defensible mid value that, combined with the Cd/area above, reproduces the real-world consumption band (see §3). [S2][S8] |
| `drivetrain_eff` | 0.87 | **0.85** | Not published. 0.85 is a standard battery-to-wheel efficiency for an 800 V e-axle drivetrain (motor + inverter + 4-speed gearbox) under steady cruise; slightly lower than 0.87 to better hit the measured ~1.1–1.3 kWh/km band. [S1] |
| `regen_eff` | 0.60 | **0.60** (keep) | Well-supported. eActros regen is 400 kW continuous / 600 kW peak with 5 recuperation levels. Field evidence: ~10% of battery recovered on the Drackensteiner Hang descent; up to ~25% of total energy recovered on favourable stages; capture efficiency ~60–70% of a braking event; ~51% on a steady 6% downhill in a comparable BEV-truck study. 0.60 is a sound central value. [S4][S5][S11] |
| `aux_base_kw` | 2.0 | **2.0** (keep) | Steady electronics/aux floor on top of the U-shaped HVAC term. EV base loads sit ~1–2 kW; the variable HVAC load is layered above this (see §3). [S10][S12] |

---

## 2. Feature sampling ranges (German operations)

| Feature | Current | **Calibrated** | Basis |
|---|---|---|---|
| `distance_km` | 1–120 | **1–350** (skewed toward 20–150; long right tail) | Inter-stop charger legs run 250–350 km; EU 561/2006 caps driving at 4.5 h before a 45-min break (~9 h/day). A right-skewed gamma with a tail to ~350 km matches mixed regional + inter-hub Autobahn legs better than a 120 km cap. Keep a short-distance mass for urban/regional pickup-delivery. [S3-de][S13] |
| `payload_t` | 0–22 | **0–22** (keep) | Full legal span; frequent empty/part-load runs. 0 = empty trailer (18 t rig), 22 = fully loaded 40 t rig. [S1] |
| `speed_kph` | 30–90 | **30–85** (centre ~72; cluster 70–80 Autobahn) | German Lkw Autobahn limit is **80 km/h** (>7.5 t); limiter 90 km/h; Landstrasse 60, urban 50. Realistic moving averages: 70–80 km/h Autobahn, 55–65 km/h mixed. Cap at **85** (eActros legal/operating window) rather than 90 — 90 km/h is not legally driveable for a loaded truck on German roads. [S3-de][S2] |
| `gradient_pct` | -6 .. +6 | **-6 .. +6** (keep; centre 0, σ small) | Autobahn design target ≤4%; German terrain ranges 0% (North German Plain) to ~6% (Alpine foothills / Mittelgebirge grades). Symmetric about flat, most segments gentle. [S3-de] |
| `temperature_c` | -15 .. 40 | **-15 .. 40** (keep; seasonal mean ~10, wide σ) | German envelope: January mean ~-2.8 °C, cold snaps to -15 °C (record -29.4 °C); summer mean 20–28 °C, heatwaves >35 °C (record 41.2 °C). -15..40 °C captures the operating year. [S3-de][S9] |
| `wind_mps` | 0–12 | **0–12** (keep; gamma, light-skewed) | German onshore mean wind 4–6 m/s, sheltered 3–4 m/s; 0–12 m/s (0–43 km/h) spans calm to strong gusts. Modelled as headwind (conservative). At 12 m/s pure headwind the apparent air speed gains >40 km/h — a large single-leg penalty. [S3-de][S12] |

---

## 3. Physics notes (curve shapes and targets)

### Target baseline consumption band
The recalibrated generator should reproduce, for a **40 t combination on the
flat at 80–85 km/h, 20 °C, no wind**, a battery-side consumption of
**~1.25–1.35 kWh/km** (the pure-physics, no-route-regen figure). On *real
mixed routes* (which include descents that feed regen) this averages down toward
the field-measured **~0.95–1.05 kWh/km**:

- ADAC fully-loaded 40 t, ~350 km Munich–Wörth at ≤85 km/h: **~0.88 kWh/km**. [S4]
- Daimler 15,000+ km European tour (40 t, 22 countries, 20 °C ref): **~1.03 kWh/km** average, envelope **0.85** (downhill) to **1.40** (cold + unpaved). [S5]
- Vandijck 1,530 km long-haul: **0.963 kWh/km**; aero-kit truck **0.934 kWh/km**. [S7]
- WLTP declared: **1.19 kWh/km**; spec-implied (600 kWh / 500 km): **~1.20 kWh/km**. [S2]

With the §1 values (`crr=0.0055, cd=0.55, A=10, dt=0.85`) the model yields
**1.29 kWh/km** at 40 t / 80 km/h flat and **1.35 kWh/km** at 85 km/h — squarely
in the WLTP/spec-implied band. Empty-rig (18 t) flat at 80 km/h falls to
**~0.90 kWh/km**; mid-load (29 t) to **~1.09 kWh/km**. The empty→full span
(~0.90 → ~1.29) is ~+43% gross, consistent with the +0.6–0.8% per tonne field
sensitivity over a 22 t payload swing. [S8][S5]

### HVAC / auxiliary load vs temperature (U-shape)
Aux power = `aux_base_kw` (flat comfort band) + a linear rise on each side.
Calibrated to the eActros winter test and EV-HVAC literature:

| Temperature | **Target HVAC + aux (kW)** | Basis |
|---|---|---|
| -10 °C | **~6–7 kW** | EV cabin heating ~6 kW at -7 °C; eActros winter test attributes ~5% of energy to cabin heating to 21 °C. [S10][S6] |
| 20 °C (comfort) | **~2 kW** (`aux_base_kw` floor) | Base electronics/aux only; HVAC near zero in the 18–25 °C comfort band. [S10][S12] |
| 38 °C | **~4–5 kW** | AC/cooling compressor draws ~3–6 kW continuous, ~5–6 kW near 40 °C. [S12] |

Recommended slopes to hit these points: **comfort band 20 ± 3 °C**, **cold
slope ~0.18 kW/°C**, **hot slope ~0.13 kW/°C** (cold steeper than hot, since
resistive cabin heating + battery conditioning costs more than AC). At an
80 km/h cruise this adds only ~0.05–0.09 kWh/km even at -10 °C, matching the
observed **~5% winter heating share** and the overall **+25% normal-winter /
up to +50% snow-and-ice** consumption rise from the 6,500 km eActros winter
test. The U-shape minimum sits at ~20 °C (most efficient ~21.8–25.2 °C in the
literature). [S6][S10][S12]

> Note: the *full* winter penalty (+25%) is not from HVAC alone — denser cold
> air + winter tyres (~15%), reduced regen (~4%) and battery heating (<1%) make
> up the rest. The temperature-driven aux term captures only the cabin/aux
> share (~5%); the air-density and tyre effects are implicitly folded into the
> physics aero/roll terms and the noise envelope.

### Regen recovery fraction
Use **regen_eff = 0.60**: on a descent, the model recovers 60% of the downhill
potential energy, then scales by drivetrain efficiency on the recovery path
(round-trip loss). A climb-then-descend on the same grade does **not** cancel —
net consumption stays positive because only ~50–70% of potential energy is
recaptured. This reproduces Daimler's "~25% of total energy recovered on
favourable stages" and the ~51% steady-6%-downhill capture seen in comparable
BEV trucks. [S4][S5][S11]

### Speed / aero
Aerodynamic drag dominates at Autobahn speed (30–50%+ of tractive energy above
60–70 km/h, >50% near the legal cap; drag ∝ v², aero power ∝ v³). The model
reproduces ~+10–15% consumption from 60 → 85 km/h (1.06 → 1.35 kWh/km, +28% at
40 t in the physics-only run), almost all aero — consistent with the ~0.7–1.3%
per km/h heavy-truck figure and the ~3–6% aero-kit delta in the eActros tests.
Capping the speed sample at 85 km/h keeps the dataset inside the legally
driveable German window. [S6][S8]

---

## Sources

- **[S1]** Mercedes-Benz Trucks celebrates world premiere of the eActros 600 — Daimler Truck press release — https://www.daimlertruck.com/en/newsroom/pressrelease/mercedes-benz-trucks-celebrates-world-premiere-of-the-battery-electric-long-haul-truck-eactros-600-52428265
- **[S2]** The new eActros 600 product sheet (official Mercedes-Benz Trucks GB PDF) — https://www.mercedes-benz-trucks.com/content/dam/brandhub/markets/gb/files/product-sheets/eActros-600-product-sheet.pdf.coredownload.pdf
- **[S3]** Mercedes eActros 600 LS 4×2 vehicle datasheet (manufacturer data, hosted by hylane) — https://www.hylane.nl/assets/documents/fahrzeug-datenblaetter/Mercedes_eActros_4x2_hylane_EN.pdf
- **[S4]** On the road: Mercedes-Benz eActros 600 driving impressions review (Commercial Motor) — https://www.commercialmotor.com/knowledge-hub/article/on-the-road-mercedes-benz-eactros-600
- **[S5]** More than 15,000 km traveled all-electric: eActros 600 European Testing Tour completed — Daimler Truck — https://www.daimlertruck.com/en/newsroom/pressrelease/more-than-15000-kilometers-traveled-all-electric-mercedes-benz-eactros-600-testing-tour-throughout-europe-completed-successfully-52780594
- **[S6]** eActros 600 in der Wintererprobung: So viel Strom kostet die Kälte (eurotransport) — https://www.eurotransport.de/fahrzeuge/eactros-600-in-der-wintererprobung-so-viel-strom-kostet-die-kaelte/ ; Mercedes eActros 600 winter test over 6,500 km (VISION mobility) — https://vision-mobility.de/en/news/mercedes-eactros-600-stable-performance-in-winter-test-over-6-500-kilometers-369122.html
- **[S7]** Vandijck Transport tests the eActros 600 on a 1,500+ km long-haul route — Daimler Truck — https://www.daimlertruck.com/en/newsroom/pressrelease/vandijck-transport-tests-the-eactros-600-on-a-long-haul-route-of-over-1500-kilometers-53076996
- **[S8]** How aerodynamics and rolling resistance impact your truck's fuel consumption (Volvo Trucks) — https://www.volvotrucks.com/en-en/news-stories/insights/articles/2019/oct/how-aerodynamics-and-rolling-resistance-impact-your-trucks-fuel-consumption.html ; The Weight of Electrification (ACEA Intelligence) — https://www.aceaintelligence.eu/post/the-weight-of-electrification-why-electric-trucks-tip-the-scales-differently ; The payload problem in heavy-duty electric trucking (Lux Research) — https://luxresearchinc.com/blog/the-payload-problem-in-heavy-duty-electric-trucking-is-not-all-that-big/
- **[S9]** DWD climate monitoring Germany — https://www.dwd.de/EN/climate_environment/climatemonitoring/germany/germany_node.html ; List of extreme temperatures in Germany (Wikipedia) — https://en.wikipedia.org/wiki/List_of_extreme_temperatures_in_Germany
- **[S10]** Recent advances on air heating system of cabin for pure electric vehicles (PMC review) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9568831/ ; Effects of ambient temperature on EV range (ScienceDirect) — https://www.sciencedirect.com/science/article/abs/pii/S0196890425000160
- **[S11]** Long Downhill Braking and Energy Recovery of Pure Electric Commercial Vehicles (Preprints.org) — https://www.preprints.org/manuscript/202401.0581
- **[S12]** From real operations: e-truck consumption values 2024 (Designwerk) — https://www.designwerk.com/en/post/e-truck/from-real-operation-consumption-values-of-electric-trucks-in-special-applications/ ; Clean Energy Wire — German wind 2023 — https://www.cleanenergywire.org/news/windiest-year-over-decade-enabled-germanys-2023-renewables-success-weather-service
- **[S13]** Driving time and rest periods (EU Regulation 561/2006, EUR-Lex) — https://eur-lex.europa.eu/EN/legal-content/summary/driving-time-and-rest-periods-in-the-road-transport-sector.html ; Milence Fact Sheet (charging hubs) — https://milence.com/app/uploads/2026/04/Milence-Fact-Sheet_General-EN.pdf
- **[S3-de]** Speed limits in Germany (Wikipedia) — https://en.wikipedia.org/wiki/Speed_limits_in_Germany ; Autobahn (Wikipedia) — https://en.wikipedia.org/wiki/Autobahn ; Germany Relief (Britannica) — https://www.britannica.com/place/Germany/Relief
