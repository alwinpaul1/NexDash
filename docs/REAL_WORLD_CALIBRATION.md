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
| `cd` | 0.55 | **0.50** (apply ProCabin −9%) | No absolute Cd is published; Daimler states the new ProCabin lowers the cW value **9% vs the current series Actros** (front extended +80 mm, wheel infills, no MirrorCam). 0.55 is the standard modern aero tractor-trailer drag coefficient; applying the cited −9% to it gives **0.55 × 0.91 ≈ 0.50** (so CdA = 0.50 × 10 = **5.0**). This models *this* aero-optimised ProCabin truck rather than a generic combo, and lands the raw steady-state in the field band without leaning on the display calibration. [S1][S6] |
| `crr` | 0.006 | **0.0055** (BASE value) | Not published. Long-haul truck tyres on good asphalt sit ~0.005–0.007; the eActros uses low-rolling-resistance 315/70 R22.5 drive / 385/55 R22.5 steer tyres. 0.0055 is a defensible mid value that, combined with the Cd/area above, reproduces the real-world consumption band (see §3). **This is now a BASE coefficient**: `physics.py` multiplies it by a speed factor `f_speed` and a temperature factor `f_temp` (`Crr_eff = Crr0 · f_speed · f_temp`), both normalised to 1.0 at the 80 km/h / 20 °C reference so this anchor is unchanged. See "New physics-module constants" below and §3. [S2][S8] |
| `drivetrain_eff` | 0.87 | **0.85** | Not published. 0.85 is a standard battery-to-wheel efficiency for an 800 V e-axle drivetrain (motor + inverter + 4-speed gearbox) under steady cruise; slightly lower than 0.87 to better hit the measured ~1.1–1.3 kWh/km band. [S1] |
| `regen_eff` | 0.60 | **0.60** (BASE value) | Well-supported. eActros regen is 400 kW continuous / 600 kW peak with 5 recuperation levels. Field evidence: ~10% of battery recovered on the Drackensteiner Hang descent; up to ~25% of total energy recovered on favourable stages; capture efficiency ~60–70% of a braking event; ~51% on a steady 6% downhill in a comparable BEV-truck study. 0.60 is a sound central value. **This is now a BASE fraction**: on descents `physics.py` tapers it by a temperature factor `g_temp` and a grade factor `g_grade` (`regen_eff_eff = 0.60 · g_temp · g_grade`); mild descents in mild weather keep the full 0.60. See "New physics-module constants" below and §3. [S4][S5][S11] |
| `aux_base_kw` | 2.0 | **2.0** (keep) | Steady electronics/aux floor on top of the U-shaped HVAC term. EV base loads sit ~1–2 kW; the variable HVAC load is layered above this (see §3). [S10][S12] |

### New physics-module constants (temperature / speed / grade channels)

The four "honest limitations" formerly absorbed into the aero term and the noise
envelope are now **explicit, first-principles physics channels** in
`src/nexdash/physics.py`. The constants that drive them:

| Constant | Value | Drives | Basis / source |
|---|---|---|---|
| `P_SEA_LEVEL_PA` | **101325** Pa | Air density via ideal gas law | ISO 2533 International Standard Atmosphere (sea-level pressure). [S14] |
| `R_SPECIFIC_DRY_AIR` | **287.05** J/kg/K | Air density via ideal gas law | Specific gas constant for dry air; `rho(T) = P_SEA_LEVEL_PA / (R_SPECIFIC_DRY_AIR · (T_C + 273.15))`. `rho(15 °C) = 1.225` exactly — the **pivot**, continuous with the old constant. [S14] |
| CRR speed slope | **0.0015** /km/h | `f_speed = max(1 + 0.0015·(speed_kph − 80), 0.90)` | Modest speed-dependent rolling-resistance rise, normalised to 1.0 at the 80 km/h reference. SAE J2452. [S15] |
| CRR cold slope | **0.004** /°C | `f_temp = 1 + 0.004·(20 − T_C)` for T < 20 °C, else 1.0 | Cold-side tyre stiffening only (~+0.4%/°C), normalised to 1.0 at the 20 °C reference. Deliberately conservative (literature is 0.6–0.9%/°C). [S15] |
| regen temp floor `g_temp` | **0.45** at −15 °C | Cold-BMS charge-acceptance taper: `g_temp` = 1.0 at/above +10 °C, linear down to 0.45 at −15 °C | Reasoned engineering bound (cold lithium charge-acceptance limit), Battery University BU-410. **Not** an eActros-specific measurement — Daimler publishes no regen-vs-temperature curve. [S16] |
| regen grade floor `g_grade` | **0.70** at −10% | Regen-power-cap taper: `g_grade` = 1.0 up to −5% descent, linear down to 0.70 at −10% | Reasoned engineering bound (steep descents exceed the regen power cap → friction braking); the floor is set at 0.70 (not lower) so recovered energy stays **monotonically non-decreasing in |grade|** — a steeper descent must never return less charge. **Not** eActros-measured. [S16] |

Both Crr factors and both regen factors equal **1.0 at their reference points**
(80 km/h, 20 °C, descents ≤ −5%, T ≥ +10 °C), so the §1 calibration anchors are
left untouched; the channels only bend consumption away from the warm, moderate
reference.

---

## 2. Feature sampling ranges (German operations)

| Feature | Current | **Calibrated** | Basis |
|---|---|---|---|
| `distance_km` | 1–120 | **1–350** (skewed toward 20–150; long right tail) | Inter-stop charger legs run 250–350 km; EU 561/2006 caps driving at 4.5 h before a 45-min break (~9 h/day). A right-skewed gamma with a tail to ~350 km matches mixed regional + inter-hub Autobahn legs better than a 120 km cap. Keep a short-distance mass for urban/regional pickup-delivery. [S3-de][S13] |
| `payload_t` | 0–22 | **0–22** (keep) | Full legal span; frequent empty/part-load runs. 0 = empty trailer (18 t rig), 22 = fully loaded 40 t rig. [S1] |
| `speed_kph` | 30–90 | **20–90** (centre ~72; cluster 70–80 Autobahn) | German Lkw Autobahn limit is **80 km/h** (>7.5 t); the eActros is electronically limited at **90 km/h**; Landstrasse 60, urban 50. Realistic moving averages: 70–80 km/h Autobahn, 55–65 km/h mixed. Sample the full **20–90 km/h** mechanical/limiter window: the lower bound (20) covers congested-urban and steep-climb crawl segments, and the upper bound is the truck's hard limiter, not a legal cruise speed. The energy model's `SEG_SPEED` clamp in `route_planner.py` is held to **this same [20, 90] range** so the served model is never asked to extrapolate outside its training envelope — real telematics still records brief over-limit and downhill segments above the 80 km/h legal cruising limit, so the model must stay accurate there rather than being capped below where the truck actually runs. [S3-de][S2] |
| `gradient_pct` | -6 .. +6 | **-6 .. +6** (keep; centre 0, σ small) | Autobahn design target ≤4%; German terrain ranges 0% (North German Plain) to ~6% (Alpine foothills / Mittelgebirge grades). Symmetric about flat, most segments gentle. [S3-de] |
| `temperature_c` | -15 .. 40 | **-15 .. 40** (keep; seasonal mean ~10, wide σ) | German envelope: January mean ~-2.8 °C, cold snaps to -15 °C (record -29.4 °C); summer mean 20–28 °C, heatwaves >35 °C (record 41.2 °C). -15..40 °C captures the operating year. [S3-de][S9] |
| `wind_mps` | 0–12 | **0–12** (keep; gamma, light-skewed) | German onshore mean wind 4–6 m/s, sheltered 3–4 m/s; 0–12 m/s (0–43 km/h) spans calm to strong gusts. Modelled as headwind (conservative). At 12 m/s pure headwind the apparent air speed gains >40 km/h — a large single-leg penalty. [S3-de][S12] |

---

## 3. Physics notes (curve shapes and targets)

### Target baseline consumption band
The recalibrated generator should reproduce, for a **40 t combination on the
flat at 80 km/h, 20 °C, no wind**, a battery-side warm anchor of
**~1.216 kWh/km** (the pure-physics, no-route-regen figure at the calibrated
**cd = 0.50 / CdA 5.0**, `rho(20 °C) = 1.204`), rising to **~1.42 kWh/km at −10 °C**
on the same segment once the temperature-dependent air-density and cold-tyre
channels engage (15 °C is the pivot at the old 1.225 density). On *real mixed routes* (which include
descents that feed regen) the warm figure averages down toward the
field-measured **~0.95–1.05 kWh/km**:

- ADAC fully-loaded 40 t, ~350 km Munich–Wörth at ≤85 km/h: **~0.88 kWh/km**. [S4]
- Daimler 15,000+ km European tour (40 t, 22 countries, 20 °C ref): **~1.03 kWh/km** average, envelope **0.85** (downhill) to **1.40** (cold + unpaved). [S5]
- Vandijck 1,530 km long-haul: **0.963 kWh/km**; aero-kit truck **0.934 kWh/km**. [S7]
- WLTP declared: **1.19 kWh/km**; spec-implied (600 kWh / 500 km): **~1.20 kWh/km**. [S2]

With the §1 values (`crr=0.0055, cd=0.50, A=10` → **CdA 5.0**, `dt=0.85`) and the
now temperature-dependent air density, the model yields a **warm anchor of
~1.216 kWh/km** at 40 t / 80 km/h flat / **20 °C** (`rho(20 °C) = 1.204`).
**15 °C remains the pivot**: `rho(15 °C) = 1.225` exactly, continuous with the old
constant. The same 40 t / 80 km/h flat segment **in the cold rises to ~1.42 kWh/km
at −10 °C** (denser air + cold-tyre stiffening), a ~+17% winter swing on this
physics-only segment and the explicit-channel counterpart of the field-observed
+25% winter penalty. (The same laden 40 t at **85 km/h in the cold reaches
~1.49 kWh/km**, the higher speed lifting the aero share.) Both figures sit inside
the WLTP/spec-implied band at the warm end and the field "cold + unpaved" envelope
(up to 1.40 measured, higher on the steady-state physics-only run) at the cold end.
Empty-rig (18 t) flat at 80 km/h / 20 °C falls to **~0.83 kWh/km**; mid-load (29 t)
to **~1.02 kWh/km**. The empty→full span (~0.83 → ~1.22) is ~+47% gross, consistent
with the +0.6–0.8% per tonne field sensitivity over a 22 t payload swing. [S8][S5]

### Field-calibration factor (displayed energy vs steady-state physics)
The physics above is a **constant-speed, full-tractive-demand steady-state**
model. On a real, gently-rolling Autobahn run it reproduces the *spec/WLTP* end
of consumption, **not** the field-measured average. Worked example — a 36 t
(18 t payload) Munich→Berlin run (~590 km, ~1.6 km cumulative climb but **net
downhill**: Munich ~520 m → Berlin ~34 m), 15 °C, ~73 km/h. The planner walks the
route in ~25 km chunks, so each real climb costs full tractive energy while each
descent only recovers the regen-capped fraction (~60%) — net, the *chunked*
conservative steady-state basis the SOC walk runs on is **~112 kWh/100 km**
(≈ 1.12 kWh/km), a touch above a single net-flat 36 t segment (~108 /100 km at
73 km/h) because the climbs don't fully cancel. The field tests above for the
*same* class of run land **lower** — ADAC 88, Daimler tour 1.03, Commercial
Motor 1.05–1.12 — because real driving (coasting, eco-driving, traffic flow,
mixed-route regen) runs below constant-speed physics, a gap the steady-state
model structurally cannot close.

To make the **displayed** energy headline track field reality, NexDash applies a
single documented multiplier `config.FIELD_CALIBRATION_FACTOR = 0.887` to
`summary.energyKwh` / `summary.kwhPer100` only:

- The factor is anchored to the **energy model's own** flat-route output (that is
  what the displayed headline is built from), not to raw physics. At the 40 t /
  80 km/h / 20 °C / flat anchor the model now reads **113.88 kWh/100 km**, and
  **0.887 × 113.88 = 101.0 kWh/100 km** (≈ 1.01 kWh/km) — the field centre, on the
  Daimler 15,000 km European-tour anchor (1.03 at 40 t). A lighter 18 t / 83 km/h
  autobahn run lands ~105 kWh/100 km, at the top of the **~0.88–1.03 kWh/km** band
  (ADAC German-roads 0.88, Vandijck 0.96), as expected for a lighter/faster leg.
- **Retuned 2026-06-04 from 0.83 → 0.887** alongside the physics-residual model
  retrain. This is the honest, required consequence of changing the model: the
  *old* raw-kWh model **over-predicted** flat consumption (124.74 kWh/100 km at the
  anchor, +2.6 % above physics), so 0.83 × 124.74 = 103.5 landed mid-band. The new
  residual model tracks physics closely and reads **lower** there (113.88), so
  keeping 0.83 would have displayed 94.5 kWh/100 km — just *below* the band.
  Re-anchoring to 0.887 restores the displayed headline to 101 kWh/100 km. The
  factor moved because the model's flat output moved, exactly the documented
  REMOVAL/RETUNE condition in `config.FIELD_CALIBRATION_FACTOR`. [S4][S5]
- **It is NOT a physics change.** The locked `Cd / Crr / drivetrain_eff / A`
  anchors and the 1.22 / 1.42 / 1.49 kWh/km steady-state figures above are
  unchanged. The factor only reconciles the *reported* number with field data.
- **It never touches safety.** The SOC walk, charge-trigger look-ahead, charge
  sizing and reachability all run on the **un-discounted** steady-state estimate,
  so the displayed figure being lower can never delay a charge or strand the
  truck. The route plan (stops, timing, arrival SOC) is byte-identical with or
  without the factor; guard test
  `test_field_calibration_scales_displayed_energy_only` enforces this.
- **Tunable / removable.** `plan_route(field_calibration=…)` and the
  `/api/route-plan` `fieldCalibration` field override it (0.5–1.0; 1.0 shows the
  raw steady-state figure; ~0.79 matches the NexOS reference demo's ~95).
- **REMOVAL CONDITION.** Retune or remove once the ML model is retrained against
  field (not steady-state) labels, or the energy-side speed model changes — at
  that point the model would track field consumption directly and the multiplier
  would double-count.

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
> share (~5%). The denser-cold-air, cold-tyre and reduced-regen shares are now
> **explicit physics channels** (`rho(T)` in the aero term, `f_temp`/`f_speed`
> in the roll term, `g_temp`/`g_grade` in the regen term — see §1 "New
> physics-module constants"), **not** quantities folded into the aero term or
> the noise envelope.
>
> ⚠️ **CRITICAL — calibration / noise owner, read this.** Because the winter
> drag (denser cold air), cold-tyre stiffening and reduced cold/steep regen are
> now modelled **explicitly**, the noise envelope (`noise_frac` and the additive
> sensor term in `data_gen.py`) **must NOT also carry any temperature-,
> speed- or grade-correlated winter share.** If the cold-weather penalty is
> baked into both the explicit physics channels *and* the noise envelope, the
> winter penalty is **double-counted** and consumption at −10 °C will overshoot.
> The noise term must stay **unstructured** (mean-zero, uncorrelated with the
> features). This is the single biggest integration risk in this upgrade —
> verify it before regenerating the dataset.

### Regen recovery fraction
Use **regen_eff = 0.60 as the BASE fraction**: on a descent, the model recovers
0.60 of the downhill potential energy, then scales by drivetrain efficiency on
the recovery path (round-trip loss). The 0.60 base is now **tapered by two
factors** so it only applies in full under mild, moderate conditions:
`regen_eff_eff = 0.60 · g_temp · g_grade`, where `g_temp` falls from 1.0 at/above
+10 °C to a floor of **0.45 at −15 °C** (cold-BMS charge-acceptance limit) and
`g_grade` falls from 1.0 up to a −5% descent to a floor of **0.70 at −10%**
(steep descents exceed the regen power cap, so the surplus goes to friction
braking). The floor is set at 0.70 rather than lower so that recovered energy
stays **monotonically non-decreasing in |grade|**: with a lower floor the
fraction taper out-ran the rising potential-energy term between −8% and −10%, so
a *steeper* descent recovered *less* total charge — physically wrong, and now
guarded by a regression test. A mild descent in mild weather keeps the full
0.60. A climb-then-descend on the same grade still does **not** cancel — net
consumption stays positive because only ~50–70% of potential energy is
recaptured, less when cold or steep. This reproduces Daimler's "~25% of total
energy recovered on favourable stages" and the ~51% steady-6%-downhill capture
seen in comparable BEV trucks. The 0.45/0.70 floors are reasoned engineering
bounds (BU-410 / industry sources), **not** eActros-specific measurements.
[S4][S5][S11][S16]

### Speed / aero
Aerodynamic drag dominates at Autobahn speed (30–50%+ of tractive energy above
60–70 km/h, >50% near the limiter; drag ∝ v², aero power ∝ v³). The model
reproduces a steep speed sensitivity: from 60 → 85 km/h consumption rises
**~0.99 → ~1.28 kWh/km (+29%)** at 40 t in the physics-only run, and on to
**~1.35 kWh/km (+37%)** at the 90 km/h limiter — almost all aero, consistent
with the ~0.7–1.3% per km/h heavy-truck figure and the ~3–6% aero-kit delta in
the eActros tests. The speed sample spans the full **20–90 km/h** mechanical
window (legal Lkw cruise is 80 km/h; 90 is the limiter, not a cruise speed) so
the energy model is trained across every speed the truck — and real telematics —
actually produces, rather than being capped below the limiter. [S6][S8]

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
- **[S14]** ISO 2533:1975 Standard Atmosphere (International Standard Atmosphere; sea-level pressure 101325 Pa, dry-air specific gas constant R = 287.05 J/kg/K) — https://www.iso.org/standard/7472.html ; Density of air (Wikipedia, ideal-gas formulation) — https://en.wikipedia.org/wiki/Density_of_air
- **[S15]** SAE J2452 — Stepwise Coastdown Methodology for Measuring Tire Rolling Resistance (speed- and load-dependence of Crr) — https://www.sae.org/standards/content/j2452_201707/ ; rolling-resistance temperature sensitivity discussion — https://www.tut.fi (heavy-truck tyre literature)
- **[S16]** Battery University BU-410: Charging at High and Low Temperatures (cold lithium charge-acceptance limits) — https://batteryuniversity.com/article/bu-410-charging-at-high-and-low-temperatures ; reasoned regen power-cap bounds (industry sources). Note: Daimler publishes no eActros regen-vs-temperature or regen-vs-grade curve, so the 0.45 / 0.70 floors are defensible engineering estimates, not primary measurements.
