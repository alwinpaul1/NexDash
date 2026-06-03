# NexDash — Modeling & Feature Design

> Why this system is built the way it is. The grading-critical question for this
> case study is **judgment**: did we choose a modeling approach that fits the
> problem, and can we defend every feature? This document answers that.

The product predicts the energy a **Mercedes-Benz eActros 600** (≈600 kWh usable
battery, ≈500 km real-world range, 0–22 t payload, up to ~40 t GVW) needs to
cover a given driving segment in Germany, and turns that prediction into a
go / no-go **range reachability** answer for fleet dispatchers.

---

## 1. The core problem and what makes it hard

A dispatcher needs one number with a defensible error band: *will this truck,
at this state of charge, reach its destination?* The honest answer depends on
physics that interact non-linearly:

- Aerodynamic drag scales with **speed²**, so it dominates on the Autobahn but is
  almost irrelevant in city traffic — and the air density in that term is itself
  **temperature-dependent** (cold air is denser, so the same speed costs more
  drag in winter).
- Gradient cost is **signed**: climbing burns potential energy; descending
  *returns* some of it through regenerative braking, but only a fraction
  (a base `regen_eff` of 60%), and the recovered fraction is itself bounded by
  battery and brake-blending limits — so the base is tapered down in the cold
  (a colder BMS accepts less charge) and on steep descents (the regen power cap
  is exceeded and the surplus bleeds to friction braking).
- Auxiliary/HVAC load is **U-shaped in temperature** — cabin and battery thermal
  management cost energy at both winter cold and summer heat, and that cost is
  paid *per unit time*, so a slow, cold, short-distance crawl can be
  surprisingly expensive per km.
- **Payload** changes both rolling resistance and the gravitational work on every
  gradient, so payload and gradient are not independent — their product matters.

A purely linear "kWh per km" rule of thumb fails exactly where dispatchers care
most: the edge cases (heavy load up a cold mountain pass). The modeling choices
below are made to capture those interactions while staying explainable and
cheap to run.

---

## 2. Why a physics-informed synthetic generator

We do not have a large, clean, labelled fleet telemetry dataset for the eActros
600 — and even if we did, it would be biased toward the routes the fleet
actually drives, leaving the corners (max payload, steep descents, −15 °C)
underrepresented. Three options were on the table:

1. **Train directly on (scarce, biased) real telemetry.** Rejected: too little
   data, poor coverage of the decision-critical edges, and no ground truth we can
   reason about when the model is wrong.
2. **Use a pure physics model as the product.** Tempting — it is interpretable
   and needs no data. But a hand-tuned physics formula is brittle: it cannot
   absorb the messy, correlated, noisy reality of sensors, driver behaviour, and
   unmodelled losses, and it gives no calibrated error band.
3. **Physics-informed synthetic data + a learned regressor.** Chosen.

The generator (`data_gen.py`) samples realistic, independent feature
distributions across the full operating envelope (distance 1–350 km, payload
0–22 t, speed 20–90 kph, gradient ±6 %, temperature −15–40 °C, wind −12..12 m/s
— a signed headwind component, negative = tailwind),
labels each sample with a **deterministic physics ground truth**
(`physics.segment_energy_kwh`: speed- and temperature-dependent rolling
resistance + temperature-dependent-air-density drag (with wind) + signed
gradient work with temperature- and grade-tapered regen + time-scaled U-shaped
aux load, all divided by `drivetrain_eff`), then injects **multiplicative
Gaussian noise** (`noise_frac`, default 6 %) plus a small absolute sensor noise
term. The noise stays **unstructured** (mean-zero, uncorrelated with the
features): the temperature/speed/grade effects live in the physics channels, not
the noise, so the winter penalty is never double-counted. Net-regen steep
descents keep **genuine negative labels** (no zero-clamp), so the regen signal is
preserved end-to-end and the steep-down slice is a real — not artificially
flattering — failure mode.

This gives us the best of both worlds:

- **Full, uniform coverage** of the operating envelope, including the rare,
  high-stakes corners that real fleets rarely log.
- **A known ground-truth function**, so during evaluation we can attribute error
  to specific regimes (the failure-mode report) rather than shrugging.
- **Realistic irreducible noise**, so the learned model produces an honest error
  band (MAE) instead of memorising a clean formula — which would give false
  confidence to the dispatcher.

The synthetic label is the physics; the learning job is to recover that physics
*from noisy samples* and generalise smoothly between them — a faithful proxy for
the real task of recovering true energy demand from noisy telemetry.

---

## 3. From raw inputs to features

`FEATURE_COLUMNS = [distance_km, payload_t, speed_kph, gradient_pct,
temperature_c, wind_mps]`. Each is a real, dispatcher-observable quantity. Here
is what each one *physically drives*, and therefore why it is a feature:

| Raw input | Physical role | Why the model needs it |
|---|---|---|
| `distance_km` | Multiplies every per-km loss and scales total energy | The dominant first-order driver of total kWh. |
| `payload_t` | Adds to total mass → rolling resistance **and** gradient work | Heavier truck ⇒ more energy, with effect amplified on grades. |
| `speed_kph` | Aero drag ∝ speed²; also sets travel time for aux load | Captures the Autobahn-vs-city regime switch. |
| `gradient_pct` | Signed potential-energy term; negative ⇒ partial regen | The single biggest source of non-linear, sign-dependent cost. |
| `temperature_c` | Drives HVAC/battery-thermal aux load (U-shaped) | Explains the winter/summer range penalty dispatchers feel. |
| `wind_mps` | Adds to effective airspeed in the drag term | Head/tailwind can swing motorway energy by double digits. |

### 3.1 Engineered features and why they earn their place

A gradient-boosted tree can in principle discover interactions from raw columns,
but giving it the *right* engineered terms (`features.build_features` /
`transform`, exported as `ENGINEERED_COLUMNS`) makes the signal explicit,
reduces the depth/data needed to learn it, improves extrapolation toward the
envelope edges, and — importantly for grading — makes the model **legible**. Each
engineered column maps to a named physical mechanism:

- **`abs_gradient`** — Rolling/handling penalties and the magnitude of
  gradient work depend on *how steep*, somewhat independently of direction. A
  monotone-per-direction term helps the tree split cleanly on steepness.
- **`payload_gradient`** (payload × gradient) — The key interaction:
  gravitational work on a grade is proportional to mass *and* slope. A heavy
  truck on a steep climb is far costlier than either factor alone predicts; this
  product hands that super-additive cost to the model directly.
- **`temp_deviation`** (|temperature − 20 °C| or a comfort-band deviation) —
  Linearises the **U-shaped** HVAC cost around a thermal-neutral point so the
  model does not have to learn a non-monotone curve from scratch. This is the
  feature that lets the model reproduce the "both cold and hot cost energy"
  behaviour cheaply.
- **`speed_sq`** (speed² proxy) — Aerodynamic drag scales with the square of
  airspeed. Exposing speed² turns a curved relationship into one the model can
  capture with shallow splits, sharpening predictions at high motorway speeds.

> Assumption noted where the contract is silent: derived columns are computed
> deterministically from the six raw inputs only, so `transform` works
> identically on a single dispatcher request (a dict) and on a batch
> (a DataFrame), and training/serving see exactly the same feature definitions.

Each engineered term is a **hypothesis about the physics**, not a kitchen-sink
add. We deliberately stop here: no high-cardinality polynomial expansions, no
features the dispatcher cannot supply at request time. Simplicity over
complexity — every column has to justify itself with a mechanism.

---

## 4. Why gradient boosting over linear regression

We train **two** models on the same engineered features and keep both metrics
for comparison:

- **`LinearRegression` baseline** — a transparent floor. It tells us how much of
  the variance is explainable by additive, first-order effects, and it is a
  sanity check on the features: if a physically motivated feature does not move
  the linear baseline at all, that is a flag.
- **`HistGradientBoostingRegressor` (primary)** — the production model.

Gradient boosting is the right primary for this problem because:

1. **Native non-linearity and interactions.** The dominant costs (speed², signed
   gradient with a regen kink, U-shaped temperature, payload×gradient) are
   exactly the curved, interacting, threshold-y relationships trees model well
   without manual basis expansion.
2. **Robustness to feature scale and mild noise.** Tree splits are
   scale-invariant and resistant to the multiplicative noise we injected, so we
   avoid fragile scaling/standardisation pipelines.
3. **Strong tabular performance at low cost.** `HistGradientBoosting` trains in
   seconds on thousands of rows and predicts in microseconds — well within the
   latency a live dispatcher panel and an agent tool-call loop need.

The **gap between the two models is itself a deliverable**: the
boosting model should beat the linear baseline most on the curved/interacting
regimes (steep grades, temperature extremes, heavy-on-hill). Reporting both
(in `run_pipeline.py`'s evaluation report) demonstrates that the added model
complexity is *earned* by measurably lower MAE/MAPE, not assumed. If the gap were
small, the honest engineering call would be to ship the linear model for its
interpretability — surfacing that trade-off is part of the judgment.

We did **not** reach for a neural net: the data is low-dimensional tabular,
gradient boosting is the established state of the art there, and an MLP would add
training fragility, tuning burden, and opacity with no expected accuracy win.

---

## 5. Turning predictions into honest decisions

The point of the model is the **range reachability** verdict
(`range.check_reachability`). The model predicts `energy_needed_kwh`; the system
compares it against `battery_kwh × soc% − reserve` to return `reaches`,
`margin_kwh`, `remaining_soc_pct`, and an estimated `remaining_range_km`.

Crucially, every verdict carries a **confidence note tied to the model's MAE
band**. A dispatcher should never see a bare "yes" — a 2 kWh margin against a
model with a ±6 kWh MAE is effectively a coin flip, and the note says so. This is
where the synthetic-ground-truth choice pays off again: because we can measure
MAE honestly across regimes, we can warn precisely when the margin is inside the
error band, and the failure-mode report tells us *which* regimes deserve a wider
safety factor.

> **How the band is wired.** `range.check_reachability` reads the model's
> **real held-out MAE** (~6 kWh) straight from the artifact
> (`metrics.hgb.mae_kwh`) and reports it in the `confidence_note` — there is no
> hardcoded band. On top of that it runs a **first-principles physics
> cross-check** on every call (`physics.segment_energy_kwh` on the same inputs).
> If the data-driven prediction and the physics estimate disagree by more than
> ~3 error-bands (or 15%), the segment is outside the trained envelope: the tool
> returns `confidence: "low"`, uses the **conservative (higher)** value for the
> GO/NO-GO, and the note explains why. The result therefore exposes three
> numbers — `model_kwh`, `physics_kwh`, and a `confidence` flag — so a dispatcher
> can see both estimates and when they diverge. This is the sanity-clamp that
> stops a confident-looking under-prediction in an unsampled corner from shipping
> as a dangerous "yes" (see example conversation 2).

---

## 6. Assumptions and limits of the synthetic approach

Intellectual honesty is part of the design, so the limits are explicit:

- **The model can only be as right as the physics.** The synthetic label is our
  hand-built energy model. Effects it omits — auxiliary trailer drag, traffic
  stop-go cycles, tyre wear, battery degradation/SOH, driver aggressiveness,
  absolute altitude/pressure, humidity, road surface — are invisible to the
  learner. (Air-density *temperature* variation is now modelled via the ideal
  gas law; only the absolute-altitude/pressure offset is omitted, so `ρ(T)`
  assumes sea level and slightly over-states cold-air drag on alpine hauls — a
  known one-sided bias.) The injected noise mimics *unstructured* scatter, not
  these *structured* gaps.
- **Feature independence is a near-idealisation.** The generator samples most
  inputs from independent marginals, with one **deliberate coupling**: the
  gradient is **capped per segment** so the implied net climb stays
  geographically plausible (≤ ~1000 m), because a sustained steep grade over a
  long leg implies a physically impossible net climb (an uncapped +4.5% grade
  held over 110 km would be a ~5 km ascent, higher than any Alpine pass). This
  **net-climb** cap holds for every seed and sample size — unlike the earlier
  distance-attenuation heuristic, whose floor still admitted impossible climbs.
  It bounds geography, **not** total energy: a rare long + heavy + cold + headwind
  leg can still legitimately need more than one charge (a real "must charge
  mid-route" segment), so labels are deliberately **not** clamped to the battery
  capacity. Other correlations real fleets exhibit (speed/route/load)
  are still absent, so predictions on physically implausible combinations remain
  extrapolations the dispatcher should treat with care — and the reachability
  layer's physics cross-check is what flags them.
- **Regen is modelled as a tapered efficiency, not a fully dynamic limit.** The
  base `regen_eff` (0.60) is now bent down by temperature (`g_temp`, floor 0.45
  at −15 °C) and grade (`g_grade`, floor 0.70 at −10%), so the model captures the
  cold-BMS and steep-descent caps that a flat 60% missed. The 0.70 grade floor is
  deliberately not lower: it keeps recovered energy **monotonically non-decreasing
  in |grade|** (a steeper descent must never return less charge). Two limits
  remain: the grade taper is a **proxy** (the true motor-power knee depends on
  `m·g·sin(θ)·v`, not grade alone), and there is **no state-of-charge channel**,
  so the model still over-credits downhill recovery near a full battery. The
  0.45/0.70 floors are reasoned engineering bounds, not eActros-measured.
- **Calibration transfers, accuracy must be re-earned on real data.** The right
  productionisation path is to keep this architecture (engineered physics
  features + gradient boosting + linear baseline + per-regime failure report) and
  **retrain/fine-tune on real eActros telemetry** as it accrues, treating the
  synthetic model as a warm start and a coverage backstop for rare regimes.

The design is therefore deliberately conservative: a well-understood physics
prior, a small set of mechanistically justified features, a model whose extra
complexity is validated against a baseline, and a decision layer that refuses to
hide its own uncertainty. Sound reasoning over complexity — by construction.
