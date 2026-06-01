# NexDash — Long-Term Path: From Synthetic Seed to Telematics-Fed Range Intelligence

This document describes the credible engineering path from the current seed system — a
physics-grounded synthetic dataset, a `HistGradientBoostingRegressor` energy model, a
reachability service, and an agentic dispatcher — to a production system that continuously
learns from real Mercedes-Benz eActros 600 telematics across a German fleet.

The seed is intentionally honest about its limits: the "ground truth" today is a
deterministic physics model (`nexdash/physics.py`), and the ML model is trained to imitate
it with added noise. That is the right starting point for an offline demo, but it is also
the single biggest gap to close. Everything below is structured around replacing synthetic
ground truth with measured energy consumption, without ever shipping a regression to
dispatchers who make routing decisions on these numbers.

---

## 0. Where we are, and the one assumption that must die

**Current state.**
- `data_gen.py` synthesizes features and labels energy via the physics model.
- `model.py` fits a primary GBM plus a `LinearRegression` baseline and records both metrics.
- `evaluate.py` reports MAE/RMSE/MAPE/R², plus slice metrics by temperature, gradient, and payload.
- `range.py` converts a prediction into a reach/no-reach verdict with a reserve buffer.
- The dispatcher (`agent.py`) and MCP server expose these as tools.

**The assumption that must die:** *energy consumption is the physics formula plus noise.*
Reality diverges from `segment_energy_kwh` for reasons the formula does not model — battery
thermal management overhead, HVAC behavior with real cabin/cargo loads, driver style,
traffic stop-go cycles, tire wear, road surface, battery state-of-health (SOH) degradation
over the vehicle's life, and regen that depends on braking discipline and grade profiles
rather than a single average gradient. The whole long-term plan is the disciplined,
measured replacement of synthetic labels with telematics-measured energy.

---

## 1. Data ingestion and labeling from real trucks

### 1.1 What the truck emits
eActros 600 vehicles expose telematics via the Mercedes-Benz / Daimler Truck fleet API
(Fleetboard / Truck Data Center) and, at lower level, the FMS / J1939 CAN bus. Per trip we
can capture, at 1 Hz or event-driven cadence:

- GNSS position, speed, heading, odometer.
- High-voltage battery: SOC, SOH, pack voltage/current (→ integrated kWh in/out), pack temperature.
- HVAC and auxiliary power draw, cabin/ambient temperature.
- Drivetrain torque/power, regen energy counter.
- Payload/axle-load sensors (GVW estimate), trailer presence.
- Ambient: temperature, and via map/weather join — gradient profile, wind, precipitation.

The **label we actually want** is *measured net traction + auxiliary energy for a road
segment*, derived primarily from the **coulomb-counted / energy-counter delta** (∫ pack
current·voltage dt) over the segment, reconciled with SOC deltas. This is the real
`energy_kwh` that replaces the physics output.

### 1.2 Ingestion architecture
```
Truck (FMS/CAN) ──► Telematics gateway ──► Streaming ingest (Kafka/Kinesis)
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                        ▼
                  Raw landing (object store,                Real-time features
                  immutable, partitioned by                 (online store for
                  vehicle/day)                              live range checks)
                          │
                          ▼
                  Segmentation + labeling job (batch)
                          │
                          ▼
                  Curated training table (offline feature store)
```

- **Raw landing zone:** every message persisted immutably (Parquet, partitioned by
  `vehicle_id` / `date`). Never mutate raw; all corrections happen downstream so we can
  always re-derive labels when the labeling logic improves.
- **Schema contract & validation:** each ingested batch passes schema + range checks
  (a Great Expectations / Pandera suite) mirroring the bounds already encoded in
  `data_gen.py` (distance 1–350 km, payload 0–22 t, speed 30–85 kph, gradient ±6%
  capped per segment, temperature −15..40 °C, wind −12..12 m/s). Out-of-contract rows are quarantined, not
  silently dropped — quarantine volume is itself a monitored signal.

### 1.3 Segmentation and label derivation
Continuous telematics must be cut into the same *segment* abstraction the model consumes
(one row = one homogeneous stretch). The labeling job:

1. Splits a trip into segments where speed band, gradient band, and load are roughly
   constant (or uses fixed map-matched road segments).
2. Computes segment features matching `FEATURE_COLUMNS`
   (`distance_km, payload_t, speed_kph, gradient_pct, temperature_c, wind_mps`).
3. Derives the label `energy_kwh` from the battery energy-counter delta over that segment,
   with the physics model retained as a **sanity bound**: a measured value implausibly far
   from `segment_energy_kwh` (e.g. negative beyond max regen, or 3× rolling+aero+grade) is
   flagged for review rather than trusted blindly.
4. Tags provenance: `vehicle_id`, `firmware_version`, `battery_soh`, `data_source` (real
   vs synthetic), and a `label_quality` score (sensor completeness, GNSS dropout, map-match
   confidence).

### 1.4 Hybrid training corpus and de-weighting synthetic data
We do not throw away the physics seed. We move through phases:

- **Phase A (today):** 100% synthetic.
- **Phase B (early telematics):** real data is scarce and biased toward the routes/seasons
  driven first. Train on synthetic + real, **down-weighting** synthetic samples (sample
  weights) and using physics predictions as an *input feature* (a strong prior) rather than
  the label. This is residual learning: the model learns the *correction* to physics.
- **Phase C (mature):** train predominantly on real labels; synthetic data is reserved to
  cover rare regimes (extreme cold + heavy payload + steep grade) that real operations
  under-sample, preventing catastrophic extrapolation.

`label_quality` and recency feed per-sample weights so clean, recent, in-distribution data
dominates.

---

## 2. Retraining cadence and triggers

Retraining is **both scheduled and event-driven**. Cadence alone is wasteful or too slow;
triggers alone are unpredictable. We run both.

### 2.1 Scheduled cadence
- **Weekly** automated retrain candidate while real data is still accumulating and seasonal
  conditions shift quickly (a German winter→spring transition materially changes HVAC and
  battery-thermal energy).
- **Monthly** once the data distribution stabilizes, with a quarterly "deep" retrain that
  also revisits feature engineering and hyperparameters.

### 2.2 Event triggers (any one fires a candidate build)
- **Drift trigger:** input-drift or prediction-drift alarm crosses threshold (§4).
- **Performance trigger:** rolling production MAE/MAPE on freshly-labeled segments breaches
  the SLO (e.g. MAPE > 12% over a trailing window, vs. the seed's offline baseline).
- **Fleet-change trigger:** new vehicles, a firmware/BMS update, a new depot or route
  corridor, or a battery-chemistry/SOH cohort shift.
- **Seasonal trigger:** onset of a new temperature regime not well represented in training.
- **Data-volume trigger:** N new high-quality labeled segments accumulated (e.g. +50k).

Every retrain is **deterministic and reproducible** (pinned seed, pinned data snapshot
hash, pinned dependency lockfile), extending the seed's existing `seed=42` discipline so a
model version can always be rebuilt byte-for-byte.

---

## 3. Proving a new model is genuinely better *before* deploy

A new model is **guilty until proven innocent**. No challenger reaches dispatchers without
clearing every gate below, in order. This is the heart of the long-term safety story:
dispatchers route real trucks on these numbers, and an over-optimistic range estimate that
strands a vehicle is far worse than a conservative one.

### 3.1 Frozen backtest (offline gate)
- A **held-out, time-forward** test set (the most recent weeks, never seen in training) —
  not a random split — because we must measure performance on the *future*, mirroring how
  the model is actually used. Random splits leak temporal structure and flatter the model.
- A **frozen golden set** of curated, high-quality labeled segments that never changes, so
  metric movements reflect the model, not the test data.
- Champion and challenger are scored with the existing `evaluate()` and
  `failure_mode_report()` functions. Challenger must win on the **headline metric** (MAPE /
  MAE) **and not regress** on any safety-critical slice.

### 3.2 Slice / sub-population gates (no silent regressions)
Reuse and extend `failure_mode_report` slices (temperature: cold<0 / mild / hot>30;
gradient: steep-down / flat / steep-up>4%; payload: light<7 / mid / heavy>15) plus new
real-world slices: per-depot, per-firmware, per-SOH-band, per-driver-style cohort.

Gate rule: challenger may not regress more than a small tolerance (e.g. +2% MAPE relative)
on **any** slice, even if the aggregate improves. Aggregate wins that hide a "heavy payload
in the cold" regression are rejected — that is precisely the regime where a wrong range
estimate strands a truck.

### 3.3 Safety-asymmetric scoring
Range underestimation (predict *more* energy than real → conservative, truck arrives with
margin) is safer than overestimation (predict *less* → stranded truck). We add an
**asymmetric loss / acceptance metric** that penalizes optimistic errors more heavily, and
gate on the **over-prediction rate at the reachability boundary** specifically. The seed's
`range.check_reachability` already bakes in a `reserve_pct`; the challenger must not erode
the real-world reliability that buffer is meant to provide.

### 3.4 Calibration and uncertainty
Beyond point error, the challenger must be **calibrated**: predicted confidence bands
should match observed error coverage. The seed already reports the model's real held-out
MAE in `range.py`'s `confidence_note` and runs a first-principles physics cross-check that
flags `confidence: "low"` when the two estimates diverge (**implemented** — this is the
seed's sanity-clamp). Validating that band against *measured* error is now also
**implemented** (`nexdash.calibration`, surfaced as report Section 5): split-conformal
intervals are calibrated on a held-out half and their realized coverage audited on a
disjoint half at 80/90/95%, with a bootstrap PASS/FAIL per level (FAIL only when the band
*under*-covers — the over-confident direction — since conformal guarantees coverage >= the
level), a Mondrian per-gradient-regime breakdown, and an Expected Calibration Error. A
companion auto-miner (`nexdash.failure_miner`) searches the held-out residuals for the
worst multi-feature error pockets the hand-picked slices miss. The promotion gate
(`nexdash.promote`) already rejects challengers that regress a slice or raise the
optimistic-error rate; folding the conformal coverage check into that gate (reject the
sharper-but-over-confident challenger) is the remaining online step.

### 3.5 Champion / challenger + shadow mode (online gate)
After offline gates pass:
1. **Shadow mode:** the challenger runs on **live production traffic in parallel** with the
   champion. It serves no dispatcher; its predictions are logged alongside the champion's
   and later compared against the measured energy once trips complete. This catches
   train/serve skew, feature-pipeline bugs, and latency issues that offline tests can't.
2. **Backtested champion/challenger:** replay the last N weeks of real requests through both
   models and compare against realized labels.
3. **Canary rollout:** if shadow metrics confirm the offline verdict, route a small slice of
   real traffic (e.g. one depot, 5–10% of range checks) to the challenger with tight
   monitoring and an automatic abort. Expand stepwise (5% → 25% → 50% → 100%) only while
   live slice metrics hold.

Only a model that wins offline backtests, regresses on no slice, is calibrated, behaves in
shadow, and survives canary is **promoted to champion**.

---

## 4. Drift detection and monitoring

Models silently rot as the world moves. We monitor four layers, each with alerting.

### 4.1 Input (feature) drift
- Per-feature distribution monitoring vs. the training reference: Population Stability Index
  (PSI), KS test, and Jensen–Shannon divergence on `FEATURE_COLUMNS` and the engineered
  columns from `features.py`.
- Multivariate drift (e.g. a density/novelty detector) to catch new *combinations*
  (heavy payload + steep grade + extreme cold) even when each marginal looks normal — these
  are exactly the seed's documented failure-mode slices.
- Out-of-range / quarantine rate from the ingestion contract (§1.2) is a first-line alarm.

### 4.2 Prediction drift
- Monitor the distribution of model outputs (kWh, kWh/km) over time. A shift in predictions
  without a corresponding known cause (season, fleet change) is a warning even before labels
  arrive — labels lag, predictions don't.

### 4.3 Residual / performance monitoring (the ground-truth check)
- Once trips complete and segments are labeled, compute **realized residuals**
  (`predicted − measured`) continuously. This is the only signal grounded in truth.
- Track rolling MAE/RMSE/MAPE overall **and per slice** (same bins as
  `failure_mode_report`), plus the over-prediction rate at the reach boundary.
- Watch **bias**, not just spread: a persistent sign in residuals (e.g. systematically
  optimistic in winter) is a safety alarm even if MAE looks fine.

### 4.4 Operational / system monitoring
- Prediction latency, error rate, model-load failures (the seed's API already returns a 503
  if the model is missing — that condition becomes a paged alert in production).
- Feature freshness / pipeline lag, online/offline feature parity checks.

### 4.5 Alerting and routing
- Tiered alerts: **warning** (notify ML on-call, raises retrain priority) vs. **critical**
  (safety-slice MAPE breach, sustained optimistic bias → page + auto-fallback to a more
  conservative reserve or to the physics model as a floor).
- Drift and performance breaches feed directly into the retrain triggers in §2.2, closing
  the monitor→retrain loop.

---

## 5. Versioning, lineage, and rollback

### 5.1 What we version
Every artifact is content-addressed and joined by a single `model_version`:
- **Code** (git SHA), **dependencies** (a lockfile hash; the seed's `requirements.txt` would be pinned to exact versions as part of this step).
- **Data snapshot** (immutable dataset hash; the seed already writes a deterministic
  `dataset.csv`).
- **Feature transform** (`features.py` `ENGINEERED_COLUMNS` + transform logic version) —
  versioned with the model so train/serve cannot diverge.
- **Trained model + metrics** (the seed's `EnergyModel.save/load` joblib gains a metadata
  sidecar: training data hash, code SHA, full `evaluate`/`failure_mode_report` output,
  champion comparison).
- A **model registry** (e.g. MLflow) tracks lineage and the Staging→Shadow→Canary→Production
  lifecycle.

### 5.2 Rollback
- The **previous champion is always retained** and hot-loadable. Promotion is a pointer
  flip; rollback is the reverse and must complete in minutes.
- **Automatic rollback** triggers on a critical live alarm during canary or post-promotion
  (safety-slice breach, sustained optimistic bias, latency/error spike).
- **Fail-safe floor:** if no ML model is trustworthy, the service degrades to the
  deterministic physics model (`segment_energy_kwh`) with a widened reserve. The seed's
  reservation buffer in `check_reachability` makes this degradation graceful rather than
  catastrophic — dispatchers get conservative numbers, not wrong ones.
- Every promotion and rollback is an auditable, logged event tied to the metrics that
  justified it.

---

## 6. The feedback loop from dispatcher outcomes

The richest long-term signal is what happens **after** a dispatcher acts on a range check.

### 6.1 Closing the loop
1. The agent issues a reachability verdict (predicted energy, margin, remaining
   SOC/range) and **logs the prediction** with its `model_version`, inputs, and a request ID.
2. The truck drives the route. Telematics returns **measured** segment energy and the actual
   arrival SOC.
3. A reconciliation job joins prediction ↔ outcome on the request ID, producing a labeled
   row *and* an outcome verdict:
   - **Correct-reachable:** said it would reach, it did (with margin close to predicted).
   - **False-reachable (critical):** said reach, truck arrived dangerously low or stranded —
     the worst case; over-weighted in retraining and an immediate slice-monitor input.
   - **False-unreachable:** said no-reach but it would have made it — costs efficiency
     (unnecessary detour/charge), tracked to detect over-conservatism.
4. These outcomes feed: (a) the residual monitor (§4.3), (b) the asymmetric safety metric
   (§3.3), (c) the training corpus with quality/recency weights (§1.4), and (d) the retrain
   triggers (§2.2).

### 6.2 Human and dispatcher feedback
- Dispatchers can flag a verdict ("this was wrong — truck stranded" / "this was overly
  cautious"). These flags are high-value, sampled for manual label review, and tracked as a
  product KPI.
- The agent already exposes structured tool calls (`predict_energy`, `check_reachability`);
  logging tool inputs/outputs gives a clean, replayable audit trail for both debugging and
  building the next training set.

### 6.3 Guardrails against feedback poisoning
- Because dispatchers may **act on** predictions (e.g. add charging stops when range looks
  tight), naive feedback can bias the model toward its own behavior. We log the *plan vs.
  executed* route and prefer outcomes from trips driven as planned, or correct for
  intervention, so the loop measures reality rather than echoing the model.

---

## 7. Roadmap summary

| Horizon | Milestone |
|---|---|
| **Now (seed)** | Physics ground truth, synthetic dataset, GBM + linear baseline, slice eval, reachability + agent. |
| **0–3 mo** | Telematics ingestion + raw landing + schema contracts; segment labeling; physics-as-feature residual model (Phase B); offline backtest + slice gates wired into a registry. |
| **3–6 mo** | Drift + residual monitoring with alerting; shadow mode; champion/challenger; canary rollout; automated rollback to previous champion / physics floor. |
| **6–12 mo** | Outcome feedback loop closed; asymmetric safety metric + calibration gates; event-driven retraining; per-depot/per-SOH/per-driver slice models; mostly-real corpus (Phase C). |
| **12 mo+** | Per-vehicle SOH-aware personalization; route-profile (full grade/traffic) modeling instead of single-gradient segments; continuous learning with full lineage and audit. |

---

## 8. Guiding principles

1. **Conservative beats accurate at the boundary.** A stranded truck is the failure mode we
   optimize against; optimistic errors are penalized harder than pessimistic ones.
2. **Never trust a model you can't roll back.** Previous champion and physics floor are
   always one pointer-flip away.
3. **No silent slice regressions.** Aggregate wins never excuse a worse cold/heavy/steep
   regime — those are where range matters most.
4. **Measure the future, not a shuffle.** Time-forward backtests and shadow mode, not random
   splits, decide promotion.
5. **Reproducibility is non-negotiable.** Code + data + features + seed are pinned per
   version, extending the seed's deterministic design.
6. **The loop must reflect reality, not echo the model.** Dispatcher actions are accounted
   for so feedback teaches the model about the world, not about itself.
