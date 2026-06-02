# NexDash Route Planner — Implementation Notes

Running log of design decisions, deviations, tradeoffs, and open questions while
reviewing and extending the route planner (backend + frontend). Newest entries on top.

---

## Context / current state (2026-06-02)

The route planner is mature, and **all of this session's work is merged to `main`
(PRs #18–#25)** — nothing is uncommitted. The arc, in order:
- **#18** field-calibration baseline + eActros 600 spec card.
- **#19** per-road speed limits (Tier S) + visible per-leg km/h + 80 km/h truck cap + SOC-colour anchor + `minSocFloor` gauge.
- **#20** plain-language hints under the two SOC sliders.
- **#21** per-stop charging cost (€).
- **#22** charge-time/ETA reconciled to the matched station's real power.
- **#23** aero recalibration **cd 0.55 → 0.50** (ProCabin −9%, CdA 5.0), `FIELD_CALIBRATION_FACTOR 0.80 → 0.83`, `WIND_MPS 3.0 → 0.0`, **fastest-free charger selection** (`rankChargersByTime`).
- **#24** **Speed Limit Profile** chart + exact break placement (4:30/4:30) + read-only-audit fixes (stale anchor text, reconcile epsilon, gauge label, list key).
- **#25** **bulletproof EU-561 breaks** (a charge is COUNTED as a break but a dedicated 45-min rest is ALWAYS inserted for compliance — matches NexOS; we no longer treat a charge as *the* mandatory break) + docs recalibration for cd=0.50.

Net result (Berlin→Munich, 18 t): **~108 kWh/100 km** display (raw ~130, was 140), a **400 kW free charger**, **2 breaks** (dedicated rest + charge), EU 561 compliant.

> The **earlier-engagement history** (PRs #8/#9 + the Pass-2 VRP optimiser exploration, 2026-05-31/06-01) lives in **`implementation-notes.html`**. Several items there are **superseded**: the VRP stop-order optimiser is now **dormant** (stop order follows the typed sequence — no auto-reorder), and its energy figures (cd 0.55, ~132 kWh/100 km) predate the cd 0.50 recalibration.

---

## Audit triage (2026-06-02) — 5-agent read-only audit + lead synthesis

**Fixed (real, low-risk, verified):**
- **Stale dispatcher-facing anchor** — `route_planner.py` assumptions text said
  "~1.27 kWh/km warm anchor" (cd=0.55 era); now "~1.22 … at the calibrated CdA 5.0".
- **Reconcile match epsilon** — `routePlanner.js` stop-shift used `+1e-6` against
  0.1 km-rounded distances, which could drop a legit charge-before-stop match; widened
  to `+0.01` (10 m).
- **Gauge label** — SocGauge "Min" → "Min floor" (it shows the operator's *set* floor,
  not the achieved trip low — the exact ambiguity that confused earlier this session).
- **Stable list key** — ChargingStopsList keyed by lat/lng instead of array index.

**Dropped as FALSE POSITIVE (verified against code + screenshots):**
- "ChargeCard shows 0 kWh / no cost" — the re-label is `{ ...seg, station }`; `...seg`
  already preserves `kWh`/`costEur` from the backend segment, and live screenshots show
  "450 kWh · ≈€202" in the timeline card. The audit misread the `station` sub-object as
  the whole return. Not a bug.
- "Divergent `ci`/`cj` counters in reconcile / segment relabel" — both loops walk
  `segments` in identical order, so they stay aligned. The synthesis itself re-adjudicated
  and dropped these.

**Deferred / documented (not fixed, with reason):**
- `/api/optimize` contract gaps (no `chargeTargetSoc`/`fieldCalibration`/`departure`
  forwarding, weak `origin`/matrix validation). **The endpoint is DORMANT** — the
  frontend hard-codes `const opt = null` (VRP never auto-invoked, per the earlier
  typed-order decision). Real gaps, zero current impact. Fix only if the VRP is ever
  wired to a UI action. (See OQ3.)
- Docs staleness — README / REAL_WORLD_CALIBRATION.md still cite cd=0.55-era anchors
  (1.265 / 1.47 / 1.55 kWh/km). The app-facing text is fixed (above); the docs need a
  recalibration pass (warm ~1.216, recompute cold) — deferred to avoid shipping a
  hand-guessed cold figure. (See OQ4.)
- `legTimings`/`speedLimits` lack Pydantic structural validation — current code degrades
  gracefully (drops to heuristic speed) on malformed input; low risk. Deferred.
- Multi-day midnight-rollover in `absFromHhmm` reconcile — only bites 3+ day trips with
  large charge deltas; edge case, deferred. (OQ5)

## Design decisions

### D1 — Per-zone speed display + ETA source (2026-06-02)
**Decision:** ETA stays anchored to **TomTom's measured truck duration** (traffic +
posted-limit aware). Separately, add a **speed-limit profile** (speed vs distance,
capped at the 80 km/h truck limit) so slow zones (30/50 km/h) are *visible* — the
prior caveat was that a 4 km 30-zone is diluted into a ~73 km/h leg average.

**Why:** The user deferred ("decide yourself … depend on TomTom"). Driving at exactly
each posted limit everywhere would make the ETA ~30 min *faster* than TomTom's real
measured time (real trucks lose ~6% to ramps/accel/hills/traffic) — re-introducing the
optimism the user explicitly ruled out one turn earlier. So: trust TomTom for time,
surface the posted limits as a profile for visibility. The drive-leg averages (~73–76)
stay realistic; the profile shows the road's limits (the truck's *target*, which it
averages just under). This mirrors how a sat-nav shows limit signs while the ETA
assumes you average below them.

**Verified earlier this turn (live probe):** our driving time == TomTom's traffic-aware
duration exactly (8.05 h, 73 km/h), all segments ≤ 80 km/h, posted mix on this route
is 567 km @ 80 + 13 @ 50 + 4 @ 30 + 3 @ 60 + 1 @ 40.

### D2 — Break placement precision (2026-06-02)
**Decision:** split the chunk at the exact 4.5 h point so the break lands on the cap
(4:30/4:30) rather than at the previous ~25 km chunk boundary (~4:27).
**Why / tradeoff:** ETA-neutral by construction (partial + remainder = same chunk +
same 45-min break). Pure precision/parity with NexOS. Cost: a small split inside the
safety-critical SOC walk — mitigated by conservation (km/energy/time preserved) and the
full test suite passing.

---

## Deviations from a naive spec

- **Speeds are anchored to TomTom, not driven at posted limits.** A naive reading of
  "truck speed = the zone's limit" would set 80 on the autobahn and 30 in towns and let
  the ETA fall out of that. We deliberately do NOT, because it makes the ETA optimistic
  vs real measured time. See D1.

---

## Tradeoffs considered

- **Charge-as-break vs always a dedicated rest → chose bulletproof (#25).** We initially
  let a ≥45-min charge *satisfy* the EU-561 break (CORTE-guidance-backed, faster). After the
  NexOS comparison we switched to the legally-bulletproof posture: the charge is COUNTED as
  a break but a dedicated 45-min rest is ALWAYS inserted for compliance — never relying on
  the un-codified charge-as-break reading. Matches NexOS. (OQ1 resolved.)

---

## Open questions (for later confirmation — not blocking)

- **OQ1 — Charge-as-break legality. RESOLVED → bulletproof.** Per the NexOS screenshot,
  NexOS keeps a *dedicated* rest for the 4.5 h compliance AND counts the charge as a 2nd
  break (2 breaks). We now match: a ≥45-min charge is **counted** as a break but **no
  longer resets the 4.5 h clock**, so a dedicated 45-min rest is always inserted for
  compliance — we never lean on the CORTE-only "charging = the break" reading. Verified
  live (Berlin→Munich: dedicated rest at 4:30 + charge counted = 2 breaks, EU 561 ok).
  Edge case: when a charge and the 4.5 h mark nearly coincide, the dedicated rest can land
  a few minutes of driving after the charge (legal, slightly redundant; rare).
- **OQ2 — Charge target = 95% soft ceiling.** We over-charge (arrive 33–52%) like NexOS
  (61%). A "charge only what's needed + cushion" policy would be faster but trades away
  buffer. Left as the conservative long-haul default.
- **OQ3 — `/api/optimize` (VRP stop-order) is dormant.** `optimizeRoute` hard-codes
  `opt = null`, so stop order always follows the typed sequence (your earlier explicit
  rule). The endpoint still exists with contract gaps. Confirm whether the VRP should
  ever be wired to an explicit "suggest a better order" action; until then it's dead code.
- **OQ4 — Docs recalibration. RESOLVED.** Recomputed every anchor at cd=0.50 / CdA 5.0
  (via `artifacts/doc_anchors.py`) and updated README.md + REAL_WORLD_CALIBRATION.md:
  warm 1.265→**1.216**, cold (−10 °C) 1.47→**1.42** (+17%), the questionable "22 t/85 =
  1.55" replaced with the correct same-laden-truck high-speed cold **1.49**, empty-rig
  0.90→**0.83**, mid-load 1.09→**1.02**, and the factor note 0.80/1.265 → **0.83 × 1.216
  ≈ 1.01** (config docstring was already updated in #23). The worked example now uses the
  live-verified Berlin→Munich raw ~130 × 0.83 = ~108 kWh/100 km.
- **OQ5 — Multi-day reconcile clock.** For 3+ day routes the HH:MM-based rollover in
  charge-time reconciliation could misdate a shifted segment; switching to absolute
  timestamps would harden it. Low priority.
