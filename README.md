# NexDash — EV Truck Range Intelligence

> *"Truck 14 is sitting at 62% charge in Hamburg. It's loaded to 18 tonnes and has a 240 km run down to Bremen on the A1. It's −4 °C and windy. Will it make it without a charging stop?"*

That single question — asked dozens of times a day by a fleet dispatcher — is what NexDash answers. We model a **Mercedes-Benz eActros 600** (≈600 kWh usable battery, ≈500 km real-world range, 0–22 t payload, GVW up to ~40 t) and turn raw trip parameters into a confident, explainable **reach / no-reach** verdict with an energy margin and a remaining-range estimate.

NexDash is a small but complete AI-engineering system:

1. A **physics ground-truth simulator** for segment energy.
2. A **synthetic dataset generator** built on that physics.
3. A **machine-learning energy model** (gradient boosting, with a linear baseline for honest comparison).
4. A **range-reachability service** wrapping the model with an operational reserve.
5. A **route planner** that enriches a trip with real Open-Meteo elevation, temperature and wind before predicting.
6. An **LLM dispatcher agent** that answers natural-language questions using tools — exposed over a **CLI**, an **MCP** server, and an `/api/chat` endpoint.
7. A **FastAPI backend** plus a **bonus** Vite + React console (route planner + a floating chat widget). The chat UI is a *plus* per the brief; the CLI / MCP / `/api/chat` are the required Part-2 interface.

---

## Why a model and not just the physics?

The physics simulator (`nexdash.physics`) is deterministic and explainable, so it makes an excellent *teacher*. But in the real world energy draw is noisy: driver behaviour, traffic, HVAC habits, tyre wear and battery ageing all add variance the textbook equations don't capture. We therefore use the physics to **bootstrap a labelled dataset**, add realistic noise, and train an ML model that (a) tolerates that noise, (b) is trivially retrainable on *real* telemetry once it arrives (see [Long-term: real data](#long-term-real-data)), and (c) gives us a measurable error band we can surface to dispatchers as confidence.

---

## Quickstart

Requires Python ≥ 3.10.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies + the package (editable)
pip install -r requirements.txt
pip install -e .

# 3. Generate data, train, evaluate, and write the report — end to end
python run_pipeline.py             # writes models/energy_model.joblib + reports/evaluation_report.md
```

Everything below assumes step 3 has run (it creates the trained model the agent, API and CLI all load).

```bash
# Ask the dispatcher agent a question (needs MINIMAX_API_KEY — see API keys below)
python -m nexdash.cli --once "Can a truck at 62% SOC reach 240 km with 18 t in the cold?"
python -m nexdash.cli                       # interactive REPL

# Launch the API (FastAPI on http://localhost:8000)
python dashboard/server.py
```

A `Makefile` wraps the common flows: `make setup`, `make data`, `make train`, `make test`, `make serve`, `make agent`.

**Bonus React console.** The brief notes a chat UI is a plus. A Vite + React front-end (route planner + a floating chat widget) lives in `frontend/` and talks to the same FastAPI backend. With the backend running, start it separately:

```bash
cd frontend
npm install
npm run dev        # Vite dev server, proxies API calls to http://localhost:8000
```

The React console is optional — everything in Part 1/2 (model, evaluation, agent, CLI, MCP, `/api/chat`) works without it.

### API keys

**LLM (agent/CLI).** The dispatcher runs on **MiniMax** via its OpenAI-compatible API, model **`MiniMax-M3`**. Put the key in a `.env` at the repo root (gitignored; loaded automatically by `dashboard/server.py`), or export it:

```bash
export MINIMAX_API_KEY=...            # default model: MiniMax-M3
# optional: pick a different model
export NEXDASH_LLM_MODEL=MiniMax-M3
```

The provider is selected by which key is present: `MINIMAX_API_KEY` → MiniMax (OpenAI-compatible adapter); otherwise `ANTHROPIC_API_KEY` → Anthropic Claude (`claude-opus-4-8`). A thin adapter translates the OpenAI tool-calling schema to/from the Anthropic-style loop, so the tool layer and tests are provider-agnostic. The physics, dataset, model and evaluation all work **entirely offline** — only the LLM agent/CLI need a key; without one, `/api/chat` degrades to a friendly message rather than failing.

**Maps (bonus React console only).** The route planner uses TomTom (routing, EV-charging POIs, live traffic) and MapTiler tiles. Set these in `frontend/.env` if you run the console:

```bash
VITE_TOMTOM_API_KEY=...
VITE_MAPTILER_API_KEY=...     # optional; falls back to TomTom / CARTO tiles
```

> **What TomTom does — and doesn't — decide (honest scope).** TomTom supplies the
> *road geometry*, *live traffic*, and *real charging-station* names/power/availability
> for display. It does **not** decide the charging plan: **stop placement comes from our
> own physics-grounded SOC simulation** (`nexdash.route_planner`), and each synthetic stop
> is then matched to the nearest real DC-fast POI for the map. A consequence to be candid
> about: a displayed station whose real power is too low to refill within the simulated
> window is **not** currently detected — the energy decision is the model's, the station
> card is cosmetic. (We evaluated TomTom's Long-Distance EV Routing as a *cross-check*
> oracle, not a replacement — the case study is about our model; see `docs/LONG_TERM.md`.)
>
> **Security.** `VITE_TOMTOM_API_KEY` ships to the browser (unavoidable for client-side
> maps). Per TomTom's guidance, **domain-whitelist** the key (restrict per-domain and
> per-product) so a leaked key can't be abused; the demo key is also rate/transaction
> limited. For production, proxy the calls through the FastAPI backend so the key never
> leaves the server. Calls retry with backoff on 429/5xx and time out (`tomtomFetch`),
> and every external call fails soft to a local fallback.

### Registering the MCP server

NexDash ships an MCP server (`nexdash.mcp_server`) exposing `predict_energy` and `check_reachability` as tools. Add it to any MCP client (e.g. Claude Desktop) like so:

```json
{
  "mcpServers": {
    "nexdash": {
      "command": "python",
      "args": ["-m", "nexdash.mcp_server"],
      "env": { "PYTHONPATH": "/absolute/path/to/NexDash/src" }
    }
  }
}
```

Run it standalone for a quick smoke test with `python -m nexdash.mcp_server`.

---

## Repository map

```
NexDash/
├── run_pipeline.py            # End-to-end: generate → train → evaluate → report (deterministic)
├── pyproject.toml             # Package metadata (src/ layout, console script `nexdash`)
├── requirements.txt
├── Makefile                   # setup / data / train / test / serve / agent
├── src/nexdash/
│   ├── config.py              # eActros 600 constants, physics globals, paths
│   ├── physics.py             # Deterministic segment-energy ground truth + breakdown
│   ├── data_gen.py            # Synthetic dataset sampler (CLI: python -m nexdash.data_gen)
│   ├── features.py            # Feature columns + engineering (transform/build_features)
│   ├── model.py               # EnergyModel: HistGradientBoosting + LinearRegression baseline
│   ├── evaluate.py            # Metrics, failure-mode slicing, matplotlib plots
│   ├── range.py               # check_reachability: SOC → reach/no-reach + margin
│   ├── geodata.py             # Open-Meteo elevation/temperature/wind enrichment (fails soft, cached)
│   ├── route_planner.py       # Plans a trip end-to-end (geodata → segments → predictions)
│   ├── model_info.py          # Headline model metrics for /api/model-info (fail-soft)
│   ├── tools.py               # tool schemas + dispatch() (provider-agnostic)
│   ├── mcp_server.py          # FastMCP server exposing the same tools
│   ├── agent.py               # DispatcherAgent: tool-use loop
│   └── cli.py                 # REPL / --once front-end for the agent
├── dashboard/
│   └── server.py              # FastAPI: /api/predict, /api/route-plan, /api/model-info, /api/chat, /api/health
├── frontend/                  # BONUS: Vite + React route-planner console + chat widget; calls the same FastAPI
├── data/                      # Generated dataset.csv lands here
├── models/                    # Trained energy_model.joblib lands here
├── reports/                   # evaluation_report.md + figures/ (generated)
├── docs/
│   ├── DESIGN.md              # Physics + feature + modelling rationale
│   ├── LONG_TERM.md           # Retraining, versioning, drift strategy
│   └── REAL_WORLD_CALIBRATION.md  # eActros 600 parameters tied to cited sources
├── examples/                  # Example agent conversations (a reach case + a no-reach case)
└── tests/                     # pytest suite (LLM/network mocked)
```

---

## Modeling approach

NexDash predicts **energy consumption (kWh) for a single driving segment** and uses that to reason about range.

**Ground truth — physics.** `segment_energy_kwh(...)` sums the four classic loads, all divided by drivetrain efficiency:

- **Rolling resistance** — `Crr · m · g · distance`, scaling with total mass (kerb + payload). `Crr` is a **base** coefficient bent by speed and temperature (`Crr_eff = Crr0 · f_speed · f_temp`): a modest rise above 80 km/h and cold-tyre stiffening below 20 °C, both normalised to 1.0 at the 80 km/h / 20 °C reference.
- **Aerodynamic drag** — `½ · ρ(T) · Cd · A · v² · distance`, using the effective air speed (`speed + headwind`); the dominant term at motorway pace. Air density is **temperature-dependent** via the ideal gas law (`ρ(T) = 101325 / (287.05 · (T + 273.15))`), so cold air (`ρ(−10 °C) = 1.341`) costs more drag than warm (`ρ(35 °C) = 1.146`); `ρ(15 °C) = 1.225` is the pivot.
- **Gradient (potential energy)** — `m · g · Δh`; downhill segments recover a **base** `regen_eff` (60%) of the would-be energy via regenerative braking, which is why steep descents can net *negative* kWh. The base is tapered on descents by temperature (`g_temp`, floor 0.45 at −15 °C: cold BMS charge-acceptance) and grade (`g_grade`, floor 0.70 at −10%: steep descents exceed the regen power cap and bleed to friction braking).
- **Auxiliary / HVAC** — a base load plus a cabin-conditioning term that rises at **both** cold and hot extremes, scaled by travel time (`distance / speed`).

**Uphill vs. downhill — the gradient channel in one table.** The same 50 km / 15 t / 70 km/h / 10 °C / no-wind leg, varying *only* the road gradient. Climbing spends potential energy; descending **recovers** it via regen, so a steep descent can net *negative* kWh (energy returned to the pack). Reproduce with `python -c "from nexdash.physics import segment_energy_kwh as e; from nexdash.config import TRUCK; [print(g, e(distance_km=50, payload_t=15, speed_kph=70, gradient_pct=g, temperature_c=10, wind_mps=0, truck=TRUCK)) for g in (8,4,0,-4,-8)]"`:

| Gradient | Energy (kWh) | Behaviour |
| --- | --- | --- |
| +8 % | **475.1** | uphill — lifting ~33 t gross up the climb dominates |
| +4 % | 264.7 | uphill |
| 0 % (flat) | 53.3 | baseline rolling + aero + aux |
| −4 % | **−38.3** | downhill — regen returns more than the segment spends (net negative) |
| −8 % | −96.6 | steeper descent recovers more |

Regen is capped, not unlimited: it is tapered by temperature and grade (a steeper descent never recovers *less* than a gentler one — guaranteed by the `g_grade` floor and a regression test), and only gravitational potential energy is recovered, never kinetic braking-to-stop. The evaluation's failure-mode table reports `steep_up (>+4%)` and `steep_down (<−4%)` slices separately so each direction's error is visible.

`energy_breakdown(...)` returns each component separately for transparency.

**Features.** The model consumes the six raw inputs — `distance_km, payload_t, speed_kph, gradient_pct, temperature_c, wind_mps` — plus engineered terms that encode the physics structure so a tree model finds the patterns faster: a **payload × gradient** interaction (mass amplifies grade), **|gradient|**, **temperature deviation from 20 °C** (HVAC is a U-shape, not linear), and a **speed² proxy** (drag is quadratic). Each is documented in `nexdash.features`.

**Estimator.** A `HistGradientBoostingRegressor` is the production model; a `LinearRegression` baseline is trained alongside it so every report shows *how much the non-linear model actually buys us*. `EnergyModel` wraps a scikit-learn pipeline and accepts raw feature dicts (it calls `features.transform` internally), so callers never duplicate engineering logic.

Full rationale, formulas and assumptions: **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## Evaluation

`run_pipeline.py` writes `reports/evaluation_report.md` with **real, regenerated numbers** plus figures. The values below are the **actual results** from the committed run (`n=6000`, `seed=42`, 4,800 train / 1,200 held-out test). The truck spec and synthetic operating envelope are now **calibrated to real Mercedes-Benz eActros 600 figures** (German long-haul ops) — see [`docs/REAL_WORLD_CALIBRATION.md`](docs/REAL_WORLD_CALIBRATION.md) for every cited assumption.

**Held-out test set (gradient-boosting primary, n=1,200):** MAE **5.970 kWh**, RMSE 9.133 kWh, MAPE 14.06 %, R² 0.9836.

The headline metric is the **MAE in kWh (5.97)** — that is the number a dispatcher should reason about per segment. We also report a *fleet-intuitive* proxy: 5.97 kWh is **0.995 %** of the eActros 600's ~600 kWh usable battery (the energy it spends across its ~500 km real-world range). This proxy is flattering on long segments (whose absolute energies are large), so treat the per-kWh MAE — and the per-regime failure table below — as the real accuracy signal, not the percentage.

**Read these numbers honestly.** Two framings keep them grounded:

- **Per-SEGMENT, not per-trip.** Every figure above is the error on a *single leg*. A real route is many segments and the errors accumulate (roughly with the number of legs), so the **trip-level** error is several times larger than any single-segment figure and is **not** bounded by the 0.995 % proxy. Treat that percentage as a per-leg sanity scale, never a promise that a whole route lands inside a 10 % reserve.
- **Circular evaluation.** These metrics measure how faithfully the ML model reproduces our own `nexdash.physics.segment_energy_kwh` — the *same* function that generated the labels. They bound model-vs-PHYSICS error, **not** model-vs-REALITY error, which is unknown until real eActros telematics arrive. A low MAE proves the model re-learned our physics, not that the physics matches a real truck.

**Verified uncertainty + auto-discovered failures (report Section 5).** Beyond the point metrics, the pipeline now *proves* its confidence rather than asserting it (`nexdash.calibration`): it calibrates **split-conformal** prediction intervals on a held-out half and audits their realized coverage on a disjoint half at 80/90/95%, flagging a band **FAIL** only when it *under*-covers (the over-confident, dangerous direction) and **CONSERVATIVE** when it over-covers — with a per-gradient-regime (Mondrian) breakdown and an Expected Calibration Error. A companion auto-miner (`nexdash.failure_miner`) fits a shallow tree on the held-out error to surface the worst multi-feature pockets the hand-picked slices miss (e.g. very long legs → ~3× MAE), each gated by a support floor and a bootstrapped lift CI so flukes don't ship. Both are deterministic, offline, and disclosed under the same circular-evaluation caveat (coverage-of-physics, not of-reality).

The model-vs-baseline comparison is computed on the **same 1,200-row held-out test set** as the headline (both estimators are fit on all 4,800 training rows; the metrics are stored on the model artifact, so the served model, this table, and the headline all describe one model on one test set — there is no hidden inner split, and the HGB MAE here equals the headline MAE):

| Metric              | Gradient Boosting (primary) | Linear baseline |
| ------------------- | --------------------------- | --------------- |
| MAE (kWh)           | 5.970                       | 25.120          |
| RMSE (kWh)          | 9.133                       | 33.490          |
| MAPE (%)            | 14.06                       | 95.08           |
| R²                  | 0.9836                      | 0.7794          |

The gradient-boosting model cuts MAE several-fold versus the linear baseline (5.97 vs 25.12 kWh). Three diagnostic figures are saved to `reports/figures/`: **predicted-vs-actual**, **residual-vs-temperature**, and **error-by-payload**.

### Where it fails and why

A model is only trustworthy if you know its blind spots. `failure_mode_report(...)` slices error by operating regime, ranked by **absolute MAE (kWh)** — what actually strands a truck — not MAPE, which explodes on near-zero-denominator downhill slices. The two most decision-relevant failures are **steep uphill** and **heavy payload**:

- **Steep climbs (gradient > +4 %).** This is the dangerous slice. Held-out MAE here is **8.05 kWh** (n=20) versus the overall 5.97 kWh. Steep *climbs* convert payload mass into potential energy fastest, so any miss is a large *absolute* kWh miss — exactly the regime that can strand a truck. (Small sample: n=20 < 30, so treat as indicative, not precise.)
- **Heavy payload (> 15 t).** MAE **7.25 kWh** (n=378) — the worst-error payload bin. Payload scales rolling resistance and gradient potential energy linearly, so heavy loads both consume the most and leave the most room to be wrong in absolute terms.
- **Steep descents (gradient < −4 %).** Small absolute error (MAE 2.70 kWh, n=18) but a now-genuine failure regime: these net-regen segments carry **real negative labels** (no zero-clamp), so the regen signal is preserved and the slice is no longer artificially flattering. Its *percentage* error is loud (MAPE 31.24 %) because regen drives net energy near zero and the denominator collapses — which is precisely why we rank by MAE, not MAPE. (Small sample: n=18 < 30.)
- **Temperature.** Cold is **not** the hardest regime to *predict*, even though the upgraded physics makes it the most *expensive*: the cold consumption channels (denser air + cold tyres + reduced regen) are smooth and learnable, so the error slices stay flat — cold MAE 5.58 (n=166), mild 6.06 (n=979), hot 5.60 (n=55). The `mild (0–30 °C)` slice actually carries the largest *absolute* error simply because it holds the most rows and the biggest-energy segments. A winter run costs more energy (see below) but is not meaningfully harder to predict than a mild one; don't conflate the two.

The reachability service exposes uncertainty on every verdict in two ways. First, `range.check_reachability` reads the model's **real held-out MAE** (~6 kWh) straight from the artifact (`metrics.hgb.mae_kwh`) and reports it in the `confidence_note` — no more hardcoded band. Second, it runs a **first-principles physics cross-check** on every call: if the data-driven model and the physics estimate disagree by more than ~3 error-bands (or 15%), it returns `confidence: "low"`, uses the conservative (higher) value for the GO/NO-GO, and the note explains the segment is outside the trained envelope. New result keys: `confidence`, `model_kwh`, `physics_kwh`. The route console surfaces the verdict in **red** when the margin is within the error band rather than pretending a knife-edge result is safe.

**What this evaluation can and cannot tell us.** The data is synthetic: labels are a physics ground truth plus realistic noise (multiplicative ~6% + a small additive sensor term), which sets a **noise floor** (~5–6% MAPE / ~4 kWh MAE — the error a *perfect* predictor still scores against the noisy labels) that bounds achievable accuracy from below. The model's residual sits a couple of multiples above that floor, so there is genuine headroom; it is the right *order of magnitude*, not a tuning failure, but it is **not** the irreducible scatter either. The gradient is now **capped per segment** so the implied net climb stays geographically plausible (≤ ~1000 m), because a sustained steep grade over a long leg would otherwise imply a physically impossible net climb; this **net-climb** cap holds for every seed and sample size. (A rare long + heavy + cold + headwind leg can still legitimately need *more than one charge* — a real "must charge mid-route" segment — so labels are deliberately **not** clamped to the battery capacity.) Net-regen steep descents keep **genuine negative labels** (no 0.05 kWh zero-clamp), so the regen signal is preserved and the noise stays unstructured. That distribution fix is also the resolution of an earlier mistake of ours — we had documented conversation 2 as a "tree extrapolation blow-up" (the model inventing ~752 kWh), but on re-checking, the physics label for those exact inputs is ~769 kWh, so the *old* model was faithfully reproducing an **unphysical training label**. The real defect was the data distribution, not the model; we fixed the generator and the failure narrative now ranks slices by MAE.

**What used to be a limitation and now isn't.** Four simplifications the earlier write-up listed as honest gaps are now **modelled as explicit physics channels**, not constants:

- **Air density** is temperature-dependent via the ideal gas law (was a constant 1.225). `ρ(−10 °C) = 1.341` vs `ρ(35 °C) = 1.146`; `ρ(15 °C) = 1.225` is the pivot.
- **Rolling resistance** rises with speed and with cold (`Crr_eff = Crr0 · f_speed · f_temp`; was a flat 0.0055 independent of speed and temperature).
- **Regen** is tapered on descents by temperature and grade (`0.60 · g_temp · g_grade`, floors 0.45 cold / 0.70 steep; was a flat 60%). The steep-grade floor is 0.70 (not a lower value) so that recovered energy stays **monotonically non-decreasing in |grade|** — a steeper descent must never return *less* charge.
- **Temperature** therefore acts through *four* channels — cold-air drag, cold-tyre roll, reduced regen and the U-shaped HVAC aux — not HVAC alone.

The combined winter signal is visible end-to-end: a flat 40 t / 80 km/h segment costs **~1.265 kWh/km at 20 °C** but **~1.47 kWh/km at −10 °C** (a ~+16% swing; a faster, lighter 22 t / 85 km/h flat run reaches **~1.55 kWh/km** in the cold, where the higher speed and warmer-anchor split push the aero share up), the explicit-channel counterpart of the field-observed +25% winter penalty.

**The honest residual limitations that remain** — fixing the big four does not make the model perfect:

- **Steady-state per segment.** Each segment is modelled at constant speed; there are no acceleration transients, and only **gravitational** potential energy is regenerated — kinetic energy lost in braking-to-stop is not recovered.
- **No absolute altitude / pressure.** `ρ(T)` assumes sea level, so alpine hauls slightly **over-state cold-air drag** (Germany is mostly 0–1000 m, so this is a known *one-sided* bias). Fixing it needs an elevation input and a dataset regeneration.
- **Linear ramps, reasoned floors.** `f_temp`, `g_temp` and `g_grade` are linear approximations of smooth BMS/tyre curves, and the regen floors (0.45 cold / 0.70 steep) are **reasoned engineering bounds, not eActros measurements** (Daimler publishes no regen-vs-temperature curve) — treat their magnitudes as defensible estimates pending primary data.
- **Grade-only regen proxy.** The regen power-cap taper keys off **grade**, but the true motor-cap knee depends on `m·g·sin(θ)·v`, not grade alone.
- **No state-of-charge channel** (a full battery also refuses regen) and **no humidity** (a sub-1% effect that partly cancels the cold-density gain).
- **Conservative cold-Crr slope.** The 0.4%/°C cold rolling slope is deliberately below the literature 0.6–0.9%/°C (surface-temperature figures conflate tyre self-heating), so the cold rolling penalty here is a **lower bound**.

The candid framing stands: "no weaknesses" is impossible. We fixed the big four and the bullets above name what remains. The steep-up slice is small (n=20 < 30, indicative only), rare *combinations* of features are still under-represented, and accuracy must be re-earned on real telemetry — the synthetic numbers establish that the pipeline works, not that it is calibrated to a real eActros.

---

## Agent design

The dispatcher experience is an **LLM tool-use loop**, not free-form generation over numbers.

- **Tools (`nexdash.tools`).** Two tool schemas — `predict_energy` and `check_reachability` — with thin, JSON-serializable wrappers and a `dispatch(name, args)` router. The model is *forbidden in the system prompt from inventing numbers*; it must call a tool.
- **Agent (`nexdash.agent.DispatcherAgent`).** `ask(question)` sends the question plus `TOOL_SPECS` to the configured LLM (MiniMax-M3 by default, or Anthropic Claude), executes any tool calls via `dispatch`, feeds the result back, and loops until the model returns plain-language text. A thin OpenAI⇄Anthropic adapter lets the same loop drive either provider; MiniMax is a reasoning model, so its `<think>…</think>` is stripped from the user-facing reply (kept in history for continuity). The system prompt frames it as a fleet dispatcher that always states the **margin** and a **caveat**. The client is injectable, so tests run fully mocked with no network.
- **MCP (`nexdash.mcp_server`).** The exact same two tools are registered on a `FastMCP` server named `nexdash`, delegating to `nexdash.tools` so there is no logic drift — any MCP-capable host gets NexDash's reasoning without importing Python.
- **CLI (`nexdash.cli`).** A friendly REPL (or `--once`) over the agent, with a clear message and exit code if no LLM key (`MINIMAX_API_KEY` / `ANTHROPIC_API_KEY`) is set.
- **HTTP + chat UI (bonus).** `POST /api/chat` wires `DispatcherAgent.chat` over the same loop, returning the reply plus which tools were used, and degrades gracefully (no crash) when no key is set or the provider is rate-limited. The React console in `frontend/` exposes this as a floating **chat widget** (`ChatWidget` / `ChatPanel`) that **renders the agent's Markdown** (tables, lists, caveat) — the brief's "chat UI is a plus."

This separation means the *numbers* are deterministic and testable while the *explanation* is the only thing the LLM is responsible for, and the CLI, MCP server and `/api/chat` all share one tool source of truth.

Two faithful end-to-end transcripts — a comfortable reach and a no-reach case — live in **[examples/](examples/)**; their numbers are copied verbatim from live `check_reachability` output (including the no-reach case, where the physics cross-check catches an out-of-envelope under-prediction and the verdict falls back to the conservative estimate).

---

## Long-term: real data

The synthetic-physics approach is a launchpad, not the destination. As real eActros telemetry arrives, NexDash is designed to swap the data source without touching the model interface. The strategy — **continuous retraining, model versioning, and drift detection** — is detailed in **[docs/LONG_TERM.md](docs/LONG_TERM.md)**, and the core of it is now **implemented and runnable**, not just described:

- **Retraining.** Telemetry replaces the synthetic generator as the labelled source; `features.transform` and `EnergyModel` are unchanged, so the pipeline is a one-line data swap. The physics simulator survives as a sanity oracle and a cold-start prior for routes with little real data.
- **Versioning & lineage.** Every trained model now gets a **content-addressed provenance record** (`nexdash.registry`): a training-data SHA-256 + code (git) SHA + seed + full held-out metrics, written as a JSON sidecar next to the artifact and into `models/registry/` — so every deployed verdict maps to a specific model + dataset. The `model_version` is surfaced in the report header and `/api/model-info`. (The joblib bytes are deliberately left untouched, so the pipeline stays byte-reproducible.)
- **"Is the new version actually better?"** — `python -m nexdash.promote <champion> <challenger>` scores both on one frozen held-out set and **promotes only if** a paired-bootstrap 95% CI on the MAE improvement excludes zero, **no failure-mode slice regresses**, and the **optimistic-error rate does not rise** — the fraction of held-out rows where the model under-predicts energy (the direction that strands a truck), the safety-asymmetric check. An aggregate win that hides a cold/steep regression is rejected.
- **Drift.** `python -m nexdash.drift <reference.csv> <new_batch.csv>` computes per-feature **PSI + KS** against the training reference (standard 0.1 / 0.25 tiers) plus a realized-residual monitor when the batch carries labels, and exits non-zero on significant drift — a ready retraining trigger.

---

## Examples

Two annotated dispatcher transcripts live in **[examples/](examples/)**: [`conversation_1.md`](examples/conversation_1.md) (a comfortable, high-confidence reach) and [`conversation_2.md`](examples/conversation_2.md) (a no-reach case where a charging stop is advised, in which the physics cross-check flags low confidence and the verdict falls back to the conservative estimate). Both show the agent's tool call, the verbatim tool result, and the plain-language answer; every number matches what the live `check_reachability` tool returns.

---

## Tests

```bash
pytest            # or: make test
```

The suite covers physics invariants, dataset shape, feature engineering, model round-trips, evaluation metrics, range logic, and the agent tool-use loop. All LLM and network calls are mocked, so the tests are deterministic and run offline.

---

## License

MIT.
