# NexDash — EV Truck Range Intelligence

> *"Truck 14 is at 62% charge in Hamburg, loaded to 18 t, with a 240 km run to Bremen on the A1. It's −4 °C and windy. Does it make it without charging?"*

NexDash answers that dispatcher question for a **Mercedes-Benz eActros 600** (~600 kWh usable battery, ~500 km range, 0–22 t payload). The pieces: a deterministic **physics model** generates a justified synthetic dataset, a **HistGradientBoosting** model learns from it to predict per-segment energy (kWh), and a **reachability service** turns those predictions into a reach / no-reach verdict with an energy margin. On top of that sits an **LLM dispatcher agent** (MiniMax-M3) that answers plain-language questions by calling these as tools. You can reach it over a CLI, an MCP server, an HTTP API, or a React console that plans a whole trip — real route geometry, weather/elevation, charging stops, EU-561 driver hours.

The detailed architecture and design rationale live in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (the former long README). This README is the short, run-focused version plus the four case-study sections.

---

## Quickstart — how to run

Requires **Python ≥ 3.10** (and Node ≥ 18 only for the optional React console).

```bash
# 1. Clone
git clone <repo-url> NexDash
cd NexDash

# 2. Virtual environment + install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 3. Train + evaluate end to end (deterministic, seed=42, 6,000 samples).
#    Writes data/dataset.csv, models/energy_model.joblib, reports/evaluation_report.md
python run_pipeline.py
```

Steps 1–3 run **fully offline** — no API key for the model, evaluation, or report. The LLM agent is the only part that needs a key (next).

### Set the LLM key (only for the agent / CLI / chat)

The dispatcher runs on **MiniMax-M3** via its OpenAI-compatible API. Put the key in a `.env` at the repo root (gitignored, auto-loaded by the server), or export it:

```bash
export MINIMAX_API_KEY=<your-key>
export NEXDASH_LLM_MODEL=MiniMax-M3   # optional, this is the default
```

### Ask the agent

```bash
# One-shot
python -m nexdash.cli --once "Can a truck at 62% SOC reach 240 km with 18 t in the cold?"

# Interactive REPL
python -m nexdash.cli
```

### Run the app (API + web chat)

```bash
# Backend — FastAPI on http://localhost:8000
#   serves /api/predict, /api/route-plan, /api/model-info, /api/chat, /api/health
python dashboard/server.py

# Bonus React console — Vite dev server on http://localhost:5173 (proxies API to :8000)
#   route planner + floating chat widget; needs frontend/.env with TomTom/MapTiler keys
cd frontend
npm install
npm run dev
```

In the web console, the **chat widget** plans a whole trip from plain language and fills the result panel. If `MINIMAX_API_KEY` isn't set, `/api/chat` returns a friendly message instead of erroring out, and the rest of the app keeps working.

### Tests and Make shortcuts

```bash
pytest            # deterministic, offline (LLM + network mocked)
```

`make setup` · `make train` · `make test` · `make serve` · `make agent` wrap the common flows.

**MCP server (bonus).** `predict_energy` and `check_reachability` are exposed as an MCP server. Register it in any MCP host:

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

Smoke-test standalone with `python -m nexdash.mcp_server`.

---

## 1. Approach

NexDash predicts **energy (kWh) for one driving segment**, then reasons about range from that.

**Physics teacher.** `nexdash.physics.segment_energy_kwh` sums the four classic loads, all divided by drivetrain efficiency 0.85: rolling resistance (`Crr · m · g · d`, with a speed- and cold-dependent `Crr`), aerodynamic drag (`½ ρ(T) Cd A v² d`, with temperature-dependent air density from the ideal gas law), signed gradient work (`m g Δh`, with regen credited on descents — base 0.60, tapered by temperature and grade), and a U-shaped HVAC/auxiliary load. It's deterministic and you can read off where every kWh went, which is exactly what you want from a label generator.

**Synthetic dataset.** `data_gen` samples the German long-haul envelope (distance 1–350 km, payload 0–22 t, speed 20–90 km/h, gradient ±6 % capped per row to stay geographically plausible, temperature −15–40 °C, wind ±12 m/s) and labels each segment with the physics, then adds ~6 % multiplicative noise and a small additive sensor term. Net-regen descents keep **genuine negative labels** — no zero-clamp — and the implied net climb is capped at ~1500 m so a single row can't imply a mountain that isn't there.

**Model.** The primary model is a `HistGradientBoostingRegressor` trained on 4,800 samples with **13 features**: the 6 raw inputs plus 7 mechanistic terms (`abs_gradient`, `payload_x_gradient`, `temp_dev_from_20`, `speed_sq`, `payload_x_distance`, `distance_x_gradient`, `net_climb`). A `LinearRegression` baseline trains on the same data so you can see what the non-linearity actually buys. The reachability service turns predictions into verdicts, attaches the held-out MAE as the uncertainty band, and runs a **first-principles physics cross-check**: when model and physics diverge by more than 15 %, it flags `confidence: "low"` and falls back to the conservative physics value. That's the out-of-envelope detector.

Full physics and feature rationale: **[docs/DESIGN.md](docs/DESIGN.md)**. Calibration to real eActros figures (with cited sources): **[docs/REAL_WORLD_CALIBRATION.md](docs/REAL_WORLD_CALIBRATION.md)**.

---

## 2. Evaluation — and where it fails (honestly)

From the committed run (`reports/evaluation_report.md`, n=6,000, seed=42, 1,200-row held-out test):

| Metric    | HistGradientBoosting (primary) | Linear baseline |
| --------- | ------------------------------ | --------------- |
| MAE (kWh) | **5.661**                      | 11.684          |
| RMSE      | 8.998                          | 17.195          |
| MAPE      | 12.40 %                        | 42.92 %         |
| R²        | 0.9890                         | 0.9598          |

The headline is **MAE 5.66 kWh** — about 0.94 % of the ~600 kWh battery, and roughly half the linear baseline's error. On calibration (split-conformal), 90 % nominal intervals cover 90.8 % empirically [88.3–93.0 %], which passes, with ECE 0.0133.

**Where it breaks.** Slices are ranked by absolute MAE, since that's the metric that actually strands a truck:

- **Steep climbs (>+4 % grade)** — the safety-critical slice, and the weakest one. MAE **8.29 kWh** (n=36). Climbs turn payload mass into potential energy faster than anything else, so a miss here is a big absolute miss. The signed `distance_x_gradient` feature plus a 1500 m net-climb ceiling fixed the old climb saturation, and at ~25 km chunks the model now tracks physics within ~10 % across ±6 %. It's still the part I trust least.
- **Heavy payload (>15 t)** — MAE **6.52 kWh** (n=378). Payload scales rolling resistance and gravity work linearly, so heavy loads consume the most and leave the most absolute room to get it wrong.
- **Cold (<0 °C)** — MAE **6.07 kWh** (n=166). Cold is the most *expensive* regime, but not the hardest to predict. The cold channels — denser air, cold tyres, reduced regen — are smooth and learnable, so error stays flat across temperature. Expensive and hard-to-predict are not the same thing.
- **Steep descents (<−4 %)** — MAE **2.45 kWh** (n=33). Small and operationally harmless, but the MAPE is loud (15.77 %) because regen drives net energy near zero. The model credits regen correctly now (25 km / −6 %: physics −40 kWh, model −31 kWh). This is the whole reason we headline MAE instead of MAPE.
- **Auto-discovered pocket:** long legs with gradient (`>116 km`) hit MAE **16.25 kWh** (2.87× lift). Trees can't predict above the energy labels they were shown, so the tail is a structural extrapolation limit — not something you tune away.

**The caveat that frames everything.** Every metric here measures model-vs-**physics**, not model-vs-**reality** — the labels come from the same `segment_energy_kwh` the model learns. A low MAE proves the model re-learned our physics. It does not prove the physics matches a real truck. The ~6 % label noise sets a floor around 4–6 kWh MAE that no predictor can beat; the rare multi-feature corners (heavy and steep and cold all at once) are under-represented; and the accuracy has to be re-earned on real telemetry before anyone bets a delivery on it. The point metrics show the pipeline works. They don't show it's calibrated to a real eActros. Full failure analysis: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#evaluation)**.

---

## 3. Agent design

The dispatcher is an **LLM tool-use loop** (`nexdash.agent.DispatcherAgent`), not the model doing arithmetic in its head. It runs on **MiniMax-M3 only**, and the system prompt forbids inventing numbers — if it wants a number, it has to call a tool.

`ask(question)` sends the question and the tool specs to MiniMax-M3, runs any tool calls through `nexdash.tools.dispatch`, feeds the results back, and loops (up to 8 turns) until plain text comes back. MiniMax is a reasoning model, so its `<think>…</think>` block is stripped from the reply. Three deterministic tools:

- **`predict_energy`** — distance, payload, speed, gradient, temperature, wind → predicted kWh (+ diagnostic plots).
- **`check_reachability`** — SOC + segment → reach/no-reach, margin_kwh, remaining SOC %, remaining range, confidence flag, physics cross-check note.
- **`plan_route`** — origin/destination cities, payload, start SOC, temperature → geocodes, fetches a real truck route (TomTom), enriches with Open-Meteo elevation/weather, simulates SOC per ~25 km segment, inserts DC-charging stops, checks EU-561 driver hours, returns the full plan.

The same tools sit behind every interface — **CLI** (`nexdash.cli`), **MCP server** (`nexdash.mcp_server`, FastMCP), **HTTP** (`POST /api/chat`), and the **React chat widget**. The numbers are shared and deterministic; the only thing the LLM generates is the explanation around them. The CLI exits with a clear message if `MINIMAX_API_KEY` is missing, and `/api/chat` degrades instead of crashing.

Two real transcripts: **[examples/conversation_1.md](examples/conversation_1.md)** (high-confidence reach) and **[examples/conversation_2.md](examples/conversation_2.md)** (no-reach, where the physics cross-check flags low confidence and falls back to the conservative estimate).

---

## 4. Long-term — retrain, prove a better version, detect drift

Synthetic physics is the launchpad, not the destination. When real eActros telemetry shows up, the data source swaps out without touching the model interface. Full design: **[docs/LONG_TERM.md](docs/LONG_TERM.md)**. The core is already runnable:

- **Retraining path (A→B→C).** Phase A (today): 100 % synthetic. Phase B (early telematics): mix synthetic and real, down-weight the synthetic, and feed physics in as an input feature so the model learns the residual. Phase C (mature): mostly real labels, with synthetic held back for the rare regimes (cold, heavy, steep) so the model doesn't fall off a cliff when it extrapolates. Cadence is weekly while data is still accumulating, then monthly or quarterly; event triggers fire on drift, a performance breach, a fleet or firmware change, or the start of a season. Every retrain is deterministic — pinned seed, data SHA, lockfile.
- **Proving a new model — guilty until proven innocent.** The bar is safety-asymmetric. `python -m nexdash.promote <champion> <challenger>` promotes **only if** a paired-bootstrap 95 % CI on the MAE improvement excludes zero, **no failure-mode slice regresses**, and the **optimistic-error rate** (under-prediction, the direction that strands a truck) does not rise. After that: a conformal calibration audit that fails on under-coverage, shadow mode on live traffic, and a 5→25→50→100 % canary that auto-aborts on any live slice breach. An aggregate win that quietly hides a cold or steep regression gets rejected.
- **Drift.** `python -m nexdash.drift <reference.csv> <new_batch.csv>` computes per-feature **PSI + KS** (0.1 / 0.25 tiers), adds a realized-residual monitor once labels are present, and exits non-zero when drift is significant. Four layers are watched: input drift, prediction drift, residual/performance (once labels arrive), and operational. Tiered alerts feed either a retrain priority or a page plus auto-fallback.
- **Versioning & rollback.** Every model is content-addressed by data SHA, code SHA, seed, and full metrics via `nexdash.registry`. The previous champion stays hot-loadable, so rollback is a pointer flip — minutes, and automatic on a critical live alarm. The fail-safe floor: if no model is trustworthy, drop back to the deterministic physics model with a widened reserve.

---

## License

MIT.
