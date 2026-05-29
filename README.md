# NexDash — EV Truck Range Intelligence

> *"Truck 14 is sitting at 62% charge in Hamburg. It's loaded to 18 tonnes and has a 240 km run down to Bremen on the A1. It's −4 °C and windy. Will it make it without a charging stop?"*

That single question — asked dozens of times a day by a fleet dispatcher — is what NexDash answers. We model a **Mercedes-Benz eActros 600** (≈600 kWh usable battery, ≈500 km real-world range, 0–22 t payload, GVW up to ~40 t) and turn raw trip parameters into a confident, explainable **reach / no-reach** verdict with an energy margin and a remaining-range estimate.

NexDash is a small but complete AI-engineering system:

1. A **physics ground-truth simulator** for segment energy.
2. A **synthetic dataset generator** built on that physics.
3. A **machine-learning energy model** (gradient boosting, with a linear baseline for honest comparison).
4. A **range-reachability service** wrapping the model with an operational reserve.
5. An **LLM dispatcher agent** that answers natural-language questions using tools — also exposed over **MCP**.
6. A **FastAPI + dashboard** front-end for live range checks.

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
# Ask the dispatcher agent a question (needs ANTHROPIC_API_KEY)
python -m nexdash.cli --once "Can a truck at 62% SOC reach 240 km with 18 t in the cold?"
python -m nexdash.cli                       # interactive REPL

# Launch the dashboard (FastAPI on http://localhost:8000)
python dashboard/server.py
```

A `Makefile` wraps the common flows: `make setup`, `make data`, `make train`, `make test`, `make serve`, `make agent`.

### API keys

The agent and CLI call the Anthropic API. Copy `.env.example` to `.env` and set your key, or export it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The model id used is the latest Claude: **`claude-opus-4-8`**. The physics, dataset, model, evaluation and dashboard work entirely offline — only the LLM agent/CLI need a key.

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
│   ├── tools.py               # Anthropic tool schemas + dispatch()
│   ├── mcp_server.py          # FastMCP server exposing the same tools
│   ├── agent.py               # DispatcherAgent: tool-use loop
│   └── cli.py                 # REPL / --once front-end for the agent
├── dashboard/
│   ├── server.py              # FastAPI: GET / , POST /api/predict
│   ├── index.html             # EV-green admin dashboard + live Range Check panel
│   └── app.js                 # Fetch wiring for the Range Check panel
├── data/                      # Generated dataset.csv lands here
├── models/                    # Trained energy_model.joblib lands here
├── reports/                   # evaluation_report.md + figures/ (generated)
├── docs/
│   ├── DESIGN.md              # Physics + feature + modelling rationale
│   └── LONG_TERM.md           # Retraining, versioning, drift strategy
├── examples/                  # Runnable usage snippets (physics, prediction, range, agent)
└── tests/                     # pytest suite (LLM/network mocked)
```

---

## Modeling approach

NexDash predicts **energy consumption (kWh) for a single driving segment** and uses that to reason about range.

**Ground truth — physics.** `segment_energy_kwh(...)` sums the four classic loads, all divided by drivetrain efficiency:

- **Rolling resistance** — `Crr · m · g · distance`, scaling with total mass (kerb + payload).
- **Aerodynamic drag** — `½ · ρ · Cd · A · v² · distance`, using the effective air speed (`speed + headwind`); the dominant term at motorway pace.
- **Gradient (potential energy)** — `m · g · Δh`; downhill segments recover `regen_eff` (60%) of the would-be energy via regenerative braking, which is why steep descents can net *negative* kWh.
- **Auxiliary / HVAC** — a base load plus a cabin-conditioning term that rises at **both** cold and hot extremes, scaled by travel time (`distance / speed`).

`energy_breakdown(...)` returns each component separately for transparency.

**Features.** The model consumes the six raw inputs — `distance_km, payload_t, speed_kph, gradient_pct, temperature_c, wind_mps` — plus engineered terms that encode the physics structure so a tree model finds the patterns faster: a **payload × gradient** interaction (mass amplifies grade), **|gradient|**, **temperature deviation from 20 °C** (HVAC is a U-shape, not linear), and a **speed² proxy** (drag is quadratic). Each is documented in `nexdash.features`.

**Estimator.** A `HistGradientBoostingRegressor` is the production model; a `LinearRegression` baseline is trained alongside it so every report shows *how much the non-linear model actually buys us*. `EnergyModel` wraps a scikit-learn pipeline and accepts raw feature dicts (it calls `features.transform` internally), so callers never duplicate engineering logic.

Full rationale, formulas and assumptions: **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## Evaluation

`run_pipeline.py` writes `reports/evaluation_report.md` with **real, regenerated numbers** plus figures. The values below are the **actual results** from the committed run (`n=6000`, `seed=42`, 4,800 train / 1,200 held-out test). The truck spec and synthetic operating envelope are now **calibrated to real Mercedes-Benz eActros 600 figures** (German long-haul ops) — see [`docs/REAL_WORLD_CALIBRATION.md`](docs/REAL_WORLD_CALIBRATION.md) for every cited assumption.

**Held-out test set (gradient-boosting primary):** MAE **8.070 kWh**, RMSE 16.543 kWh, MAPE 13.05 %, R² 0.9854 — a range-error proxy of **1.416 %** of a nominal 500 km full-charge trip.

The model-vs-baseline comparison (internal validation split) shows how much the non-linear model buys us over a transparent linear reference:

| Metric              | Gradient Boosting (primary) | Linear baseline |
| ------------------- | --------------------------- | --------------- |
| MAE (kWh)           | 7.683                       | 39.139          |
| RMSE (kWh)          | 14.961                      | 62.690          |
| MAPE (%)            | 11.91                       | 172.51          |
| R²                  | 0.9856                      | 0.7468          |

The gradient-boosting model cuts held-out MAE by roughly **5×** versus the linear baseline. Three diagnostic figures are saved to `reports/figures/`: **predicted-vs-actual**, **residual-vs-temperature**, and **error-by-payload**.

### Where it fails and why

A model is only trustworthy if you know its blind spots. `failure_mode_report(...)` slices error by operating regime; the patterns we consistently observe:

- **Temperature extremes (cold < 0 °C, hot > 30 °C).** HVAC is the hardest term to model — its U-shape and time-dependence mean the tails carry the largest MAPE. A dispatcher planning a winter run should treat the margin as *narrower* than the headline number.
- **Steep gradients (|grade| > 4%).** Regen recovery on descents is governed by a fixed efficiency in the physics, but real recovery depends on speed, battery state and braking style — so steep-down and steep-up bins show elevated error and more variance.
- **Heavy payloads (> 15 t).** These are under-represented at the *combined* extremes (e.g. heavy **and** cold **and** uphill), so the model extrapolates rather than interpolates and the error band widens.

The reachability service exposes this honestly: every verdict carries a `confidence_note` referencing the model's MAE band, and the dashboard turns the verdict **red** when the margin is within that band rather than pretending a knife-edge result is safe.

---

## Agent design

The dispatcher experience is an **LLM tool-use loop**, not free-form generation over numbers.

- **Tools (`nexdash.tools`).** Two Anthropic tool schemas — `predict_energy` and `check_reachability` — with thin, JSON-serializable wrappers and a `dispatch(name, args)` router. The model is *forbidden in the system prompt from inventing numbers*; it must call a tool.
- **Agent (`nexdash.agent.DispatcherAgent`).** `ask(question)` sends the question plus `TOOL_SPECS` to Claude, executes any `tool_use` blocks via `dispatch`, feeds the `tool_result` back, and loops until the model returns plain-language text. The system prompt frames it as a fleet dispatcher assistant that always states the **margin** and a **caveat**. The client is injectable, so tests run fully mocked with no network.
- **MCP (`nexdash.mcp_server`).** The exact same two tools are registered on a `FastMCP` server named `nexdash`, so any MCP-capable host gets NexDash's reasoning without importing Python.
- **CLI (`nexdash.cli`).** A friendly REPL (or `--once`) over the agent, with a clear message if `ANTHROPIC_API_KEY` is missing.

This separation means the *numbers* are deterministic and testable while the *explanation* is the only thing the LLM is responsible for.

---

## Long-term: real data

The synthetic-physics approach is a launchpad, not the destination. As real eActros telemetry arrives, NexDash is designed to swap the data source without touching the model interface. The strategy — **continuous retraining, model versioning, and drift detection** — is detailed in **[docs/LONG_TERM.md](docs/LONG_TERM.md)**:

- **Retraining.** Telemetry replaces the synthetic generator as the labelled source; `features.transform` and `EnergyModel` are unchanged, so the pipeline is a one-line data swap. The physics simulator survives as a sanity oracle and a cold-start prior for routes with little real data.
- **Versioning.** Models are persisted with joblib alongside their metrics and training-data hash, so every deployed verdict is traceable to a specific model + dataset, and rollback is trivial.
- **Drift.** Monitor input-feature distributions (e.g. seasonal temperature shift, new payload mixes) and live prediction residuals against actual consumption; trigger retraining when MAPE on recent trips exceeds the validated band.

---

## Examples

Runnable, self-contained snippets live in **[examples/](examples/)** — physics breakdowns, a single ML prediction from raw features, a reachability check, and a mocked agent conversation. They double as the fastest way to learn the public API.

---

## Tests

```bash
pytest            # or: make test
```

The suite covers physics invariants, dataset shape, feature engineering, model round-trips, evaluation metrics, range logic, and the agent tool-use loop. All LLM and network calls are mocked, so the tests are deterministic and run offline.

---

## License

MIT.
