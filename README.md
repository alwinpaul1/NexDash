# NexDash Case Study

> *"Truck 14 is at 62% charge in Hamburg, loaded to 18 t, with a 240 km run to Bremen on the A1. It's −4 °C and windy. Does it make it without charging?"*

That is the question a dispatcher asks all day, and a wrong answer means a 40-tonne truck stuck on the Autobahn. NexDash answers it for a **Mercedes-Benz eActros 600** (about 600 kWh of usable battery, roughly 500 km of range, 0 to 22 t of payload).

It has three parts. A physics model writes a synthetic training set. A machine-learning model learns from that set and predicts how much energy (in kWh) a truck burns on one leg of a trip. A reachability check turns that prediction into a plain yes-or-no with a safety margin. On top sits a chat agent (MiniMax-M3) that takes a question in normal language, calls these as tools, and explains the answer. You can reach it from a command line, an MCP server, an HTTP API, or a React console that plans a whole trip with real roads, weather, charging stops, and EU driver-hour rules.

The long version, with the full design rationale, lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. This README is the short, run-it version plus the four sections the case study asks for.

---

## How to run

You need **Python 3.10 or newer** (and Node 18+ only if you want the React console).

```bash
# 1. Clone
git clone <repo-url> NexDash
cd NexDash

# 2. Make a virtual environment and install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 3. Train and evaluate, start to finish (seed=42, 6,000 samples).
#    Writes data/dataset.csv, models/energy_model.joblib, reports/evaluation_report.md
python run_pipeline.py
```

Steps 1 to 3 run **fully offline**. No API key is needed to train, evaluate, or read the report. Only the chat agent needs a key.

### Give the agent a key

The agent runs on **MiniMax-M3** through its OpenAI-compatible API. Put the key in a `.env` file at the repo root (it is gitignored and loaded automatically), or export it:

```bash
export MINIMAX_API_KEY=<your-key>
export NEXDASH_LLM_MODEL=MiniMax-M3   # optional, this is already the default
```

### Ask the agent

```bash
# One question
python -m nexdash.cli --once "Can a truck at 62% SOC reach 240 km with 18 t in the cold?"

# Interactive prompt
python -m nexdash.cli
```

### Run the app (API plus web chat)

```bash
# Backend: FastAPI on http://localhost:8000
#   serves /api/predict, /api/route-plan, /api/model-info, /api/chat, /api/health
python dashboard/server.py

# Optional React console: Vite dev server on http://localhost:5173 (proxies the API to :8000)
#   route planner plus a floating chat widget; needs frontend/.env with TomTom/MapTiler keys
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5173** in your browser — that's the dashboard (the Vite dev server proxies `/api` to the backend on `:8000`, so keep both running). The chat widget plans a whole trip from plain language and fills in the result panel. If `MINIMAX_API_KEY` is missing, `/api/chat` returns a friendly note instead of an error, and the rest of the app keeps working.

### API keys (and where to get them)

Copy `frontend/.env.example` to `frontend/.env` and drop your keys in. Each is **free** to obtain:

| Key | Put in | Needed for | Get it at |
|-----|--------|------------|-----------|
| `VITE_TOMTOM_API_KEY` | `frontend/.env` | Geocoding, truck routing, map tiles (the route planner) | [developer.tomtom.com](https://developer.tomtom.com) |
| `MINIMAX_API_KEY` | `.env` (repo root) | The chat agent (MiniMax-M3) | [platform.minimaxi.com](https://platform.minimaxi.com) |
| `VITE_MAPTILER_API_KEY` | `frontend/.env` | *Optional* — nicer map tiles (falls back to free OSM/CARTO) | [maptiler.com](https://www.maptiler.com) |
| `TOMTOM_API_KEY` | `.env` (repo root) | *Optional* — server-side routing for the agent's `plan_route`/MCP (else reuses the Vite key) | same as TomTom above |

Weather and elevation come from **Open-Meteo, which needs no key** (free, no signup). The app degrades gracefully: no TomTom key → the planner can't geocode/route; no MiniMax key → chat shows a friendly note; no MapTiler key → free fallback map tiles.

### Tests and shortcuts

```bash
pytest            # offline and deterministic (the LLM and network are mocked)
```

`make setup`, `make train`, `make test`, `make serve`, and `make agent` wrap the common steps.

### Deploy (one URL, Railway / any Docker host)

The repo ships a `Dockerfile` that builds the **whole app as one container** — FastAPI serves the `/api` routes *and* the built React frontend, so it's a single origin (no CORS, no second deploy). The image builds the frontend, trains the model (deterministic), and starts the server on `$PORT`.

On **Railway**: New Project → Deploy from this GitHub repo (it auto-detects the `Dockerfile`/`railway.json`), then add the variables and deploy:

| Variable | Scope | Value |
|----------|-------|-------|
| `VITE_TOMTOM_API_KEY` | build | your TomTom key (baked into the frontend bundle) |
| `TOMTOM_API_KEY` | runtime | same TomTom key (server-side routing) |
| `MINIMAX_API_KEY` | runtime | your MiniMax key (chat agent) |
| `VITE_MAPTILER_API_KEY` | build | *optional* — nicer map tiles |

Railway exposes the variables to the Docker build automatically. The health check is `/api/health`. Locally you can `docker build -t nexdash . && docker run -p 8000:8000 -e TOMTOM_API_KEY=… -e MINIMAX_API_KEY=… nexdash`.

**MCP server (bonus).** The same tools are exposed as an MCP server so any MCP host (Claude Desktop, Cursor, your own agent…) can call them directly — no NexDash LLM in the loop. It exposes four tools: `predict_energy`, `check_reachability`, `plan_route`, and `model_info`.

### Hosted — connect in 30 seconds (no install)

A public instance is already running over **Streamable HTTP**:

```
https://nexdash-mcp-production.up.railway.app/mcp
```

Add it as a **custom connector** — nothing to install, no key on your side:

- **Claude (Pro / Max):** Customize → **Connectors** → **Add custom connector** → paste the URL → **Add**. No OAuth needed (leave Advanced settings blank).
- **Claude (Team / Enterprise):** an owner adds it under **Organization settings → Connectors → Add → Custom → Web** → paste the URL; members then click **Connect**.
- **Cursor / your own agent:** point it at the same URL with the **Streamable HTTP** transport. For a stdio-only client, bridge it: `npx mcp-remote https://nexdash-mcp-production.up.railway.app/mcp`.

Once connected, the four tools appear and you can ask things like *"plan a route Berlin → Munich, 18 t, depart 9am"*.

> Notes: Claude connects from **Anthropic's cloud**, not your device, so the server has to be public — this one is. It's an **open demo endpoint** (no auth) and `plan_route` uses the host's TomTom key, so please don't hammer it. To run your own hosted copy, deploy the Docker image with `SERVICE_MODE=mcp` (the container then serves Streamable HTTP at `/mcp` on `$PORT`).

### Run it yourself (local, stdio)

For a private/local setup, register it as a stdio server:

1. Install and train once, so the deps and the model exist:
   ```bash
   pip install -e .          # makes `python -m nexdash.mcp_server` importable
   python run_pipeline.py    # writes models/energy_model.joblib (required)
   ```
2. Smoke-test it standalone (it'll wait on stdio — Ctrl-C to exit):
   ```bash
   python -m nexdash.mcp_server
   ```
3. Register it in your MCP host's config (e.g. `claude_desktop_config.json`). Point `command` at the **interpreter that has the package installed** (your venv), and set the keys:
   ```json
   {
     "mcpServers": {
       "nexdash": {
         "command": "/absolute/path/to/NexDash/.venv/bin/python",
         "args": ["-m", "nexdash.mcp_server"],
         "env": {
           "PYTHONPATH": "/absolute/path/to/NexDash/src",
           "TOMTOM_API_KEY": "<your-tomtom-key>"
         }
       }
     }
   }
   ```

Notes:
- If `pip install -e .` ran in that environment, you can drop the `PYTHONPATH` line (the package is already importable).
- `TOMTOM_API_KEY` is only needed for `plan_route` (geocode + routing); `predict_energy` / `check_reachability` / `model_info` run fully offline.
- The trained model must be present on that machine — copy `models/energy_model.joblib` over, or run `python run_pipeline.py` there (deterministic, seed 42).

---

## 1. Approach

NexDash predicts the **energy a truck uses on one segment of a drive**, then reasons about range from there.

**The physics part writes the data.** There is no public energy dataset for 40-tonne electric trucks, so we build one. A physics function (`nexdash.physics.segment_energy_kwh`) adds up the four things that drain the battery: the tyres rolling on the road, air pushing back as the truck moves, the climb up or down a hill (with some energy won back on descents through regenerative braking), and the heating or cooling for the cab. Cold weather and speed make each of these worse, and the function accounts for that. It is fully deterministic, so for any segment you can see exactly where each kWh went. That is what you want from a label generator.

**The synthetic dataset.** `data_gen` samples the kind of legs a German long-haul truck actually drives: 1 to 350 km, 0 to 22 t of payload, 20 to 90 km/h, gentle to steep grades, and temperatures from −15 to 40 °C. It labels each segment with the physics, then adds a little random noise so the data is not unrealistically clean. Long downhill runs keep their genuine negative numbers (the truck gains charge), and no single leg is allowed to imply a mountain that does not exist.

**The model.** The main model is a gradient-boosted tree (`HistGradientBoostingRegressor`) trained on 4,800 samples. Here is the one design choice that matters most: it does **not** try to predict the raw energy number. Instead it predicts the small gap between the physics estimate and the noisy label, and we add that gap back to the physics at prediction time. The reason is safety. A tree can never predict higher than the largest value it saw in training, so a raw model flat-lined and *under-predicted* on long climbs and long distances. That is the exact direction that strands a truck. By letting the physics carry the heavy lifting and asking the model only for a small correction, predictions now track physics even past the training range. A plain linear model still trains on the raw number so you can see what the tree buys you. The reachability check then takes the more cautious of the model and the physics, attaches the test error as an uncertainty band, and flags low confidence when the two disagree by more than 15%.

Full physics and feature reasoning: **[docs/DESIGN.md](docs/DESIGN.md)**. How we calibrated against real eActros figures, with sources: **[docs/REAL_WORLD_CALIBRATION.md](docs/REAL_WORLD_CALIBRATION.md)**.

---

## 2. Evaluation, and where it fails (honestly)

From the committed run (`reports/evaluation_report.md`, 6,000 samples, seed=42, 1,200 held out for testing):

| Metric    | Gradient-boosted tree (main) | Linear baseline |
| --------- | ---------------------------- | --------------- |
| MAE (kWh) | **4.43**                     | 12.21           |
| RMSE      | 8.01                         | 17.63           |
| MAPE      | 6.73 %                       | 62.93 %         |
| R²        | 0.9913                       | 0.9578          |

The headline is **4.43 kWh of average error**, which is about 0.74% of the 600 kWh battery and roughly a third of the linear baseline's error. On calibration, the 90% confidence band covers 87.5% of cases (passing), while the 95% band covers 93.0%, which slightly under-covers and counts as a borderline fail.

**Where it breaks.** Slices are ranked by absolute error in kWh, since that is what actually strands a truck. The good news first: the old under-prediction on long climbs is gone, because the model now follows the physics past the training range instead of flat-lining below it.

- **Steep climbs (grade above +4%)** are the safety-critical case. Error is 6.60 kWh against the noisy labels, but only about 2.6 kWh against the clean physics. Most of that gap is the irreducible label noise, not model error.
- **Heavy loads (above 15 t)** sit at 5.21 kWh. Heavy trucks burn the most energy, so there is simply more room to be off by a few kWh.
- **Cold (below 0 °C)** is 4.75 kWh. Cold is the most expensive regime to drive in, but not the hardest to predict, since the cold effects are smooth and the model learns them well.
- **Steep descents (below −4%)** are 1.07 kWh, small and harmless, though the percentage error looks loud because the net energy is near zero. Regen is credited correctly here, which is why we report kWh and not percentages.
- **Long legs with any gradient** (over ~124 km and above 0.5% grade) are the worst pocket at 18.4 kWh. Again, most of that is label noise on large numbers, the model's true error there is about 6.6 kWh, and the cautious physics fallback still backs it up.

**The caveat that frames all of it.** Every number here measures the model against our **physics**, not against a **real truck**, because the model learns from labels the physics wrote. A low error proves the model relearned our physics well. It does not prove the physics matches a real eActros. The label noise sets a floor of about 4 kWh that no model can beat, the rare corner cases (heavy and steep and cold all at once) are thin in the data, and the accuracy has to be re-earned on real telemetry before anyone bets a delivery on it. The numbers show the pipeline works. They do not show it is calibrated to a real truck yet. Full failure analysis: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#evaluation)**.

---

## 3. Agent design

The agent is a tool-use loop (`nexdash.agent.DispatcherAgent`), not the model doing math in its head. It runs on **MiniMax-M3 only**, and the system prompt forbids it from making up numbers. If it wants a number, it has to call a tool.

`ask(question)` sends the question and the tool list to MiniMax-M3, runs any tool calls through `nexdash.tools.dispatch`, feeds the results back, and loops (up to 8 turns) until plain text comes out. MiniMax is a reasoning model, so its internal `<think>` block is stripped before the reply is shown. There are three tools:

- **`predict_energy`** takes distance, payload, speed, gradient, temperature, and wind, and returns predicted kWh plus diagnostic plots.
- **`check_reachability`** takes a starting charge and a segment, and returns reach or no-reach, the margin in kWh, remaining charge, remaining range, a confidence flag, and the physics cross-check note.
- **`plan_route`** takes two cities, payload, starting charge, and temperature, then geocodes them, pulls a real truck route from TomTom, adds elevation and weather from Open-Meteo, simulates charge over the trip, drops in charging stops, checks EU driver-hour limits, and returns the full plan.

The same three tools sit behind every interface: the CLI (`nexdash.cli`), the MCP server (`nexdash.mcp_server`), the HTTP endpoint (`POST /api/chat`), and the React chat widget. The numbers are shared and deterministic. The only thing the model writes is the explanation around them. The CLI exits with a clear message if the key is missing, and `/api/chat` degrades gracefully instead of crashing.

Two real transcripts: **[examples/conversation_1.md](examples/conversation_1.md)** (a confident reach) and **[examples/conversation_2.md](examples/conversation_2.md)** (a no-reach, where the physics cross-check flags low confidence and falls back to the safer estimate).

---

## 4. The long-term plan: retrain, prove, and catch drift

Synthetic physics is the starting point, not the destination. When real eActros telemetry arrives, the data source swaps in without changing the model interface. Full design: **[docs/LONG_TERM.md](docs/LONG_TERM.md)**. The pieces below already run today:

- **Retraining, in three phases.** Phase A (now) is 100% synthetic. Phase B (early real data) mixes synthetic and real, gives the synthetic less weight, and feeds the physics in as an input so the model keeps learning a small correction. Phase C (mature) is mostly real data, with synthetic held back for the rare cold, heavy, and steep cases so the model does not fall off a cliff at the edges. Retraining happens weekly while data is still coming in, then monthly or quarterly, with extra runs triggered by drift, a drop in performance, a fleet or firmware change, or a new season. Every retrain is reproducible, with a pinned seed and a recorded data version.
- **Proving a new model is actually better.** The new model is guilty until proven innocent. `python -m nexdash.promote <champion> <challenger>` promotes a challenger **only if** its error improvement is statistically real, **no failure slice gets worse**, and the rate of dangerous under-predictions does not rise. After that come a calibration audit, a shadow run on live traffic, and a slow rollout (5%, then 25, 50, 100) that aborts the moment any live slice slips. A model that wins on average but quietly hurts the cold or steep cases gets rejected.
- **Catching drift.** `python -m nexdash.drift <reference.csv> <new_batch.csv>` measures how far the new data has moved from the old and exits with an error when the shift is significant. Four things are watched: the inputs, the predictions, the real error once labels arrive, and the operational signals. Alerts feed either a retrain or a page plus an automatic fallback.
- **Versioning and rollback.** Every model is tagged by its data, its code, its seed, and its full metrics through `nexdash.registry`. The previous model stays loaded and ready, so a rollback is a single pointer flip that takes minutes, and it happens automatically on a critical alarm. The floor: if no model is trustworthy, fall back to the plain physics model with a wider safety margin.
