"""FastAPI server for the NexDash EV Truck Range Intelligence dashboard.

Serves the static dashboard UI and exposes a JSON range-check endpoint that
delegates to :func:`nexdash.range.check_reachability`. The trained energy model
is loaded once at application startup so requests stay fast; if the model
artifact is missing the ``/api/predict`` endpoint returns a clear ``503`` JSON
payload telling the operator to run ``run_pipeline.py`` first.

Run locally::

    python dashboard/server.py

This starts uvicorn on ``0.0.0.0:8000``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexdash.config import DEFAULT_MODEL_PATH
from nexdash.model import EnergyModel
from nexdash import range as range_module
from nexdash import route_planner as route_planner_module
from nexdash import fleet as fleet_module
from nexdash import agent as agent_module

logger = logging.getLogger("nexdash.dashboard")

# Directory that holds index.html / app.js and any other static assets.
DASHBOARD_DIR = Path(__file__).resolve().parent
INDEX_FILE = DASHBOARD_DIR / "index.html"


class PredictRequest(BaseModel):
    """Inputs for a single range-check request from the dashboard panel."""

    soc_pct: float = Field(..., description="Current state of charge, percent (0-100).")
    distance_km: float = Field(..., description="Segment distance to travel (km).")
    payload_t: float = Field(..., description="Cargo payload (tonnes).")
    speed_kph: float = Field(..., description="Average travel speed (km/h).")
    gradient_pct: float = Field(..., description="Net road gradient (percent; negative = downhill).")
    temperature_c: float = Field(..., description="Ambient temperature (degrees Celsius).")
    wind_mps: float = Field(0.0, description="Headwind component (m/s); optional, defaults to 0.")


class RoutePlanRequest(BaseModel):
    """Inputs for a full route-plan SOC simulation from the NexOS planner.

    The frontend computes road geometry + total distance/time via the TomTom
    truck-routing API and sends those totals here; the backend runs the heavy
    model-driven SOC drain, charging-stop insertion and EU 561 driver-hours.
    """

    waypoints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered [{lat,lng,label?}] origin + destinations.",
    )
    distanceKm: float = Field(..., description="Total route distance (km).")
    durationS: float = Field(..., description="Total driving time (seconds).")
    startSoc: float = Field(100.0, description="Starting state of charge (%).")
    minSoc: float = Field(15.0, description="SOC floor never to dip below (%).")
    payloadKg: float = Field(0.0, description="Cargo payload (kg).")
    reservePct: float = Field(10.0, description="Safety-reserve buffer above min SOC (%).")
    maxChargeKw: float = Field(350.0, description="Max charging power (kW).")
    departure: Optional[str] = Field(None, description="ISO local departure datetime.")
    temperatureC: float = Field(15.0, description="Ambient temperature (deg C).")
    geometry: Optional[list[list[float]]] = Field(
        None,
        description=(
            "Optional [[lat, lng], ...] road polyline from the routing engine. "
            "When present the backend enriches it (elevation gradient + weather "
            "via Open-Meteo) and simulates SOC per enriched segment; when absent "
            "the flat-route approximation is used."
        ),
    )


def _load_model_if_present(app: FastAPI) -> None:
    """Pre-load the energy model into ``app.state`` if the artifact exists.

    Loading at startup both warms the prediction cache and lets us surface a
    clear error immediately rather than on the first request. The presence flag
    drives the ``503`` behaviour of ``/api/predict``.
    """

    app.state.model_available = False
    app.state.model = None

    if not Path(DEFAULT_MODEL_PATH).exists():
        logger.warning(
            "Energy model artifact not found at %s. /api/predict will return 503 "
            "until you run: python run_pipeline.py",
            DEFAULT_MODEL_PATH,
        )
        return

    try:
        # EnergyModel.load also primes the module-level cache used by
        # range.check_reachability -> model.predict_energy.
        app.state.model = EnergyModel.load(DEFAULT_MODEL_PATH)
        app.state.model_available = True
        logger.info("Loaded energy model from %s", DEFAULT_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive startup guard
        logger.exception("Failed to load energy model: %s", exc)
        app.state.model_available = False
        app.state.model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; nothing special to tear down."""

    _load_model_if_present(app)
    yield


app = FastAPI(
    title="NexDash EV Truck Range Intelligence",
    description="Range-check API and dashboard for the Mercedes-Benz eActros 600 fleet.",
    version="1.0.0",
    lifespan=lifespan,
)

# Open CORS so the dashboard (and external tools) can call the API freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_MISSING_MESSAGE = (
    "Energy model artifact not found. Train it first by running "
    "`python run_pipeline.py` from the project root, then restart the server."
)


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the single-page dashboard UI."""

    if not INDEX_FILE.exists():  # pragma: no cover - misconfigured deployment
        return JSONResponse(
            status_code=500,
            content={"error": f"index.html not found at {INDEX_FILE}"},
        )
    return FileResponse(INDEX_FILE)


@app.get("/api/health", include_in_schema=True)
async def health() -> dict:
    """Lightweight health check reporting whether the model is loaded."""

    return {
        "status": "ok",
        "model_available": bool(getattr(app.state, "model_available", False)),
        "model_path": str(DEFAULT_MODEL_PATH),
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Run a range-check for the supplied trip parameters.

    Returns the full :func:`nexdash.range.check_reachability` dict. If the
    trained model is unavailable, responds with ``503`` and a clear message.
    """

    if not getattr(app.state, "model_available", False):
        return JSONResponse(
            status_code=503,
            content={"error": "model_unavailable", "detail": _MODEL_MISSING_MESSAGE},
        )

    try:
        result = range_module.check_reachability(
            soc_pct=req.soc_pct,
            distance_km=req.distance_km,
            payload_t=req.payload_t,
            speed_kph=req.speed_kph,
            gradient_pct=req.gradient_pct,
            temperature_c=req.temperature_c,
            wind_mps=req.wind_mps,
            model_path=DEFAULT_MODEL_PATH,
        )
    except Exception as exc:  # pragma: no cover - surfaces unexpected failures
        logger.exception("Range check failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "range_check_failed", "detail": str(exc)},
        )

    return result


@app.post("/api/route-plan")
async def route_plan(req: RoutePlanRequest):
    """Run the model-driven SOC simulation for a full multi-stop route.

    Delegates to :func:`nexdash.route_planner.plan_route`. Returns the
    ``socProfile`` / ``segments`` / ``chargingStops`` / ``summary`` portion of
    the frontend ``PlanResult`` (the frontend owns ``geometry``). If the trained
    model is unavailable, responds with ``503`` and a clear message.
    """

    if not getattr(app.state, "model_available", False):
        return JSONResponse(
            status_code=503,
            content={"error": "model_unavailable", "detail": _MODEL_MISSING_MESSAGE},
        )

    try:
        result = route_planner_module.plan_route(
            distance_km=req.distanceKm,
            duration_s=req.durationS,
            start_soc=req.startSoc,
            min_soc=req.minSoc,
            payload_kg=req.payloadKg,
            reserve_pct=req.reservePct,
            max_charge_kw=req.maxChargeKw,
            departure=req.departure,
            temperature_c=req.temperatureC,
            waypoints=req.waypoints,
            geometry=req.geometry,
            model_path=DEFAULT_MODEL_PATH,
        )
    except Exception as exc:  # pragma: no cover - surfaces unexpected failures
        logger.exception("Route plan failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "route_plan_failed", "detail": str(exc)},
        )

    return result


@app.get("/api/fleet")
async def fleet() -> JSONResponse:
    """Return the deterministic mock fleet, scored through the real model.

    Delegates to :func:`nexdash.fleet.fleet_status` (which lazy-loads the model)
    and derives the status/at-risk counts the dispatcher console summarises. The
    fleet helper is fully fail-soft: if the model artifact is missing the
    per-truck model fields collapse to ``None`` (those trucks are not counted as
    at-risk) but the roster still renders. Any unexpected failure returns a
    clear ``500`` JSON payload rather than an opaque error.
    """

    try:
        trucks = fleet_module.fleet_status(model_path=DEFAULT_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - surfaces unexpected failures
        logger.exception("Fleet status failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "fleet_status_failed", "detail": str(exc)},
        )

    counts = {
        "inTransit": sum(1 for t in trucks if t["status"] == "in_transit"),
        "available": sum(1 for t in trucks if t["status"] == "available"),
        "charging": sum(1 for t in trucks if t["status"] == "charging"),
        "maintenance": sum(1 for t in trucks if t["status"] == "maintenance"),
        "atRisk": sum(1 for t in trucks if t.get("atRisk") is True),
    }
    return JSONResponse(content={"trucks": trucks, "counts": counts})


@app.get("/api/model-info")
async def model_info() -> JSONResponse:
    """Return the trained energy model's headline metrics for the console.

    Delegates to :func:`nexdash.fleet.model_info`, which prefers the metrics
    stored on the model artifact and falls back to parsing the evaluation
    report. It is fail-soft (missing pieces become ``None``); only an
    unexpected failure produces a ``500``.
    """

    try:
        info = fleet_module.model_info(model_path=DEFAULT_MODEL_PATH)
    except Exception as exc:  # pragma: no cover - surfaces unexpected failures
        logger.exception("Model info failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "model_info_failed", "detail": str(exc)},
        )

    return JSONResponse(content=info)


class ChatMessage(BaseModel):
    """A single chat turn from the dispatcher conversation."""

    role: str = Field(..., description='"user" or "assistant".')
    content: str = Field(..., description="Plain-text message content.")


class ChatRequest(BaseModel):
    """The conversation history for the dispatcher chat agent."""

    messages: list[ChatMessage] = Field(default_factory=list)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    """Run the LLM dispatcher agent on the conversation.

    The agent (``nexdash.agent.DispatcherAgent``) uses Claude with the
    deterministic energy/reachability tools. Returns the reply plus the list of
    tools the agent invoked this turn (so the UI can show them). If the server
    has no ``ANTHROPIC_API_KEY`` it returns a friendly degraded message rather
    than a hard error.
    """

    history = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        agent = agent_module.DispatcherAgent(model_path=DEFAULT_MODEL_PATH)
        result = agent.chat(history)
        return JSONResponse(
            content={"reply": result.get("reply", ""), "tools": result.get("tools", [])}
        )
    except agent_module.MissingAPIKeyError:
        return JSONResponse(
            content={
                "reply": (
                    "⚠️ The dispatcher agent isn't connected yet — set "
                    "MINIMAX_API_KEY (or ANTHROPIC_API_KEY) on the server and "
                    "restart to chat live. Meanwhile the Route Planner and range "
                    "tools still work."
                ),
                "tools": [],
                "degraded": True,
            }
        )
    except agent_module.AgentError as exc:
        # Recoverable provider issue (e.g. a rate-limited free model): degrade
        # gracefully with the explanation rather than a hard 500.
        return JSONResponse(
            content={"reply": f"⚠️ {exc}", "tools": [], "degraded": True}
        )
    except Exception as exc:  # pragma: no cover - surfaces unexpected failures
        logger.exception("Chat failed: %s", exc)
        return JSONResponse(
            status_code=500, content={"error": "chat_failed", "detail": str(exc)}
        )


# Mount the dashboard directory for static assets (app.js, css, images, ...).
# Mounted last so explicit routes above take precedence over the static handler.
app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=False), name="static")


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a repo-root ``.env`` into the environment.

    Minimal, dependency-free, and only invoked from ``main()`` (never at import
    time), so it can't leak the (gitignored) ``.env`` keys into the test suite.
    Existing environment variables always win.
    """
    import os
    from pathlib import Path

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``python dashboard/server.py``."""

    import uvicorn

    _load_dotenv()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
