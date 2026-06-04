# NexDash — single-container deploy (Railway / any Docker host).
# One service: FastAPI serves the /api routes AND the built React frontend, so
# the whole app is one URL with no CORS and no second deploy.

# ----------------------------------------------------------------------------
# Stage 1 — build the React frontend (Vite bakes VITE_* keys into the bundle).
# ----------------------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Client-side keys are baked in at build time. Railway passes service variables
# of the same name as build args automatically (declare them as ARG here).
ARG VITE_TOMTOM_API_KEY=""
ARG VITE_MAPTILER_API_KEY=""
# VITE_API_BASE stays empty so the frontend calls /api on the SAME origin.
ENV VITE_TOMTOM_API_KEY=$VITE_TOMTOM_API_KEY \
    VITE_MAPTILER_API_KEY=$VITE_MAPTILER_API_KEY \
    VITE_API_BASE=""
RUN npm run build   # -> /app/frontend/dist

# ----------------------------------------------------------------------------
# Stage 2 — Python runtime: install, train the model, serve.
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

# Python deps (sklearn pinned to 1.7.0 — same version trains & loads the model).
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

# App source.
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY run_pipeline.py ./
RUN pip install -e .

# Train the energy model at build (deterministic seed=42) so the .joblib is
# baked into the image and freshly pickled with the installed sklearn 1.7.0.
RUN python run_pipeline.py

# Built frontend from stage 1.
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Railway sets $PORT; main() binds to it (falls back to 8000 locally).
EXPOSE 8000
CMD ["python", "dashboard/server.py"]
