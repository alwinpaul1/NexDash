"""Model registry + provenance for NexDash — the long-term-thinking backbone.

This turns ``docs/LONG_TERM.md`` section 5 ("content-addressed versioning +
lineage") from prose into working code. Every trained model gets a record of:

* **what produced it** — a SHA-256 of the training data plus the code (git) SHA
  and seed, combined into a content-addressed ``model_version``; and
* **how good it was** — the full held-out metrics and failure-mode slices.

That lineage is what makes the promotion gate (:mod:`nexdash.promote`) and drift
monitor (:mod:`nexdash.drift`) meaningful: you cannot prove a new version is
genuinely better, or roll back to a known-good one, if you cannot identify which
data + code produced an artifact.

Design choice — provenance is a *sidecar*, never inside the joblib
--------------------------------------------------------------------
The trained ``energy_model.joblib`` payload is left byte-for-byte unchanged so
the pipeline stays exactly reproducible across runs (a property the evaluation
relies on). Provenance lives in JSON next to the artifact (``<model>.provenance.json``)
and, for history, in ``models/registry/<model_version>.json``. The ``trained_at``
timestamp therefore never perturbs the model bytes. All functions are fail-soft
and fully offline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .config import MODELS_DIR

__all__ = [
    "REGISTRY_DIR",
    "dataset_sha256",
    "git_sha",
    "make_version",
    "build_provenance",
    "write_sidecar",
    "read_sidecar",
    "register",
    "list_registry",
]

#: Directory of historical registry entries (one JSON per ``model_version``).
REGISTRY_DIR: Path = MODELS_DIR / "registry"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_sha256(path: Union[str, Path]) -> str:
    """Return the SHA-256 hex digest of the dataset file, or ``"unknown"``.

    Streamed so a large CSV never loads fully into memory. Fail-soft: a missing
    or unreadable file yields ``"unknown"`` rather than raising, because
    provenance must never be the thing that breaks a training run.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unknown"


def git_sha() -> str:
    """Return the current git commit SHA, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def make_version(data_hash: str, code_sha: str) -> str:
    """Compose a short, content-addressed model version from data + code hashes.

    Deterministic for a fixed (training data, code) state — the same data and
    commit always yield the same version, so re-registering overwrites rather
    than accumulates.
    """
    return f"{(data_hash or 'nodata')[:12]}-{(code_sha or 'nogit')[:8]}"


def build_provenance(
    dataset_path: Union[str, Path],
    metrics: dict[str, Any],
    *,
    seed: int = 42,
    failure_modes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a provenance record for a freshly trained model.

    Args:
        dataset_path: Path to the training dataset CSV (hashed for lineage).
        metrics: The model's held-out metrics dict (e.g. ``EnergyModel.metrics``).
        seed: The training seed.
        failure_modes: Optional failure-mode slice report to store alongside.

    Returns:
        A JSON-serializable dict with ``model_version``, ``dataset_sha256``,
        ``git_sha``, ``seed``, ``trained_at`` (UTC ISO), ``metrics`` and
        ``failure_modes``.
    """
    data_hash = dataset_sha256(dataset_path)
    code_sha = git_sha()
    return {
        "model_version": make_version(data_hash, code_sha),
        "dataset_sha256": data_hash,
        "git_sha": code_sha,
        "seed": int(seed),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "failure_modes": failure_modes,
    }


def _sidecar_path(model_path: Union[str, Path]) -> Path:
    """The provenance sidecar path that sits next to a model artifact."""
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".provenance.json")


def write_sidecar(model_path: Union[str, Path], provenance: dict[str, Any]) -> Path:
    """Write the provenance JSON next to the model artifact; return its path."""
    sidecar = _sidecar_path(model_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def read_sidecar(model_path: Union[str, Path]) -> Optional[dict[str, Any]]:
    """Read the provenance sidecar for a model artifact, or ``None`` (fail-soft)."""
    try:
        return json.loads(_sidecar_path(model_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def register(
    provenance: dict[str, Any], *, registry_dir: Union[str, Path] = REGISTRY_DIR
) -> Path:
    """Persist a provenance record into the content-addressed registry history.

    The entry is written to ``<registry_dir>/<model_version>.json``. Because the
    version is content-addressed, re-registering the same data+code overwrites
    the same file rather than accumulating duplicates.
    """
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    version = str(provenance.get("model_version", "unknown"))
    entry_path = registry_dir / f"{version}.json"
    entry_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return entry_path


def list_registry(*, registry_dir: Union[str, Path] = REGISTRY_DIR) -> list[dict[str, Any]]:
    """Return all registry entries (newest ``trained_at`` first), fail-soft."""
    d = Path(registry_dir)
    if not d.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for p in d.glob("*.json"):
        try:
            entries.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    entries.sort(key=lambda e: str(e.get("trained_at", "")), reverse=True)
    return entries
