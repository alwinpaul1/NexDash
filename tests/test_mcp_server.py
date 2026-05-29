"""Tests for :mod:`nexdash.mcp_server`.

These tests verify the FastMCP wiring *without* starting the stdio server and
*without* loading a trained model:

* the module imports cleanly;
* a module-level ``FastMCP`` instance named ``"nexdash"`` exists;
* both intended tools (``predict_energy`` and ``check_reachability``) are
  registered in the FastMCP tool registry and are introspectable;
* each registered tool is callable and delegates to :mod:`nexdash.tools`
  (the heavy model/range layer is monkeypatched so no ``.joblib`` model is
  needed and no network/disk model load happens).

The FastMCP registry is introspected via the documented ``list_tools``
coroutine and, as a defensive cross-check, via the internal
``_tool_manager``. Both are exercised so the test still encodes intent
(``the two tools are exposed``) even if one access path changes.

Why these assertions matter (Rule 9): the MCP server is the integration
surface MCP-aware clients (Claude Desktop, IDEs) bind to. If the instance
name drifts, or a tool fails to register, or a registered tool no longer
forwards to ``nexdash.tools``, every external client silently loses the
capability. Each assertion guards one of those contract points.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _registered_tool_names(mcp) -> set[str]:
    """Return the set of tool names registered on a FastMCP instance.

    Uses the public ``list_tools`` coroutine (the same surface MCP clients
    see). The server is never started; we only introspect the registry.
    """
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def _tool_by_name(mcp, name: str):
    """Fetch the internal FunctionTool object for ``name`` from the registry.

    Goes through the internal ``_tool_manager`` so the test can reach the
    underlying Python callable (``.fn``) and invoke it directly without
    spinning up the async MCP call machinery.
    """
    for tool in mcp._tool_manager.list_tools():
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name!r} not found in registry")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mcp_server():
    """Import the module under test once and hand back the ``mcp`` instance."""
    module = importlib.import_module("nexdash.mcp_server")
    return module, module.mcp


# ---------------------------------------------------------------------------
# Import + instance identity
# ---------------------------------------------------------------------------
def test_module_imports(mcp_server):
    """The MCP server module imports without side effects beyond registration."""
    module, _ = mcp_server
    assert module is not None


def test_mcp_instance_exists_and_named_nexdash(mcp_server):
    """A FastMCP instance named exactly ``"nexdash"`` is exposed.

    The instance name is what shows up in client MCP configs; a typo here
    breaks every downstream ``mcpServers["nexdash"]`` binding.
    """
    from mcp.server.fastmcp import FastMCP

    _, mcp = mcp_server
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "nexdash"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
def test_both_tools_registered(mcp_server):
    """Both contracted tools are present in the FastMCP registry."""
    _, mcp = mcp_server
    names = _registered_tool_names(mcp)
    assert "predict_energy" in names, f"missing predict_energy; have {names}"
    assert "check_reachability" in names, f"missing check_reachability; have {names}"


def test_registry_access_paths_agree(mcp_server):
    """Public ``list_tools`` and internal ``_tool_manager`` agree on the set.

    Cross-checking the two access paths makes the test robust to which one
    the FastMCP version favours while still asserting the same intent.
    """
    _, mcp = mcp_server
    public = _registered_tool_names(mcp)
    internal = {t.name for t in mcp._tool_manager.list_tools()}
    assert {"predict_energy", "check_reachability"} <= public
    assert public == internal


@pytest.mark.parametrize("tool_name", ["predict_energy", "check_reachability"])
def test_tool_has_description_and_schema(mcp_server, tool_name):
    """Each tool advertises a non-empty description and an input schema.

    MCP clients render these to the user/LLM; empty metadata would make the
    tool effectively undiscoverable even though it is registered.
    """
    _, mcp = mcp_server
    tool = _tool_by_name(mcp, tool_name)
    assert tool.description and tool.description.strip()
    # FunctionTool stores the JSON input schema under ``parameters``.
    schema = tool.parameters
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "properties" in schema


# ---------------------------------------------------------------------------
# Tools are callable and delegate to nexdash.tools
# ---------------------------------------------------------------------------
def test_predict_energy_tool_callable_and_delegates(mcp_server, monkeypatch):
    """The registered ``predict_energy`` tool runs and forwards to nexdash.tools.

    We monkeypatch ``nexdash.tools.predict_energy`` (the model loader) so no
    trained ``.joblib`` is required. Invoking the registered tool's ``.fn``
    must therefore produce a JSON-serializable dict carrying that stub value.
    """
    from nexdash import tools as nexdash_tools

    monkeypatch.setattr(nexdash_tools, "predict_energy", lambda *a, **k: 42.0)

    _, mcp = mcp_server
    tool = _tool_by_name(mcp, "predict_energy")
    result = tool.fn(
        distance_km=50,
        payload_t=10,
        speed_kph=70,
        gradient_pct=1.0,
        temperature_c=15,
    )
    assert isinstance(result, dict)
    assert result["energy_kwh"] == pytest.approx(42.0)
    assert result["inputs"]["distance_km"] == pytest.approx(50.0)
    # default wind_mps must flow through to the underlying call.
    assert result["inputs"]["wind_mps"] == pytest.approx(0.0)


def test_check_reachability_tool_callable_and_delegates(mcp_server, monkeypatch):
    """The registered ``check_reachability`` tool runs and forwards to nexdash.tools.

    ``nexdash.tools.check_reachability`` is monkeypatched to a deterministic
    stub so the test exercises only the MCP-server -> tools delegation, not
    the physics/ML stack or any model file on disk.
    """
    from nexdash import tools as nexdash_tools

    sentinel = {
        "reaches": True,
        "energy_needed_kwh": 80.0,
        "margin_kwh": 12.0,
    }

    captured = {}

    def fake_check(**kwargs):
        captured.update(kwargs)
        return dict(sentinel)

    monkeypatch.setattr(nexdash_tools, "check_reachability", fake_check)

    _, mcp = mcp_server
    tool = _tool_by_name(mcp, "check_reachability")
    result = tool.fn(
        soc_pct=80,
        distance_km=120,
        payload_t=15,
        speed_kph=80,
        gradient_pct=2.0,
        temperature_c=-5,
    )
    assert result == sentinel
    # The reserve default contracted on the server (10.0) must reach the layer.
    assert captured["reserve_pct"] == pytest.approx(10.0)
    assert captured["soc_pct"] == pytest.approx(80.0)


if __name__ == "__main__":  # pragma: no cover - manual invocation convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
