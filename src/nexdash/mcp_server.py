"""FastMCP server exposing NexDash EV-truck range intelligence as MCP tools.

This module wraps the deterministic/ML prediction layer of NexDash (the
Mercedes-Benz eActros 600 energy model) as a Model Context Protocol (MCP)
server so that MCP-aware clients (Claude Desktop, IDEs, custom agents) can call
``predict_energy`` and ``check_reachability`` directly.

The two exposed tools are thin delegations to :mod:`nexdash.tools`, which in
turn call :func:`nexdash.model.predict_energy` and
:func:`nexdash.range.check_reachability`. Keeping the real logic in
``nexdash.tools`` means the MCP server, the in-process Anthropic agent, and the
FastAPI dashboard all share one source of truth.

Running the server
------------------
The server speaks MCP over stdio (the default transport for ``FastMCP.run``)::

    python -m nexdash.mcp_server

Registering in an MCP client config
-----------------------------------
Add an entry to your client's MCP server configuration. For Claude Desktop the
file is ``claude_desktop_config.json`` (macOS:
``~/Library/Application Support/Claude/claude_desktop_config.json``)::

    {
      "mcpServers": {
        "nexdash": {
          "command": "python",
          "args": ["-m", "nexdash.mcp_server"],
          "env": {
            "PYTHONPATH": "/absolute/path/to/NexDash/src"
          }
        }
      }
    }

If the ``nexdash`` package is installed into the active environment (e.g. via
``pip install -e .``) the ``env``/``PYTHONPATH`` block can be omitted and you
can point ``command`` at that environment's interpreter. The trained model must
exist on disk first; run ``python run_pipeline.py`` once to generate it.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from nexdash import tools

mcp = FastMCP("nexdash")


@mcp.tool()
def predict_energy(
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    wind_mps: float = 0.0,
) -> dict:
    """Predict energy consumption (kWh) for one eActros 600 trip segment.

    Uses the trained NexDash energy model to estimate how many kilowatt-hours
    the truck will draw from its battery over a single segment.

    Args:
        distance_km: Segment length in kilometres (typ. 1-120).
        payload_t: Cargo payload in tonnes (0-22).
        speed_kph: Average travel speed in km/h (30-90).
        gradient_pct: Net road gradient in percent; negative is downhill.
        temperature_c: Ambient temperature in degrees Celsius (-15 to 40).
        wind_mps: Headwind component in metres/second (default 0.0).

    Returns:
        JSON-serializable dict with the predicted energy (kWh) and the echoed
        input features.
    """
    try:
        return tools.predict_energy_tool(
            distance_km=distance_km,
            payload_t=payload_t,
            speed_kph=speed_kph,
            gradient_pct=gradient_pct,
            temperature_c=temperature_c,
            wind_mps=wind_mps,
        )
    except Exception as exc:  # noqa: BLE001 - MCP boundary: an out-of-range arg must
        # return a structured error to the client, not crash the tool call (mirrors
        # the in-process agent._run_tool contract).
        return {"error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def check_reachability(
    soc_pct: float,
    distance_km: float,
    payload_t: float,
    speed_kph: float,
    gradient_pct: float,
    temperature_c: float,
    wind_mps: float = 0.0,
    reserve_pct: float = 10.0,
) -> dict:
    """Decide whether an eActros 600 can reach a destination on current charge.

    Combines the energy prediction with the current state of charge and a safety
    reserve to determine reachability, then reports the energy margin and the
    estimated remaining state of charge / range on arrival.

    Args:
        soc_pct: Current battery state of charge in percent (0-100).
        distance_km: Trip distance in kilometres.
        payload_t: Cargo payload in tonnes (0-22).
        speed_kph: Average travel speed in km/h (30-90).
        gradient_pct: Net road gradient in percent; negative is downhill.
        temperature_c: Ambient temperature in degrees Celsius (-15 to 40).
        wind_mps: Headwind component in metres/second (default 0.0).
        reserve_pct: Battery reserve to keep untouched, in percent
            (default 10.0).

    Returns:
        JSON-serializable dict with ``reaches`` (bool), energy needed/available,
        usable energy after reserve, margin, remaining SOC/range estimates, and
        a confidence note referencing the model's error band.
    """
    try:
        return tools.check_reach_tool(
            soc_pct=soc_pct,
            distance_km=distance_km,
            payload_t=payload_t,
            speed_kph=speed_kph,
            gradient_pct=gradient_pct,
            temperature_c=temperature_c,
            wind_mps=wind_mps,
            reserve_pct=reserve_pct,
        )
    except Exception as exc:  # noqa: BLE001 - MCP boundary: e.g. speed<=0 now raises a
        # clear ValueError in check_reachability; return it as a structured error
        # rather than letting it crash the MCP tool call.
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    mcp.run()
