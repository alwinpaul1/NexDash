# Example Conversation 1 — Reachability with a Cold-Weather Twist

This transcript shows a typical interaction with the NexDash **DispatcherAgent**
(`nexdash.agent.DispatcherAgent`). The agent is a fleet dispatcher assistant for
the Mercedes-Benz eActros 600. It always uses tools for any numeric claim and then
explains the result in plain language, including the safety margin and a caveat.

The user's question involves a cold morning, so auxiliary/HVAC (cabin + battery
heating) load is higher than usual — a detail the physics-trained model captures
and the agent surfaces in its answer.

---

**User:**
> Truck NX-07 is at 60% SOC carrying 12 tonnes. Can it reach München? It's about
> 95 km on the autobahn, mostly flat, cruising around 80 km/h. One catch: it's a
> cold morning, about -8 °C with a light headwind.

---

**Agent (turn 1 — issues a tool call):**

The agent decides it needs real numbers and calls `check_reachability`.

```json
{
  "type": "tool_use",
  "name": "check_reachability",
  "input": {
    "soc_pct": 60,
    "distance_km": 95,
    "payload_t": 12,
    "speed_kph": 80,
    "gradient_pct": 0,
    "temperature_c": -8,
    "wind_mps": 3.0
  }
}
```

---

**Tool result (`check_reachability` returns JSON):**

```json
{
  "energy_needed_kwh": 134.2,
  "energy_available_kwh": 360.0,
  "usable_after_reserve_kwh": 300.0,
  "reaches": true,
  "margin_kwh": 165.8,
  "remaining_soc_pct": 37.6,
  "remaining_range_km": 117.4,
  "confidence_note": "Estimate from energy model (HistGradientBoosting), MAE ~3.1 kWh; treat the margin as +/- ~1 trip-segment of uncertainty.",
  "model": "HistGradientBoostingRegressor"
}
```

> **How to read this:** battery is 600 kWh usable, so 60% SOC = 360 kWh available.
> Holding back a 10% reserve (60 kWh) leaves 300 kWh usable. The model predicts the
> trip needs ~134 kWh — the cold air pushes HVAC/battery-heating load up, which is why
> consumption is a bit heavier than a mild-weather run of the same distance. Even so,
> the truck arrives with a comfortable buffer.

---

**Agent (turn 2 — final plain-language answer):**

> Yes — NX-07 reaches München with room to spare.
>
> At 60% SOC you have **360 kWh** on board. Keeping a 10% safety reserve, that's
> **300 kWh usable**. The 95 km run at 80 km/h with 12 t of payload is predicted to
> draw **~134 kWh**. The –8 °C morning and light headwind do raise consumption — cabin
> and battery heating alone add a meaningful chunk versus a mild day — but the trip
> still leaves a **margin of ~166 kWh** above your usable budget.
>
> You should pull in at roughly **38% SOC**, with an estimated **~117 km** of remaining
> range after arrival, so a return leg or a depot detour is well within reach.
>
> **Caveat:** this is a model estimate (typical error ~3 kWh, around ±2% on a trip this
> size). If the headwind stiffens, traffic forces a lot of stop-start, or the cabin is
> kept very warm for the full drive, real consumption could climb a few kWh — but you're
> nowhere near the reserve, so the margin absorbs it comfortably.
