# Example Conversation 2 — Borderline / Failure Case (Charging Stop Advised)

This transcript shows the **DispatcherAgent** (`nexdash.agent.DispatcherAgent`)
handling a hard case for the Mercedes-Benz eActros 600: a long leg with a **heavy
payload**, a **sustained steep climb**, and **deep cold**. These three factors stack —
mass drives both rolling resistance and the gradient (potential-energy) term, while
the cold raises HVAC/battery-heating load. The result is a trip the truck cannot
finish on its current charge, and the agent advises a charging stop rather than
papering over the shortfall.

---

**User:**
> NX-11 is at 45% SOC, fully loaded at 22 tonnes. I need it to do a 110 km mountain
> leg toward the Alps — there's a long pull averaging about +4.5% gradient — at
> 70 km/h. It's –12 °C up there. Does it make it, or do I need to stage a charge?

---

**Agent (turn 1 — issues a tool call):**

The agent gathers concrete numbers before answering, calling `check_reachability`.

```json
{
  "type": "tool_use",
  "name": "check_reachability",
  "input": {
    "soc_pct": 45,
    "distance_km": 110,
    "payload_t": 22,
    "speed_kph": 70,
    "gradient_pct": 4.5,
    "temperature_c": -12,
    "wind_mps": 0.0
  }
}
```

---

**Tool result (`check_reachability` returns JSON):**

```json
{
  "energy_needed_kwh": 281.4,
  "energy_available_kwh": 270.0,
  "usable_after_reserve_kwh": 210.0,
  "reaches": false,
  "margin_kwh": -71.4,
  "remaining_soc_pct": -1.9,
  "remaining_range_km": 0.0,
  "confidence_note": "Estimate from energy model (HistGradientBoosting), MAE ~3.1 kWh; shortfall far exceeds the error band, so the 'no-go' is confident.",
  "model": "HistGradientBoostingRegressor"
}
```

> **How to read this:** 45% SOC = 270 kWh raw, but after the 10% reserve only **210 kWh**
> is usable. The model predicts **~281 kWh** for this leg — the 22 t mass on a sustained
> +4.5% climb dominates (lifting 40 t of GVW uphill is expensive), and –12 °C piles on
> heating load. The shortfall (**~71 kWh**) is more than twenty times the model's typical
> error, so this isn't a close call the uncertainty could rescue.

---

**Agent (turn 2 — final plain-language answer):**

> No — NX-11 will not make the full 110 km mountain leg on its current charge. **Stage a
> charge before the climb.**
>
> At 45% SOC the truck has 270 kWh, but with the 10% safety reserve only **210 kWh** is
> usable. This leg is predicted to need **~281 kWh** — the full 22 t load on a long
> +4.5% pull is the big driver (you're hauling ~40 t of gross weight uphill), and the
> –12 °C cold adds significant heating load on top. That leaves a **shortfall of ~71 kWh**:
> the truck would run into its reserve well before the summit, around the **last stretch
> of the climb**, not at the destination.
>
> **Recommendation:** top up before you start. To cover ~281 kWh plus the reserve you want
> the battery near **~55–60% SOC at the trailhead** (roughly +10–15 percentage points over
> where you are now). A short fast charge at a depot or en-route charger before the grade
> turns this from a no-go into a comfortable run. If charging before the climb isn't
> possible, split the leg or reduce payload.
>
> **Caveat:** these are model estimates (typical error ~3 kWh). But here the gap is ~71 kWh
> — far outside any uncertainty band — so the "won't make it" verdict is firm, not marginal.
> Heavy load, steep sustained grade, and deep cold are exactly the conditions where range
> drops fastest, so don't gamble the reserve on this one.
