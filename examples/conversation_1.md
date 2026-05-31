# Example Conversation 1 — A Comfortable Reach

This transcript shows a typical interaction with the NexDash **DispatcherAgent**
(`nexdash.agent.DispatcherAgent`). The agent is a fleet dispatcher assistant for
the Mercedes-Benz eActros 600. It always uses tools for any numeric claim and then
explains the result in plain language, including the safety margin and a caveat.

The leg here sits comfortably inside the model's trained envelope: a moderate
distance, mid payload, gentle grade, and only mild cold. Every estimate the model
makes is well-supported by training data, so the verdict comes back high-confidence
and the data-driven and physics estimates agree almost exactly.

> The tool-result JSON below is copied verbatim from a live
> `check_reachability` call with the exact inputs shown (no API key needed to
> reproduce it):
> `PYTHONPATH=src python -c "from nexdash.range import check_reachability; import json; print(json.dumps(check_reachability(soc_pct=60, distance_km=120, payload_t=12, speed_kph=80, gradient_pct=0.5, temperature_c=2, wind_mps=3), indent=2))"`.

---

**User:**
> Truck NX-07 is at 60% SOC carrying 12 tonnes. Can it reach München? It's about
> 120 km on the autobahn, mostly flat, cruising around 80 km/h. It's a chilly
> morning, about 2 °C with a light headwind.

---

**Agent (turn 1 — issues a tool call):**

The agent decides it needs real numbers and calls `check_reachability`.

```json
{
  "type": "tool_use",
  "name": "check_reachability",
  "input": {
    "soc_pct": 60,
    "distance_km": 120,
    "payload_t": 12,
    "speed_kph": 80,
    "gradient_pct": 0.5,
    "temperature_c": 2,
    "wind_mps": 3.0
  }
}
```

---

**Tool result (`check_reachability` returns JSON — verbatim from the live tool):**

```json
{
  "energy_needed_kwh": 211.605,
  "energy_available_kwh": 360.0,
  "usable_after_reserve_kwh": 300.0,
  "reaches": true,
  "margin_kwh": 88.395,
  "remaining_soc_pct": 24.73,
  "remaining_range_km": 50.1,
  "confidence": "high",
  "model_kwh": 211.605,
  "physics_kwh": 212.273,
  "confidence_note": "Estimate from a HistGradientBoosting energy model whose held-out mean absolute error is about +/-6 kWh (physics cross-check agrees within 1 kWh). Treat margins smaller than this band as uncertain and keep the safety reserve."
}
```

> **How to read this:** battery is 600 kWh usable, so 60% SOC = 360 kWh available.
> Holding back a 10% reserve (60 kWh) leaves 300 kWh usable. The model predicts the
> trip needs ~212 kWh, leaving an **88 kWh margin** — the truck arrives with a
> comfortable buffer.
>
> **On the confidence band:** the note reports the model's *real* held-out MAE
> (about ±6 kWh, read live from the model artifact, not a hardcoded guess). It also
> runs a first-principles physics cross-check on every call: here the data-driven
> model (211.6 kWh) and the physics estimate (212.3 kWh) agree within ~1 kWh, so the
> verdict is flagged **high confidence**. The 88 kWh margin dwarfs the ~6 kWh error
> band many times over, so this reach is not a knife-edge call.

---

**Agent (turn 2 — final plain-language answer):**

> Yes — NX-07 reaches München with room to spare.
>
> At 60% SOC you have **360 kWh** on board. Keeping a 10% safety reserve, that's
> **300 kWh usable**. The 120 km run at 80 km/h with 12 t of payload is predicted to
> draw **~212 kWh**. The chilly 2 °C morning and light headwind nudge consumption up
> a little, but the trip still leaves a **margin of ~88 kWh** above your usable
> budget.
>
> You should pull in at roughly **25% SOC**, with an estimated **~50 km** of remaining
> range after arrival.
>
> **Caveat:** this is a model estimate, with a typical error of about ±6 kWh — and an
> independent physics cross-check agrees with the model to within ~1 kWh here, so it's
> a high-confidence result. That error band is tiny next to your 88 kWh margin, so
> you're nowhere near the reserve. If the headwind stiffens or traffic forces a lot of
> stop-start, real consumption could climb a few kWh — but the margin absorbs it
> comfortably.
