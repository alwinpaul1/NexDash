# Example Conversation 1 — A Comfortable Reach

This transcript shows a typical interaction with the NexDash **DispatcherAgent**
(`nexdash.agent.DispatcherAgent`). The agent is a fleet dispatcher assistant for
the Mercedes-Benz eActros 600. It always uses tools for any numeric claim and then
explains the result in plain language, including the safety margin and a caveat.

The leg here sits comfortably inside the model's trained envelope: a moderate
distance, mid payload, gentle grade, and only mild cold. The verdict comes back
**high-confidence** — and, candidly, this leg also shows that point-wise model
error varies: the data-driven estimate and the first-principles physics cross-check
differ by ~24 kWh (more than the 6 kWh headline MAE), yet the ~100 kWh margin makes
the reach verdict robust either way.

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
  "energy_needed_kwh": 196.754,
  "energy_available_kwh": 360.0,
  "usable_after_reserve_kwh": 300.0,
  "reaches": true,
  "margin_kwh": 103.246,
  "remaining_soc_pct": 27.21,
  "remaining_range_km": 63.0,
  "confidence": "high",
  "model_kwh": 196.754,
  "physics_kwh": 220.815,
  "confidence_note": "Estimate from a HistGradientBoosting energy model whose held-out mean absolute error is about +/-6 kWh (physics cross-check agrees within 24 kWh). Treat margins smaller than this band as uncertain and keep the safety reserve."
}
```

> **How to read this:** battery is 600 kWh usable, so 60% SOC = 360 kWh available.
> Holding back a 10% reserve (60 kWh) leaves 300 kWh usable. The model predicts the
> trip needs ~197 kWh, leaving a **103 kWh margin** — the truck arrives with a
> comfortable buffer.
>
> **On the confidence band — and an honest note:** the note reports the model's
> *real* held-out MAE (about ±6 kWh, read live from the artifact, not a hardcoded
> guess). It also runs a first-principles physics cross-check on every call. Here the
> data-driven model (197 kWh) and the physics estimate (221 kWh) differ by **~24 kWh**
> — larger than the 6 kWh *average* test error, a reminder that point-wise error is
> not uniform (see the report's failure-mode section). The gap is still inside the
> "flag if they diverge" threshold, so the verdict stays **high confidence**. Crucially
> the decision is robust to the disagreement: even on the *more pessimistic* physics
> number (221 kWh), there is still a ~79 kWh margin, so this reach is not a knife-edge
> call.

---

**Agent (turn 2 — final plain-language answer):**

> Yes — NX-07 reaches München with room to spare.
>
> At 60% SOC you have **360 kWh** on board. Keeping a 10% safety reserve, that's
> **300 kWh usable**. The 120 km run at 80 km/h with 12 t of payload is predicted to
> draw **~197 kWh**, leaving a **margin of ~103 kWh** above your usable budget. You
> should pull in at roughly **27% SOC**, with an estimated **~63 km** of remaining
> range after arrival.
>
> **Caveat:** this is a model estimate (typical error ±6 kWh). On this particular leg
> a first-principles physics check comes out ~24 kWh *higher* than the model — point-
> wise error varies more than the average — but even on that more cautious number you
> keep a ~79 kWh margin, so you are nowhere near the reserve. If the headwind stiffens
> or traffic forces a lot of stop-start, real consumption could climb further; the
> margin absorbs it comfortably, and I'd re-check if conditions change materially.
