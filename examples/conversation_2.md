# Example Conversation 2 — Honest No-Go via the Safety Mechanism

This transcript shows the **DispatcherAgent** (`nexdash.agent.DispatcherAgent`)
handling a hard case for the Mercedes-Benz eActros 600: a long leg with a **heavy
payload**, a **sustained steep climb**, and **deep cold**. These three factors stack —
mass drives both rolling resistance and the gradient (potential-energy) term, while
the cold raises HVAC/battery-heating load.

This case is deliberately chosen because it lands **outside the model's trained
envelope**. The data generator caps each segment's net climb at ~1000 m — a sustained
+4.5% grade simply cannot persist over 110 km in real terrain (that would be a ~5 km net
climb, higher than any German road), so no such segment appears in training. Asked to
predict one anyway, the data-driven model **under-predicts badly** (it has never seen
this combination and reverts toward the bulk of its training mass). On its own that is
a dangerous, optimistic error.

The system catches it. On every call, `check_reachability` runs a first-principles
physics cross-check. When the data-driven model and physics disagree sharply, it flags
**low confidence**, falls back to the **conservative (higher) physics value** for the
GO/NO-GO, and says so. Here that mechanism turns a silent under-prediction into a
correct, safe "no" — the system defending against its own extrapolation.

> The tool-result JSON below is copied verbatim from a live `check_reachability`
> call with the exact inputs shown. Reproduce with:
> `PYTHONPATH=src python -c "from nexdash.range import check_reachability; import json; print(json.dumps(check_reachability(soc_pct=80, distance_km=110, payload_t=22, speed_kph=70, gradient_pct=4.5, temperature_c=-12), indent=2))"`.

---

**User:**
> NX-11 is at 80% SOC, fully loaded at 22 tonnes. I need it to do a 110 km mountain
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
    "soc_pct": 80,
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

**Tool result (`check_reachability` returns JSON — verbatim from the live tool):**

```json
{
  "energy_needed_kwh": 782.189,
  "energy_available_kwh": 480.0,
  "usable_after_reserve_kwh": 420.0,
  "reaches": false,
  "margin_kwh": -362.189,
  "remaining_soc_pct": 0.0,
  "remaining_range_km": 0.0,
  "confidence": "low",
  "model_kwh": 256.642,
  "physics_kwh": 782.189,
  "confidence_note": "LOW CONFIDENCE: the data-driven model (257 kWh) and a first-principles physics estimate (782 kWh) disagree sharply, which means this segment is outside the envelope the model was trained on. The more conservative (higher) estimate of 782 kWh (physics) is used for this decision; treat it as indicative only and keep a wide reserve."
}
```

> **How to read this — and why the two numbers differ.** 80% SOC = 480 kWh raw, but
> after the 10% reserve only **420 kWh** is usable. Two estimates are reported:
> `model_kwh` (257 kWh) is the data-driven model, and `physics_kwh` (782 kWh) is the
> first-principles cross-check. They disagree by 3×.
>
> That disagreement is the signal. A sustained +4.5% grade over 110 km is a corner the
> data generator never produces (it caps the per-segment net climb at ~1000 m, so such
> a grade cannot hold over that distance), so the data-driven model is extrapolating and
> lands **optimistically low** at 257 kWh. The physics term is dominated by lifting
> ~40 t of gross weight up a 4.5% grade for 110 km — that alone is several hundred kWh —
> so 782 kWh is the mechanistically defensible figure.
>
> Because the gap is far wider than the model's normal ±6 kWh band, the tool flags
> **`confidence: "low"`** and uses the **conservative 782 kWh** for the verdict. Note
> the exact danger this defuses: the model alone says just **257 kWh — comfortably
> *under* the 420 kWh usable budget**, so trusted blindly it would optimistically
> green-light the trip. The physics estimate (782 kWh) blows past the budget, and
> because the tool falls back to the conservative value when the two disagree this
> sharply, the **no-go is firm**. This is the system catching its own under-prediction
> rather than quietly shipping a dangerous "yes".

---

**Agent (turn 2 — final plain-language answer):**

> No — NX-11 will not make the full 110 km mountain leg on its current charge. **Stage a
> charge before the climb.**
>
> At 80% SOC the truck has 480 kWh, but with the 10% safety reserve only **420 kWh** is
> usable. This leg sits outside what the model was trained on — a sustained +4.5% pull
> over 110 km doesn't occur in normal terrain — so I don't fully trust the model's raw
> number. To stay safe, the tool falls back to a first-principles physics estimate of
> **~782 kWh** for lifting ~40 t of gross weight up that grade in –12 °C, and flags the
> result as low confidence.
>
> **Why I'm confident in the "no" even though I distrust the model here.** The model's
> own estimate is ~257 kWh — which is actually *below* your 420 kWh usable budget, so on
> its own it would have said "yes". That's the trap: this leg is an out-of-envelope
> extrapolation, and the model under-predicts it badly. The first-principles physics
> estimate is ~782 kWh, far above the budget, and the tool falls back to that conservative
> number when the two disagree this sharply — so the **"won't make it" conclusion is firm**,
> and the truck would hit its reserve well before the summit.
>
> **Recommendation:** top up before you start, or split the leg / reduce payload. A short
> fast charge at a depot or en-route charger before the grade turns this from a no-go into
> a manageable run. Heavy load, steep sustained grade, and deep cold are exactly the
> conditions where range drops fastest — and exactly the kind of edge case where I lean on
> the conservative estimate rather than gamble the reserve.
