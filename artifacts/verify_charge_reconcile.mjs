// Unit-tests the REAL reconcileChargeDurations() exported from routePlanner.js:
// a charge at a station below the truck's max power must run longer, shifting the
// charge segment, all later segments, the ETA/total/charging time, and any
// delivery stop AFTER the charge (recomputing on-time) — while a full-power
// station changes nothing and a stop BEFORE the charge is untouched.
import { reconcileChargeDurations } from "../frontend/src/lib/routePlanner.js";

let failures = 0;
const eq = (label, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) failures++;
  console.log(`  ${ok ? "✓" : "✗"} ${label}: got ${JSON.stringify(got)}${ok ? "" : ` — want ${JSON.stringify(want)}`}`);
};

function basePlan(stationKw, stops) {
  return {
    segments: [
      { type: "drive", startTime: "12:00", endTime: "16:15", durationMin: 255, socStart: 100, socEnd: 19 },
      { type: "charge", startTime: "16:15", endTime: "17:31", durationMin: 76, socStart: 19, socEnd: 95, kWh: 457, costEur: 206 },
      { type: "drive", startTime: "17:31", endTime: "21:14", durationMin: 223, socStart: 95, socEnd: 30 },
    ],
    chargingStops: [{ index: 0, distKm: 325, maxPowerKw: stationKw, chargeMinutes: 102, durationMin: 76, kWh: 457 }],
    stops,
    summary: { etaLabel: "21:14", etaIso: "2026-06-02T21:14", totalTimeH: 9.23, chargingTimeMin: 76, drivingTimeH: 7.97, assumptions: [] },
    maxChargeKw: 400,
    departure: "2026-06-02T12:00",
  };
}

// --- Scenario 1: 300 kW station (below 400 cap) -> +25 min ----------------- //
console.log("Scenario 1 — 300 kW station (charge 76 -> ~101 min, ETA +25):");
{
  const p = basePlan(300, [
    { index: 0, distKm: 608, etaLabel: "21:14", etaIso: "2026-06-02T21:14", deliverBy: "2026-06-02T21:00", onTime: true, isFinal: true },
  ]);
  reconcileChargeDurations(p);
  eq("charge segment durationMin", p.segments[1].durationMin, 101);
  eq("charge segment endTime", p.segments[1].endTime, "17:56");
  eq("post-charge drive startTime", p.segments[2].startTime, "17:56");
  eq("post-charge drive endTime", p.segments[2].endTime, "21:39");
  eq("summary etaLabel", p.summary.etaLabel, "21:39");
  eq("summary chargingTimeMin", p.summary.chargingTimeMin, 101);
  eq("summary totalTimeH", p.summary.totalTimeH, 9.65);
  eq("station chargeMinutes (card in sync)", p.chargingStops[0].chargeMinutes, 101);
  eq("stop etaLabel shifted", p.stops[0].etaLabel, "21:39");
  eq("stop onTime recomputed (now late)", p.stops[0].onTime, false);
  eq("assumption appended", p.summary.assumptions.length, 1);
}

// --- Scenario 2: 400 kW station (== cap) -> no change ----------------------- //
console.log("Scenario 2 — 400 kW station (== cap, nothing changes):");
{
  const p = basePlan(400, [
    { index: 0, distKm: 608, etaLabel: "21:14", etaIso: "2026-06-02T21:14", deliverBy: "2026-06-02T21:00", onTime: true, isFinal: true },
  ]);
  reconcileChargeDurations(p);
  eq("charge durationMin unchanged", p.segments[1].durationMin, 76);
  eq("etaLabel unchanged", p.summary.etaLabel, "21:14");
  eq("chargingTimeMin unchanged", p.summary.chargingTimeMin, 76);
  eq("stop onTime unchanged", p.stops[0].onTime, true);
  eq("no assumption added", p.summary.assumptions.length, 0);
}

// --- Scenario 3: charge delta pushes ETA past midnight --------------------- //
console.log("Scenario 3 — midnight rollover (200 kW, 50 -> 100 min):");
{
  const p = {
    segments: [
      { type: "drive", startTime: "20:00", endTime: "23:00", durationMin: 180 },
      { type: "charge", startTime: "23:00", endTime: "23:50", durationMin: 50 },
      { type: "drive", startTime: "23:50", endTime: "00:30", durationMin: 40 },
    ],
    chargingStops: [{ index: 0, distKm: 200, maxPowerKw: 200, durationMin: 50 }],
    stops: [],
    summary: { etaLabel: "00:30", etaIso: "2026-06-03T00:30", totalTimeH: 4.5, chargingTimeMin: 50, assumptions: [] },
    maxChargeKw: 400,
    departure: "2026-06-02T20:00",
  };
  reconcileChargeDurations(p);
  eq("charge endTime (past midnight)", p.segments[1].endTime, "00:40");
  eq("final drive startTime", p.segments[2].startTime, "00:40");
  eq("ETA past midnight", p.summary.etaLabel, "01:20");
  eq("chargingTimeMin", p.summary.chargingTimeMin, 100);
}

// --- Scenario 4: a stop BEFORE the charge must NOT shift -------------------- //
console.log("Scenario 4 — pre-charge stop stays put, post-charge stop shifts:");
{
  const p = basePlan(300, [
    { index: 0, distKm: 200, etaLabel: "14:30", etaIso: "2026-06-02T14:30", deliverBy: null, onTime: null, isFinal: false },
    { index: 1, distKm: 608, etaLabel: "21:14", etaIso: "2026-06-02T21:14", deliverBy: "2026-06-02T21:00", onTime: true, isFinal: true },
  ]);
  reconcileChargeDurations(p);
  eq("pre-charge stop ETA unchanged", p.stops[0].etaLabel, "14:30");
  eq("post-charge stop ETA shifted", p.stops[1].etaLabel, "21:39");
}

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
